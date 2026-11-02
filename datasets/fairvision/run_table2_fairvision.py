# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.

from pathlib import Path
import json
import math
import random
import gc

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = Path("PATH_TO_FAIRVISION_MANIFEST_CSV")

FEATURE_DIR = ROOT / "datasets/fairvision/results/eyeclip_slice_features"

KG_DESC = (
    ROOT
    / "datasets/fairvision/results/kg_128_descriptors.csv"
)

EXISTING_META = Path("PATH_TO_FAIRVISION_BASE_TO_META_PREDICTIONS")

EXISTING_VAL = Path("PATH_TO_FAIRVISION_VAL_PREDICTIONS")

OUT = (
    ROOT
    / "datasets/fairvision/results/table2_seed42"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42

# EXACT finalized FairVision visual configuration.
D_MODEL = 256
N_HEADS = 4
N_BLOCKS = 2
DROPOUT = 0.1
BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-5
VISUAL_EPOCHS = 8
MAX_CANONICAL_Z = 512

# Dataset-specific train-only selection.
KG_C_GRID = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

FUSION_C_GRID = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


# ============================================================
# PRE-FLIGHT -- fail before expensive GPU training
# ============================================================

required_paths = [
    MANIFEST,
    FEATURE_DIR,
    KG_DESC,
    EXISTING_META,
    EXISTING_VAL,
]

for p in required_paths:
    if not p.exists():
        raise FileNotFoundError(
            f"PRE-FLIGHT missing: {p}"
        )

if DEVICE.type != "cuda":
    raise RuntimeError(
        "CUDA unavailable. Refusing expensive visual cross-fit on CPU."
    )


manifest = pd.read_csv(MANIFEST)
kg = pd.read_csv(KG_DESC)
old_meta = pd.read_csv(EXISTING_META)
old_val = pd.read_csv(EXISTING_VAL)

for df, name in [
    (manifest, "manifest"),
    (kg, "KG"),
    (old_meta, "existing meta"),
    (old_val, "existing val"),
]:
    if "case_id" not in df.columns:
        raise RuntimeError(
            f"{name}: missing case_id"
        )
    df["case_id"] = df["case_id"].astype(str)


if len(kg) != 7000:
    raise RuntimeError(
        f"Expected 7000 KG rows, got {len(kg)}"
    )

if kg["case_id"].nunique() != 7000:
    raise RuntimeError(
        "KG case IDs are not unique."
    )

if "split" not in kg.columns or "label" not in kg.columns:
    raise RuntimeError(
        f"KG metadata incomplete: {list(kg.columns[:8])}"
    )

kg["label"] = kg["label"].astype(int)

train6000 = (
    kg[kg["split"] == "train"]
    .copy()
    .reset_index(drop=True)
)

val1000 = (
    kg[kg["split"] == "val"]
    .copy()
    .reset_index(drop=True)
)

if len(train6000) != 6000:
    raise RuntimeError(
        f"Expected train6000, got {len(train6000)}"
    )

if len(val1000) != 1000:
    raise RuntimeError(
        f"Expected val1000, got {len(val1000)}"
    )


# Existing visual schemas.
for df, name, expected_n in [
    (old_meta, "meta1200", 1200),
    (old_val, "val1000", 1000),
]:
    required = {
        "case_id",
        "label",
        "prob_amd",
        "prediction",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{name}: missing {sorted(missing)}; "
            f"columns={list(df.columns)}"
        )

    if len(df) != expected_n:
        raise RuntimeError(
            f"{name}: expected {expected_n}, got {len(df)}"
        )

    if df["case_id"].nunique() != expected_n:
        raise RuntimeError(
            f"{name}: duplicate case IDs"
        )

    df["label"] = df["label"].astype(int)
    df["prob_amd"] = df["prob_amd"].astype(float)

    expected_pred = (
        df["prob_amd"].to_numpy() >= 0.5
    ).astype(int)

    if not np.array_equal(
        expected_pred,
        df["prediction"].to_numpy(int),
    ):
        raise RuntimeError(
            f"{name}: prob_amd/prediction mismatch"
        )


# Official validation must align exactly.
if set(old_val["case_id"]) != set(val1000["case_id"]):
    raise RuntimeError(
        "Existing visual val1000 and KG val1000 case sets differ."
    )


# Labels must align.
tmp = old_val[
    ["case_id", "label"]
].merge(
    val1000[
        ["case_id", "label"]
    ],
    on="case_id",
    suffixes=("_visual", "_kg"),
    validate="one_to_one",
)

if not np.array_equal(
    tmp["label_visual"],
    tmp["label_kg"],
):
    raise RuntimeError(
        "Visual/KG official-val labels differ."
    )


# ============================================================
# Exact 128-D structural representation
# ============================================================

KG_META_COLS = {
    "case_id",
    "split",
    "label",
    "condition",
}

KG_COLS = [
    c
    for c in kg.columns
    if c not in KG_META_COLS
]

if len(KG_COLS) != 128:
    raise RuntimeError(
        f"Expected exact 128-D KG descriptors; got {len(KG_COLS)}"
    )

for c in KG_COLS:
    kg[c] = pd.to_numeric(
        kg[c],
        errors="coerce",
    )

train6000 = (
    kg[kg["split"] == "train"]
    .copy()
    .reset_index(drop=True)
)

val1000 = (
    kg[kg["split"] == "val"]
    .copy()
    .reset_index(drop=True)
)


# ============================================================
# Construct genuine 5-fold OOF partition
#
# Fold 1 = existing clean meta1200 holdout.
# Folds 2-5 = stratified partition of the remaining base4800.
#
# Therefore every one of the 6000 training volumes is predicted
# by a visual/KG model that did NOT train on that volume.
# ============================================================

meta_ids = set(
    old_meta["case_id"]
)

if not meta_ids.issubset(
    set(train6000["case_id"])
):
    raise RuntimeError(
        "Existing meta1200 is not a subset of official train6000."
    )

base4800 = (
    train6000[
        ~train6000["case_id"].isin(meta_ids)
    ]
    .copy()
    .reset_index(drop=True)
)

if len(base4800) != 4800:
    raise RuntimeError(
        f"Expected base4800, got {len(base4800)}"
    )

fold_assignment = train6000[
    ["case_id", "label"]
].copy()

fold_assignment["fold"] = -1

fold_assignment.loc[
    fold_assignment["case_id"].isin(meta_ids),
    "fold",
] = 1

skf4 = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=SEED,
)

for j, (_, hold_idx) in enumerate(
    skf4.split(
        np.zeros(len(base4800)),
        base4800["label"].to_numpy(),
    ),
    start=2,
):
    hold_ids = set(
        base4800.iloc[hold_idx]["case_id"]
    )

    fold_assignment.loc[
        fold_assignment["case_id"].isin(hold_ids),
        "fold",
    ] = j


if (fold_assignment["fold"] < 1).any():
    raise RuntimeError(
        "Incomplete fold assignment."
    )

if fold_assignment["case_id"].nunique() != 6000:
    raise RuntimeError(
        "Fold assignment does not cover 6000 unique cases."
    )

fold_sizes = (
    fold_assignment
    .groupby("fold")
    .size()
    .to_dict()
)

if fold_sizes != {
    1: 1200,
    2: 1200,
    3: 1200,
    4: 1200,
    5: 1200,
}:
    raise RuntimeError(
        f"Unexpected fold sizes: {fold_sizes}"
    )

fold_assignment.to_csv(
    OUT / "crossfit_fold_assignments.csv",
    index=False,
)


# ============================================================
# Frozen EyeCLIP slice features
# ============================================================

def feature_path(case_id):
    return (
        FEATURE_DIR
        / f"{case_id}.npy"
    )


# Cheap existence check before GPU training.
missing_features = [
    cid
    for cid in train6000["case_id"]
    if not feature_path(cid).exists()
]

if missing_features:
    raise RuntimeError(
        "Missing train EyeCLIP files. "
        f"First missing: {missing_features[:10]}"
    )


def load_eyeclip(case_id):
    path = feature_path(case_id)

    x = np.load(
        path
    ).astype(
        np.float32,
        copy=False,
    )

    if (
        x.ndim != 2
        or x.shape[1] != 512
    ):
        raise RuntimeError(
            f"{case_id}: unexpected EyeCLIP shape {x.shape}"
        )

    if not np.isfinite(x).all():
        raise RuntimeError(
            f"{case_id}: non-finite EyeCLIP values"
        )

    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    x = (
        x
        / np.clip(
            norms,
            1e-12,
            None,
        )
    )

    return x.astype(
        np.float32,
        copy=False,
    )


# Cache only mean-pooled vectors (~12 MB), not all slice embeddings.
print("Precomputing 6000 normalized mean-pooled vectors...")

POOLED = {}

for i, case_id in enumerate(
    train6000["case_id"],
    start=1,
):
    POOLED[case_id] = (
        load_eyeclip(case_id)
        .mean(axis=0)
        .astype(np.float32)
    )

    if i % 1000 == 0:
        print(
            f"  pooled {i}/6000"
        )


def fit_visual_scaler(frame):
    pooled = np.stack([
        POOLED[str(case_id)]
        for case_id
        in frame["case_id"]
    ])

    mu = (
        pooled.mean(axis=0)
        .astype(np.float32)
    )

    sigma = (
        pooled.std(axis=0)
        .astype(np.float32)
    )

    sigma = np.clip(
        sigma,
        1e-6,
        None,
    )

    return mu, sigma


class VisualDataset(Dataset):
    def __init__(
        self,
        frame,
        mu,
        sigma,
    ):
        self.df = (
            frame
            .copy()
            .reset_index(drop=True)
        )

        self.mu = mu
        self.sigma = sigma

    def __len__(self):
        return len(self.df)

    def __getitem__(
        self,
        idx,
    ):
        row = self.df.iloc[idx]

        case_id = str(
            row["case_id"]
        )

        x = load_eyeclip(
            case_id
        )

        x = (
            x
            - self.mu[None, :]
        ) / self.sigma[None, :]

        return {
            "case_id":
                case_id,

            "x":
                torch.from_numpy(
                    x.astype(
                        np.float32,
                        copy=False,
                    )
                ),

            "label":
                torch.tensor(
                    int(row["label"]),
                    dtype=torch.long,
                ),
        }


def collate_visual(batch):
    B = len(batch)

    zmax = max(
        item["x"].shape[0]
        for item in batch
    )

    x = torch.zeros(
        B,
        zmax,
        512,
        dtype=torch.float32,
    )

    valid = torch.zeros(
        B,
        zmax,
        dtype=torch.bool,
    )

    labels = []
    cases = []

    for i, item in enumerate(batch):
        z = item["x"].shape[0]

        x[i, :z] = item["x"]
        valid[i, :z] = True

        labels.append(
            item["label"]
        )

        cases.append(
            item["case_id"]
        )

    return {
        "x": x,
        "valid": valid,
        "label": torch.stack(labels),
        "case_id": cases,
    }


# ============================================================
# EXACT graph-free B2 visual architecture
# ============================================================

def sinusoidal_encoding(
    length,
    dim,
    device,
):
    if length == 1:
        pos = torch.zeros(
            1,
            device=device,
            dtype=torch.float32,
        )
    else:
        pos = torch.linspace(
            0,
            MAX_CANONICAL_Z - 1,
            steps=length,
            device=device,
            dtype=torch.float32,
        )

    pe = torch.zeros(
        length,
        dim,
        device=device,
        dtype=torch.float32,
    )

    div = torch.exp(
        torch.arange(
            0,
            dim,
            2,
            device=device,
            dtype=torch.float32,
        )
        * (
            -math.log(10000.0)
            / dim
        )
    )

    pe[:, 0::2] = torch.sin(
        pos[:, None]
        * div[None, :]
    )

    pe[:, 1::2] = torch.cos(
        pos[:, None]
        * div[None, :]
    )

    return pe


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dropout,
    ):
        super().__init__()

        if dim % heads != 0:
            raise ValueError(
                "dim must be divisible by heads"
            )

        self.heads = heads
        self.head_dim = (
            dim // heads
        )

        self.qkv = nn.Linear(
            dim,
            3 * dim,
        )

        self.out = nn.Linear(
            dim,
            dim,
        )

        self.drop = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        valid,
    ):
        B, Z, D = x.shape

        qkv = (
            self.qkv(x)
            .reshape(
                B,
                Z,
                3,
                self.heads,
                self.head_dim,
            )
            .permute(
                2,
                0,
                3,
                1,
                4,
            )
        )

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        logits = (
            q
            @ k.transpose(
                -2,
                -1,
            )
        ) / math.sqrt(
            self.head_dim
        )

        logits = (
            logits.masked_fill(
                (~valid)[
                    :,
                    None,
                    None,
                    :
                ],
                -1e9,
            )
        )

        attn = torch.softmax(
            logits,
            dim=-1,
        )

        attn = self.drop(
            attn
        )

        out = (
            attn @ v
        )

        out = (
            out.transpose(
                1,
                2,
            )
            .reshape(
                B,
                Z,
                D,
            )
        )

        out = self.out(
            out
        )

        return (
            out
            * valid[
                :,
                :,
                None,
            ]
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dropout,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            dim
        )

        self.attn = (
            MultiHeadSelfAttention(
                dim,
                heads,
                dropout,
            )
        )

        self.norm2 = nn.LayerNorm(
            dim
        )

        self.ff = nn.Sequential(
            nn.Linear(
                dim,
                4 * dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                4 * dim,
                dim,
            ),
            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        x,
        valid,
    ):
        x = (
            x
            + self.attn(
                self.norm1(x),
                valid,
            )
        )

        x = (
            x
            + self.ff(
                self.norm2(x)
            )
        )

        return (
            x
            * valid[
                :,
                :,
                None,
            ]
        )


class AttentionPool(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super().__init__()

        self.score = nn.Linear(
            dim,
            1,
        )

    def forward(
        self,
        x,
        valid,
    ):
        score = (
            self.score(x)
            .squeeze(-1)
        )

        score = (
            score.masked_fill(
                ~valid,
                -1e9,
            )
        )

        alpha = torch.softmax(
            score,
            dim=1,
        )

        return torch.sum(
            alpha[
                :,
                :,
                None,
            ]
            * x,
            dim=1,
        )


class B2Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.proj = nn.Linear(
            512,
            D_MODEL,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                D_MODEL,
                N_HEADS,
                DROPOUT,
            )
            for _ in range(
                N_BLOCKS
            )
        ])

        self.pool = AttentionPool(
            D_MODEL
        )

        self.classifier = nn.Linear(
            D_MODEL,
            2,
        )

    def forward(
        self,
        x,
        valid,
    ):
        x = self.proj(
            x
        )

        pe = sinusoidal_encoding(
            x.shape[1],
            D_MODEL,
            x.device,
        )

        x = (
            x
            + pe[
                None,
                :,
                :
            ]
        )

        x = (
            x
            * valid[
                :,
                :,
                None,
            ]
        )

        for block in self.blocks:
            x = block(
                x,
                valid,
            )

        x = self.pool(
            x,
            valid,
        )

        return self.classifier(
            x
        )


probe = B2Model()

PARAM_COUNT = sum(
    p.numel()
    for p in probe.parameters()
)

if PARAM_COUNT != 1711619:
    raise RuntimeError(
        f"Unexpected B2 parameter count: {PARAM_COUNT}"
    )

del probe


# ============================================================
# Visual cross-fit
# ============================================================

def train_visual_fold(
    train_frame,
    hold_frame,
    fold,
):
    mu, sigma = (
        fit_visual_scaler(
            train_frame
        )
    )

    tr_ds = VisualDataset(
        train_frame,
        mu,
        sigma,
    )

    ho_ds = VisualDataset(
        hold_frame,
        mu,
        sigma,
    )

    generator = (
        torch.Generator()
    )

    # Same fixed seed for every model,
    # matching the finalized FairVision branch.
    generator.manual_seed(
        SEED
    )

    tr_loader = DataLoader(
        tr_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate_visual,
        pin_memory=True,
    )

    ho_loader = DataLoader(
        ho_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_visual,
        pin_memory=True,
    )

    seed_everything(
        SEED
    )

    model = (
        B2Model()
        .to(DEVICE)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(
        1,
        VISUAL_EPOCHS + 1,
    ):
        model.train()

        total_loss = 0.0
        total_n = 0

        for batch in tr_loader:
            x = (
                batch["x"]
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            valid = (
                batch["valid"]
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            y = (
                batch["label"]
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                x,
                valid,
            )

            loss = (
                F.cross_entropy(
                    logits,
                    y,
                )
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                float(
                    loss.detach().cpu()
                )
                * len(y)
            )

            total_n += len(y)

        print(
            f"Fold {fold} | "
            f"epoch {epoch:02d}/{VISUAL_EPOCHS} | "
            f"loss={total_loss / total_n:.6f}"
        )

    model.eval()

    rows = []

    with torch.no_grad():
        for batch in ho_loader:
            logits = model(
                batch["x"].to(
                    DEVICE,
                    non_blocking=True,
                ),
                batch["valid"].to(
                    DEVICE,
                    non_blocking=True,
                ),
            )

            probs = (
                torch.softmax(
                    logits,
                    dim=1,
                )[:, 1]
                .cpu()
                .numpy()
            )

            labels = (
                batch["label"]
                .numpy()
            )

            for (
                case_id,
                label,
                prob,
            ) in zip(
                batch["case_id"],
                labels,
                probs,
            ):
                rows.append({
                    "case_id":
                        str(case_id),

                    "label":
                        int(label),

                    "prob_visual":
                        float(prob),

                    "fold":
                        int(fold),
                })

    pred = pd.DataFrame(
        rows
    )

    if len(pred) != len(hold_frame):
        raise RuntimeError(
            f"Fold {fold}: incomplete visual predictions."
        )

    del model
    del optimizer
    del tr_loader
    del ho_loader
    del tr_ds
    del ho_ds

    gc.collect()
    torch.cuda.empty_cache()

    return pred


visual_parts = []

# Fold 1: reuse already-valid clean base4800 -> meta1200 predictions.
fold1 = (
    old_meta[
        [
            "case_id",
            "label",
            "prob_amd",
        ]
    ]
    .rename(
        columns={
            "prob_amd":
                "prob_visual"
        }
    )
    .copy()
)

fold1["fold"] = 1

expected_fold1 = set(
    fold_assignment.loc[
        fold_assignment["fold"] == 1,
        "case_id",
    ]
)

if set(fold1["case_id"]) != expected_fold1:
    raise RuntimeError(
        "Existing meta1200 is not exact fold 1."
    )

visual_parts.append(
    fold1
)

print()
print("=" * 96)
print("FULL-6000 VISUAL CROSSFIT")
print("=" * 96)
print("Fold 1: reusing existing clean base4800 -> meta1200 predictions.")
print("Folds 2-5: training new fixed 8-epoch models.")
print()


for fold in [
    2,
    3,
    4,
    5,
]:
    cache = (
        OUT
        / f"visual_oof_fold{fold}.csv"
    )

    hold_ids = set(
        fold_assignment.loc[
            fold_assignment["fold"]
            == fold,
            "case_id",
        ]
    )

    if cache.exists():
        cached = pd.read_csv(
            cache
        )

        cached["case_id"] = (
            cached["case_id"]
            .astype(str)
        )

        required = {
            "case_id",
            "label",
            "prob_visual",
            "fold",
        }

        if (
            required.issubset(
                cached.columns
            )
            and len(cached) == 1200
            and cached["case_id"].nunique()
                == 1200
            and set(cached["case_id"])
                == hold_ids
            and set(cached["fold"])
                == {fold}
        ):
            print(
                f"Fold {fold}: valid cached "
                f"predictions found -- reusing."
            )

            visual_parts.append(
                cached
            )
            continue

        raise RuntimeError(
            f"Fold {fold}: cache exists but failed integrity check: {cache}"
        )

    hold = (
        train6000[
            train6000[
                "case_id"
            ].isin(
                hold_ids
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    train = (
        train6000[
            ~train6000[
                "case_id"
            ].isin(
                hold_ids
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    if (
        len(train) != 4800
        or len(hold) != 1200
    ):
        raise RuntimeError(
            f"Fold {fold}: expected 4800/1200, "
            f"got {len(train)}/{len(hold)}"
        )

    if (
        set(train["case_id"])
        & set(hold["case_id"])
    ):
        raise RuntimeError(
            f"Fold {fold}: train/hold overlap"
        )

    print()
    print(
        "=" * 96
    )
    print(
        f"VISUAL FOLD {fold}/5 | "
        f"train=4800 | hold=1200"
    )
    print(
        "=" * 96
    )

    pred = train_visual_fold(
        train,
        hold,
        fold,
    )

    pred.to_csv(
        cache,
        index=False,
    )

    visual_parts.append(
        pred
    )


visual_oof = pd.concat(
    visual_parts,
    ignore_index=True,
)

if (
    len(visual_oof) != 6000
    or visual_oof[
        "case_id"
    ].nunique() != 6000
):
    raise RuntimeError(
        "Visual OOF is not exact 6000 unique cases."
    )

if set(
    visual_oof["case_id"]
) != set(
    train6000["case_id"]
):
    raise RuntimeError(
        "Visual OOF case set != train6000."
    )

visual_oof.to_csv(
    OUT
    / "visual_oof_6000.csv",
    index=False,
)


# ============================================================
# KG model
# ============================================================

def fit_kg(
    train_frame,
    C,
):
    raw = (
        train_frame[
            KG_COLS
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    medians = raw.median(
        axis=0,
        skipna=True,
    )

    clean = (
        raw
        .fillna(medians)
        .fillna(0.0)
    )

    std = clean.std(
        axis=0,
        ddof=0,
    )

    keep = (
        std[
            np.isfinite(std)
            & (std > 1e-12)
        ]
        .index
        .tolist()
    )

    if not keep:
        raise RuntimeError(
            "All KG features constant."
        )

    scaler = (
        StandardScaler()
    )

    X = scaler.fit_transform(
        clean[
            keep
        ].to_numpy(
            np.float64
        )
    )

    y = (
        train_frame[
            "label"
        ]
        .to_numpy(
            int
        )
    )

    clf = LogisticRegression(
        C=float(C),
        solver="liblinear",
        max_iter=5000,
        random_state=SEED,
    )

    clf.fit(
        X,
        y,
    )

    return {
        "medians":
            medians,

        "keep":
            keep,

        "scaler":
            scaler,

        "clf":
            clf,
    }


def predict_kg(
    bundle,
    frame,
):
    Xdf = (
        frame[
            KG_COLS
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(
            bundle[
                "medians"
            ]
        )
        .fillna(0.0)
    )

    X = (
        Xdf[
            bundle[
                "keep"
            ]
        ]
        .to_numpy(
            np.float64
        )
    )

    X = (
        bundle[
            "scaler"
        ]
        .transform(
            X
        )
    )

    return (
        bundle[
            "clf"
        ]
        .predict_proba(
            X
        )[:, 1]
    )


# ============================================================
# KG C selection using EXACT SAME 5 training-only folds
# ============================================================

print()
print("=" * 96)
print("KG C SELECTION — FULL TRAIN6000 OOF")
print("=" * 96)

kg_selection_rows = []
kg_oof_by_C = {}


for C in KG_C_GRID:
    parts = []
    fold_aucs = []

    for fold in range(
        1,
        6,
    ):
        hold_ids = set(
            fold_assignment.loc[
                fold_assignment[
                    "fold"
                ] == fold,
                "case_id",
            ]
        )

        tr = (
            train6000[
                ~train6000[
                    "case_id"
                ].isin(
                    hold_ids
                )
            ]
            .copy()
            .reset_index(drop=True)
        )

        ho = (
            train6000[
                train6000[
                    "case_id"
                ].isin(
                    hold_ids
                )
            ]
            .copy()
            .reset_index(drop=True)
        )

        bundle = fit_kg(
            tr,
            C,
        )

        p = predict_kg(
            bundle,
            ho,
        )

        auc = roc_auc_score(
            ho["label"],
            p,
        )

        fold_aucs.append(
            float(auc)
        )

        parts.append(
            pd.DataFrame({
                "case_id":
                    ho["case_id"],

                "label":
                    ho["label"],

                "prob_kg":
                    p,

                "fold":
                    fold,
            })
        )

    full = pd.concat(
        parts,
        ignore_index=True,
    )

    if (
        len(full) != 6000
        or full[
            "case_id"
        ].nunique() != 6000
    ):
        raise RuntimeError(
            f"KG OOF incomplete for C={C}"
        )

    mean_auc = float(
        np.mean(
            fold_aucs
        )
    )

    sd_auc = float(
        np.std(
            fold_aucs
        )
    )

    kg_selection_rows.append({
        "C":
            C,

        "mean_fold_AUROC":
            mean_auc,

        "sd_fold_AUROC":
            sd_auc,

        "fold1_AUROC":
            fold_aucs[0],

        "fold2_AUROC":
            fold_aucs[1],

        "fold3_AUROC":
            fold_aucs[2],

        "fold4_AUROC":
            fold_aucs[3],

        "fold5_AUROC":
            fold_aucs[4],
    })

    kg_oof_by_C[
        float(C)
    ] = full

    print(
        f"C={C:<7g} | "
        f"mean fold AUROC="
        f"{mean_auc:.6f} "
        f"+/- {sd_auc:.6f}"
    )


kg_selection = pd.DataFrame(
    kg_selection_rows
)

best_kg_auc = (
    kg_selection[
        "mean_fold_AUROC"
    ]
    .max()
)

selected_kg_C = float(
    kg_selection.loc[
        np.isclose(
            kg_selection[
                "mean_fold_AUROC"
            ],
            best_kg_auc,
            rtol=0,
            atol=1e-12,
        ),
        "C",
    ]
    .min()
)

print(
    "Selected KG C:",
    selected_kg_C,
)

kg_selection.to_csv(
    OUT
    / "kg_C_selection.csv",
    index=False,
)

kg_oof = (
    kg_oof_by_C[
        selected_kg_C
    ]
    .copy()
)


# ============================================================
# Complete genuine 6000 OOF base predictions
# ============================================================

oof = (
    visual_oof[
        [
            "case_id",
            "label",
            "prob_visual",
            "fold",
        ]
    ]
    .merge(
        kg_oof[
            [
                "case_id",
                "label",
                "prob_kg",
                "fold",
            ]
        ],
        on=[
            "case_id",
            "label",
            "fold",
        ],
        how="inner",
        validate="one_to_one",
    )
)

if (
    len(oof) != 6000
    or oof[
        "case_id"
    ].nunique() != 6000
):
    raise RuntimeError(
        "Merged OOF base predictions are incomplete."
    )


def metric_dict(
    y,
    p,
):
    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    pred = (
        p >= 0.5
    ).astype(int)

    return {
        "AUROC":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "AUPRC":
            float(
                average_precision_score(
                    y,
                    p,
                )
            ),

        "BalAcc":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),

        "MacroF1":
            float(
                f1_score(
                    y,
                    pred,
                    average="macro",
                )
            ),

        "Accuracy":
            float(
                accuracy_score(
                    y,
                    pred,
                )
            ),
    }


print()
print("=" * 96)
print("OOF BASE PERFORMANCE — TRAIN6000")
print("=" * 96)

mv_oof = metric_dict(
    oof["label"],
    oof["prob_visual"],
)

mk_oof = metric_dict(
    oof["label"],
    oof["prob_kg"],
)

print(
    "Visual OOF | "
    f"AUROC={mv_oof['AUROC']:.6f} | "
    f"AUPRC={mv_oof['AUPRC']:.6f}"
)

print(
    "KG OOF     | "
    f"AUROC={mk_oof['AUROC']:.6f} | "
    f"AUPRC={mk_oof['AUPRC']:.6f}"
)

oof.to_csv(
    OUT
    / "train6000_OOF_base_predictions.csv",
    index=False,
)


# ============================================================
# Fusion C selection from genuine OOF predictions only
# ============================================================

def logit(p):
    p = np.clip(
        np.asarray(
            p,
            dtype=np.float64,
        ),
        1e-5,
        1.0 - 1e-5,
    )

    return np.log(
        p
        / (
            1.0 - p
        )
    )


X_meta = np.column_stack([
    logit(
        oof[
            "prob_visual"
        ]
    ),

    logit(
        oof[
            "prob_kg"
        ]
    ),
])

y_meta = (
    oof[
        "label"
    ]
    .to_numpy(
        int
    )
)

meta_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

fusion_selection_rows = []

print()
print("=" * 96)
print("FUSION C SELECTION — TRAIN6000 OOF ONLY")
print("=" * 96)


for C in FUSION_C_GRID:
    aucs = []

    for (
        tr_idx,
        ho_idx,
    ) in meta_cv.split(
        X_meta,
        y_meta,
    ):
        scaler = (
            StandardScaler()
        )

        Xtr = scaler.fit_transform(
            X_meta[
                tr_idx
            ]
        )

        Xho = scaler.transform(
            X_meta[
                ho_idx
            ]
        )

        clf = LogisticRegression(
            C=float(C),
            solver="liblinear",
            max_iter=5000,
            random_state=SEED,
        )

        clf.fit(
            Xtr,
            y_meta[
                tr_idx
            ],
        )

        p = (
            clf.predict_proba(
                Xho
            )[:, 1]
        )

        aucs.append(
            float(
                roc_auc_score(
                    y_meta[
                        ho_idx
                    ],
                    p,
                )
            )
        )

    mean_auc = float(
        np.mean(
            aucs
        )
    )

    sd_auc = float(
        np.std(
            aucs
        )
    )

    fusion_selection_rows.append({
        "C":
            C,

        "mean_AUROC":
            mean_auc,

        "sd_AUROC":
            sd_auc,
    })

    print(
        f"C={C:<7g} | "
        f"mean AUROC="
        f"{mean_auc:.6f} "
        f"+/- {sd_auc:.6f}"
    )


fusion_selection = pd.DataFrame(
    fusion_selection_rows
)

best_fusion_auc = (
    fusion_selection[
        "mean_AUROC"
    ]
    .max()
)

selected_fusion_C = float(
    fusion_selection.loc[
        np.isclose(
            fusion_selection[
                "mean_AUROC"
            ],
            best_fusion_auc,
            rtol=0,
            atol=1e-12,
        ),
        "C",
    ]
    .min()
)

print(
    "Selected fusion C:",
    selected_fusion_C,
)

fusion_selection.to_csv(
    OUT
    / "fusion_C_selection.csv",
    index=False,
)


# ============================================================
# Freeze fusion on ALL 6000 genuine OOF predictions
# ============================================================

meta_scaler = (
    StandardScaler()
)

Xm = (
    meta_scaler
    .fit_transform(
        X_meta
    )
)

meta_clf = (
    LogisticRegression(
        C=selected_fusion_C,
        solver="liblinear",
        max_iter=5000,
        random_state=SEED,
    )
)

meta_clf.fit(
    Xm,
    y_meta,
)

print()
print(
    "Frozen standardized fusion coefficients:",
    f"Visual={meta_clf.coef_[0][0]:.6f},",
    f"KG={meta_clf.coef_[0][1]:.6f},",
    f"intercept={meta_clf.intercept_[0]:.6f}",
)


# ============================================================
# FINAL:
# KG refit on full6000
# Visual full6000 -> val1000 already exists and is reused
# Official val1000 has NEVER participated in selection
# ============================================================

full_kg = fit_kg(
    train6000,
    selected_kg_C,
)

p_val_kg = predict_kg(
    full_kg,
    val1000,
)

val_kg_pred = pd.DataFrame({
    "case_id":
        val1000[
            "case_id"
        ],

    "label":
        val1000[
            "label"
        ],

    "prob_kg":
        p_val_kg,
})

val = (
    old_val[
        [
            "case_id",
            "label",
            "prob_amd",
        ]
    ]
    .rename(
        columns={
            "prob_amd":
                "prob_visual"
        }
    )
    .merge(
        val_kg_pred,
        on=[
            "case_id",
            "label",
        ],
        how="inner",
        validate="one_to_one",
    )
)

if len(val) != 1000:
    raise RuntimeError(
        "Final official-val merge is not 1000."
    )

X_val_meta = np.column_stack([
    logit(
        val[
            "prob_visual"
        ]
    ),

    logit(
        val[
            "prob_kg"
        ]
    ),
])

X_val_meta = (
    meta_scaler
    .transform(
        X_val_meta
    )
)

val[
    "prob_fusion"
] = (
    meta_clf
    .predict_proba(
        X_val_meta
    )[:, 1]
)


mv = metric_dict(
    val["label"],
    val["prob_visual"],
)

mk = metric_dict(
    val["label"],
    val["prob_kg"],
)

mf = metric_dict(
    val["label"],
    val["prob_fusion"],
)


print()
print("=" * 96)
print("FAIRVISION FINAL — OFFICIAL VAL1000")
print("FULL-6000 GENUINE CROSSFITTED FUSION")
print("=" * 96)

print(
    f"Selected KG C     : {selected_kg_C}"
)

print(
    f"Selected fusion C : {selected_fusion_C}"
)

print()

for name, m in [
    ("Visual only", mv),
    ("KG only", mk),
    ("Visual + KG", mf),
]:
    print(
        f"{name:<18} "
        f"AUROC={m['AUROC']:.6f} | "
        f"AUPRC={m['AUPRC']:.6f} | "
        f"BalAcc={m['BalAcc']:.6f} | "
        f"MacroF1={m['MacroF1']:.6f} | "
        f"Acc={m['Accuracy']:.6f}"
    )

print()

print(
    "Fusion - Visual AUROC :",
    f"{mf['AUROC'] - mv['AUROC']:+.6f}",
)

print(
    "Fusion - Visual AUPRC :",
    f"{mf['AUPRC'] - mv['AUPRC']:+.6f}",
)

print(
    "Fusion - Visual BalAcc:",
    f"{mf['BalAcc'] - mv['BalAcc']:+.6f}",
)

print(
    "Fusion - Visual MacroF1:",
    f"{mf['MacroF1'] - mv['MacroF1']:+.6f}",
)

print(
    "Fusion - Visual Acc   :",
    f"{mf['Accuracy'] - mv['Accuracy']:+.6f}",
)


# ============================================================
# Save
# ============================================================

val.to_csv(
    OUT
    / "official_val1000_predictions.csv",
    index=False,
)

pd.DataFrame([
    {
        "method":
            "Visual only",
        **mv,
    },
    {
        "method":
            "KG only",
        **mk,
    },
    {
        "method":
            "Visual + KG",
        **mf,
    },
]).to_csv(
    OUT
    / "summary.csv",
    index=False,
)

with open(
    OUT
    / "protocol.json",
    "w",
) as f:
    json.dump(
        {
            "dataset":
                "FairVision",

            "task":
                "AMD vs no-AMD",

            "train_n":
                6000,

            "official_val_n":
                1000,

            "crossfit_folds":
                5,

            "fold1_reused_existing_clean_holdout":
                True,

            "visual_epochs":
                VISUAL_EPOCHS,

            "visual_lr":
                LR,

            "visual_weight_decay":
                WEIGHT_DECAY,

            "visual_batch_size":
                BATCH_SIZE,

            "visual_loss":
                "unweighted cross entropy",

            "visual_backbone":
                "frozen EyeCLIP slice embeddings",

            "kg_dimension":
                128,

            "kg_C_grid":
                KG_C_GRID,

            "selected_kg_C":
                selected_kg_C,

            "fusion_C_grid":
                FUSION_C_GRID,

            "selected_fusion_C":
                selected_fusion_C,

            "fusion_training":
                "all 6000 genuine OOF visual+KG predictions",

            "official_val_used_for_selection":
                False,

            "test3000_used":
                False,

            "decision_threshold":
                0.5,
        },
        f,
        indent=2,
    )

print()
print("=" * 96)
print("DONE")
print("=" * 96)
print("Saved:", OUT)
