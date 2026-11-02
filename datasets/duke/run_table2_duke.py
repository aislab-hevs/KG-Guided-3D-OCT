# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.


from pathlib import Path
import json
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_CSV = Path("PATH_TO_DUKE_CANONICAL_DATASET") / "manifest.csv"

FEATURE_DIR = (
    ROOT / "datasets/duke/results/eyeclip_slice_features"
)

KG_DESC_CSV = (
    ROOT / "datasets/duke/results/kg_128_descriptors.csv"
)

OUT = (
    ROOT / "datasets/duke/results/table2_seed42"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Same visual architecture/training hyperparameters as the
# finalized primary-cohort model.
D_MODEL = 256
N_HEADS = 4
N_BLOCKS = 2
DROPOUT = 0.1
BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-4
MAX_CANONICAL_Z = 512

# Source-selected fixed hyperparameters.
# These are NOT re-selected on Srinivasan.
VISUAL_EPOCHS = 7
KG_C = 0.1
META_C = 1.0

# Srinivasan evaluation.
OUTER_FOLDS = 5

# Needed only to create leakage-free base predictions
# for fitting the fusion model inside each outer training fold.
FUSION_INNER_FOLDS = 5


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ============================================================
# Cohort
# ============================================================

manifest = pd.read_csv(MANIFEST_CSV)

required = {"case_id", "disease", "label"}
missing = required - set(manifest.columns)
if missing:
    raise RuntimeError(f"Manifest missing columns: {sorted(missing)}")

frame = (
    manifest[
        manifest["disease"].isin(["AMD", "DME"])
    ][["case_id", "disease", "label"]]
    .copy()
    .reset_index(drop=True)
)

frame["case_id"] = frame["case_id"].astype(str)
frame["disease"] = frame["disease"].astype(str)
frame["label"] = frame["label"].astype(int)

if len(frame) != 30:
    raise RuntimeError(f"Expected 30 Srinivasan AMD/DME acquisitions, found {len(frame)}")

if frame["case_id"].duplicated().any():
    raise RuntimeError("Duplicate Srinivasan case_id values.")

if not (
    ((frame["disease"] == "AMD") & (frame["label"] == 0))
    | ((frame["disease"] == "DME") & (frame["label"] == 1))
).all():
    raise RuntimeError("Unexpected AMD/DME label mapping.")

if (frame["disease"] == "AMD").sum() != 15:
    raise RuntimeError("Expected 15 AMD acquisitions.")

if (frame["disease"] == "DME").sum() != 15:
    raise RuntimeError("Expected 15 DME acquisitions.")

# Srinivasan manifest has no separate subject identifier.
# Each acquisition is therefore the evaluation unit.
frame["group_id"] = frame["case_id"]

print("=" * 88)
print("SRINIVASAN METHOD REPLICATION")
print("=" * 88)
print("Device:", DEVICE)
print("Acquisitions:", len(frame))
print("AMD:", int((frame["label"] == 0).sum()))
print("DME:", int((frame["label"] == 1).sum()))
print("Evaluation unit: OCT acquisition")
print("Visual epochs (source-selected):", VISUAL_EPOCHS)
print("KG C (source-selected):", KG_C)
print("Fusion C (source-selected):", META_C)
print("No Srinivasan-specific hyperparameter selection is performed.")


# ============================================================
# Fold helper
# ============================================================

def stratified_folds(df, n_splits, seed):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    y = df["label"].to_numpy()

    out = []

    for train_idx, hold_idx in skf.split(
        np.zeros(len(df)),
        y,
    ):
        tr = (
            df.iloc[train_idx]
            .copy()
            .reset_index(drop=True)
        )

        ho = (
            df.iloc[hold_idx]
            .copy()
            .reset_index(drop=True)
        )

        if set(tr["case_id"]) & set(ho["case_id"]):
            raise RuntimeError("Fold overlap detected.")

        if tr["label"].nunique() != 2 or ho["label"].nunique() != 2:
            raise RuntimeError("A fold is missing one disease class.")

        out.append((tr, ho))

    return out


# ============================================================
# Frozen EyeCLIP slice features
# ============================================================

def load_eyeclip(case_id):
    path = FEATURE_DIR / f"{case_id}.npy"

    if not path.exists():
        raise FileNotFoundError(path)

    x = np.load(path).astype(np.float32)

    if x.ndim != 2 or x.shape[1] != 512:
        raise RuntimeError(
            f"{case_id}: unexpected EyeCLIP shape {x.shape}"
        )

    if not np.isfinite(x).all():
        raise RuntimeError(
            f"{case_id}: non-finite EyeCLIP feature values"
        )

    # Same per-slice L2 normalization as the primary pipeline.
    norm = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    x = x / np.clip(
        norm,
        1e-12,
        None,
    )

    return x


# Check all 30 visual files before training.
for case_id in frame["case_id"]:
    _ = load_eyeclip(case_id)

print("EyeCLIP check: PASS (30/30)")


def fit_visual_scaler(train_frame):
    # Same source protocol:
    # mean-pool normalized slice embeddings per acquisition,
    # then estimate feature-wise mean/std across training acquisitions.
    pooled = np.stack([
        load_eyeclip(case_id).mean(axis=0)
        for case_id in train_frame["case_id"]
    ])

    mu = pooled.mean(axis=0).astype(np.float32)
    sigma = pooled.std(axis=0).astype(np.float32)

    sigma = np.clip(
        sigma,
        1e-6,
        None,
    )

    return mu, sigma


class VisualDataset(Dataset):
    def __init__(
        self,
        df,
        mu,
        sigma,
    ):
        self.df = (
            df.copy()
            .reset_index(drop=True)
        )
        self.mu = mu
        self.sigma = sigma

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = load_eyeclip(
            str(row["case_id"])
        )

        x = (
            x - self.mu[None, :]
        ) / self.sigma[None, :]

        return {
            "case_id": str(row["case_id"]),
            "x": torch.from_numpy(
                x.astype(
                    np.float32,
                    copy=False,
                )
            ),
            "label": torch.tensor(
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

        labels.append(item["label"])
        cases.append(item["case_id"])

    return {
        "x": x,
        "valid": valid,
        "label": torch.stack(labels),
        "case_id": cases,
    }


# ============================================================
# B2 graph-free cross-slice Transformer
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
        pos[:, None] * div[None, :]
    )

    pe[:, 1::2] = torch.cos(
        pos[:, None] * div[None, :]
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
            raise ValueError("dim must be divisible by heads")

        self.heads = heads
        self.head_dim = dim // heads

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
            @ k.transpose(-2, -1)
        ) / math.sqrt(
            self.head_dim
        )

        logits = logits.masked_fill(
            (~valid)[:, None, None, :],
            -1e9,
        )

        attn = torch.softmax(
            logits,
            dim=-1,
        )

        attn = self.drop(attn)

        out = attn @ v

        out = (
            out.transpose(1, 2)
            .reshape(
                B,
                Z,
                D,
            )
        )

        out = self.out(out)

        return (
            out
            * valid[:, :, None]
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dropout,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = MultiHeadSelfAttention(
            dim,
            heads,
            dropout,
        )

        self.norm2 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(
                dim,
                4 * dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                4 * dim,
                dim,
            ),
            nn.Dropout(dropout),
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
            * valid[:, :, None]
        )


class AttentionPool(nn.Module):
    def __init__(self, dim):
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

        score = score.masked_fill(
            ~valid,
            -1e9,
        )

        alpha = torch.softmax(
            score,
            dim=1,
        )

        return torch.sum(
            alpha[:, :, None]
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
            for _ in range(N_BLOCKS)
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
        x = self.proj(x)

        pe = sinusoidal_encoding(
            x.shape[1],
            D_MODEL,
            x.device,
        )

        x = (
            x
            + pe[None, :, :]
        )

        x = (
            x
            * valid[:, :, None]
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

        return self.classifier(x)


probe = B2Model()
print(
    "B2 trainable parameters:",
    sum(p.numel() for p in probe.parameters())
)
del probe


# ============================================================
# Visual train / predict
# ============================================================

def build_visual_loaders(
    train_frame,
    eval_frame,
    seed,
):
    mu, sigma = fit_visual_scaler(
        train_frame
    )

    tr_ds = VisualDataset(
        train_frame,
        mu,
        sigma,
    )

    ev_ds = VisualDataset(
        eval_frame,
        mu,
        sigma,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate_visual,
        pin_memory=(DEVICE.type == "cuda"),
    )

    ev_loader = DataLoader(
        ev_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_visual,
        pin_memory=(DEVICE.type == "cuda"),
    )

    return (
        tr_loader,
        ev_loader,
        mu,
        sigma,
    )


def train_visual_epoch(
    model,
    loader,
    optimizer,
):
    model.train()

    total_loss = 0.0
    total_n = 0

    for batch in loader:
        x = batch["x"].to(DEVICE)
        valid = batch["valid"].to(DEVICE)
        y = batch["label"].to(DEVICE)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            x,
            valid,
        )

        loss = F.cross_entropy(
            logits,
            y,
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            float(loss.detach().cpu())
            * len(y)
        )
        total_n += len(y)

    return (
        total_loss
        / max(total_n, 1)
    )


@torch.no_grad()
def predict_visual(
    model,
    loader,
):
    model.eval()

    rows = []

    for batch in loader:
        logits = model(
            batch["x"].to(DEVICE),
            batch["valid"].to(DEVICE),
        )

        p = (
            torch.softmax(
                logits,
                dim=1,
            )[:, 1]
            .cpu()
            .numpy()
        )

        y = batch["label"].numpy()

        for case_id, label, prob in zip(
            batch["case_id"],
            y,
            p,
        ):
            rows.append({
                "case_id": case_id,
                "label": int(label),
                "prob_visual": float(prob),
            })

    return pd.DataFrame(rows)


def fit_visual_fixed_epochs(
    train_frame,
    eval_frame,
    epochs,
    seed,
    quiet=False,
):
    (
        tr_loader,
        ev_loader,
        _,
        _,
    ) = build_visual_loaders(
        train_frame,
        eval_frame,
        seed=seed,
    )

    seed_everything(seed + 1)

    model = B2Model().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    last_loss = None

    for epoch in range(
        1,
        epochs + 1,
    ):
        last_loss = train_visual_epoch(
            model,
            tr_loader,
            optimizer,
        )

        if not quiet:
            print(
                f"      visual epoch "
                f"{epoch:02d}/{epochs} "
                f"loss={last_loss:.4f}"
            )

    pred = predict_visual(
        model,
        ev_loader,
    )

    del model
    del optimizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        pred,
        float(last_loss),
    )


# ============================================================
# KG structural descriptors
# ============================================================

kg_raw = pd.read_csv(
    KG_DESC_CSV
)

if "case_id" not in kg_raw.columns:
    raise RuntimeError(
        "Srinivasan KG descriptor file has no case_id column."
    )

kg_raw["case_id"] = (
    kg_raw["case_id"]
    .astype(str)
)

if len(kg_raw) != 30:
    raise RuntimeError(
        f"Expected 30 Srinivasan KG descriptor rows, got {len(kg_raw)}"
    )

if kg_raw["case_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate Srinivasan KG descriptor case_id values."
    )

KG_COLS = [
    c
    for c in kg_raw.columns
    if c != "case_id"
]

if len(KG_COLS) != 128:
    raise RuntimeError(
        f"Expected 128 KG descriptors, got {len(KG_COLS)}"
    )

if set(kg_raw["case_id"]) != set(frame["case_id"]):
    raise RuntimeError(
        "Srinivasan manifest/KG case sets do not match."
    )

print(
    "KG structural descriptor dimension:",
    len(KG_COLS),
)
print("KG check: PASS (30/30)")


def kg_rows_for(df):
    wanted = list(
        df["case_id"].astype(str)
    )

    order = {
        case_id: i
        for i, case_id in enumerate(wanted)
    }

    d = (
        kg_raw[
            kg_raw["case_id"].isin(
                set(wanted)
            )
        ]
        .copy()
    )

    d["_order"] = (
        d["case_id"]
        .map(order)
    )

    d = (
        d.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    if list(d["case_id"]) != wanted:
        raise RuntimeError(
            "KG row order/case mismatch."
        )

    # Add labels from the fold frame, not from case-name parsing.
    meta = (
        df[
            ["case_id", "label"]
        ]
        .copy()
    )

    d = d.merge(
        meta,
        on="case_id",
        how="left",
        validate="one_to_one",
    )

    return d


def fit_kg_bundle(
    train_frame,
    C=KG_C,
):
    tr = kg_rows_for(
        train_frame
    )

    raw = (
        tr[KG_COLS]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    medians = raw.median(
        axis=0,
        skipna=True,
    )

    Xdf = (
        raw.fillna(medians)
        .fillna(0.0)
    )

    std = Xdf.std(
        axis=0,
        ddof=0,
    )

    keep_cols = (
        std[
            std > 1e-12
        ]
        .index
        .tolist()
    )

    if not keep_cols:
        raise RuntimeError(
            "All KG descriptor columns are constant."
        )

    X = (
        Xdf[keep_cols]
        .to_numpy(
            dtype=np.float64
        )
    )

    y = tr["label"].to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(
        C=C,
        solver="liblinear",
        max_iter=5000,
        random_state=SEED,
    )

    clf.fit(
        Xs,
        y,
    )

    return {
        "C": float(C),
        "feature_cols": list(KG_COLS),
        "keep_cols": keep_cols,
        "medians": medians.to_dict(),
        "scaler": scaler,
        "clf": clf,
    }


def predict_kg_bundle(
    bundle,
    eval_frame,
):
    ev = kg_rows_for(
        eval_frame
    )

    med = pd.Series(
        bundle["medians"]
    )

    Xdf = (
        ev[
            bundle["feature_cols"]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(med)
        .fillna(0.0)
    )

    X = (
        Xdf[
            bundle["keep_cols"]
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    Xs = (
        bundle["scaler"]
        .transform(X)
    )

    p = (
        bundle["clf"]
        .predict_proba(Xs)[:, 1]
    )

    return pd.DataFrame({
        "case_id": ev["case_id"].astype(str),
        "label": ev["label"].astype(int),
        "prob_kg": p.astype(float),
    })


# ============================================================
# Fusion
# ============================================================

def to_logit(p):
    p = np.clip(
        np.asarray(
            p,
            dtype=np.float64,
        ),
        1e-5,
        1.0 - 1e-5,
    )

    return np.log(
        p / (1.0 - p)
    )


def make_meta_X(
    df,
):
    return np.column_stack([
        to_logit(
            df["prob_visual"]
        ),
        to_logit(
            df["prob_kg"]
        ),
    ])


def fit_meta_model(
    base_oof,
    C=META_C,
):
    X = make_meta_X(
        base_oof
    )

    y = (
        base_oof["label"]
        .to_numpy()
    )

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(
        C=C,
        solver="liblinear",
        max_iter=5000,
        random_state=SEED,
    )

    clf.fit(
        Xs,
        y,
    )

    return {
        "C": float(C),
        "scaler": scaler,
        "clf": clf,
    }


def predict_meta(
    bundle,
    base_pred,
):
    X = make_meta_X(
        base_pred
    )

    Xs = (
        bundle["scaler"]
        .transform(X)
    )

    return (
        bundle["clf"]
        .predict_proba(Xs)[:, 1]
    )


# ============================================================
# Inner cross-fit used ONLY to train fusion inside outer TRAIN
# ============================================================


# ============================================================
# TABLE-4 THRESHOLD RULE
# max BAcc -> max Macro-F1 -> closest to 0.5
# ============================================================

def select_threshold_table4(y, p):

    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=float)

    unique = np.unique(p)

    if len(unique) < 2:
        raise RuntimeError(
            "Validation probabilities collapsed."
        )

    candidates = np.concatenate([
        [0.0],
        (unique[:-1] + unique[1:]) / 2.0,
        [1.0],
    ])

    best = None

    for threshold in candidates:

        pred = (
            p >= threshold
        ).astype(int)

        bacc = balanced_accuracy_score(
            y,
            pred,
        )

        mf1 = f1_score(
            y,
            pred,
            average="macro",
        )

        key = (
            float(bacc),
            float(mf1),
            -abs(
                float(threshold)
                - 0.5
            ),
        )

        if (
            best is None
            or key > best[0]
        ):
            best = (
                key,
                float(threshold),
            )

    return best[1]


def select_branch_median_threshold(
    base_oof,
    prob_col,
):

    rows = []

    groups = sorted(
        base_oof[
            "threshold_inner_fold"
        ].unique()
    )

    if len(groups) != FUSION_INNER_FOLDS:
        raise RuntimeError(
            f"{prob_col}: expected "
            f"{FUSION_INNER_FOLDS} inner folds, "
            f"found {len(groups)}"
        )

    for inner_fold in groups:

        sub = (
            base_oof[
                base_oof[
                    "threshold_inner_fold"
                ] == inner_fold
            ]
            .copy()
        )

        threshold = select_threshold_table4(
            sub["label"],
            sub[prob_col],
        )

        rows.append({
            "inner_fold":
                int(inner_fold),
            "threshold":
                float(threshold),
            "n_validation":
                int(len(sub)),
        })

    median_threshold = float(
        np.median([
            r["threshold"]
            for r in rows
        ])
    )

    return (
        median_threshold,
        rows,
    )


def select_meta_median_threshold(
    base_oof,
    outer_id,
):

    base_oof = (
        base_oof
        .reset_index(drop=True)
        .copy()
    )

    y = (
        base_oof["label"]
        .to_numpy(dtype=int)
    )

    skf = StratifiedKFold(
        n_splits=FUSION_INNER_FOLDS,
        shuffle=True,
        random_state=
            SEED
            + 60000
            + outer_id,
    )

    rows = []

    for meta_fold, (
        train_idx,
        val_idx,
    ) in enumerate(
        skf.split(
            np.zeros(
                len(base_oof)
            ),
            y,
        ),
        start=1,
    ):

        meta_train = (
            base_oof
            .iloc[train_idx]
            .reset_index(drop=True)
        )

        meta_val = (
            base_oof
            .iloc[val_idx]
            .reset_index(drop=True)
        )

        bundle = fit_meta_model(
            meta_train,
            C=META_C,
        )

        p = predict_meta(
            bundle,
            meta_val,
        )

        threshold = select_threshold_table4(
            meta_val["label"],
            p,
        )

        rows.append({
            "inner_fold":
                int(meta_fold),
            "threshold":
                float(threshold),
            "n_validation":
                int(len(meta_val)),
        })

    median_threshold = float(
        np.median([
            r["threshold"]
            for r in rows
        ])
    )

    return (
        median_threshold,
        rows,
    )


def make_inner_base_oof(
    outer_train,
    outer_id,
):
    inner_folds = stratified_folds(
        outer_train,
        n_splits=FUSION_INNER_FOLDS,
        seed=SEED + 40000 + outer_id,
    )

    visual_parts = []
    kg_parts = []

    for inner_id, (
        inner_train,
        inner_hold,
    ) in enumerate(
        inner_folds,
        start=1,
    ):
        print(
            f"    fusion inner "
            f"{inner_id}/{FUSION_INNER_FOLDS}: "
            f"{len(inner_train)} train / "
            f"{len(inner_hold)} hold"
        )

        visual_pred, _ = (
            fit_visual_fixed_epochs(
                inner_train,
                inner_hold,
                epochs=VISUAL_EPOCHS,
                seed=(
                    SEED
                    + 50000
                    + 1000 * outer_id
                    + inner_id
                ),
                quiet=True,
            )
        )

        kg_bundle = fit_kg_bundle(
            inner_train,
            C=KG_C,
        )

        kg_pred = predict_kg_bundle(
            kg_bundle,
            inner_hold,
        )

        visual_pred = visual_pred.copy()
        visual_pred[
            "threshold_inner_fold"
        ] = int(inner_id)

        visual_parts.append(
            visual_pred
        )
        kg_parts.append(
            kg_pred
        )

    v = pd.concat(
        visual_parts,
        ignore_index=True,
    )

    k = pd.concat(
        kg_parts,
        ignore_index=True,
    )

    oof = v.merge(
        k,
        on=[
            "case_id",
            "label",
        ],
        validate="one_to_one",
    )

    if len(oof) != len(outer_train):
        raise RuntimeError(
            "Incomplete fusion-training OOF predictions."
        )

    if oof["case_id"].nunique() != len(outer_train):
        raise RuntimeError(
            "Duplicate/missing fusion-training OOF cases."
        )

    if set(oof["case_id"]) != set(
        outer_train["case_id"]
    ):
        raise RuntimeError(
            "Fusion-training OOF case set mismatch."
        )

    return oof


# ============================================================
# Metrics
# ============================================================

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

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),

        "macro_F1":
            float(
                f1_score(
                    y,
                    pred,
                    average="macro",
                )
            ),

        "accuracy":
            float(
                accuracy_score(
                    y,
                    pred,
                )
            ),
    }


def print_metrics(
    name,
    y,
    p,
):
    m = metric_dict(
        y,
        p,
    )

    print(
        f"{name:<14} "
        f"AUROC={m['AUROC']:.4f} | "
        f"AUPRC={m['AUPRC']:.4f} | "
        f"BalAcc={m['balanced_accuracy']:.4f} | "
        f"MacroF1={m['macro_F1']:.4f} | "
        f"Acc={m['accuracy']:.4f}"
    )

    return m


# ============================================================
# Outer 5-fold evaluation
# ============================================================

outer_folds = stratified_folds(
    frame,
    n_splits=OUTER_FOLDS,
    seed=SEED,
)

fold_assignment_rows = []
outer_pred_parts = []
median_threshold_rows = []
inner_threshold_rows = []
fold_metric_rows = []
fusion_coef_rows = []

for outer_id, (
    outer_train,
    outer_hold,
) in enumerate(
    outer_folds,
    start=1,
):
    print()
    print("=" * 88)
    print(
        f"OUTER FOLD "
        f"{outer_id}/{OUTER_FOLDS}"
    )
    print("=" * 88)

    print(
        "Outer train:",
        len(outer_train),
        "| AMD=",
        int((outer_train["label"] == 0).sum()),
        "| DME=",
        int((outer_train["label"] == 1).sum()),
    )

    print(
        "Outer hold :",
        len(outer_hold),
        "| AMD=",
        int((outer_hold["label"] == 0).sum()),
        "| DME=",
        int((outer_hold["label"] == 1).sum()),
    )

    for c in outer_hold["case_id"]:
        fold_assignment_rows.append({
            "case_id": c,
            "outer_fold": outer_id,
        })

    # --------------------------------------------------------
    # 1. Full outer-train visual model -> outer hold
    # --------------------------------------------------------
    print("\n  [1/3] Visual-only outer model")

    visual_hold, final_visual_loss = (
        fit_visual_fixed_epochs(
            outer_train,
            outer_hold,
            epochs=VISUAL_EPOCHS,
            seed=(
                SEED
                + 10000
                + outer_id
            ),
            quiet=False,
        )
    )

    # --------------------------------------------------------
    # 2. Full outer-train KG model -> outer hold
    # --------------------------------------------------------
    print("\n  [2/3] KG-only outer model")

    kg_bundle = fit_kg_bundle(
        outer_train,
        C=KG_C,
    )

    kg_hold = predict_kg_bundle(
        kg_bundle,
        outer_hold,
    )

    # --------------------------------------------------------
    # 3. Fusion:
    #    fit only from leakage-free inner OOF base predictions
    #    generated inside outer TRAIN.
    # --------------------------------------------------------
    print("\n  [3/3] Visual + KG fusion")

    fusion_train_oof = (
        make_inner_base_oof(
            outer_train,
            outer_id=outer_id,
        )
    )


    # --------------------------------------------------------
    # Validation threshold selection.
    #
    # IMPORTANT:
    # each inner model gets its OWN validation threshold;
    # different model probability distributions are NOT
    # pooled before optimizing the threshold.
    # --------------------------------------------------------

    (
        threshold_visual,
        visual_threshold_details,
    ) = select_branch_median_threshold(
        fusion_train_oof,
        "prob_visual",
    )

    (
        threshold_kg,
        kg_threshold_details,
    ) = select_branch_median_threshold(
        fusion_train_oof,
        "prob_kg",
    )

    (
        threshold_fusion,
        fusion_threshold_details,
    ) = select_meta_median_threshold(
        fusion_train_oof,
        outer_id,
    )

    for branch, details in [
        (
            "Visual only",
            visual_threshold_details,
        ),
        (
            "KG only",
            kg_threshold_details,
        ),
        (
            "Visual + KG",
            fusion_threshold_details,
        ),
    ]:
        for row in details:
            inner_threshold_rows.append({
                "outer_fold":
                    int(outer_id),
                "method":
                    branch,
                **row,
            })

    median_threshold_rows.append({
        "outer_fold":
            int(outer_id),

        "threshold_visual":
            float(
                threshold_visual
            ),

        "threshold_kg":
            float(
                threshold_kg
            ),

        "threshold_fusion":
            float(
                threshold_fusion
            ),
    })

    print()
    print(
        "    Fold-specific-validation "
        "MEDIAN thresholds:"
    )

    print(
        f"      Visual={threshold_visual:.8f} | "
        f"KG={threshold_kg:.8f} | "
        f"Fusion={threshold_fusion:.8f}"
    )

    meta_bundle = fit_meta_model(
        fusion_train_oof,
        C=META_C,
    )

    hold_base = visual_hold.merge(
        kg_hold,
        on=[
            "case_id",
            "label",
        ],
        validate="one_to_one",
    )

    hold_base[
        "prob_fusion"
    ] = predict_meta(
        meta_bundle,
        hold_base,
    )

    hold_base[
        "outer_fold"
    ] = outer_id

    outer_pred_parts.append(
        hold_base
    )

    coef = (
        meta_bundle["clf"]
        .coef_[0]
    )

    fusion_coef_rows.append({
        "outer_fold": outer_id,
        "visual_coef_standardized": float(coef[0]),
        "kg_coef_standardized": float(coef[1]),
        "intercept": float(
            meta_bundle["clf"]
            .intercept_[0]
        ),
        "visual_final_train_loss": float(
            final_visual_loss
        ),
        "kg_keep_features": int(
            len(kg_bundle["keep_cols"])
        ),
    })

    # Outer-fold diagnostics, not the final pooled estimate.
    for model_name, col in [
        ("Visual only", "prob_visual"),
        ("KG only", "prob_kg"),
        ("Visual + KG", "prob_fusion"),
    ]:
        m = metric_dict(
            hold_base["label"],
            hold_base[col],
        )

        row = {
            "outer_fold": outer_id,
            "method": model_name,
            **m,
        }

        fold_metric_rows.append(
            row
        )

    print("\n  Outer-hold fold metrics:")
    print_metrics(
        "Visual only",
        hold_base["label"],
        hold_base["prob_visual"],
    )
    print_metrics(
        "KG only",
        hold_base["label"],
        hold_base["prob_kg"],
    )
    print_metrics(
        "Visual + KG",
        hold_base["label"],
        hold_base["prob_fusion"],
    )


# ============================================================
# Complete held-out OOF predictions
# ============================================================

pred = pd.concat(
    outer_pred_parts,
    ignore_index=True,
)

if len(pred) != 30:
    raise RuntimeError(
        f"Expected 30 held-out predictions, got {len(pred)}"
    )

if pred["case_id"].nunique() != 30:
    raise RuntimeError(
        "Each Srinivasan acquisition must have exactly one held-out prediction."
    )

if set(pred["case_id"]) != set(
    frame["case_id"]
):
    raise RuntimeError(
        "Final held-out prediction case set mismatch."
    )

order = {
    c: i
    for i, c in enumerate(
        frame["case_id"]
    )
}

pred["_order"] = (
    pred["case_id"]
    .map(order)
)

pred = (
    pred.sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)


# ============================================================
# Final pooled acquisition-level metrics
# ============================================================

print()
print("=" * 88)
print("FINAL SRINIVASAN HELD-OUT 5-FOLD RESULTS")
print("=" * 88)

summary_rows = []

for model_name, col in [
    ("Visual only", "prob_visual"),
    ("KG only", "prob_kg"),
    ("Visual + KG", "prob_fusion"),
]:
    m = print_metrics(
        model_name,
        pred["label"],
        pred[col],
    )

    summary_rows.append({
        "method": model_name,
        **m,
    })


summary = pd.DataFrame(
    summary_rows
)

visual_auc = float(
    summary.loc[
        summary["method"] == "Visual only",
        "AUROC",
    ].iloc[0]
)

fusion_auc = float(
    summary.loc[
        summary["method"] == "Visual + KG",
        "AUROC",
    ].iloc[0]
)

print()
print(
    "Fusion - Visual AUROC:",
    f"{fusion_auc - visual_auc:+.4f}",
)


# ============================================================
# Save
# ============================================================

pred.to_csv(
    OUT / "duke_heldout_predictions.csv",
    index=False,
)

summary.to_csv(
    OUT / "srinivasan_native_true_summary.csv",
    index=False,
)

pd.DataFrame(
    fold_metric_rows
).to_csv(
    OUT / "duke_fold_metrics.csv",
    index=False,
)

pd.DataFrame(
    fold_assignment_rows
).to_csv(
    OUT / "srinivasan_native_true_outer_fold_assignments.csv",
    index=False,
)

pd.DataFrame(
    fusion_coef_rows
).to_csv(
    OUT / "srinivasan_native_true_fusion_coefficients.csv",
    index=False,
)

protocol = {
    "cohort": "Srinivasan AMD/DME",
    "n_acquisitions": 30,
    "n_AMD": 15,
    "n_DME": 15,
    "evaluation_unit": "acquisition",
    "outer_folds": OUTER_FOLDS,
    "fusion_inner_folds": FUSION_INNER_FOLDS,
    "seed": SEED,
    "visual_architecture": {
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_blocks": N_BLOCKS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_canonical_z": MAX_CANONICAL_Z,
    },
    "fixed_source_selected_hyperparameters": {
        "visual_epochs": VISUAL_EPOCHS,
        "kg_C": KG_C,
        "fusion_C": META_C,
    },
    "srinivasan_specific_hyperparameter_tuning": False,
    "feature_dir": str(FEATURE_DIR),
    "kg_descriptor_csv": str(KG_DESC_CSV),
    "manifest_csv": str(MANIFEST_CSV),
}

with open(
    OUT / "srinivasan_native_true_protocol.json",
    "w",
) as f:
    json.dump(
        protocol,
        f,
        indent=2,
    )

print()
print("Saved:")
for p in [
    OUT / "duke_heldout_predictions.csv",
    OUT / "srinivasan_native_true_summary.csv",
    OUT / "duke_fold_metrics.csv",
    OUT / "srinivasan_native_true_outer_fold_assignments.csv",
    OUT / "srinivasan_native_true_fusion_coefficients.csv",
    OUT / "srinivasan_native_true_protocol.json",
]:
    print(" ", p)

print()
print("DONE")



# ============================================================
# DUKE MEDIAN-THRESHOLD CHECK
# Use the held-out probabilities produced by the current run.
# ============================================================

print()
print("=" * 100)
print(
    "DUKE — INNER-FOLD THRESHOLDS -> MEDIAN "
    "-> CURRENT RUN HELD-OUT PREDICTIONS"
)
print("=" * 100)

OLD_OUT = OUT

OLD_PRED = (
    OLD_OUT
    / "duke_heldout_predictions.csv"
)

OLD_FOLD = (
    OLD_OUT
    / "duke_fold_metrics.csv"
)

old_pred = pd.read_csv(
    OLD_PRED
)

old_fold = pd.read_csv(
    OLD_FOLD
)

threshold_df = (
    pd.DataFrame(
        median_threshold_rows
    )
    .sort_values(
        "outer_fold"
    )
    .reset_index(
        drop=True
    )
)

inner_threshold_df = (
    pd.DataFrame(
        inner_threshold_rows
    )
    .sort_values([
        "outer_fold",
        "method",
        "inner_fold",
    ])
    .reset_index(
        drop=True
    )
)

if len(threshold_df) != 5:
    raise RuntimeError(
        f"Expected 5 outer threshold rows, "
        f"got {len(threshold_df)}"
    )

if len(inner_threshold_df) != 75:
    raise RuntimeError(
        f"Expected 75 inner threshold rows, "
        f"got {len(inner_threshold_df)}"
    )

old_pred = old_pred.merge(
    threshold_df,
    on="outer_fold",
    how="left",
    validate="many_to_one",
)

old_pred[
    "pred_visual_median"
] = (
    old_pred[
        "prob_visual"
    ].to_numpy(dtype=float)
    >=
    old_pred[
        "threshold_visual"
    ].to_numpy(dtype=float)
).astype(int)

old_pred[
    "pred_kg_median"
] = (
    old_pred[
        "prob_kg"
    ].to_numpy(dtype=float)
    >=
    old_pred[
        "threshold_kg"
    ].to_numpy(dtype=float)
).astype(int)

old_pred[
    "pred_fusion_median"
] = (
    old_pred[
        "prob_fusion"
    ].to_numpy(dtype=float)
    >=
    old_pred[
        "threshold_fusion"
    ].to_numpy(dtype=float)
).astype(int)


specs = [
    (
        "Visual only",
        "prob_visual",
        "pred_visual_median",
    ),
    (
        "KG only",
        "prob_kg",
        "pred_kg_median",
    ),
    (
        "Visual + KG",
        "prob_fusion",
        "pred_fusion_median",
    ),
]


rows = []

for outer_fold in range(1, 6):

    sub = (
        old_pred[
            old_pred[
                "outer_fold"
            ] == outer_fold
        ]
        .copy()
    )

    y = (
        sub["label"]
        .to_numpy(dtype=int)
    )

    for method, pcol, dcol in specs:

        p = (
            sub[pcol]
            .to_numpy(dtype=float)
        )

        d = (
            sub[dcol]
            .to_numpy(dtype=int)
        )

        rows.append({
            "outer_fold":
                outer_fold,

            "method":
                method,

            "AUROC":
                float(
                    roc_auc_score(
                        y,
                        p,
                    )
                ),

            "balanced_accuracy":
                float(
                    balanced_accuracy_score(
                        y,
                        d,
                    )
                ),

            "macro_F1":
                float(
                    f1_score(
                        y,
                        d,
                        average="macro",
                    )
                ),

            "accuracy":
                float(
                    accuracy_score(
                        y,
                        d,
                    )
                ),
        })


result = pd.DataFrame(
    rows
)


# ============================================================
# HARD AUROC INVARIANCE TEST
# ============================================================

check = (
    old_fold[
        [
            "outer_fold",
            "method",
            "AUROC",
        ]
    ]
    .merge(
        result[
            [
                "outer_fold",
                "method",
                "AUROC",
            ]
        ],
        on=[
            "outer_fold",
            "method",
        ],
        suffixes=(
            "_old",
            "_new",
        ),
        validate="one_to_one",
    )
)

check[
    "abs_diff"
] = (
    check["AUROC_old"]
    -
    check["AUROC_new"]
).abs()

if not np.allclose(
    check["AUROC_old"],
    check["AUROC_new"],
    atol=1e-12,
    rtol=0.0,
):
    print(
        check.to_string(
            index=False
        )
    )
    raise RuntimeError(
        "STOP: AUROC changed."
    )

print()
print(
    "AUROC INVARIANCE: PASS"
)


# ============================================================
# OLD FIXED-0.5 vs MEDIAN-VALIDATION THRESHOLD
# ============================================================

print()
print(
    "OLD FIXED-0.5 vs "
    "INNER-FOLD-MEDIAN VALIDATION THRESHOLD"
)
print()

for method in [
    "Visual only",
    "KG only",
    "Visual + KG",
]:

    old_sub = (
        old_fold[
            old_fold[
                "method"
            ] == method
        ]
    )

    new_sub = (
        result[
            result[
                "method"
            ] == method
        ]
    )

    old_auc = float(
        old_sub[
            "AUROC"
        ].mean()
    )

    old_bacc = float(
        old_sub[
            "balanced_accuracy"
        ].mean()
    )

    new_auc = float(
        new_sub[
            "AUROC"
        ].mean()
    )

    new_bacc = float(
        new_sub[
            "balanced_accuracy"
        ].mean()
    )

    print(
        f"{method:<16} "
        f"AUROC {old_auc:.4f} -> {new_auc:.4f} | "
        f"BalAcc {old_bacc:.4f} -> {new_bacc:.4f}"
    )


print()
print(
    "FINAL MEDIAN THRESHOLDS:"
)

print(
    threshold_df.to_string(
        index=False
    )
)

print()
print(
    "ALL INNER-FOLD THRESHOLDS:"
)

print(
    inner_threshold_df.to_string(
        index=False
    )
)


# ============================================================
# Save check
# ============================================================

threshold_df.to_csv(
    OUT
    / "duke_median_thresholds.csv",
    index=False,
)

inner_threshold_df.to_csv(
    OUT
    / "duke_inner_fold_thresholds.csv",
    index=False,
)

result.to_csv(
    OUT
    / "duke_fold_metrics_median_VALIDATION_SELECTED.csv",
    index=False,
)

old_pred.to_csv(
    OUT
    / "duke_exact_old_predictions_median_VALIDATION_SELECTED.csv",
    index=False,
)

check.to_csv(
    OUT
    / "duke_AUROC_invariance_check.csv",
    index=False,
)

print()
print(
    "Saved:",
    OUT,
)
print("=" * 100)
