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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_CSV = Path("PATH_TO_NOOR_CANONICAL_DATASET") / "manifest.csv"

FEATURE_DIR = (
    ROOT / "datasets/noor/results/eyeclip_slice_features"
)

KG_DESC_CSV = (
    ROOT / "datasets/noor/results/kg_128_descriptors.csv"
)

OUT = (
    ROOT / "datasets/noor/results/table2_seed42"
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

# Predefined search spaces.
# Selection is performed ONLY inside each outer-training fold.
VISUAL_EPOCH_CANDIDATES = [3, 5, 7, 9, 11]
KG_C_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0]
META_C_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0]

# Defaults only for function signatures; final outer models do NOT
# use these values unless selected by inner CV.
KG_C_DEFAULT = 0.1
META_C_DEFAULT = 1.0

# Noor evaluation.
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

if len(frame) != 98:
    raise RuntimeError(f"Expected 98 Noor AMD/DME acquisitions, found {len(frame)}")

if frame["case_id"].duplicated().any():
    raise RuntimeError("Duplicate Noor case_id values.")

if not (
    ((frame["disease"] == "AMD") & (frame["label"] == 0))
    | ((frame["disease"] == "DME") & (frame["label"] == 1))
).all():
    raise RuntimeError("Unexpected AMD/DME label mapping.")

if (frame["disease"] == "AMD").sum() != 48:
    raise RuntimeError("Expected 48 AMD acquisitions.")

if (frame["disease"] == "DME").sum() != 50:
    raise RuntimeError("Expected 50 DME acquisitions.")

# Noor manifest has no separate subject identifier.
# Each acquisition is therefore the evaluation unit.
frame["group_id"] = frame["case_id"]

print("=" * 88)
print("NOOR METHOD REPLICATION")
print("=" * 88)
print("Device:", DEVICE)
print("Acquisitions:", len(frame))
print("AMD:", int((frame["label"] == 0).sum()))
print("DME:", int((frame["label"] == 1).sum()))
print("Evaluation unit: OCT acquisition")
print("Visual epoch search:", VISUAL_EPOCH_CANDIDATES)
print("KG C search:", KG_C_CANDIDATES)
print("Fusion C search:", META_C_CANDIDATES)
print("Hyperparameters selected by nested CV inside each outer-training fold.")


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


# Check all 98 visual files before training.
for case_id in frame["case_id"]:
    _ = load_eyeclip(case_id)

print("EyeCLIP check: PASS (98/98)")


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
        "Noor KG descriptor file has no case_id column."
    )

kg_raw["case_id"] = (
    kg_raw["case_id"]
    .astype(str)
)

if len(kg_raw) != 98:
    raise RuntimeError(
        f"Expected 98 Noor KG descriptor rows, got {len(kg_raw)}"
    )

if kg_raw["case_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate Noor KG descriptor case_id values."
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
        "Noor manifest/KG case sets do not match."
    )

print(
    "KG structural descriptor dimension:",
    len(KG_COLS),
)
print("KG check: PASS (98/98)")


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
    C=KG_C_DEFAULT,
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
    C=META_C_DEFAULT,
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
# Dataset-specific hyperparameter selection
# Performed ONLY inside each outer TRAIN partition
# ============================================================

def select_visual_epochs(
    outer_train,
    outer_id,
):
    folds = stratified_folds(
        outer_train,
        n_splits=FUSION_INNER_FOLDS,
        seed=SEED + 20000 + outer_id,
    )

    scores = []

    print("    Visual epoch selection:")

    for epochs in VISUAL_EPOCH_CANDIDATES:

        parts = []

        # Same initialization/data-order seed across epoch
        # candidates within a given inner fold.
        for inner_id, (
            inner_train,
            inner_hold,
        ) in enumerate(
            folds,
            start=1,
        ):
            pred, _ = fit_visual_fixed_epochs(
                inner_train,
                inner_hold,
                epochs=epochs,
                seed=(
                    SEED
                    + 21000
                    + 1000 * outer_id
                    + inner_id
                ),
                quiet=True,
            )

            parts.append(pred)

        oof = pd.concat(
            parts,
            ignore_index=True,
        )

        auc = float(
            roc_auc_score(
                oof["label"],
                oof["prob_visual"],
            )
        )

        scores.append(
            (epochs, auc)
        )

        print(
            f"      epochs={epochs:2d} "
            f"inner-AUROC={auc:.4f}"
        )

    # Highest AUROC; shortest epoch wins exact ties.
    selected = sorted(
        scores,
        key=lambda x: (
            -x[1],
            x[0],
        ),
    )[0]

    print(
        f"      SELECTED epochs={selected[0]} "
        f"(AUROC={selected[1]:.4f})"
    )

    return int(selected[0]), float(selected[1])


def select_kg_C(
    outer_train,
    outer_id,
):
    folds = stratified_folds(
        outer_train,
        n_splits=FUSION_INNER_FOLDS,
        seed=SEED + 30000 + outer_id,
    )

    scores = []

    print("    KG C selection:")

    for C in KG_C_CANDIDATES:

        parts = []

        for inner_id, (
            inner_train,
            inner_hold,
        ) in enumerate(
            folds,
            start=1,
        ):
            bundle = fit_kg_bundle(
                inner_train,
                C=C,
            )

            pred = predict_kg_bundle(
                bundle,
                inner_hold,
            )

            parts.append(pred)

        oof = pd.concat(
            parts,
            ignore_index=True,
        )

        auc = float(
            roc_auc_score(
                oof["label"],
                oof["prob_kg"],
            )
        )

        scores.append(
            (float(C), auc)
        )

        print(
            f"      C={C:g} "
            f"inner-AUROC={auc:.4f}"
        )

    # Highest AUROC; smaller C wins exact ties.
    selected = sorted(
        scores,
        key=lambda x: (
            -x[1],
            x[0],
        ),
    )[0]

    print(
        f"      SELECTED C={selected[0]:g} "
        f"(AUROC={selected[1]:.4f})"
    )

    return float(selected[0]), float(selected[1])


def select_meta_C(
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
        .to_numpy()
    )

    skf = StratifiedKFold(
        n_splits=FUSION_INNER_FOLDS,
        shuffle=True,
        random_state=SEED + 60000 + outer_id,
    )

    scores = []

    print("    Fusion C selection:")

    for C in META_C_CANDIDATES:

        pred_all = np.full(
            len(base_oof),
            np.nan,
            dtype=float,
        )

        for train_idx, hold_idx in skf.split(
            np.zeros(len(base_oof)),
            y,
        ):
            meta_train = (
                base_oof
                .iloc[train_idx]
                .reset_index(drop=True)
            )

            meta_hold = (
                base_oof
                .iloc[hold_idx]
                .reset_index(drop=True)
            )

            bundle = fit_meta_model(
                meta_train,
                C=C,
            )

            pred_all[hold_idx] = (
                predict_meta(
                    bundle,
                    meta_hold,
                )
            )

        if np.isnan(pred_all).any():
            raise RuntimeError(
                "Incomplete meta-model CV predictions."
            )

        auc = float(
            roc_auc_score(
                y,
                pred_all,
            )
        )

        scores.append(
            (float(C), auc)
        )

        print(
            f"      C={C:g} "
            f"inner-AUROC={auc:.4f}"
        )

    selected = sorted(
        scores,
        key=lambda x: (
            -x[1],
            x[0],
        ),
    )[0]

    print(
        f"      SELECTED C={selected[0]:g} "
        f"(AUROC={selected[1]:.4f})"
    )

    return float(selected[0]), float(selected[1])




# ============================================================
# TABLE-4-MATCHED VALIDATION-ONLY OPERATING THRESHOLD
# max BAcc -> max Macro-F1 -> closest to 0.5
# ============================================================

def select_threshold_table4(
    y,
    p,
):
    y = np.asarray(
        y,
        dtype=np.int64,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    unique = np.unique(p)

    if len(unique) < 2:
        raise RuntimeError(
            "Validation probabilities collapsed."
        )

    candidates = np.concatenate(
        [
            [0.0],
            (
                unique[:-1]
                + unique[1:]
            ) / 2.0,
            [1.0],
        ]
    )

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


def make_meta_threshold_oof(
    base_oof,
    outer_id,
    C,
):
    """
    Leakage-free meta-level probabilities used ONLY for
    threshold selection inside the current outer-training fold.
    """
    base_oof = (
        base_oof
        .reset_index(drop=True)
        .copy()
    )

    y = (
        base_oof["label"]
        .to_numpy()
    )

    skf = StratifiedKFold(
        n_splits=FUSION_INNER_FOLDS,
        shuffle=True,
        random_state=SEED + 60000 + outer_id,
    )

    pred_all = np.full(
        len(base_oof),
        np.nan,
        dtype=float,
    )

    for train_idx, hold_idx in skf.split(
        np.zeros(len(base_oof)),
        y,
    ):
        meta_train = (
            base_oof
            .iloc[train_idx]
            .reset_index(drop=True)
        )

        meta_hold = (
            base_oof
            .iloc[hold_idx]
            .reset_index(drop=True)
        )

        bundle = fit_meta_model(
            meta_train,
            C=C,
        )

        pred_all[hold_idx] = predict_meta(
            bundle,
            meta_hold,
        )

    if np.isnan(pred_all).any():
        raise RuntimeError(
            "Incomplete meta threshold OOF predictions."
        )

    return pred_all


# ============================================================
# Inner cross-fit used ONLY to train fusion inside outer TRAIN
# ============================================================

def make_inner_base_oof(
    outer_train,
    outer_id,
    visual_epochs,
    kg_C,
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
                epochs=visual_epochs,
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
            C=kg_C,
        )

        kg_pred = predict_kg_bundle(
            kg_bundle,
            inner_hold,
        )

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
fold_metric_rows = []
fusion_coef_rows = []
tuning_rows = []

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
    # Hyperparameter selection using OUTER TRAIN only
    # --------------------------------------------------------
    print("\n  [0/3] Inner-CV hyperparameter selection")

    selected_visual_epochs, visual_inner_auc = (
        select_visual_epochs(
            outer_train,
            outer_id,
        )
    )

    selected_kg_C, kg_inner_auc = (
        select_kg_C(
            outer_train,
            outer_id,
        )
    )

    # --------------------------------------------------------
    # 1. Full outer-train visual model -> outer hold
    # --------------------------------------------------------
    print("\n  [1/3] Visual-only outer model")

    visual_hold, final_visual_loss = (
        fit_visual_fixed_epochs(
            outer_train,
            outer_hold,
            epochs=selected_visual_epochs,
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
        C=selected_kg_C,
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
            visual_epochs=selected_visual_epochs,
            kg_C=selected_kg_C,
        )
    )

    selected_meta_C, meta_inner_auc = (
        select_meta_C(
            fusion_train_oof,
            outer_id,
        )
    )

    # --------------------------------------------------------
    # Validation-selected operating thresholds
    # OUTER HOLD IS NOT USED HERE.
    # --------------------------------------------------------

    threshold_y = (
        fusion_train_oof["label"]
        .to_numpy(dtype=np.int64)
    )

    threshold_visual = select_threshold_table4(
        threshold_y,
        fusion_train_oof[
            "prob_visual"
        ].to_numpy(dtype=float),
    )

    threshold_kg = select_threshold_table4(
        threshold_y,
        fusion_train_oof[
            "prob_kg"
        ].to_numpy(dtype=float),
    )

    fusion_threshold_oof = make_meta_threshold_oof(
        fusion_train_oof,
        outer_id=outer_id,
        C=selected_meta_C,
    )

    threshold_fusion = select_threshold_table4(
        threshold_y,
        fusion_threshold_oof,
    )

    print(
        "    Validation-selected thresholds:"
    )
    print(
        f"      Visual={threshold_visual:.8f} | "
        f"KG={threshold_kg:.8f} | "
        f"Fusion={threshold_fusion:.8f}"
    )

    meta_bundle = fit_meta_model(
        fusion_train_oof,
        C=selected_meta_C,
    )

    tuning_rows.append({
        "outer_fold": outer_id,
        "visual_epochs": selected_visual_epochs,
        "visual_inner_auc": visual_inner_auc,
        "kg_C": selected_kg_C,
        "kg_inner_auc": kg_inner_auc,
        "fusion_C": selected_meta_C,
        "fusion_inner_auc": meta_inner_auc,
    })

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

    hold_base["threshold_visual"] = float(
        threshold_visual
    )
    hold_base["threshold_kg"] = float(
        threshold_kg
    )
    hold_base["threshold_fusion"] = float(
        threshold_fusion
    )

    hold_base["pred_visual_validation_selected"] = (
        hold_base["prob_visual"].to_numpy(dtype=float)
        >= threshold_visual
    ).astype(int)

    hold_base["pred_kg_validation_selected"] = (
        hold_base["prob_kg"].to_numpy(dtype=float)
        >= threshold_kg
    ).astype(int)

    hold_base["pred_fusion_validation_selected"] = (
        hold_base["prob_fusion"].to_numpy(dtype=float)
        >= threshold_fusion
    ).astype(int)

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

if len(pred) != 98:
    raise RuntimeError(
        f"Expected 98 held-out predictions, got {len(pred)}"
    )

if pred["case_id"].nunique() != 98:
    raise RuntimeError(
        "Each Noor acquisition must have exactly one held-out prediction."
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
print("LEGACY FIXED-0.5 DIAGNOSTIC — NOT FINAL TABLE-2 RESULT")
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
    OUT / "noor_heldout_predictions.csv",
    index=False,
)

summary.to_csv(
    OUT / "noor_summary.csv",
    index=False,
)

pd.DataFrame(
    fold_metric_rows
).to_csv(
    OUT / "noor_fold_metrics.csv",
    index=False,
)

pd.DataFrame(
    fold_assignment_rows
).to_csv(
    OUT / "noor_outer_fold_assignments.csv",
    index=False,
)

pd.DataFrame(
    fusion_coef_rows
).to_csv(
    OUT / "noor_fusion_coefficients.csv",
    index=False,
)

pd.DataFrame(
    tuning_rows
).to_csv(
    OUT / "noor_selected_hyperparameters.csv",
    index=False,
)

protocol = {
    "cohort": "Noor AMD/DME",
    "n_acquisitions": 98,
    "n_AMD": 48,
    "n_DME": 50,
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
    "hyperparameter_selection": {
        "strategy": "nested_cv_outer_train_only",
        "selection_metric": "AUROC",
        "visual_epoch_candidates": VISUAL_EPOCH_CANDIDATES,
        "kg_C_candidates": KG_C_CANDIDATES,
        "fusion_C_candidates": META_C_CANDIDATES,
        "classification_threshold":
            "validation-selected inside each outer-training fold",
        "threshold_selection":
            "maximum balanced accuracy; macro-F1 tie-break; closest-to-0.5 final tie-break",
    },
    "noor_specific_hyperparameter_tuning": True,
    "outer_hold_used_for_selection": False,
    "feature_dir": str(FEATURE_DIR),
    "kg_descriptor_csv": str(KG_DESC_CSV),
    "manifest_csv": str(MANIFEST_CSV),
}

with open(
    OUT / "noor_protocol.json",
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
    OUT / "noor_heldout_predictions.csv",
    OUT / "noor_summary.csv",
    OUT / "noor_fold_metrics.csv",
    OUT / "noor_outer_fold_assignments.csv",
    OUT / "noor_fusion_coefficients.csv",
    OUT / "noor_protocol.json",
]:
    print(" ", p)

print()
print("DONE")



# ============================================================
# TABLE 2 VALIDATION-SELECTED EVALUATION
# ============================================================

print()
print("=" * 96)
print("AUTHORITATIVE NOOR TABLE-2 RESULT — VALIDATION-SELECTED OPERATING POINTS")
print("=" * 96)

VALIDATION_SELECTED_PRED = OUT / "noor_heldout_predictions.csv"

if not VALIDATION_SELECTED_PRED.exists():
    raise RuntimeError(
        f"Missing held-out predictions: {VALIDATION_SELECTED_PRED}"
    )

vpred = pd.read_csv(VALIDATION_SELECTED_PRED)

required_validation_selected = {
    "label",
    "prob_visual",
    "prob_kg",
    "prob_fusion",
    "pred_visual_validation_selected",
    "pred_kg_validation_selected",
    "pred_fusion_validation_selected",
    "threshold_visual",
    "threshold_kg",
    "threshold_fusion",
    "outer_fold",
}

missing_validation_selected = (
    required_validation_selected
    - set(vpred.columns)
)

if missing_validation_selected:
    raise RuntimeError(
        "Missing VALIDATION_SELECTED prediction columns: "
        + str(sorted(missing_validation_selected))
    )


def final_validation_selected_metrics(
    y,
    p,
    pred,
):
    y = np.asarray(
        y,
        dtype=int,
    )
    p = np.asarray(
        p,
        dtype=float,
    )
    pred = np.asarray(
        pred,
        dtype=int,
    )

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

        "BalancedAcc":
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


model_specs = [
    (
        "Visual",
        "prob_visual",
        "pred_visual_validation_selected",
    ),
    (
        "KG",
        "prob_kg",
        "pred_kg_validation_selected",
    ),
    (
        "Visual+KG",
        "prob_fusion",
        "pred_fusion_validation_selected",
    ),
]

summary_rows = []

for (
    name,
    prob_col,
    pred_col,
) in model_specs:

    m = final_validation_selected_metrics(
        vpred["label"],
        vpred[prob_col],
        vpred[pred_col],
    )

    summary_rows.append({
        "Model": name,
        **m,
    })

    print(
        f"{name:<12} "
        f"AUROC={m['AUROC']:.4f} | "
        f"AUPRC={m['AUPRC']:.4f} | "
        f"BalAcc={m['BalancedAcc']:.4f} | "
        f"MacroF1={m['MacroF1']:.4f} | "
        f"Acc={m['Accuracy']:.4f}"
    )


# ------------------------------------------------------------
# 10,000 ordinary acquisition-level bootstrap replicates.
# Thresholds remain frozen exactly as selected inside each
# outer-training fold.
# ------------------------------------------------------------

rng = np.random.RandomState(
    SEED + 91000
)

N_BOOT = 10000

bootstrap_rows = []

y_all = (
    vpred["label"]
    .to_numpy(dtype=int)
)

n_all = len(vpred)

for (
    name,
    prob_col,
    pred_col,
) in model_specs:

    p_all = (
        vpred[prob_col]
        .to_numpy(dtype=float)
    )

    d_all = (
        vpred[pred_col]
        .to_numpy(dtype=int)
    )

    auc_values = []
    bacc_values = []

    for _ in range(N_BOOT):

        idx = rng.randint(
            0,
            n_all,
            size=n_all,
        )

        yb = y_all[idx]

        if np.unique(yb).size < 2:
            continue

        pb = p_all[idx]
        db = d_all[idx]

        auc_values.append(
            roc_auc_score(
                yb,
                pb,
            )
        )

        bacc_values.append(
            balanced_accuracy_score(
                yb,
                db,
            )
        )

    auc_lo, auc_hi = np.percentile(
        auc_values,
        [2.5, 97.5],
    )

    ba_lo, ba_hi = np.percentile(
        bacc_values,
        [2.5, 97.5],
    )

    bootstrap_rows.append({
        "Model": name,
        "AUROC_CI_low": float(auc_lo),
        "AUROC_CI_high": float(auc_hi),
        "BalancedAcc_CI_low": float(ba_lo),
        "BalancedAcc_CI_high": float(ba_hi),
        "valid_bootstrap": len(auc_values),
    })

    print(
        f"{name:<12} "
        f"AUROC CI=[{auc_lo:.4f}, {auc_hi:.4f}] | "
        f"BalAcc CI=[{ba_lo:.4f}, {ba_hi:.4f}] | "
        f"bootstrap={len(auc_values)}/{N_BOOT}"
    )


pd.DataFrame(
    summary_rows
).to_csv(
    OUT / "noor_summary_VALIDATION_SELECTED.csv",
    index=False,
)

pd.DataFrame(
    bootstrap_rows
).to_csv(
    OUT / "noor_bootstrap_VALIDATION_SELECTED.csv",
    index=False,
)

threshold_check = (
    vpred[
        [
            "outer_fold",
            "threshold_visual",
            "threshold_kg",
            "threshold_fusion",
        ]
    ]
    .drop_duplicates()
    .sort_values("outer_fold")
)

threshold_check.to_csv(
    OUT / "noor_thresholds_VALIDATION_SELECTED.csv",
    index=False,
)

print()
print("Per-fold frozen thresholds:")
print(
    threshold_check.to_string(
        index=False
    )
)

print()
print("AUTHORITATIVE FILES:")
print(
    OUT / "noor_summary_VALIDATION_SELECTED.csv"
)
print(
    OUT / "noor_bootstrap_VALIDATION_SELECTED.csv"
)
print(
    OUT / "noor_thresholds_VALIDATION_SELECTED.csv"
)
print("=" * 96)
