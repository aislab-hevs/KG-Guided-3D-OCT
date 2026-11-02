# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.

from pathlib import Path
import json
import math
import random

import joblib
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

FEATURE_DIR = (
    ROOT / "datasets/huang/results/eyeclip_slice_features"
)

SPLIT_CSV = Path("PATH_TO_HUANG_SUBJECT_SPLIT_CSV")

KG_DESC_CSV = (
    ROOT
    / "datasets/huang/results/kg_descriptors/train_val_graph_descriptors.csv"
)

OUT = (
    ROOT
    / "datasets/huang/results/table2_seed42"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Frozen visual architecture/training hyperparameters.
D_MODEL = 256
N_HEADS = 4
N_BLOCKS = 2
DROPOUT = 0.1

BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-4

MAX_CANONICAL_Z = 512

# Cross-fitting protocol.
OUTER_FOLDS = 5
INNER_VISUAL_FOLDS = 3
MAX_EPOCHS = 50

KG_INNER_FOLDS = 4

KG_C_GRID = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

META_C_GRID = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


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
# Cohort helpers
# ============================================================

def subject_table(frame):
    bad = (
        frame.groupby("group_id")
        ["disease"]
        .nunique()
    )

    if (bad > 1).any():
        raise RuntimeError(
            "Inconsistent disease labels "
            "within a subject/group."
        )

    return (
        frame.groupby(
            "group_id",
            as_index=False,
        )
        .agg(
            disease=("disease", "first")
        )
        .reset_index(drop=True)
    )


def subject_folds(
    frame,
    n_splits,
    seed,
):
    """
    Subject-disjoint disease-stratified folds.
    """
    s = subject_table(frame)

    y = (
        s["disease"]
        == "DME"
    ).astype(int).to_numpy()

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    out = []

    for train_idx, hold_idx in skf.split(
        np.zeros(len(s)),
        y,
    ):
        train_groups = set(
            s.iloc[train_idx]["group_id"]
        )

        hold_groups = set(
            s.iloc[hold_idx]["group_id"]
        )

        tr = (
            frame[
                frame["group_id"]
                .isin(train_groups)
            ]
            .copy()
            .reset_index(drop=True)
        )

        ho = (
            frame[
                frame["group_id"]
                .isin(hold_groups)
            ]
            .copy()
            .reset_index(drop=True)
        )

        assert (
            set(tr["group_id"])
            .isdisjoint(
                set(ho["group_id"])
            )
        )

        out.append(
            (tr, ho)
        )

    return out


def subject_weights(groups):
    groups = np.asarray(
        groups
    )

    unique, counts = np.unique(
        groups,
        return_counts=True,
    )

    count_map = dict(
        zip(
            unique,
            counts,
        )
    )

    w = np.asarray(
        [
            1.0 / count_map[g]
            for g in groups
        ],
        dtype=np.float64,
    )

    return (
        w
        * len(w)
        / w.sum()
    )


# ============================================================
# Metrics
# ============================================================

def aggregate_subjects(
    y,
    p,
    groups,
):
    d = pd.DataFrame({
        "group_id": groups,
        "label": y,
        "prob_DME": p,
    })

    bad = (
        d.groupby("group_id")
        ["label"]
        .nunique()
    )

    if (bad > 1).any():
        raise RuntimeError(
            "Inconsistent labels "
            "within a subject/group."
        )

    return (
        d.groupby(
            "group_id",
            as_index=False,
        )
        .agg(
            label=("label", "first"),
            prob_DME=(
                "prob_DME",
                "mean",
            ),
            n_volumes=(
                "prob_DME",
                "size",
            ),
        )
    )


def metric_dict(
    y,
    p,
):
    y = np.asarray(y)
    p = np.asarray(p)

    pred = (
        p >= 0.5
    ).astype(int)

    return {
        "accuracy":
            float(
                accuracy_score(
                    y,
                    pred,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y,
                    pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "auroc":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "auprc":
            float(
                average_precision_score(
                    y,
                    p,
                )
            ),
    }


def report(
    name,
    y,
    p,
    groups,
):
    vm = metric_dict(
        y,
        p,
    )

    sub = aggregate_subjects(
        y,
        p,
        groups,
    )

    sm = metric_dict(
        sub["label"],
        sub["prob_DME"],
    )

    print(
        f"\n{name}"
    )

    print(
        "-" * len(name)
    )

    print(
        "Volume  | "
        f"AUROC={vm['auroc']:.4f} | "
        f"AUPRC={vm['auprc']:.4f} | "
        f"BalAcc={vm['balanced_accuracy']:.4f} | "
        f"MacroF1={vm['macro_f1']:.4f} | "
        f"Acc={vm['accuracy']:.4f}"
    )

    print(
        "Subject | "
        f"AUROC={sm['auroc']:.4f} | "
        f"AUPRC={sm['auprc']:.4f} | "
        f"BalAcc={sm['balanced_accuracy']:.4f} | "
        f"MacroF1={sm['macro_f1']:.4f} | "
        f"Acc={sm['accuracy']:.4f}"
    )

    return (
        vm,
        sm,
        sub,
    )


# ============================================================
# Canonical TRAIN / VALIDATION
# ============================================================

split_all = pd.read_csv(
    SPLIT_CSV
)

split_all["case_id"] = (
    split_all["case_id"]
    .astype(str)
)

split_all["group_id"] = (
    split_all["group_id"]
    .astype(str)
)

assert len(split_all) == 192

assert (
    split_all["split"]
    == "train"
).sum() == 126

assert (
    split_all["split"]
    == "val"
).sum() == 34

assert (
    split_all["split"]
    == "test"
).sum() == 32

train_df = (
    split_all[
        split_all["split"]
        == "train"
    ]
    .copy()
    .reset_index(drop=True)
)

val_df = (
    split_all[
        split_all["split"]
        == "val"
    ]
    .copy()
    .reset_index(drop=True)
)

assert (
    train_df["group_id"]
    .nunique()
    == 94
)

assert (
    val_df["group_id"]
    .nunique()
    == 20
)

assert (
    set(
        train_df["group_id"]
    )
    .isdisjoint(
        set(
            val_df["group_id"]
        )
    )
)

print(
    "=" * 82
)

print(
    "FINAL SUBJECT-DISJOINT "
    "CROSSFIT VISUAL + KG FUSION"
)

print(
    "=" * 82
)

print(
    f"Device: {DEVICE}"
)

print(
    "TRAIN:",
    len(train_df),
    "volumes /",
    train_df["group_id"]
    .nunique(),
    "subjects",
)

print(
    "VALIDATION:",
    len(val_df),
    "volumes /",
    val_df["group_id"]
    .nunique(),
    "subjects",
)

print(
    "TEST images/features/predictions "
    "are NOT loaded or evaluated."
)

print(
    "Visual epoch selection: "
    f"{INNER_VISUAL_FOLDS}-fold "
    "inner subject CV, "
    f"epochs 1..{MAX_EPOCHS}, "
    "maximize mean subject AUROC, "
    "earliest tie."
)


# ============================================================
# Frozen EyeCLIP slice features
# ============================================================

def load_eyeclip(case_id):
    path = (
        FEATURE_DIR
        / f"{case_id}.npy"
    )

    x = np.load(
        path
    ).astype(
        np.float32
    )

    if (
        x.ndim != 2
        or x.shape[1] != 512
    ):
        raise RuntimeError(
            f"{case_id}: "
            f"unexpected EyeCLIP "
            f"shape {x.shape}"
        )

    norm = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    x = (
        x
        / np.clip(
            norm,
            1e-12,
            None,
        )
    )

    return x


def fit_visual_scaler(frame):
    pooled = np.stack([
        load_eyeclip(
            case_id
        ).mean(axis=0)

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

    return (
        mu,
        sigma,
    )


class VisualDataset(Dataset):

    def __init__(
        self,
        frame,
        mu,
        sigma,
        weighted,
    ):
        self.df = (
            frame
            .reset_index(drop=True)
        )

        self.mu = mu
        self.sigma = sigma

        if weighted:
            counts = (
                self.df
                .groupby("group_id")
                .size()
                .to_dict()
            )

            self.weights = np.asarray(
                [
                    1.0
                    / counts[g]

                    for g
                    in self.df["group_id"]
                ],
                dtype=np.float32,
            )

        else:
            self.weights = np.ones(
                len(self.df),
                dtype=np.float32,
            )

    def __len__(self):
        return len(
            self.df
        )

    def __getitem__(
        self,
        idx,
    ):
        row = self.df.iloc[idx]

        x = load_eyeclip(
            str(
                row["case_id"]
            )
        )

        x = (
            x
            - self.mu[None, :]
        ) / self.sigma[None, :]

        y = (
            0
            if row["disease"]
            == "AMD"
            else 1
        )

        return {
            "case_id":
                str(
                    row["case_id"]
                ),

            "group_id":
                str(
                    row["group_id"]
                ),

            "x":
                torch.from_numpy(
                    x.astype(
                        np.float32,
                        copy=False,
                    )
                ),

            "label":
                torch.tensor(
                    y,
                    dtype=torch.long,
                ),

            "weight":
                torch.tensor(
                    self.weights[idx],
                    dtype=torch.float32,
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
    weights = []
    cases = []
    groups = []

    for i, item in enumerate(
        batch
    ):
        z = item["x"].shape[0]

        x[i, :z] = (
            item["x"]
        )

        valid[i, :z] = True

        labels.append(
            item["label"]
        )

        weights.append(
            item["weight"]
        )

        cases.append(
            item["case_id"]
        )

        groups.append(
            item["group_id"]
        )

    return {
        "x":
            x,

        "valid":
            valid,

        "label":
            torch.stack(
                labels
            ),

        "weight":
            torch.stack(
                weights
            ),

        "case_id":
            cases,

        "group_id":
            groups,
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
        pos[:, None]
        * div[None, :]
    )

    pe[:, 1::2] = torch.cos(
        pos[:, None]
        * div[None, :]
    )

    return pe


class MultiHeadSelfAttention(
    nn.Module
):

    def __init__(
        self,
        dim,
        heads,
        dropout,
    ):
        super().__init__()

        assert (
            dim % heads
            == 0
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
            attn
            @ v
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
            * valid[:, :, None]
        )


class TransformerBlock(
    nn.Module
):

    def __init__(
        self,
        dim,
        heads,
        dropout,
    ):
        super().__init__()

        self.norm1 = (
            nn.LayerNorm(
                dim
            )
        )

        self.attn = (
            MultiHeadSelfAttention(
                dim,
                heads,
                dropout,
            )
        )

        self.norm2 = (
            nn.LayerNorm(
                dim
            )
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
            * valid[:, :, None]
        )


class AttentionPool(
    nn.Module
):

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
            alpha[:, :, None]
            * x,
            dim=1,
        )


class B2Model(
    nn.Module
):

    def __init__(self):
        super().__init__()

        self.proj = nn.Linear(
            512,
            D_MODEL,
        )

        self.blocks = (
            nn.ModuleList([
                TransformerBlock(
                    D_MODEL,
                    N_HEADS,
                    DROPOUT,
                )

                for _ in range(
                    N_BLOCKS
                )
            ])
        )

        self.pool = (
            AttentionPool(
                D_MODEL
            )
        )

        self.classifier = (
            nn.Linear(
                D_MODEL,
                2,
            )
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

        return (
            self.classifier(
                x
            )
        )


probe_model = B2Model()

print(
    "B2 trainable parameters:",
    sum(
        p.numel()
        for p
        in probe_model.parameters()
    ),
)

del probe_model


# ============================================================
# Visual train / predict
# ============================================================

def build_visual_loaders(
    train_frame,
    eval_frame,
    seed,
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
        weighted=True,
    )

    ev_ds = VisualDataset(
        eval_frame,
        mu,
        sigma,
        weighted=False,
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        seed
    )

    tr_loader = DataLoader(
        tr_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=
            collate_visual,
        pin_memory=True,
    )

    ev_loader = DataLoader(
        ev_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=
            collate_visual,
        pin_memory=True,
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

    numerator = 0.0
    denominator = 0.0

    for batch in loader:
        x = batch["x"].to(
            DEVICE
        )

        valid = (
            batch["valid"]
            .to(DEVICE)
        )

        y = (
            batch["label"]
            .to(DEVICE)
        )

        w = (
            batch["weight"]
            .to(DEVICE)
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            x,
            valid,
        )

        each = F.cross_entropy(
            logits,
            y,
            reduction="none",
        )

        loss = (
            (each * w).sum()
            / w.sum()
        )

        loss.backward()
        optimizer.step()

        numerator += float(
            (
                each.detach()
                * w.detach()
            )
            .sum()
            .cpu()
        )

        denominator += float(
            w.detach()
            .sum()
            .cpu()
        )

    return (
        numerator
        / denominator
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
            batch["x"].to(
                DEVICE
            ),
            batch["valid"].to(
                DEVICE
            ),
        )

        p = (
            torch.softmax(
                logits,
                dim=1,
            )[:, 1]
            .cpu()
            .numpy()
        )

        y = (
            batch["label"]
            .numpy()
        )

        for (
            case_id,
            group_id,
            label,
            prob,
        ) in zip(
            batch["case_id"],
            batch["group_id"],
            y,
            p,
        ):
            rows.append({
                "case_id":
                    case_id,

                "group_id":
                    group_id,

                "label":
                    int(label),

                "prob_visual":
                    float(prob),
            })

    return pd.DataFrame(
        rows
    )


@torch.no_grad()
def visual_subject_auc(
    model,
    loader,
):
    pred = predict_visual(
        model,
        loader,
    )

    sub = aggregate_subjects(
        pred["label"]
        .to_numpy(),

        pred["prob_visual"]
        .to_numpy(),

        pred["group_id"]
        .to_numpy(),
    )

    return float(
        roc_auc_score(
            sub["label"],
            sub["prob_DME"],
        )
    )


def select_visual_epoch_cv(
    frame,
    n_splits,
    seed,
    stage_name,
):
    """
    Train all inner folds through epochs 1..MAX_EPOCHS.
    Select earliest epoch maximizing the mean
    inner-fold subject AUROC.
    """

    folds = subject_folds(
        frame,
        n_splits=n_splits,
        seed=seed,
    )

    fold_curves = []
    detail_rows = []

    print(
        f"\n{stage_name}: "
        "visual epoch selection"
    )

    for fold_id, (
        tr,
        va,
    ) in enumerate(
        folds,
        start=1,
    ):
        print(
            f"  inner {fold_id}/"
            f"{n_splits}: "
            f"{tr['group_id'].nunique()} "
            "train subjects, "
            f"{va['group_id'].nunique()} "
            "val subjects"
        )

        (
            tr_loader,
            va_loader,
            _,
            _,
        ) = build_visual_loaders(
            tr,
            va,
            seed=
                seed
                + 100
                * fold_id,
        )

        seed_everything(
            seed
            + 1000
            * fold_id
        )

        model = (
            B2Model()
            .to(DEVICE)
        )

        optimizer = (
            torch.optim.AdamW(
                model.parameters(),
                lr=LR,
                weight_decay=
                    WEIGHT_DECAY,
            )
        )

        auc_curve = []

        for epoch in range(
            1,
            MAX_EPOCHS + 1,
        ):
            loss = (
                train_visual_epoch(
                    model,
                    tr_loader,
                    optimizer,
                )
            )

            auc = (
                visual_subject_auc(
                    model,
                    va_loader,
                )
            )

            auc_curve.append(
                auc
            )

            detail_rows.append({
                "stage":
                    stage_name,

                "inner_fold":
                    fold_id,

                "epoch":
                    epoch,

                "train_loss":
                    float(loss),

                "subject_AUROC":
                    float(auc),
            })

            if (
                epoch == 1
                or epoch % 10 == 0
                or epoch == MAX_EPOCHS
            ):
                print(
                    f"    epoch "
                    f"{epoch:02d}/"
                    f"{MAX_EPOCHS} | "
                    f"loss={loss:.4f} | "
                    f"Sub AUROC="
                    f"{auc:.4f}"
                )

        fold_curves.append(
            auc_curve
        )

        del model
        del optimizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    curves = np.asarray(
        fold_curves,
        dtype=np.float64,
    )

    mean_auc = (
        curves.mean(axis=0)
    )

    std_auc = (
        curves.std(axis=0)
    )

    # np.argmax returns first maximum:
    # earliest epoch wins exact ties.
    best_idx = int(
        np.argmax(
            mean_auc
        )
    )

    best_epoch = (
        best_idx + 1
    )

    summary = pd.DataFrame({
        "epoch":
            np.arange(
                1,
                MAX_EPOCHS + 1,
            ),

        "mean_subject_AUROC":
            mean_auc,

        "std_subject_AUROC":
            std_auc,
    })

    print(
        f"  selected epoch="
        f"{best_epoch} | "
        f"mean inner Sub AUROC="
        f"{mean_auc[best_idx]:.4f} "
        f"+/- "
        f"{std_auc[best_idx]:.4f}"
    )

    return (
        best_epoch,
        pd.DataFrame(
            detail_rows
        ),
        summary,
    )


def fit_visual_fixed_epochs(
    train_frame,
    eval_frame,
    epochs,
    seed,
):
    (
        tr_loader,
        ev_loader,
        mu,
        sigma,
    ) = build_visual_loaders(
        train_frame,
        eval_frame,
        seed=seed,
    )

    seed_everything(
        seed + 1
    )

    model = (
        B2Model()
        .to(DEVICE)
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=
                WEIGHT_DECAY,
        )
    )

    last_loss = None

    for _epoch in range(
        1,
        epochs + 1,
    ):
        last_loss = (
            train_visual_epoch(
                model,
                tr_loader,
                optimizer,
            )
        )

    pred = predict_visual(
        model,
        ev_loader,
    )

    return (
        model,
        mu,
        sigma,
        pred,
        float(last_loss),
    )


# ============================================================
# KG structural descriptors
# ============================================================

kg_desc = pd.read_csv(
    KG_DESC_CSV
)

kg_desc["case_id"] = (
    kg_desc["case_id"]
    .astype(str)
)

kg_desc["group_id"] = (
    kg_desc["group_id"]
    .astype(str)
)

# TRAIN + VALIDATION descriptors only.
assert len(kg_desc) == 160

KG_META = {
    "case_id",
    "group_id",
    "disease",
    "split",
}

KG_COLS = [
    c
    for c in kg_desc.columns
    if c not in KG_META
]

print(
    "KG structural "
    "descriptor dimension:",
    len(KG_COLS),
)


def kg_rows_for(frame):
    wanted = list(
        frame["case_id"]
        .astype(str)
    )

    order = {
        case_id: i

        for i, case_id
        in enumerate(
            wanted
        )
    }

    d = (
        kg_desc[
            kg_desc["case_id"]
            .isin(
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
        d.sort_values(
            "_order"
        )
        .drop(
            columns="_order"
        )
        .reset_index(drop=True)
    )

    if (
        list(d["case_id"])
        != wanted
    ):
        missing = sorted(
            set(wanted)
            - set(
                d["case_id"]
            )
        )

        raise RuntimeError(
            "Missing KG descriptors "
            f"for cases: {missing[:10]}"
        )

    return d


def fit_kg_bundle(
    train_frame,
    C,
):
    tr = kg_rows_for(
        train_frame
    )

    medians = (
        tr[KG_COLS]
        .median(
            axis=0,
            skipna=True,
        )
    )

    Xdf = (
        tr[KG_COLS]
        .fillna(
            medians
        )
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
            "All KG descriptor "
            "columns are constant."
        )

    X = (
        Xdf[keep_cols]
        .to_numpy(
            dtype=np.float64
        )
    )

    y = (
        tr["disease"]
        == "DME"
    ).astype(int).to_numpy()

    groups = (
        tr["group_id"]
        .to_numpy()
    )

    scaler = (
        StandardScaler()
    )

    Xs = (
        scaler.fit_transform(
            X
        )
    )

    clf = LogisticRegression(
        C=C,
        solver="liblinear",
        max_iter=5000,
        random_state=SEED,
    )

    clf.fit(
        Xs,
        y,
        sample_weight=
            subject_weights(
                groups
            ),
    )

    return {
        "C":
            float(C),

        "feature_cols":
            list(KG_COLS),

        "keep_cols":
            keep_cols,

        "medians":
            medians.to_dict(),

        "scaler":
            scaler,

        "clf":
            clf,
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
            bundle[
                "feature_cols"
            ]
        ]
        .fillna(med)
        .fillna(0.0)
    )

    X = (
        Xdf[
            bundle[
                "keep_cols"
            ]
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
        .predict_proba(
            Xs
        )[:, 1]
    )

    return pd.DataFrame({
        "case_id":
            ev["case_id"],

        "group_id":
            ev["group_id"],

        "label":
            (
                ev["disease"]
                == "DME"
            ).astype(int),

        "prob_kg":
            p,
    })


def select_kg_C(
    train_frame,
    n_splits,
    seed,
    stage_name,
):
    folds = subject_folds(
        train_frame,
        n_splits=n_splits,
        seed=seed,
    )

    rows = []

    print(
        f"\n{stage_name}: "
        "KG C selection"
    )

    for C in KG_C_GRID:
        aucs = []

        for tr, va in folds:
            bundle = (
                fit_kg_bundle(
                    tr,
                    C,
                )
            )

            pred = (
                predict_kg_bundle(
                    bundle,
                    va,
                )
            )

            sub = aggregate_subjects(
                pred["label"]
                .to_numpy(),

                pred["prob_kg"]
                .to_numpy(),

                pred["group_id"]
                .to_numpy(),
            )

            aucs.append(
                float(
                    roc_auc_score(
                        sub["label"],
                        sub["prob_DME"],
                    )
                )
            )

        mean_auc = float(
            np.mean(
                aucs
            )
        )

        std_auc = float(
            np.std(
                aucs
            )
        )

        print(
            f"  C={C:<6} "
            f"mean Sub AUROC="
            f"{mean_auc:.4f} "
            f"+/- {std_auc:.4f}"
        )

        rows.append({
            "stage":
                stage_name,

            "C":
                float(C),

            "mean_subject_AUROC":
                mean_auc,

            "std_subject_AUROC":
                std_auc,
        })

    table = pd.DataFrame(
        rows
    )

    best_C = float(
        table.sort_values(
            [
                "mean_subject_AUROC",
                "C",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]["C"]
    )

    print(
        "  selected KG C:",
        best_C,
    )

    return (
        best_C,
        table,
    )


# ============================================================
# OUTER CROSSFIT
# ============================================================

outer_folds = subject_folds(
    train_df,
    n_splits=OUTER_FOLDS,
    seed=SEED,
)

visual_oof_parts = []
kg_oof_parts = []

visual_epoch_detail_parts = []
visual_epoch_summary_parts = []

kg_C_parts = []

outer_summary_rows = []


for outer_id, (
    outer_train,
    outer_hold,
) in enumerate(
    outer_folds,
    start=1,
):
    print(
        "\n"
        + "=" * 82
    )

    print(
        f"OUTER FOLD "
        f"{outer_id}/"
        f"{OUTER_FOLDS}"
    )

    print(
        "=" * 82
    )

    print(
        "Outer train:",
        len(outer_train),
        "volumes /",
        outer_train[
            "group_id"
        ].nunique(),
        "subjects",
    )

    print(
        "Outer hold :",
        len(outer_hold),
        "volumes /",
        outer_hold[
            "group_id"
        ].nunique(),
        "subjects",
    )

    stage = (
        f"outer_{outer_id}"
    )

    # --------------------------------------------------------
    # Visual epoch selection:
    # inner CV only.
    # --------------------------------------------------------

    (
        selected_epoch,
        detail,
        epoch_summary,
    ) = (
        select_visual_epoch_cv(
            outer_train,
            n_splits=
                INNER_VISUAL_FOLDS,
            seed=
                SEED
                + 10000
                + outer_id,
            stage_name=
                stage,
        )
    )

    detail[
        "outer_fold"
    ] = outer_id

    epoch_summary[
        "outer_fold"
    ] = outer_id

    visual_epoch_detail_parts.append(
        detail
    )

    visual_epoch_summary_parts.append(
        epoch_summary
    )

    # --------------------------------------------------------
    # Refit B2 from scratch on full outer TRAIN.
    # --------------------------------------------------------

    (
        visual_model,
        _,
        _,
        visual_pred,
        final_loss,
    ) = (
        fit_visual_fixed_epochs(
            outer_train,
            outer_hold,
            epochs=
                selected_epoch,
            seed=
                SEED
                + 20000
                + outer_id,
        )
    )

    visual_oof_parts.append(
        visual_pred
    )

    del visual_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # KG selection and OOF prediction.
    # --------------------------------------------------------

    (
        best_kg_C,
        kg_C_table,
    ) = (
        select_kg_C(
            outer_train,
            n_splits=
                KG_INNER_FOLDS,
            seed=
                SEED
                + 30000
                + outer_id,
            stage_name=
                stage,
        )
    )

    kg_C_table[
        "outer_fold"
    ] = outer_id

    kg_C_parts.append(
        kg_C_table
    )

    kg_bundle = (
        fit_kg_bundle(
            outer_train,
            best_kg_C,
        )
    )

    kg_pred = (
        predict_kg_bundle(
            kg_bundle,
            outer_hold,
        )
    )

    kg_oof_parts.append(
        kg_pred
    )

    outer_summary_rows.append({
        "outer_fold":
            outer_id,

        "visual_selected_epoch":
            int(
                selected_epoch
            ),

        "visual_refit_final_loss":
            float(
                final_loss
            ),

        "kg_selected_C":
            float(
                best_kg_C
            ),

        "outer_train_subjects":
            int(
                outer_train[
                    "group_id"
                ].nunique()
            ),

        "outer_hold_subjects":
            int(
                outer_hold[
                    "group_id"
                ].nunique()
            ),
    })


# ============================================================
# Complete OOF predictions
# ============================================================

visual_oof = pd.concat(
    visual_oof_parts,
    ignore_index=True,
)

kg_oof = pd.concat(
    kg_oof_parts,
    ignore_index=True,
)

oof = visual_oof.merge(
    kg_oof,
    on=[
        "case_id",
        "group_id",
        "label",
    ],
    validate="one_to_one",
)

assert len(oof) == 126

assert (
    oof["case_id"]
    .nunique()
    == 126
)

assert (
    oof["group_id"]
    .nunique()
    == 94
)

canonical_order = {
    case_id: i

    for i, case_id
    in enumerate(
        train_df["case_id"]
    )
}

oof["_order"] = (
    oof["case_id"]
    .map(
        canonical_order
    )
)

oof = (
    oof.sort_values(
        "_order"
    )
    .drop(
        columns="_order"
    )
    .reset_index(drop=True)
)

oof.to_csv(
    OUT
    / "train_OOF_base_predictions.csv",
    index=False,
)

pd.concat(
    visual_epoch_detail_parts,
    ignore_index=True,
).to_csv(
    OUT
    / "outer_visual_inner_epoch_details.csv",
    index=False,
)

pd.concat(
    visual_epoch_summary_parts,
    ignore_index=True,
).to_csv(
    OUT
    / "outer_visual_epoch_mean_curves.csv",
    index=False,
)

pd.concat(
    kg_C_parts,
    ignore_index=True,
).to_csv(
    OUT
    / "outer_KG_C_selection.csv",
    index=False,
)

pd.DataFrame(
    outer_summary_rows
).to_csv(
    OUT
    / "outer_crossfit_summary.csv",
    index=False,
)


print(
    "\n"
    + "=" * 82
)

print(
    "OOF BASE PERFORMANCE"
)

print(
    "=" * 82
)


(
    _,
    visual_oof_sm,
    _,
) = report(
    "OOF VISUAL B2",
    oof["label"]
    .to_numpy(),

    oof["prob_visual"]
    .to_numpy(),

    oof["group_id"]
    .to_numpy(),
)


(
    _,
    kg_oof_sm,
    _,
) = report(
    "OOF KG",
    oof["label"]
    .to_numpy(),

    oof["prob_kg"]
    .to_numpy(),

    oof["group_id"]
    .to_numpy(),
)


# ============================================================
# TRAIN-only learned fusion
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
        p
        / (1.0 - p)
    )


def make_meta_X(
    frame,
    visual_col="prob_visual",
    kg_col="prob_kg",
):
    return np.column_stack([
        to_logit(
            frame[
                visual_col
            ]
        ),

        to_logit(
            frame[
                kg_col
            ]
        ),
    ])


def fit_meta_model(
    X,
    y,
    groups,
    C,
):
    scaler = (
        StandardScaler()
    )

    Xs = (
        scaler.fit_transform(
            X
        )
    )

    clf = LogisticRegression(
        C=C,
        solver="liblinear",
        max_iter=5000,
        random_state=SEED,
    )

    clf.fit(
        Xs,
        y,
        sample_weight=
            subject_weights(
                groups
            ),
    )

    return {
        "C":
            float(C),

        "scaler":
            scaler,

        "clf":
            clf,
    }


def predict_meta(
    bundle,
    X,
):
    Xs = (
        bundle["scaler"]
        .transform(X)
    )

    return (
        bundle["clf"]
        .predict_proba(
            Xs
        )[:, 1]
    )


X_meta = make_meta_X(
    oof
)

y_meta = (
    oof["label"]
    .to_numpy()
)

g_meta = (
    oof["group_id"]
    .to_numpy()
)


meta_folds = subject_folds(
    train_df,
    n_splits=5,
    seed=
        SEED + 40000,
)

meta_rows = []


print(
    "\n"
    + "=" * 82
)

print(
    "TRAIN-ONLY META-FUSION "
    "C SELECTION"
)

print(
    "=" * 82
)


for C in META_C_GRID:
    aucs = []

    for (
        meta_train_frame,
        meta_val_frame,
    ) in meta_folds:

        train_cases = set(
            meta_train_frame[
                "case_id"
            ]
        )

        val_cases = set(
            meta_val_frame[
                "case_id"
            ]
        )

        tr_mask = (
            oof["case_id"]
            .isin(
                train_cases
            )
            .to_numpy()
        )

        va_mask = (
            oof["case_id"]
            .isin(
                val_cases
            )
            .to_numpy()
        )

        bundle = fit_meta_model(
            X_meta[
                tr_mask
            ],
            y_meta[
                tr_mask
            ],
            g_meta[
                tr_mask
            ],
            C,
        )

        p = predict_meta(
            bundle,
            X_meta[
                va_mask
            ],
        )

        sub = aggregate_subjects(
            y_meta[
                va_mask
            ],
            p,
            g_meta[
                va_mask
            ],
        )

        aucs.append(
            float(
                roc_auc_score(
                    sub["label"],
                    sub["prob_DME"],
                )
            )
        )

    mean_auc = float(
        np.mean(
            aucs
        )
    )

    std_auc = float(
        np.std(
            aucs
        )
    )

    print(
        f"C={C:<6} "
        f"mean Sub AUROC="
        f"{mean_auc:.4f} "
        f"+/- {std_auc:.4f}"
    )

    meta_rows.append({
        "C":
            float(C),

        "mean_subject_AUROC":
            mean_auc,

        "std_subject_AUROC":
            std_auc,
    })


meta_cv = pd.DataFrame(
    meta_rows
)

best_meta_C = float(
    meta_cv.sort_values(
        [
            "mean_subject_AUROC",
            "C",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .iloc[0]["C"]
)

print(
    "Selected meta C:",
    best_meta_C,
)


meta_bundle = (
    fit_meta_model(
        X_meta,
        y_meta,
        g_meta,
        best_meta_C,
    )
)

joblib.dump(
    meta_bundle,
    OUT
    / "final_train_only_meta_fusion.joblib",
)

meta_cv.to_csv(
    OUT
    / "meta_C_selection.csv",
    index=False,
)


meta_coef = (
    meta_bundle["clf"]
    .coef_[0]
)

meta_intercept = float(
    meta_bundle["clf"]
    .intercept_[0]
)


print(
    "Learned standardized "
    "fusion coefficients:"
)

print(
    "  Visual:",
    float(
        meta_coef[0]
    ),
)

print(
    "  KG    :",
    float(
        meta_coef[1]
    ),
)

print(
    "  Intercept:",
    meta_intercept,
)


# ============================================================
# Final full-TRAIN visual epoch selection
# ============================================================

print(
    "\n"
    + "=" * 82
)

print(
    "FINAL FULL-TRAIN "
    "BASE-MODEL SELECTION"
)

print(
    "=" * 82
)


(
    final_visual_epoch,
    final_visual_detail,
    final_visual_curve,
) = (
    select_visual_epoch_cv(
        train_df,
        n_splits=
            INNER_VISUAL_FOLDS,
        seed=
            SEED + 50000,
        stage_name=
            "full_train",
    )
)


final_visual_detail.to_csv(
    OUT
    / "final_visual_inner_epoch_details.csv",
    index=False,
)

final_visual_curve.to_csv(
    OUT
    / "final_visual_epoch_mean_curve.csv",
    index=False,
)


print(
    "Final visual epoch "
    "selected from TRAIN-only CV:",
    final_visual_epoch,
)


(
    final_b2,
    final_mu,
    final_sigma,
    val_visual,
    final_b2_loss,
) = (
    fit_visual_fixed_epochs(
        train_df,
        val_df,
        epochs=
            final_visual_epoch,
        seed=
            SEED + 60000,
    )
)


torch.save(
    {
        "model_state_dict":
            final_b2.state_dict(),

        "visual_mean":
            final_mu,

        "visual_std":
            final_sigma,

        "selected_epoch":
            int(
                final_visual_epoch
            ),

        "architecture": {
            "input_dim":
                512,

            "d_model":
                D_MODEL,

            "n_heads":
                N_HEADS,

            "n_blocks":
                N_BLOCKS,

            "dropout":
                DROPOUT,

            "max_canonical_z":
                MAX_CANONICAL_Z,
        },

        "training": {
            "lr":
                LR,

            "weight_decay":
                WEIGHT_DECAY,

            "batch_size":
                BATCH_SIZE,

            "epoch_selection":
                (
                    f"{INNER_VISUAL_FOLDS}-fold "
                    "subject-disjoint CV; "
                    "maximize mean subject AUROC; "
                    "earliest tie"
                ),
        },
    },

    OUT
    / "final_full_train_B2.pt",
)


print(
    "Final B2 refit loss:",
    f"{final_b2_loss:.4f}",
)


del final_b2

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# Final full-TRAIN KG selection
# ============================================================

(
    final_kg_C,
    final_kg_C_table,
) = (
    select_kg_C(
        train_df,
        n_splits=5,
        seed=
            SEED + 70000,
        stage_name=
            "full_train",
    )
)


final_kg_C_table.to_csv(
    OUT
    / "final_KG_C_selection.csv",
    index=False,
)


final_kg_bundle = (
    fit_kg_bundle(
        train_df,
        final_kg_C,
    )
)


joblib.dump(
    final_kg_bundle,
    OUT
    / "final_full_train_KG.joblib",
)


val_kg = (
    predict_kg_bundle(
        final_kg_bundle,
        val_df,
    )
)


# ============================================================
# VALIDATION
# ============================================================

val = val_visual.merge(
    val_kg,
    on=[
        "case_id",
        "group_id",
        "label",
    ],
    validate="one_to_one",
)


assert len(val) == 34

assert (
    val["group_id"]
    .nunique()
    == 20
)


X_val_meta = (
    make_meta_X(
        val
    )
)


val[
    "prob_fusion"
] = predict_meta(
    meta_bundle,
    X_val_meta,
)


val.to_csv(
    OUT
    / "validation_predictions.csv",
    index=False,
)


print(
    "\n"
    + "=" * 82
)

print(
    "VALIDATION -- ALL MODEL/FUSION "
    "SELECTION WAS TRAIN-ONLY"
)

print(
    "=" * 82
)


(
    b2_vm,
    b2_sm,
    b2_sub,
) = report(
    "FINAL VISUAL B2",

    val["label"]
    .to_numpy(),

    val["prob_visual"]
    .to_numpy(),

    val["group_id"]
    .to_numpy(),
)


(
    kg_vm,
    kg_sm,
    kg_sub,
) = report(
    "FINAL KG",

    val["label"]
    .to_numpy(),

    val["prob_kg"]
    .to_numpy(),

    val["group_id"]
    .to_numpy(),
)


(
    fusion_vm,
    fusion_sm,
    fusion_sub,
) = report(
    "FINAL LEARNED FUSION",

    val["label"]
    .to_numpy(),

    val["prob_fusion"]
    .to_numpy(),

    val["group_id"]
    .to_numpy(),
)


b2_sub.to_csv(
    OUT
    / "validation_visual_subject_predictions.csv",
    index=False,
)

kg_sub.to_csv(
    OUT
    / "validation_KG_subject_predictions.csv",
    index=False,
)

fusion_sub.to_csv(
    OUT
    / "validation_fusion_subject_predictions.csv",
    index=False,
)


print(
    "\nSUBJECT AUROC DELTAS"
)

print(
    "--------------------"
)

print(
    "Fusion - Visual:",
    f"{fusion_sm['auroc'] - b2_sm['auroc']:+.4f}",
)

print(
    "Fusion - KG    :",
    f"{fusion_sm['auroc'] - kg_sm['auroc']:+.4f}",
)


# ============================================================
# Summary
# ============================================================

summary = {
    "seed":
        SEED,

    "outer_folds":
        OUTER_FOLDS,

    "inner_visual_folds":
        INNER_VISUAL_FOLDS,

    "max_epochs":
        MAX_EPOCHS,

    "visual_epoch_selection":
        (
            "maximize mean inner-fold "
            "subject AUROC; earliest tie"
        ),

    "outer_selected_visual_epochs":
        [
            int(
                x[
                    "visual_selected_epoch"
                ]
            )
            for x
            in outer_summary_rows
        ],

    "final_visual_epoch":
        int(
            final_visual_epoch
        ),

    "final_KG_C":
        float(
            final_kg_C
        ),

    "meta_C":
        float(
            best_meta_C
        ),

    "meta_visual_coefficient":
        float(
            meta_coef[0]
        ),

    "meta_KG_coefficient":
        float(
            meta_coef[1]
        ),

    "meta_intercept":
        meta_intercept,

    "OOF_visual_subject_metrics":
        visual_oof_sm,

    "OOF_KG_subject_metrics":
        kg_oof_sm,

    "validation_visual_volume_metrics":
        b2_vm,

    "validation_visual_subject_metrics":
        b2_sm,

    "validation_KG_volume_metrics":
        kg_vm,

    "validation_KG_subject_metrics":
        kg_sm,

    "validation_fusion_volume_metrics":
        fusion_vm,

    "validation_fusion_subject_metrics":
        fusion_sm,

    "decision_threshold_optimized":
        False,

    "fusion_fitted_on_validation":
        False,

    "validation_used_for_epoch_selection":
        False,

    "test_images_features_predictions_evaluated":
        False,
}


with open(
    OUT / "summary.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


print(
    "\nSaved:"
)

print(
    OUT
)

print(
    "\nNo fusion coefficient "
    "was fitted on validation."
)

print(
    "No decision threshold "
    "was optimized."
)

print(
    "Validation was not used "
    "to select the visual "
    "training epoch."
)

print(
    "TEST images/features/predictions "
    "were NOT loaded or evaluated."
)
