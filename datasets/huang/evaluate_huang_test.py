# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.

from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


# ============================================================
# Frozen paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    ROOT / "datasets/huang/results/eyeclip_slice_features"
)

SPLIT_CSV = Path("PATH_TO_HUANG_SUBJECT_SPLIT_CSV")

FINAL = (
    ROOT
    / "datasets/huang/results/table2_seed42"
)

TEST_DESC = (
    FINAL
    / "test_graph_descriptors.csv"
)

B2_CKPT = (
    FINAL
    / "final_full_train_B2.pt"
)

KG_MODEL = (
    FINAL
    / "final_full_train_KG.joblib"
)

META_MODEL = (
    FINAL
    / "final_train_only_meta_fusion.joblib"
)

OUT_VOLUME = (
    FINAL
    / "FINAL_TEST_volume_predictions.csv"
)

OUT_SUBJECT = (
    FINAL
    / "FINAL_TEST_subject_predictions.csv"
)

OUT_SUMMARY = (
    FINAL
    / "FINAL_TEST_summary.json"
)

OUT_BOOT = (
    FINAL
    / "FINAL_TEST_bootstrap.csv"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 4
MAX_CANONICAL_Z = 512


# ============================================================
# Load frozen TEST cohort
# ============================================================

split_df = pd.read_csv(
    SPLIT_CSV
)

split_df["case_id"] = (
    split_df["case_id"]
    .astype(str)
)

split_df["group_id"] = (
    split_df["group_id"]
    .astype(str)
)

test_df = (
    split_df[
        split_df["split"] == "test"
    ]
    .copy()
    .reset_index(drop=True)
)

assert len(test_df) == 32
assert test_df["group_id"].nunique() == 21

test_df["label"] = (
    test_df["disease"]
    == "DME"
).astype(int)


# ============================================================
# B2 architecture -- EXACT frozen architecture
# ============================================================

class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        dim,
        heads,
        dropout,
    ):
        super().__init__()

        assert dim % heads == 0

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

        attn = self.drop(
            attn
        )

        out = (
            attn @ v
        )

        out = (
            out.transpose(1, 2)
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
            * valid[:, :, None]
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


class B2Model(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        n_blocks,
        dropout,
    ):
        super().__init__()

        self.d_model = d_model

        self.proj = nn.Linear(
            512,
            d_model,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads,
                dropout,
            )
            for _ in range(
                n_blocks
            )
        ])

        self.pool = AttentionPool(
            d_model
        )

        self.classifier = nn.Linear(
            d_model,
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
            self.d_model,
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

        pooled = self.pool(
            x,
            valid,
        )

        return self.classifier(
            pooled
        )


# ============================================================
# Load frozen B2 checkpoint
# ============================================================

ckpt = torch.load(
    B2_CKPT,
    map_location="cpu",
    weights_only=False,
)

cfg = ckpt["architecture"]

model = B2Model(
    d_model=int(
        cfg["d_model"]
    ),
    n_heads=int(
        cfg["n_heads"]
    ),
    n_blocks=int(
        cfg["n_blocks"]
    ),
    dropout=float(
        cfg["dropout"]
    ),
)

model.load_state_dict(
    ckpt["model_state_dict"]
)

model = (
    model
    .to(DEVICE)
    .eval()
)

visual_mean = np.asarray(
    ckpt["visual_mean"],
    dtype=np.float32,
)

visual_std = np.asarray(
    ckpt["visual_std"],
    dtype=np.float32,
)

assert visual_mean.shape == (512,)
assert visual_std.shape == (512,)

param_count = sum(
    p.numel()
    for p in model.parameters()
)

assert param_count == 1711619


# ============================================================
# Frozen visual dataset
# ============================================================

def load_eyeclip(case_id):
    x = np.load(
        FEATURE_DIR
        / f"{case_id}.npy"
    ).astype(np.float32)

    assert (
        x.ndim == 2
        and x.shape[1] == 512
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

    x = (
        x
        - visual_mean[None, :]
    ) / visual_std[None, :]

    return x.astype(
        np.float32,
        copy=False,
    )


class TestVisualDataset(Dataset):

    def __init__(self, frame):
        self.df = (
            frame
            .reset_index(drop=True)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(
        self,
        idx,
    ):
        row = self.df.iloc[idx]

        return {
            "case_id":
                str(row["case_id"]),

            "group_id":
                str(row["group_id"]),

            "label":
                int(row["label"]),

            "x":
                torch.from_numpy(
                    load_eyeclip(
                        str(
                            row["case_id"]
                        )
                    )
                ),
        }


def collate(batch):
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

    cases = []
    groups = []
    labels = []

    for i, item in enumerate(batch):
        z = item["x"].shape[0]

        x[i, :z] = (
            item["x"]
        )

        valid[i, :z] = True

        cases.append(
            item["case_id"]
        )

        groups.append(
            item["group_id"]
        )

        labels.append(
            item["label"]
        )

    return {
        "x": x,
        "valid": valid,
        "case_id": cases,
        "group_id": groups,
        "label":
            np.asarray(
                labels,
                dtype=int,
            ),
    }


loader = DataLoader(
    TestVisualDataset(
        test_df
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate,
    pin_memory=True,
)


# ============================================================
# Frozen B2 TEST inference
# ============================================================

visual_rows = []

with torch.no_grad():

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

        for (
            case_id,
            group_id,
            label,
            prob,
        ) in zip(
            batch["case_id"],
            batch["group_id"],
            batch["label"],
            p,
        ):
            visual_rows.append({
                "case_id":
                    case_id,

                "group_id":
                    group_id,

                "label":
                    int(label),

                "prob_visual":
                    float(prob),
            })


visual = pd.DataFrame(
    visual_rows
)

assert len(visual) == 32


# ============================================================
# Frozen KG TEST inference
# ============================================================

kg_bundle = joblib.load(
    KG_MODEL
)

desc = pd.read_csv(
    TEST_DESC
)

desc["case_id"] = (
    desc["case_id"]
    .astype(str)
)

desc["group_id"] = (
    desc["group_id"]
    .astype(str)
)

assert len(desc) == 32

feature_cols = (
    kg_bundle[
        "feature_cols"
    ]
)

keep_cols = (
    kg_bundle[
        "keep_cols"
    ]
)

medians = pd.Series(
    kg_bundle[
        "medians"
    ]
)

assert set(
    feature_cols
).issubset(
    set(desc.columns)
)

Xkg_df = (
    desc[feature_cols]
    .fillna(medians)
    .fillna(0.0)
)

Xkg = (
    Xkg_df[
        keep_cols
    ]
    .to_numpy(
        dtype=np.float64
    )
)

Xkg = (
    kg_bundle["scaler"]
    .transform(Xkg)
)

p_kg = (
    kg_bundle["clf"]
    .predict_proba(
        Xkg
    )[:, 1]
)

kg = pd.DataFrame({
    "case_id":
        desc["case_id"],

    "group_id":
        desc["group_id"],

    "label":
        (
            desc["disease"]
            == "DME"
        ).astype(int),

    "prob_kg":
        p_kg,
})


# ============================================================
# Align both frozen branches
# ============================================================

pred = visual.merge(
    kg,
    on=[
        "case_id",
        "group_id",
        "label",
    ],
    validate="one_to_one",
)

assert len(pred) == 32
assert pred["group_id"].nunique() == 21


# ============================================================
# Frozen OOF-learned fusion
# ============================================================

meta = joblib.load(
    META_MODEL
)


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


Xm = np.column_stack([
    to_logit(
        pred["prob_visual"]
    ),
    to_logit(
        pred["prob_kg"]
    ),
])

Xm = (
    meta["scaler"]
    .transform(Xm)
)

pred["prob_fusion"] = (
    meta["clf"]
    .predict_proba(
        Xm
    )[:, 1]
)


# ============================================================
# Evaluation helpers
# ============================================================

def metrics(
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

    pred_label = (
        p >= 0.5
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            pred_label,
            labels=[0, 1],
        )
        .ravel()
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "accuracy":
            float(
                accuracy_score(
                    y,
                    pred_label,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred_label,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y,
                    pred_label,
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

        "sensitivity_DME":
            float(
                sensitivity
            ),

        "specificity_AMD":
            float(
                specificity
            ),

        "TP":
            int(tp),

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),
    }


def aggregate_subject(
    frame,
    prob_col,
):
    return (
        frame.groupby(
            "group_id",
            as_index=False,
        )
        .agg(
            label=(
                "label",
                "first",
            ),
            prob=(
                prob_col,
                "mean",
            ),
            n_volumes=(
                "case_id",
                "size",
            ),
        )
    )


def print_metrics(
    title,
    m,
):
    print(
        f"\n{title}"
    )

    print(
        "-" * len(title)
    )

    print(
        f"AUROC={m['auroc']:.4f} | "
        f"AUPRC={m['auprc']:.4f} | "
        f"BalAcc={m['balanced_accuracy']:.4f} | "
        f"MacroF1={m['macro_f1']:.4f} | "
        f"Acc={m['accuracy']:.4f}"
    )

    print(
        f"Sensitivity(DME)="
        f"{m['sensitivity_DME']:.4f} | "
        f"Specificity(AMD)="
        f"{m['specificity_AMD']:.4f}"
    )

    print(
        f"TP={m['TP']} "
        f"TN={m['TN']} "
        f"FP={m['FP']} "
        f"FN={m['FN']}"
    )


# ============================================================
# FINAL TEST metrics
# ============================================================

models = {
    "VISUAL":
        "prob_visual",

    "KG":
        "prob_kg",

    "FUSION":
        "prob_fusion",
}

summary = {
    "test_volumes":
        32,

    "test_subjects":
        21,

    "models":
        {},
}

subject_frames = {}

print(
    "=" * 82
)

print(
    "FINAL UNTOUCHED TEST EVALUATION"
)

print(
    "=" * 82
)

print(
    "Volumes :",
    len(pred)
)

print(
    "Subjects:",
    pred["group_id"]
    .nunique()
)

print(
    "AMD volumes:",
    int(
        (
            pred["label"]
            == 0
        ).sum()
    )
)

print(
    "DME volumes:",
    int(
        (
            pred["label"]
            == 1
        ).sum()
    )
)


for model_name, prob_col in models.items():

    vm = metrics(
        pred["label"],
        pred[prob_col],
    )

    sub = aggregate_subject(
        pred,
        prob_col,
    )

    sm = metrics(
        sub["label"],
        sub["prob"],
    )

    subject_frames[
        model_name
    ] = sub

    print(
        f"\n{'='*82}"
    )

    print(
        model_name
    )

    print(
        "=" * 82
    )

    print_metrics(
        "VOLUME",
        vm,
    )

    print_metrics(
        "SUBJECT",
        sm,
    )

    summary[
        "models"
    ][model_name] = {
        "volume":
            vm,

        "subject":
            sm,
    }


# ============================================================
# One aligned subject-level table
# ============================================================

subject_pred = (
    subject_frames[
        "VISUAL"
    ][
        [
            "group_id",
            "label",
            "n_volumes",
            "prob",
        ]
    ]
    .rename(
        columns={
            "prob":
                "prob_visual"
        }
    )
)

subject_pred = (
    subject_pred.merge(
        subject_frames[
            "KG"
        ][
            [
                "group_id",
                "prob",
            ]
        ]
        .rename(
            columns={
                "prob":
                    "prob_kg"
            }
        ),
        on="group_id",
        validate="one_to_one",
    )
)

subject_pred = (
    subject_pred.merge(
        subject_frames[
            "FUSION"
        ][
            [
                "group_id",
                "prob",
            ]
        ]
        .rename(
            columns={
                "prob":
                    "prob_fusion"
            }
        ),
        on="group_id",
        validate="one_to_one",
    )
)

subject_pred["disease"] = np.where(
    subject_pred["label"] == 1,
    "DME",
    "AMD",
)


# ============================================================
# Paired subject bootstrap
# ============================================================

rng = np.random.default_rng(
    42
)

N_BOOT = 10000

y = (
    subject_pred["label"]
    .to_numpy()
)

pv = (
    subject_pred["prob_visual"]
    .to_numpy()
)

pk = (
    subject_pred["prob_kg"]
    .to_numpy()
)

pf = (
    subject_pred["prob_fusion"]
    .to_numpy()
)

boot_rows = []

for _ in range(
    N_BOOT
):
    idx = rng.integers(
        0,
        len(y),
        len(y),
    )

    yy = y[idx]

    if np.unique(
        yy
    ).size < 2:
        continue

    av = roc_auc_score(
        yy,
        pv[idx],
    )

    ak = roc_auc_score(
        yy,
        pk[idx],
    )

    af = roc_auc_score(
        yy,
        pf[idx],
    )

    boot_rows.append({
        "visual_AUROC":
            av,

        "kg_AUROC":
            ak,

        "fusion_AUROC":
            af,

        "fusion_minus_visual":
            af - av,

        "fusion_minus_kg":
            af - ak,
    })


boot = pd.DataFrame(
    boot_rows
)


def ci(col):
    return [
        float(
            np.percentile(
                boot[col],
                2.5,
            )
        ),
        float(
            np.percentile(
                boot[col],
                97.5,
            )
        ),
    ]


summary[
    "paired_subject_bootstrap"
] = {
    "valid_resamples":
        int(len(boot)),

    "visual_AUROC_95CI":
        ci(
            "visual_AUROC"
        ),

    "kg_AUROC_95CI":
        ci(
            "kg_AUROC"
        ),

    "fusion_AUROC_95CI":
        ci(
            "fusion_AUROC"
        ),

    "fusion_minus_visual_95CI":
        ci(
            "fusion_minus_visual"
        ),

    "fusion_minus_kg_95CI":
        ci(
            "fusion_minus_kg"
        ),

    "P_fusion_gt_visual":
        float(
            (
                boot[
                    "fusion_minus_visual"
                ]
                > 0
            )
            .mean()
        ),

    "P_fusion_gt_KG":
        float(
            (
                boot[
                    "fusion_minus_kg"
                ]
                > 0
            )
            .mean()
        ),
}


print(
    "\n"
    + "=" * 82
)

print(
    "SUBJECT-LEVEL AUROC COMPARISON"
)

print(
    "=" * 82
)

sv = summary[
    "models"
]["VISUAL"]["subject"]["auroc"]

sk = summary[
    "models"
]["KG"]["subject"]["auroc"]

sf = summary[
    "models"
]["FUSION"]["subject"]["auroc"]

print(
    "Visual:",
    f"{sv:.4f}"
)

print(
    "KG    :",
    f"{sk:.4f}"
)

print(
    "Fusion:",
    f"{sf:.4f}"
)

print(
    "Fusion - Visual:",
    f"{sf - sv:+.4f}"
)

print(
    "Fusion - KG:",
    f"{sf - sk:+.4f}"
)


print(
    "\nPAIRED SUBJECT BOOTSTRAP"
)

print(
    "------------------------"
)

print(
    "Valid resamples:",
    len(boot)
)

print(
    "Visual AUROC 95% CI:",
    ci(
        "visual_AUROC"
    )
)

print(
    "KG AUROC 95% CI:",
    ci(
        "kg_AUROC"
    )
)

print(
    "Fusion AUROC 95% CI:",
    ci(
        "fusion_AUROC"
    )
)

print(
    "Fusion - Visual 95% CI:",
    ci(
        "fusion_minus_visual"
    )
)

print(
    "Fusion - KG 95% CI:",
    ci(
        "fusion_minus_kg"
    )
)

print(
    "P(Fusion > Visual):",
    summary[
        "paired_subject_bootstrap"
    ][
        "P_fusion_gt_visual"
    ]
)

print(
    "P(Fusion > KG):",
    summary[
        "paired_subject_bootstrap"
    ][
        "P_fusion_gt_KG"
    ]
)


# ============================================================
# Persist final test results
# ============================================================

pred.to_csv(
    OUT_VOLUME,
    index=False,
)

subject_pred.to_csv(
    OUT_SUBJECT,
    index=False,
)

boot.to_csv(
    OUT_BOOT,
    index=False,
)

summary[
    "frozen_visual_epoch"
] = int(
    ckpt["selected_epoch"]
)

summary[
    "frozen_KG_C"
] = float(
    kg_bundle["C"]
)

summary[
    "decision_threshold_optimized"
] = False

summary[
    "models_refit_on_test"
] = False

summary[
    "fusion_refit_on_test"
] = False

with open(
    OUT_SUMMARY,
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
    OUT_VOLUME
)

print(
    OUT_SUBJECT
)

print(
    OUT_BOOT
)

print(
    OUT_SUMMARY
)

print(
    "\nFINAL TEST evaluation complete."
)

print(
    "No model was fitted or updated."
)

print(
    "No fusion coefficient was changed."
)

print(
    "No decision threshold was optimized."
)
