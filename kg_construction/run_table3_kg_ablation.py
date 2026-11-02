# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.

from pathlib import Path
import json
import numpy as np
import pandas as pd

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


ROOT = Path(__file__).resolve().parents[1]

CSV = (
    ROOT
    / "datasets/huang/results/kg_descriptors/train_val_graph_descriptors.csv"
)

OUT = (
    ROOT
    / "kg_construction/results/table3_kg_ablation_seed42"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42

C_GRID = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

META = {
    "case_id",
    "group_id",
    "disease",
    "split",
}


# ============================================================
# Data
# ============================================================

df = pd.read_csv(CSV)

df["case_id"] = df["case_id"].astype(str)
df["group_id"] = df["group_id"].astype(str)

train = (
    df[df["split"] == "train"]
    .copy()
    .reset_index(drop=True)
)

val = (
    df[df["split"] == "val"]
    .copy()
    .reset_index(drop=True)
)

assert len(train) == 126
assert train["group_id"].nunique() == 94

assert len(val) == 34
assert val["group_id"].nunique() == 20

ALL_FEATURES = [
    c
    for c in df.columns
    if c not in META
]

assert len(ALL_FEATURES) == 128


# ============================================================
# PREDEFINED feature groups
# ============================================================

# Acquisition-depth control included in every group.
BURDEN = [
    "graph.n_slices",
    "graph.n_lesions",
    "graph.lesion_bearing_slice_fraction",
    "lesion.volume_fraction.sum",
]


LESION_GEOMETRY_PREFIXES = [
    "lesion.bbox_norm.",
    "lesion.centroid_norm.",
    "lesion.foreground_mass_fraction.",
    "lesion.slice_count.",
    "lesion.slice_span_fraction.",
    "lesion.volume_fraction.",
    "lesion.voxel_count.",
]

LESION_GEOMETRY_EXACT = [
    "lesion.count",
]


LESION_SLICE_GRAPH_EXACT = [
    "graph.n_lesion_slice_edges",
    "graph.lesion_slice_density",
    "graph.lesion_slice_edges_per_lesion",
    "graph.lesion_bearing_slice_count",
]

LESION_SLICE_PREFIXES = [
    "lesion_slice.",
]


def starts_with_any(col, prefixes):
    return any(
        col.startswith(p)
        for p in prefixes
    )


geometry = [
    c
    for c in ALL_FEATURES
    if (
        c in LESION_GEOMETRY_EXACT
        or starts_with_any(
            c,
            LESION_GEOMETRY_PREFIXES,
        )
    )
]


lesion_slice = [
    c
    for c in ALL_FEATURES
    if (
        c in LESION_SLICE_GRAPH_EXACT
        or starts_with_any(
            c,
            LESION_SLICE_PREFIXES,
        )
    )
]


GROUPS = {
    "A_BURDEN":
        BURDEN,

    "B_PLUS_LESION_GEOMETRY":
        sorted(
            set(
                BURDEN
                + geometry
            )
        ),

    "C_PLUS_LESION_SLICE":
        sorted(
            set(
                BURDEN
                + geometry
                + lesion_slice
            )
        ),

    "D_FULL_KG":
        list(
            ALL_FEATURES
        ),
}


# ============================================================
# Validate nested hierarchy
# ============================================================

names = list(GROUPS)

for name, cols in GROUPS.items():
    missing = [
        c
        for c in cols
        if c not in ALL_FEATURES
    ]

    if missing:
        raise RuntimeError(
            f"{name}: missing columns {missing}"
        )


assert set(GROUPS[names[0]]).issubset(
    set(GROUPS[names[1]])
)

assert set(GROUPS[names[1]]).issubset(
    set(GROUPS[names[2]])
)

assert set(GROUPS[names[2]]).issubset(
    set(GROUPS[names[3]])
)


print("=" * 78)
print("KG STRUCTURAL ABLATION — VALIDATION ONLY")
print("=" * 78)

print(
    "TRAIN:",
    len(train),
    "volumes /",
    train["group_id"].nunique(),
    "subjects",
)

print(
    "VAL:",
    len(val),
    "volumes /",
    val["group_id"].nunique(),
    "subjects",
)

print("TEST: NOT LOADED")

print("\nFrozen feature hierarchy:")

for name, cols in GROUPS.items():
    print(
        f"{name:<28} {len(cols):>3} descriptors"
    )


# Save definitions BEFORE evaluation.
with open(
    OUT / "feature_groups.json",
    "w",
) as f:
    json.dump(
        GROUPS,
        f,
        indent=2,
    )


# ============================================================
# Helpers
# ============================================================

def subject_table(frame):
    s = (
        frame.groupby(
            "group_id",
            as_index=False,
        )
        .agg(
            disease=("disease", "first"),
        )
    )

    return s


def make_subject_folds(
    frame,
    n_splits,
    seed,
):
    subjects = subject_table(
        frame
    )

    y = (
        subjects["disease"]
        == "DME"
    ).astype(int).to_numpy()

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    folds = []

    for a, b in cv.split(
        np.zeros(len(subjects)),
        y,
    ):
        train_groups = set(
            subjects.iloc[a]["group_id"]
        )

        hold_groups = set(
            subjects.iloc[b]["group_id"]
        )

        folds.append(
            (
                frame[
                    frame["group_id"]
                    .isin(train_groups)
                ]
                .copy()
                .reset_index(drop=True),

                frame[
                    frame["group_id"]
                    .isin(hold_groups)
                ]
                .copy()
                .reset_index(drop=True),
            )
        )

    return folds


def subject_weights(groups):
    groups = np.asarray(groups)

    unique, counts = np.unique(
        groups,
        return_counts=True,
    )

    count_map = dict(
        zip(unique, counts)
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


def fit_bundle(
    frame,
    feature_cols,
    C,
):
    Xdf = (
        frame[feature_cols]
        .copy()
    )

    medians = (
        Xdf.median(
            axis=0,
            skipna=True,
        )
    )

    Xdf = (
        Xdf
        .fillna(medians)
        .fillna(0.0)
    )

    std = Xdf.std(
        axis=0,
        ddof=0,
    )

    keep = (
        std[
            std > 1e-12
        ]
        .index
        .tolist()
    )

    if not keep:
        raise RuntimeError(
            "No non-constant features."
        )

    X = (
        Xdf[keep]
        .to_numpy(
            dtype=np.float64
        )
    )

    scaler = StandardScaler()

    X = scaler.fit_transform(
        X
    )

    y = (
        frame["disease"]
        == "DME"
    ).astype(int).to_numpy()

    clf = LogisticRegression(
        C=C,
        solver="liblinear",
        max_iter=5000,
        random_state=SEED,
    )

    clf.fit(
        X,
        y,
        sample_weight=
            subject_weights(
                frame["group_id"]
                .to_numpy()
            ),
    )

    return {
        "feature_cols":
            list(feature_cols),

        "keep":
            keep,

        "medians":
            medians,

        "scaler":
            scaler,

        "clf":
            clf,
    }


def predict_bundle(
    bundle,
    frame,
):
    Xdf = (
        frame[
            bundle["feature_cols"]
        ]
        .copy()
    )

    Xdf = (
        Xdf
        .fillna(
            bundle["medians"]
        )
        .fillna(0.0)
    )

    X = (
        Xdf[
            bundle["keep"]
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    X = (
        bundle["scaler"]
        .transform(X)
    )

    return (
        bundle["clf"]
        .predict_proba(X)[:, 1]
    )


def aggregate_subjects(
    frame,
    prob,
):
    temp = pd.DataFrame({
        "group_id":
            frame["group_id"],

        "label":
            (
                frame["disease"]
                == "DME"
            ).astype(int),

        "prob":
            prob,
    })

    return (
        temp.groupby(
            "group_id",
            as_index=False,
        )
        .agg(
            label=("label", "first"),
            prob=("prob", "mean"),
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
        "AUROC":
            roc_auc_score(y, p),

        "AUPRC":
            average_precision_score(
                y,
                p,
            ),

        "BalancedAccuracy":
            balanced_accuracy_score(
                y,
                pred,
            ),

        "MacroF1":
            f1_score(
                y,
                pred,
                average="macro",
                zero_division=0,
            ),

        "Accuracy":
            accuracy_score(
                y,
                pred,
            ),
    }


# ============================================================
# Evaluate each predefined level
# ============================================================

results = []
predictions = {}

for group_index, (
    group_name,
    feature_cols,
) in enumerate(
    GROUPS.items(),
    start=1,
):

    print(
        "\n"
        + "=" * 78
    )

    print(
        group_name,
        f"({len(feature_cols)} descriptors)"
    )

    print(
        "=" * 78
    )

    folds = make_subject_folds(
        train,
        n_splits=5,
        seed=
            SEED
            + group_index,
    )

    c_rows = []

    for C in C_GRID:
        fold_aucs = []

        for tr, ho in folds:
            bundle = fit_bundle(
                tr,
                feature_cols,
                C,
            )

            p = predict_bundle(
                bundle,
                ho,
            )

            sub = aggregate_subjects(
                ho,
                p,
            )

            fold_aucs.append(
                roc_auc_score(
                    sub["label"],
                    sub["prob"],
                )
            )

        mean_auc = float(
            np.mean(
                fold_aucs
            )
        )

        std_auc = float(
            np.std(
                fold_aucs
            )
        )

        c_rows.append({
            "group":
                group_name,

            "C":
                C,

            "mean_train_CV_subject_AUROC":
                mean_auc,

            "std_train_CV_subject_AUROC":
                std_auc,
        })

        print(
            f"C={C:<6} "
            f"TRAIN-CV Sub AUROC="
            f"{mean_auc:.4f} "
            f"+/- {std_auc:.4f}"
        )

    c_table = pd.DataFrame(
        c_rows
    )

    best_C = float(
        c_table.sort_values(
            [
                "mean_train_CV_subject_AUROC",
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
        "Selected C:",
        best_C
    )

    c_table.to_csv(
        OUT
        / f"{group_name}_C_selection.csv",
        index=False,
    )

    final_bundle = fit_bundle(
        train,
        feature_cols,
        best_C,
    )

    p_val = predict_bundle(
        final_bundle,
        val,
    )

    predictions[
        group_name
    ] = p_val

    volume_metrics = metric_dict(
        (
            val["disease"]
            == "DME"
        ).astype(int),
        p_val,
    )

    sub = aggregate_subjects(
        val,
        p_val,
    )

    subject_metrics = metric_dict(
        sub["label"],
        sub["prob"],
    )

    print(
        "\nVALIDATION SUBJECT"
    )

    print(
        "AUROC="
        f"{subject_metrics['AUROC']:.4f} | "
        "AUPRC="
        f"{subject_metrics['AUPRC']:.4f} | "
        "BalAcc="
        f"{subject_metrics['BalancedAccuracy']:.4f} | "
        "MacroF1="
        f"{subject_metrics['MacroF1']:.4f} | "
        "Acc="
        f"{subject_metrics['Accuracy']:.4f}"
    )

    results.append({
        "group":
            group_name,

        "n_descriptors":
            len(feature_cols),

        "selected_C":
            best_C,

        **{
            f"volume_{k}": v
            for k, v
            in volume_metrics.items()
        },

        **{
            f"subject_{k}": v
            for k, v
            in subject_metrics.items()
        },
    })


# ============================================================
# Aligned validation predictions
# ============================================================

pred_df = val[
    [
        "case_id",
        "group_id",
        "disease",
    ]
].copy()

for name, p in predictions.items():
    pred_df[
        f"prob_{name}"
    ] = p

pred_df.to_csv(
    OUT
    / "validation_predictions.csv",
    index=False,
)


result_df = pd.DataFrame(
    results
)

result_df.to_csv(
    OUT
    / "ablation_results.csv",
    index=False,
)


# ============================================================
# Paired bootstrap:
# FULL KG vs simple BURDEN
# ============================================================

burden_sub = aggregate_subjects(
    val,
    predictions[
        "A_BURDEN"
    ],
)

full_sub = aggregate_subjects(
    val,
    predictions[
        "D_FULL_KG"
    ],
)

paired = burden_sub.merge(
    full_sub,
    on=[
        "group_id",
        "label",
    ],
    suffixes=(
        "_BURDEN",
        "_FULL",
    ),
    validate="one_to_one",
)

y = paired[
    "label"
].to_numpy()

pb = paired[
    "prob_BURDEN"
].to_numpy()

pf = paired[
    "prob_FULL"
].to_numpy()

observed = (
    roc_auc_score(
        y,
        pf,
    )
    - roc_auc_score(
        y,
        pb,
    )
)

rng = np.random.default_rng(
    42
)

diffs = []

for _ in range(10000):
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

    diffs.append(
        roc_auc_score(
            yy,
            pf[idx],
        )
        - roc_auc_score(
            yy,
            pb[idx],
        )
    )

diffs = np.asarray(
    diffs
)

print(
    "\n"
    + "=" * 78
)

print(
    "FULL KG vs BURDEN — "
    "PAIRED SUBJECT BOOTSTRAP"
)

print(
    "=" * 78
)

print(
    "Observed AUROC delta:",
    f"{observed:+.4f}",
)

print(
    "95% CI:",
    np.percentile(
        diffs,
        [2.5, 97.5],
    ),
)

print(
    "P(Full KG > Burden):",
    float(
        np.mean(
            diffs > 0
        )
    ),
)


print(
    "\n"
    + "=" * 78
)

print(
    "FINAL ABLATION SUMMARY"
)

print(
    "=" * 78
)

print(
    result_df[
        [
            "group",
            "n_descriptors",
            "selected_C",
            "subject_AUROC",
            "subject_AUPRC",
            "subject_BalancedAccuracy",
            "subject_MacroF1",
            "subject_Accuracy",
        ]
    ]
    .to_string(
        index=False
    )
)

print(
    "\nSaved:"
)

print(
    OUT
)

print(
    "\nTEST was NOT loaded or evaluated."
)
