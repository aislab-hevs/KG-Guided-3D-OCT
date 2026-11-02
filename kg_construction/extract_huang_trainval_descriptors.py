# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parents[1]

KGDIR = ROOT / "datasets/huang/results/patient_specific_kg"
SPLIT = Path("PATH_TO_HUANG_SUBJECT_SPLIT_CSV")

OUT = ROOT / "datasets/huang/results/kg_descriptors"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]


# ============================================================
# Helpers
# ============================================================

EXCLUDE_TOKENS = {
    "case_id",
    "disease",
    "diagnosis",
    "label",
    "class_id",
    "source_component_id",
}


def get_objects(x):
    """
    Robustly convert a JSON collection to a list of records.
    """
    if isinstance(x, list):
        return x

    if isinstance(x, dict):
        # Dict-of-records
        if all(isinstance(v, dict) for v in x.values()):
            return list(x.values())

        # One record
        return [x]

    return []


def excluded(path):
    p = path.lower()

    for token in EXCLUDE_TOKENS:
        if token in p:
            return True

    # Exclude generic identifier-like fields but NOT meaningful
    # fields such as voxel_count or shared_slice_count.
    parts = re.split(r"[./]", p)

    for part in parts:
        if part in {
            "id", "node_id", "edge_id",
            "source", "target",
            "src", "dst",
            "lesion_id", "slice_id",
        }:
            return True

    return False


def flatten_numeric(obj, prefix=""):
    """
    Flatten numeric JSON leaves.
    Numeric vectors such as centroid_norm and bbox_norm are
    expanded element-wise.
    """
    out = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)

            if excluded(path):
                continue

            out.update(
                flatten_numeric(v, path)
            )

    elif isinstance(obj, (list, tuple)):
        # Numeric vector
        if len(obj) and all(
            isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, bool)
            for v in obj
        ):
            for i, v in enumerate(obj):
                path = f"{prefix}.{i}"

                if not excluded(path):
                    out[path] = float(v)

    elif (
        isinstance(obj, (int, float, np.integer, np.floating))
        and not isinstance(obj, bool)
    ):
        if not excluded(prefix):
            out[prefix] = float(obj)

    return out


def aggregate_collection(records, prefix):
    """
    Turn a variable-size set of graph objects into fixed-size
    graph descriptors.
    """
    result = {
        f"{prefix}.count": float(len(records))
    }

    if not records:
        return result

    flattened = [
        flatten_numeric(r)
        for r in records
    ]

    keys = sorted(
        set().union(
            *(d.keys() for d in flattened)
        )
    )

    for key in keys:
        vals = [
            d[key]
            for d in flattened
            if key in d and np.isfinite(d[key])
        ]

        if not vals:
            continue

        a = np.asarray(vals, dtype=float)

        name = f"{prefix}.{key}"

        result[f"{name}.mean"] = float(a.mean())
        result[f"{name}.std"] = float(a.std())
        result[f"{name}.min"] = float(a.min())
        result[f"{name}.max"] = float(a.max())
        result[f"{name}.sum"] = float(a.sum())

    return result


def recursively_find_strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from recursively_find_strings(v)

    elif isinstance(obj, list):
        for v in obj:
            yield from recursively_find_strings(v)

    elif isinstance(obj, str):
        yield obj


def extract_graph_features(case_id):
    path = KGDIR / f"{case_id}_kg.json"

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path) as f:
        g = json.load(f)

    nodes = g.get("nodes", {})
    edges = g.get("edges", {})

    slices = get_objects(
        nodes.get("slices", [])
    )

    lesions = get_objects(
        nodes.get("lesions", [])
    )

    lesion_slice = get_objects(
        edges.get("lesion_to_slice", [])
    )

    lesion_lesion = get_objects(
        edges.get("lesion_to_lesion", [])
    )

    feat = {}

    # --------------------------------------------------------
    # Core graph counts
    # --------------------------------------------------------

    ns = len(slices)
    nl = len(lesions)
    nls = len(lesion_slice)
    nll = len(lesion_lesion)

    feat["graph.n_slices"] = float(ns)
    feat["graph.n_lesions"] = float(nl)
    feat["graph.n_lesion_slice_edges"] = float(nls)
    feat["graph.n_lesion_lesion_edges"] = float(nll)

    # Normalized graph densities.
    feat["graph.lesion_slice_density"] = (
        nls / max(1, ns * nl)
    )

    possible_ll = nl * (nl - 1) / 2

    feat["graph.lesion_lesion_density"] = (
        nll / max(1.0, possible_ll)
    )

    feat["graph.lesion_slice_edges_per_lesion"] = (
        nls / max(1, nl)
    )

    feat["graph.lesion_lesion_edges_per_lesion"] = (
        nll / max(1, nl)
    )

    # --------------------------------------------------------
    # Lesion-bearing B-scan coverage.
    #
    # This uses identifiers ONLY to count unique slice nodes;
    # the identifiers themselves never become features.
    # --------------------------------------------------------

    slice_ids = set()

    for edge in lesion_slice:
        for s in recursively_find_strings(edge):
            if re.fullmatch(r"Slice_\d+", s):
                slice_ids.add(s)

    feat["graph.lesion_bearing_slice_count"] = float(
        len(slice_ids)
    )

    feat["graph.lesion_bearing_slice_fraction"] = (
        len(slice_ids) / max(1, ns)
    )

    # --------------------------------------------------------
    # Aggregate generic lesion geometry / extent.
    # --------------------------------------------------------

    feat.update(
        aggregate_collection(
            lesions,
            "lesion",
        )
    )

    # --------------------------------------------------------
    # Aggregate lesion -> slice relation attributes.
    # --------------------------------------------------------

    feat.update(
        aggregate_collection(
            lesion_slice,
            "lesion_slice",
        )
    )

    # --------------------------------------------------------
    # Aggregate lesion <-> lesion structural relation attrs.
    # --------------------------------------------------------

    feat.update(
        aggregate_collection(
            lesion_lesion,
            "lesion_lesion",
        )
    )

    return feat


def subject_weights(groups):
    groups = np.asarray(groups)

    u, counts = np.unique(
        groups,
        return_counts=True,
    )

    count_map = dict(
        zip(u, counts)
    )

    w = np.asarray([
        1.0 / count_map[g]
        for g in groups
    ])

    w *= len(w) / w.sum()

    return w


def aggregate_subjects(y, prob, groups):
    x = pd.DataFrame({
        "group_id": groups,
        "label": y,
        "prob_DME": prob,
    })

    bad = x.groupby(
        "group_id"
    )["label"].nunique()

    if (bad > 1).any():
        raise RuntimeError(
            "Inconsistent labels within subject."
        )

    return (
        x.groupby(
            "group_id",
            as_index=False,
        )
        .agg(
            label=("label", "first"),
            prob_DME=("prob_DME", "mean"),
            n_volumes=("prob_DME", "size"),
        )
    )


def metrics(y, p):
    pred = (
        np.asarray(p) >= 0.5
    ).astype(int)

    return {
        "accuracy":
            accuracy_score(y, pred),

        "balanced_accuracy":
            balanced_accuracy_score(y, pred),

        "macro_f1":
            f1_score(
                y,
                pred,
                average="macro",
                zero_division=0,
            ),

        "auroc":
            roc_auc_score(y, p),

        "auprc":
            average_precision_score(y, p),
    }


def print_metrics(title, m):
    print("\n" + title)
    print("-" * len(title))

    for k, v in m.items():
        print(f"{k:20s}: {v:.4f}")


# ============================================================
# Split
# ============================================================

split_df = pd.read_csv(SPLIT)

assert len(split_df) == 192
assert (split_df["split"] == "train").sum() == 126
assert (split_df["split"] == "val").sum() == 34
assert (split_df["split"] == "test").sum() == 32

print("=" * 72)
print("KG-ONLY DIAGNOSTIC")
print("=" * 72)

print("Train: 126 volumes / 94 subjects")
print("Val:    34 volumes / 20 subjects")
print("Test:   LOCKED -- GRAPH FILES WILL NOT BE LOADED")

# IMPORTANT:
# filter before opening any graph JSON.
work = split_df[
    split_df["split"].isin(["train", "val"])
].copy()


# ============================================================
# Extract graph descriptors
# ============================================================

rows = []

print("\nExtracting TRAIN + VALIDATION graph descriptors...")

for _, row in work.iterrows():
    feat = extract_graph_features(
        row["case_id"]
    )

    feat["case_id"] = row["case_id"]
    feat["group_id"] = row["group_id"]
    feat["disease"] = row["disease"]
    feat["split"] = row["split"]

    rows.append(feat)

desc = pd.DataFrame(rows)

meta = [
    "case_id",
    "group_id",
    "disease",
    "split",
]

feature_cols = [
    c for c in desc.columns
    if c not in meta
]

print("Graphs loaded:", len(desc))
print("Raw structural descriptors:", len(feature_cols))

# Save complete descriptor table for check.
desc.to_csv(
    OUT / "train_val_graph_descriptors.csv",
    index=False,
)


# ============================================================
# Train / validation arrays
# ============================================================

tr = desc["split"] == "train"
va = desc["split"] == "val"

Xtr_df = desc.loc[
    tr,
    feature_cols,
].copy()

Xva_df = desc.loc[
    va,
    feature_cols,
].copy()

ytr = (
    desc.loc[tr, "disease"] == "DME"
).astype(int).to_numpy()

yva = (
    desc.loc[va, "disease"] == "DME"
).astype(int).to_numpy()

gtr = desc.loc[
    tr,
    "group_id",
].to_numpy()

gva = desc.loc[
    va,
    "group_id",
].to_numpy()

case_va = desc.loc[
    va,
    "case_id",
].to_numpy()


# ============================================================
# TRAIN-only feature cleanup
# ============================================================

# Remove columns that are entirely missing in TRAIN.
usable = [
    c for c in feature_cols
    if Xtr_df[c].notna().any()
]

Xtr_df = Xtr_df[usable]
Xva_df = Xva_df.reindex(
    columns=usable
)

# TRAIN-only median imputation.
medians = Xtr_df.median(
    axis=0,
    skipna=True,
)

Xtr_df = Xtr_df.fillna(
    medians
)

Xva_df = Xva_df.fillna(
    medians
)

# Any pathological still-missing columns are zero-filled.
Xtr_df = Xtr_df.fillna(0.0)
Xva_df = Xva_df.fillna(0.0)

# Remove constant / numerically constant descriptors using TRAIN only.
train_std = Xtr_df.std(
    axis=0,
    ddof=0,
)

keep = train_std[
    train_std > 1e-12
].index.tolist()

Xtr_df = Xtr_df[keep]
Xva_df = Xva_df[keep]

print(
    "Usable non-constant TRAIN descriptors:",
    len(keep)
)

print(
    "Train matrix:",
    Xtr_df.shape
)

print(
    "Val matrix:  ",
    Xva_df.shape
)

Xtr = Xtr_df.to_numpy(
    dtype=np.float64
)

Xva = Xva_df.to_numpy(
    dtype=np.float64
)


# ============================================================
# Select C on TRAIN subjects ONLY
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

cv_rows = []

print("\nSelecting C using TRAIN subjects only...")

for C in C_VALUES:
    fold_auc = []

    for train_idx, cv_idx in cv.split(
        Xtr,
        ytr,
        groups=gtr,
    ):
        scaler = StandardScaler()

        xa = scaler.fit_transform(
            Xtr[train_idx]
        )

        xb = scaler.transform(
            Xtr[cv_idx]
        )

        clf = LogisticRegression(
            C=C,
            solver="liblinear",
            max_iter=5000,
            random_state=SEED,
        )

        clf.fit(
            xa,
            ytr[train_idx],
            sample_weight=subject_weights(
                gtr[train_idx]
            ),
        )

        p = clf.predict_proba(
            xb
        )[:, 1]

        sub = aggregate_subjects(
            ytr[cv_idx],
            p,
            gtr[cv_idx],
        )

        fold_auc.append(
            roc_auc_score(
                sub["label"],
                sub["prob_DME"],
            )
        )

    mean_auc = float(
        np.mean(fold_auc)
    )

    std_auc = float(
        np.std(fold_auc)
    )

    print(
        f"C={C:<7} "
        f"subject AUROC={mean_auc:.4f} "
        f"+/- {std_auc:.4f}"
    )

    cv_rows.append({
        "C": C,
        "mean_subject_AUROC": mean_auc,
        "std_subject_AUROC": std_auc,
    })

cv_df = pd.DataFrame(
    cv_rows
)

best_C = float(
    cv_df.sort_values(
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
    "\nSelected C:",
    best_C
)


# ============================================================
# Fit full TRAIN
# ============================================================

scaler = StandardScaler()

Xtr_s = scaler.fit_transform(
    Xtr
)

Xva_s = scaler.transform(
    Xva
)

clf = LogisticRegression(
    C=best_C,
    solver="liblinear",
    max_iter=5000,
    random_state=SEED,
)

clf.fit(
    Xtr_s,
    ytr,
    sample_weight=subject_weights(
        gtr
    ),
)


# ============================================================
# VALIDATION ONLY
# ============================================================

pva = clf.predict_proba(
    Xva_s
)[:, 1]

volume_m = metrics(
    yva,
    pva,
)

subject_df = aggregate_subjects(
    yva,
    pva,
    gva,
)

subject_m = metrics(
    subject_df["label"].to_numpy(),
    subject_df["prob_DME"].to_numpy(),
)

print_metrics(
    "KG-ONLY VALIDATION -- VOLUME LEVEL",
    volume_m,
)

print_metrics(
    "KG-ONLY VALIDATION -- SUBJECT LEVEL",
    subject_m,
)


# ============================================================
# Coefficient check
# ============================================================

coef = pd.DataFrame({
    "feature": keep,
    "coefficient": clf.coef_[0],
})

coef["abs_coefficient"] = (
    coef["coefficient"].abs()
)

coef = coef.sort_values(
    "abs_coefficient",
    ascending=False,
)

print("\nTOP 15 STANDARDIZED LOGISTIC COEFFICIENTS")
print("-----------------------------------------")

print(
    coef.head(15).to_string(
        index=False
    )
)


# ============================================================
# Save
# ============================================================

pd.DataFrame({
    "case_id": case_va,
    "group_id": gva,
    "label": yva,
    "prob_DME": pva,
}).to_csv(
    OUT / "val_volume_predictions.csv",
    index=False,
)

subject_df.to_csv(
    OUT / "val_subject_predictions.csv",
    index=False,
)

cv_df.to_csv(
    OUT / "train_only_C_selection.csv",
    index=False,
)

coef.to_csv(
    OUT / "standardized_coefficients.csv",
    index=False,
)

pd.DataFrame({
    "feature": keep,
    "train_median": medians.reindex(keep).to_numpy(),
    "train_std_before_scaling":
        train_std.reindex(keep).to_numpy(),
}).to_csv(
    OUT / "feature_check.csv",
    index=False,
)

print("\nSaved outputs to:")
print(OUT)

print("\nDONE.")
print("Held-out TEST graph files were NOT loaded or evaluated.")
