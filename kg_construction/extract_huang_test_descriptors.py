# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.

from pathlib import Path
import contextlib
import io
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "kg_construction/extract_huang_trainval_descriptors.py"
SPLIT = Path("PATH_TO_HUANG_SUBJECT_SPLIT_CSV")

REFERENCE = (
    ROOT
    / "datasets/huang/results/kg_descriptors/train_val_graph_descriptors.csv"
)

OUT = (
    ROOT
    / "datasets/huang/results/table2_seed42/test_graph_descriptors.csv"
)

# ============================================================
# Load EXACT extractor implementation used for TRAIN/VAL.
# Stop before its original descriptor-extraction experiment.
# ============================================================

source = SOURCE.read_text()

marker = "# Extract graph descriptors"

if marker not in source:
    raise RuntimeError(
        "Could not locate descriptor extraction marker "
        "in extract_huang_trainval_descriptors.py"
    )

prefix = source.split(marker)[0]

ns = {}

with contextlib.redirect_stdout(io.StringIO()):
    exec(
        compile(
            prefix,
            str(SOURCE),
            "exec",
        ),
        ns,
    )

if "extract_graph_features" not in ns:
    raise RuntimeError(
        "extract_graph_features() was not recovered "
        "from the original source."
    )

extract_graph_features = ns["extract_graph_features"]


# ============================================================
# Frozen TEST cohort
# ============================================================

split_df = pd.read_csv(SPLIT)

split_df["case_id"] = (
    split_df["case_id"]
    .astype(str)
)

split_df["group_id"] = (
    split_df["group_id"]
    .astype(str)
)

test = (
    split_df[
        split_df["split"] == "test"
    ]
    .copy()
    .reset_index(drop=True)
)

assert len(test) == 32
assert test["group_id"].nunique() == 21


# ============================================================
# Reference TRAIN/VAL schema
# ============================================================

ref = pd.read_csv(REFERENCE)

ref["case_id"] = (
    ref["case_id"]
    .astype(str)
)

META = [
    "case_id",
    "group_id",
    "disease",
    "split",
]

feature_cols = [
    c
    for c in ref.columns
    if c not in META
]

assert len(feature_cols) == 128

assert (
    set(test["case_id"])
    .isdisjoint(
        set(ref["case_id"])
    )
)


# ============================================================
# Extract TEST graph descriptors
# ============================================================

rows = []

for i, row in test.iterrows():

    case_id = str(
        row["case_id"]
    )

    feat = dict(
        extract_graph_features(
            case_id
        )
    )

    # Metadata are never graph features.
    feat["case_id"] = case_id
    feat["group_id"] = str(
        row["group_id"]
    )
    feat["disease"] = str(
        row["disease"]
    )
    feat["split"] = "test"

    rows.append(feat)

    print(
        f"[{i+1:02d}/32] {case_id}"
    )


raw = pd.DataFrame(rows)

raw_feature_cols = [
    c
    for c in raw.columns
    if c not in META
]

unexpected = sorted(
    set(raw_feature_cols)
    - set(feature_cols)
)

absent_in_all_test = sorted(
    set(feature_cols)
    - set(raw_feature_cols)
)

print("\n" + "=" * 72)
print("TEST KG DESCRIPTOR SCHEMA CHECK")
print("=" * 72)

print("Test volumes :", len(raw))
print(
    "Test subjects:",
    raw["group_id"].nunique()
)

print(
    "Reference descriptor dimension:",
    len(feature_cols)
)

print(
    "Unexpected NEW descriptor names:",
    len(unexpected)
)

if unexpected:
    print(unexpected)

print(
    "Reference descriptors absent "
    "across all TEST graphs:",
    len(absent_in_all_test)
)

if absent_in_all_test:
    print(
        "(These will remain NaN and use the "
        "frozen TRAIN median during KG inference.)"
    )
    print(absent_in_all_test)

# New feature definitions are forbidden.
assert len(unexpected) == 0

# Force exact TRAIN/VAL feature schema and order.
final = raw.reindex(
    columns=
        META
        + feature_cols
)

assert len(final) == 32
assert final["case_id"].nunique() == 32
assert final["group_id"].nunique() == 21

assert list(
    final.columns[4:]
) == feature_cols

final.to_csv(
    OUT,
    index=False,
)

print("\nFINAL TEST DESCRIPTOR TABLE")
print("---------------------------")
print("Rows:", len(final))
print(
    "Structural descriptors:",
    len(feature_cols)
)
print(
    "AMD volumes:",
    int(
        (
            final["disease"]
            == "AMD"
        ).sum()
    )
)
print(
    "DME volumes:",
    int(
        (
            final["disease"]
            == "DME"
        ).sum()
    )
)

print("\nSaved:")
print(OUT)

print("\nNo model was fitted.")
print("No prediction was computed.")
print("TEST labels were not used to define any descriptor.")
