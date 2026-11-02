# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.

from pathlib import Path
import ast
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "kg_construction/extract_huang_trainval_descriptors.py"
)

MAIN_KG = (
    ROOT / "datasets/huang/results/patient_specific_kg"
)

DUKE_KG = (
    ROOT / "datasets/duke/results/patient_specific_kg"
)

OUT = (
    ROOT / "datasets/duke/results/kg_128_descriptors.csv"
)

# ============================================================
# Load ONLY the existing feature-extraction definitions.
# Do not execute the original training script.
# ============================================================

source_text = SOURCE.read_text()
tree = ast.parse(source_text)

needed_functions = {
    "get_objects",
    "excluded",
    "flatten_numeric",
    "aggregate_collection",
    "recursively_find_strings",
    "extract_graph_features",
}

selected_nodes = []

for node in tree.body:

    if (
        isinstance(node, ast.FunctionDef)
        and node.name in needed_functions
    ):
        selected_nodes.append(node)

    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = []

        if isinstance(node, ast.Assign):
            targets = node.targets
        else:
            targets = [node.target]

        for target in targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "EXCLUDE_TOKENS"
            ):
                selected_nodes.append(node)
                break

module = ast.Module(
    body=selected_nodes,
    type_ignores=[],
)

ast.fix_missing_locations(module)

ns = {
    "np": np,
    "pd": pd,
    "json": json,
    "re": re,
    "Path": Path,
}

exec(
    compile(
        module,
        str(SOURCE),
        "exec",
    ),
    ns,
)

required = (
    needed_functions
    | {"EXCLUDE_TOKENS"}
)

missing_defs = sorted(
    x for x in required
    if x not in ns
)

if missing_defs:
    raise RuntimeError(
        f"Could not recover extractor definitions: "
        f"{missing_defs}"
    )

extract_graph_features = ns[
    "extract_graph_features"
]


def extract_directory(kgdir):
    ns["KGDIR"] = kgdir

    files = sorted(
        kgdir.glob("*_kg.json")
    )

    rows = []

    for p in files:
        case_id = p.name[:-8]   # remove "_kg.json"

        feat = extract_graph_features(
            case_id
        )

        rows.append(
            {
                "case_id": case_id,
                **feat,
            }
        )

    return pd.DataFrame(rows)


print("=" * 80)
print("EXTRACTING MAIN-COHORT DESCRIPTORS")
print("=" * 80)

main = extract_directory(MAIN_KG)

print("Volumes:", len(main))

main_features = sorted(
    c for c in main.columns
    if c != "case_id"
)

print("Feature columns:", len(main_features))


print("\n" + "=" * 80)
print("EXTRACTING SRINIVASAN DESCRIPTORS")
print("=" * 80)

noor = extract_directory(DUKE_KG)

print("Volumes:", len(noor))

noor_features = sorted(
    c for c in noor.columns
    if c != "case_id"
)

print("Feature columns:", len(noor_features))


print("\n" + "=" * 80)
print("SCHEMA COMPARISON")
print("=" * 80)

main_set = set(main_features)
noor_set = set(noor_features)

missing_in_noor = sorted(
    main_set - noor_set
)

extra_in_noor = sorted(
    noor_set - main_set
)

print("Main feature count :", len(main_features))
print("Srinivasan feature count :", len(noor_features))
print("Missing in Srinivasan    :", len(missing_in_noor))
print("Extra in Srinivasan      :", len(extra_in_noor))

if missing_in_noor:
    print("\nMissing:")
    for x in missing_in_noor:
        print(" ", x)

if extra_in_noor:
    print("\nExtra:")
    for x in extra_in_noor:
        print(" ", x)


# ============================================================
# Require exact feature schema parity.
# ============================================================

if len(main_features) != 128:
    raise RuntimeError(
        f"Expected 128 main features, "
        f"found {len(main_features)}"
    )

if missing_in_noor or extra_in_noor:
    raise RuntimeError(
        "Noor descriptor schema does not exactly "
        "match the frozen main-cohort schema."
    )


# Force exact main feature order.
noor = noor[
    ["case_id"] + main_features
]


# Check finite values.
X = noor[
    main_features
].to_numpy(dtype=float)

n_nan = int(
    np.isnan(X).sum()
)

n_inf = int(
    np.isinf(X).sum()
)

print("\n" + "=" * 80)
print("NUMERICAL CHECK")
print("=" * 80)

print("Rows :", len(noor))
print("Cols :", len(main_features))
print("NaN  :", n_nan)
print("Inf  :", n_inf)

# ------------------------------------------------------------
# Native-true diagnostic:
#
# Missing lesion-lesion summary statistics are legitimate when
# a graph has zero lesion-lesion edges. The frozen KG classifier
# handles these values leakage-safely using TRAIN-fold medians
# (and 0.0 only if a feature is entirely missing in TRAIN).
#
# For this diagnostic we nevertheless require the missingness
# pattern to be EXACTLY the one already checked:
# SRIN_AMD_02, lesion_lesion.* descriptors only.
# ------------------------------------------------------------

if n_inf:
    raise RuntimeError(
        f"Unexpected infinite descriptor values: Inf={n_inf}"
    )

nan_mask = noor[main_features].isna()

nan_rows = set(
    noor.loc[
        nan_mask.any(axis=1),
        "case_id"
    ].astype(str)
)

nan_cols = [
    c for c in main_features
    if nan_mask[c].any()
]

if n_nan:
    if nan_rows != {"SRIN_AMD_02"}:
        raise RuntimeError(
            f"Unexpected NaN cases: {sorted(nan_rows)}"
        )

    if n_nan != 30:
        raise RuntimeError(
            f"Expected exactly 30 diagnostic NaNs, found {n_nan}"
        )

    bad_cols = [
        c for c in nan_cols
        if not c.startswith("lesion_lesion.")
    ]

    if bad_cols:
        raise RuntimeError(
            "Unexpected NaN descriptor families: "
            f"{bad_cols}"
        )

    print(
        "Expected diagnostic missingness accepted: "
        "30 lesion_lesion descriptors for SRIN_AMD_02"
    )

if len(noor) != 30:
    raise RuntimeError(
        f"Expected 30 Srinivasan volumes, "
        f"found {len(noor)}"
    )


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

noor.to_csv(
    OUT,
    index=False,
)

print("\n" + "=" * 80)
print("SUCCESS")
print("=" * 80)

print("Exact feature schema match: YES")
print("Expected descriptors       : 128")
print("Srinivasan volumes               :", len(noor))
print("Saved:")
print(OUT)

print("\nNo classifier was fitted or evaluated.")
