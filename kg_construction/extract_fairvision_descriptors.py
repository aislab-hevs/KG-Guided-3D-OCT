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
SOURCE = ROOT / "kg_construction/extract_huang_trainval_descriptors.py"

PRIMARY_KG = ROOT / "datasets/huang/results/patient_specific_kg"

FAIR_KG = (
    ROOT
    / "datasets/fairvision/results/patient_specific_kg"
)

MANIFEST = Path("PATH_TO_FAIRVISION_MANIFEST_CSV")

OUT = (
    ROOT
    / "datasets/fairvision/results/kg_128_descriptors.csv"
)


# ============================================================
# Recover EXACT primary-cohort descriptor extractor
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

        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )

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

required = needed_functions | {"EXCLUDE_TOKENS"}

missing = sorted(
    x for x in required
    if x not in ns
)

if missing:
    raise RuntimeError(
        f"Could not recover extractor definitions: {missing}"
    )

extract_graph_features = ns["extract_graph_features"]


# ============================================================
# Recover exact Huang/main-cohort 128-D schema
# ============================================================

primary_files = sorted(
    PRIMARY_KG.glob("*_kg.json")
)

if not primary_files:
    raise RuntimeError(
        f"No primary KG files found in {PRIMARY_KG}"
    )

ns["KGDIR"] = PRIMARY_KG

schema = set()

for p in primary_files:
    case_id = p.name[:-8]

    feat = extract_graph_features(case_id)

    schema.update(feat.keys())

schema = sorted(schema)

if len(schema) != 128:
    raise RuntimeError(
        "Primary-cohort descriptor schema is not 128-D: "
        f"{len(schema)}"
    )

print("=" * 88)
print("FAIRVISION FINAL KG DESCRIPTOR EXTRACTION")
print("=" * 88)
print("Primary schema :", len(schema))
print("FairVision KG  :", FAIR_KG)
print("Output         :", OUT)


# ============================================================
# FairVision manifest
# ============================================================

manifest = pd.read_csv(MANIFEST)

if "case_id" not in manifest.columns:
    raise RuntimeError(
        f"Manifest has no case_id: {list(manifest.columns)}"
    )

manifest["case_id"] = manifest["case_id"].astype(str)

manifest = manifest[
    manifest["split"].isin(["train", "val"])
].copy()

if len(manifest) != 7000:
    raise RuntimeError(
        f"Expected 7000 train+val rows, got {len(manifest)}"
    )

if manifest["case_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate case_id values in manifest."
    )


# ============================================================
# Verify KG inventory
# ============================================================

kg_files = sorted(
    FAIR_KG.glob("*_kg.json")
)

if len(kg_files) != 7000:
    raise RuntimeError(
        f"Expected 7000 KG JSONs, got {len(kg_files)}"
    )

kg_ids = {
    p.name[:-8]
    for p in kg_files
}

manifest_ids = set(
    manifest["case_id"]
)

if kg_ids != manifest_ids:

    missing_kg = sorted(
        manifest_ids - kg_ids
    )

    extra_kg = sorted(
        kg_ids - manifest_ids
    )

    raise RuntimeError(
        "KG/manifest case mismatch.\n"
        f"Missing KG: {missing_kg[:10]}\n"
        f"Extra KG: {extra_kg[:10]}"
    )


# ============================================================
# Extract exact 128-D descriptors
# ============================================================

ns["KGDIR"] = FAIR_KG

rows = []

for i, case_id in enumerate(
    manifest["case_id"],
    start=1,
):

    feat = extract_graph_features(
        case_id
    )

    extra = sorted(
        set(feat) - set(schema)
    )

    if extra:
        raise RuntimeError(
            f"{case_id}: unexpected descriptor columns: "
            f"{extra}"
        )

    row = {
        "case_id": case_id
    }

    # Exact source schema/order.
    for c in schema:
        row[c] = feat.get(
            c,
            np.nan,
        )

    rows.append(row)

    if (
        i == 1
        or i % 250 == 0
        or i == 7000
    ):
        print(
            f"[{i:04d}/7000] {case_id}"
        )


desc = pd.DataFrame(
    rows,
    columns=["case_id"] + schema,
)


# ============================================================
# Integrity check
# ============================================================

if len(desc) != 7000:
    raise RuntimeError(
        f"Expected 7000 descriptors, got {len(desc)}"
    )

if desc["case_id"].nunique() != 7000:
    raise RuntimeError(
        "Descriptor case_id values are not unique."
    )

if len(
    [
        c for c in desc.columns
        if c != "case_id"
    ]
) != 128:
    raise RuntimeError(
        "Descriptor dimension is not 128."
    )

X = (
    desc[schema]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
)

n_inf = int(
    np.isinf(
        X.to_numpy(dtype=float)
    ).sum()
)

if n_inf != 0:
    raise RuntimeError(
        f"Unexpected infinite values: {n_inf}"
    )


# ============================================================
# Attach manifest metadata without changing descriptor schema
# ============================================================

meta_cols = [
    c
    for c in [
        "case_id",
        "split",
        "label",
        "condition",
    ]
    if c in manifest.columns
]

out = (
    manifest[meta_cols]
    .merge(
        desc,
        on="case_id",
        how="left",
        validate="one_to_one",
    )
)

if len(out) != 7000:
    raise RuntimeError(
        "Final merged descriptor table is not 7000 rows."
    )

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUT,
    index=False,
)


print()
print("=" * 88)
print("FINISHED")
print("=" * 88)
print("Rows             :", len(out))
print("Unique case IDs  :", out["case_id"].nunique())
print("Descriptor dim   :", len(schema))
print("Train            :", int((out["split"] == "train").sum()))
print("Val              :", int((out["split"] == "val").sum()))
print("Infinite values  :", n_inf)
print("NaN descriptor values:", int(X.isna().sum().sum()))
print("Saved            :", OUT)
