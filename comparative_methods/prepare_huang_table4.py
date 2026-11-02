# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.

#!/usr/bin/env python3

from pathlib import Path
import os
import pandas as pd
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT = Path("PATH_TO_HUANG_SUBJECT_SPLIT_CSV")

DATA_ROOT = Path(
    os.environ.get(
        "OCT_ROOT",
        "PATH_TO_HUANG_DATASET"
    )
)

OUT_ROOT = REPO_ROOT / "comparative_methods/results/huang_table4_volumes"

META_OUT = REPO_ROOT / "comparative_methods/results/huang_table4_metadata.csv"

SEARCH_DIRS = [
    DATA_ROOT / "AMD_labeled" / "input",
    DATA_ROOT / "AMD_unlabeled",
    DATA_ROOT / "DME_labeled" / "input",
    DATA_ROOT / "DME_unlabeled",
]


def find_volume(case_id):
    hits = []

    for d in SEARCH_DIRS:
        if not d.exists():
            continue

        for ext in (".tif", ".tiff"):
            p = d / f"{case_id}{ext}"
            if p.exists():
                hits.append(p)

    if len(hits) != 1:
        raise RuntimeError(
            f"{case_id}: expected one volume, found {hits}"
        )

    return hits[0]


df = pd.read_csv(SPLIT)

df["label"] = (
    df["disease"]
    .map({
        "AMD": 0,
        "DME": 1,
    })
    .astype(int)
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

rows = []

for n, r in enumerate(
    df.itertuples(index=False),
    1,
):
    case_id = str(r.case_id)

    case_dir = (
        OUT_ROOT / case_id
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    existing = sorted(
        case_dir.glob("slice_*.tiff")
    )

    if not existing:
        vol = tifffile.imread(
            find_volume(case_id)
        )

        if vol.ndim != 3:
            raise RuntimeError(
                f"{case_id}: expected Z,H,W, got {vol.shape}"
            )

        for z in range(
            vol.shape[0]
        ):
            tifffile.imwrite(
                case_dir /
                f"slice_{z:04d}.tiff",
                vol[z],
            )

    rows.append({
        "case_id": case_id,
        "pid": str(r.group_id),
        "path": str(
            case_dir.resolve()
        ),
        "label": int(
            r.label
        ),
        "split": str(
            r.split
        ),
    })

    if (
        n % 20 == 0
        or n == len(df)
    ):
        print(
            f"{n}/{len(df)}"
        )


meta = pd.DataFrame(
    rows
)

assert len(meta) == 192

assert (
    meta.loc[
        meta.split == "train",
        "pid"
    ].nunique()
    == 94
)

assert (
    meta.loc[
        meta.split == "val",
        "pid"
    ].nunique()
    == 20
)

assert (
    meta.loc[
        meta.split == "test",
        "pid"
    ].nunique()
    == 21
)

META_OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

meta.to_csv(
    META_OUT,
    index=False
)

print()
print(
    meta.groupby("split").agg(
        volumes=("case_id", "count"),
        subjects=("pid", "nunique"),
    )
)

print()
print("Saved:", META_OUT)
