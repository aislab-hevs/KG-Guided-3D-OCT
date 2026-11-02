# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.

from pathlib import Path
from PIL import Image
import tifffile
import numpy as np
import pandas as pd
import re

RAW = Path("PATH_TO_NOOR_SOURCE_DATASET")
OUT = Path("PATH_TO_NOOR_CANONICAL_DATASET")
VOL_DIR = OUT / "volumes"

VOL_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".tif", ".tiff", ".png"}


def number_from_name(path):
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        raise RuntimeError(
            f"Cannot extract slice number: {path}"
        )
    return int(nums[-1])


def case_number(path):
    nums = re.findall(r"\d+", path.name)
    if not nums:
        raise RuntimeError(
            f"Cannot extract case number: {path}"
        )
    return int(nums[-1])


rows = []

print("=" * 80)
print("PREPARING NOOR AMD/DME EXTERNAL COHORT")
print("=" * 80)
print("RAW :", RAW)
print("OUT :", OUT)
print()
print("Spatial transforms: NONE")
print("Slice interpolation: NONE")
print("Intensity transform: NONE")
print()


for disease in ["AMD", "DME"]:

    cases = sorted(
        [
            p for p in (RAW / disease).iterdir()
            if p.is_dir()
        ],
        key=case_number,
    )

    expected_cases = 48 if disease == "AMD" else 50

    if len(cases) != expected_cases:
        raise RuntimeError(
            f"{disease}: expected {expected_cases} cases, "
            f"found {len(cases)}"
        )

    for case in cases:

        n_case = case_number(case)

        imgs = sorted(
            [
                p for p in case.iterdir()
                if p.is_file()
                and p.suffix.lower() in EXTS
            ],
            key=number_from_name,
        )

        if not imgs:
            raise RuntimeError(
                f"No images found in {case}"
            )

        nums = [
            number_from_name(p)
            for p in imgs
        ]

        expected_nums = list(
            range(1, len(imgs) + 1)
        )

        if nums != expected_nums:
            raise RuntimeError(
                f"Non-contiguous slice numbering in {case}\n"
                f"Observed: {nums}"
            )

        slices = []
        source_shapes = []
        source_exts = []

        for p in imgs:

            with Image.open(p) as im:
                arr = np.asarray(
                    im.convert("L"),
                    dtype=np.uint8,
                )

            if arr.ndim != 2:
                raise RuntimeError(
                    f"Unexpected image ndim in {p}: "
                    f"{arr.shape}"
                )

            slices.append(arr)
            source_shapes.append(arr.shape)
            source_exts.append(
                p.suffix.lower()
            )

        unique_shapes = sorted(
            set(source_shapes)
        )

        if len(unique_shapes) != 1:
            raise RuntimeError(
                f"Mixed spatial shapes inside {case}: "
                f"{unique_shapes}"
            )

        volume = np.stack(
            slices,
            axis=0,
        )

        case_id = (
            f"NOOR_{disease}_{n_case:03d}"
        )

        outfile = (
            VOL_DIR
            / f"{case_id}.tif"
        )

        tifffile.imwrite(
            outfile,
            volume,
            photometric="minisblack",
            compression=None,
        )

        z, h, w = volume.shape

        rows.append({
            "case_id": case_id,
            "disease": disease,
            "label": 0 if disease == "AMD" else 1,
            "source_case": case.name,
            "source_path": str(case),
            "source_extension":
                ",".join(sorted(set(source_exts))),
            "n_slices": z,
            "height": h,
            "width": w,
            "dtype": str(volume.dtype),
            "spatial_transform": "none",
            "depth_resampling": "none",
            "output_path": str(outfile),
        })

        print(
            f"{case_id:<15} "
            f"{volume.shape} "
            f"{rows[-1]['source_extension']}"
        )


manifest = pd.DataFrame(rows)

if len(manifest) != 98:
    raise RuntimeError(
        f"Expected 98 AMD/DME volumes, "
        f"got {len(manifest)}"
    )

if manifest["case_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate canonical case IDs."
    )


# ------------------------------------------------------------------
# Verify written TIFF metadata without modifying anything
# ------------------------------------------------------------------

for row in manifest.itertuples():

    with tifffile.TiffFile(
        row.output_path
    ) as tif:

        shape = tuple(
            tif.series[0].shape
        )

        dtype = str(
            tif.series[0].dtype
        )

    expected = (
        row.n_slices,
        row.height,
        row.width,
    )

    if shape != expected:
        raise RuntimeError(
            f"{row.case_id}: saved shape "
            f"{shape} != {expected}"
        )

    if dtype != "uint8":
        raise RuntimeError(
            f"{row.case_id}: saved dtype "
            f"{dtype}"
        )


manifest_path = (
    OUT
    / "manifest.csv"
)

manifest.to_csv(
    manifest_path,
    index=False,
)


print("\n" + "=" * 80)
print("FINAL NOOR CANONICAL COHORT")
print("=" * 80)

for disease in ["AMD", "DME"]:

    sub = manifest[
        manifest["disease"] == disease
    ]

    print(
        f"{disease}: "
        f"{len(sub)} volumes | "
        f"{sub['n_slices'].sum()} B-scans | "
        f"Z={sub['n_slices'].min()}-"
        f"{sub['n_slices'].max()}"
    )

print(
    "\nTotal volumes:",
    len(manifest)
)

print(
    "Total B-scans:",
    manifest["n_slices"].sum()
)

print(
    "\nSpatial shapes:"
)

print(
    manifest[
        ["height", "width"]
    ]
    .value_counts()
    .to_string()
)

print(
    "\nSaved manifest:"
)

print(
    manifest_path
)

print(
    "\nRaw dataset was NOT modified."
)
