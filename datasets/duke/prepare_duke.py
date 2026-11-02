# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.

from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import tifffile
import json
import re

RAW = Path("PATH_TO_DUKE_SRINIVASAN_ROOT") / "Publication_Dataset"

OUT = Path(
    "PATH_TO_DUKE_CANONICAL_DATASET"
)

VOL_DIR = OUT / "volumes"
INF_DIR = Path(
    "inference_inputs/Srinivasan_AMD_DME"
)

VOL_DIR.mkdir(parents=True, exist_ok=True)
INF_DIR.mkdir(parents=True, exist_ok=True)

def num(p):
    return int(
        re.findall(r"\d+", p.stem)[-1]
    )

rows = []

for disease in ["AMD", "DME"]:

    for i in range(1, 16):

        source_case = f"{disease}{i}"
        case_id = f"SRIN_{disease}_{i:02d}"

        src_dir = (
            RAW
            / source_case
            / "TIFFs"
            / "8bitTIFFs"
        )

        files = sorted(
            src_dir.glob("*.tif"),
            key=num,
        )

        if not files:
            raise RuntimeError(
                f"No slices: {src_dir}"
            )

        slices = []
        original_shapes = []

        for p in files:

            with Image.open(p) as im:

                arr = np.asarray(
                    im.convert("L")
                )

                original_shapes.append(
                    arr.shape
                )

                # Deterministic label-independent
                # in-plane harmonization.
                im512 = im.convert("L").resize(
                    (512, 512),
                    resample=Image.Resampling.BILINEAR,
                )

                arr512 = np.asarray(
                    im512,
                    dtype=np.uint8,
                )

            slices.append(arr512)

        vol = np.stack(
            slices,
            axis=0,
        )

        assert vol.shape == (
            len(files),
            512,
            512,
        )

        # Canonical copy
        tifffile.imwrite(
            VOL_DIR / f"{case_id}.tif",
            vol,
            photometric="minisblack",
            compression=None,
        )

        # nnU-Net input
        tifffile.imwrite(
            INF_DIR / f"{case_id}_0000.tif",
            vol,
            photometric="minisblack",
            compression=None,
        )

        with open(
            INF_DIR / f"{case_id}.json",
            "w",
        ) as f:
            json.dump(
                {"spacing": [1.0, 1.0, 1.0]},
                f,
                indent=2,
            )

        rows.append({
            "case_id": case_id,
            "source_case": source_case,
            "disease": disease,
            "label": 0 if disease == "AMD" else 1,
            "n_slices": len(files),
            "original_height":
                original_shapes[0][0],
            "original_width":
                original_shapes[0][1],
            "canonical_height": 512,
            "canonical_width": 512,
            "spatial_transform":
                "bilinear_resize_512x512",
            "depth_resampling": "none",
            "output_path":
                str(VOL_DIR / f"{case_id}.tif"),
        })

        print(
            f"{case_id:<14} "
            f"Z={len(files):3d} "
            f"{original_shapes[0]} "
            f"-> {vol.shape}"
        )

manifest = pd.DataFrame(rows)

assert len(manifest) == 30
assert manifest["case_id"].is_unique

manifest.to_csv(
    OUT / "manifest.csv",
    index=False,
)

print("\n==============================")
print("Prepared:", len(manifest))
print(
    manifest.groupby("disease").size()
)
print("Canonical:", OUT)
print("Inference :", INF_DIR)
