# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.
# Frozen EyeCLIP dependency: see README.md for the official repository and pretrained checkpoint.

from pathlib import Path
import sys

import numpy as np
import torch
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

EYECLIP_REPO = Path("PATH_TO_EYECLIP_REPOSITORY")
STATE = Path("PATH_TO_EYECLIP_CHECKPOINT")

VOL_DIR = Path("PATH_TO_NOOR_CANONICAL_VOLUMES")
OUT_DIR = ROOT / "datasets/noor/results/eyeclip_slice_features"

BATCH_SIZE = 64

sys.path.insert(0, str(EYECLIP_REPO))

from eyeclip.model import build_model
from eyeclip.clip import _transform


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("DEVICE:", device)


# ------------------------------------------------------------
# Load frozen EyeCLIP
# ------------------------------------------------------------

print("Loading EyeCLIP:", STATE)

obj = torch.load(
    STATE,
    map_location="cpu",
    weights_only=True,
)

if (
    isinstance(obj, dict)
    and "model_state_dict" in obj
    and not any(
        str(k).startswith("visual.")
        for k in obj.keys()
    )
):
    state_dict = obj["model_state_dict"]
else:
    state_dict = obj

model = build_model(state_dict)
model = model.to(device)
model.eval()

for p in model.parameters():
    p.requires_grad_(False)

input_resolution = int(model.visual.input_resolution)

print("EyeCLIP input resolution:", input_resolution)

preprocess = _transform(input_resolution)


# ------------------------------------------------------------
# Input check
# ------------------------------------------------------------

paths = sorted(VOL_DIR.glob("*.tif"))

print("Noor TIFF volumes:", len(paths))

if len(paths) != 98:
    raise RuntimeError(
        f"Expected 98 Noor volumes, found {len(paths)}"
    )

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------

@torch.no_grad()
def encode_volume(path):

    volume = tifffile.imread(path)

    if volume.ndim != 3:
        raise RuntimeError(
            f"{path.name}: expected [Z,H,W], "
            f"got {volume.shape}"
        )

    Z, H, W = volume.shape

    if volume.dtype != np.uint8:
        raise RuntimeError(
            f"{path.name}: expected uint8, "
            f"got {volume.dtype}; "
            f"range=({volume.min()}, {volume.max()})"
        )

    chunks = []

    for start in range(0, Z, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            Z,
        )

        images = []

        for z in range(start, end):

            img = Image.fromarray(
                volume[z]
            )

            images.append(
                preprocess(img)
            )

        batch = torch.stack(
            images,
            dim=0,
        ).to(device)

        features = model.encode_image(
            batch
        )

        features = (
            features
            .float()
            .cpu()
            .numpy()
        )

        if (
            features.ndim != 2
            or features.shape[1] != 512
        ):
            raise RuntimeError(
                f"{path.name}: unexpected "
                f"feature shape {features.shape}"
            )

        if not np.isfinite(features).all():
            raise RuntimeError(
                f"{path.name}: non-finite features"
            )

        chunks.append(features)

    features = np.concatenate(
        chunks,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    if features.shape != (Z, 512):
        raise RuntimeError(
            f"{path.name}: final feature shape "
            f"{features.shape}; expected {(Z, 512)}"
        )

    return volume.shape, features


slice_total = 0

for i, path in enumerate(paths, 1):

    case_id = path.stem
    out = OUT_DIR / f"{case_id}.npy"

    vol_shape, features = encode_volume(
        path
    )

    np.save(
        out,
        features,
    )

    slice_total += features.shape[0]

    print(
        f"[{i:03d}/098] "
        f"{case_id:<15} "
        f"volume={vol_shape} "
        f"features={features.shape}"
    )


# ------------------------------------------------------------
# Final check
# ------------------------------------------------------------

saved = sorted(
    OUT_DIR.glob("NOOR_*.npy")
)

bad = []

for p in saved:

    x = np.load(p)

    if (
        x.ndim != 2
        or x.shape[1] != 512
        or not np.isfinite(x).all()
    ):
        bad.append(
            (p.name, x.shape)
        )

print()
print("=" * 80)
print("NOOR EYECLIP FEATURE EXTRACTION")
print("=" * 80)

print("Volumes saved :", len(saved))
print("Total B-scans :", slice_total)
print("Bad files     :", len(bad))

if bad:
    print(bad)
    raise RuntimeError(
        "Feature check failed."
    )

if len(saved) != 98:
    raise RuntimeError(
        f"Expected 98 saved feature files, "
        f"found {len(saved)}"
    )

print(
    "PASS: all 98 Noor feature files "
    "are finite [Z,512] arrays."
)
