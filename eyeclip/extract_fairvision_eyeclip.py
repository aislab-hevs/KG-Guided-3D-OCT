# ISBI 2027 public release.
# Frozen EyeCLIP feature extraction for Harvard-FairVision.
# See README.md for the official EyeCLIP repository, checkpoint, and dataset setup.

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]

EYECLIP_REPO = Path("PATH_TO_EYECLIP_REPOSITORY")
STATE = Path("PATH_TO_EYECLIP_CHECKPOINT")

MANIFEST = Path("PATH_TO_FAIRVISION_MANIFEST_CSV")
DATA_ROOT = Path("PATH_TO_FAIRVISION_DATASET")
OUT_DIR = PROJECT / "datasets/fairvision/results/eyeclip_slice_features"

sys.path.insert(0, str(EYECLIP_REPO))

from eyeclip.model import build_model
from eyeclip.clip import _transform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all volumes; otherwise only first N rows",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_eyeclip(device):
    print("Loading EyeCLIP checkpoint:")
    print(" ", STATE)

    state_dict = torch.load(
        STATE,
        map_location="cpu",
        weights_only=True,
    )

    model = build_model(state_dict)
    model = model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    input_resolution = int(model.visual.input_resolution)
    preprocess = _transform(input_resolution)

    print("Device          :", device)
    print("Input resolution:", input_resolution)
    print("Model dtype     :", next(model.parameters()).dtype)

    return model, preprocess


def extract_volume(model, preprocess, volume, device, batch_size):
    all_features = []

    for start in range(0, len(volume), batch_size):
        end = min(start + batch_size, len(volume))

        images = [
            preprocess(Image.fromarray(volume[i]))
            for i in range(start, end)
        ]

        batch = torch.stack(images, dim=0).to(device)

        with torch.inference_mode():
            features = model.encode_image(batch)

        features = (
            features
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        all_features.append(features)

    return np.concatenate(all_features, axis=0)


def main():
    args = parse_args()

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST)

    if args.limit > 0:
        df = df.iloc[:args.limit].copy()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, preprocess = load_eyeclip(device)

    print()
    print("Manifest volumes :", len(df))
    print("Output directory :", OUT_DIR)
    print("Batch size       :", args.batch_size)
    print()

    processed = 0
    skipped = 0

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="FairVision EyeCLIP",
    ):
        case_id = row.case_id
        src = DATA_ROOT / Path(row.path)
        dst = OUT_DIR / f"{case_id}.npy"

        if dst.exists() and not args.overwrite:
            try:
                existing = np.load(dst, mmap_mode="r")

                if existing.shape == (128, 512):
                    skipped += 1
                    continue

                print(
                    f"\nInvalid cached shape for {case_id}: "
                    f"{existing.shape}; recomputing."
                )

            except Exception:
                print(
                    f"\nUnreadable cache for {case_id}; recomputing."
                )

        with np.load(src, allow_pickle=False) as z:
            volume = z["oct_bscans"]

        if volume.shape != (128, 200, 200):
            raise RuntimeError(
                f"{case_id}: unexpected OCT shape {volume.shape}"
            )

        features = extract_volume(
            model=model,
            preprocess=preprocess,
            volume=volume,
            device=device,
            batch_size=args.batch_size,
        )

        if features.shape != (128, 512):
            raise RuntimeError(
                f"{case_id}: unexpected feature shape "
                f"{features.shape}"
            )

        if not np.isfinite(features).all():
            raise RuntimeError(
                f"{case_id}: non-finite EyeCLIP features detected"
            )

        np.save(dst, features)
        processed += 1

    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print("Processed :", processed)
    print("Skipped   :", skipped)
    print("Total     :", len(df))
    print("Output    :", OUT_DIR)


if __name__ == "__main__":
    main()
