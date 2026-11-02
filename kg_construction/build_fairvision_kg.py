# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.

from pathlib import Path
import argparse
import json
import tempfile

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

from build_huang_kg import build_case


PROJECT = Path(__file__).resolve().parents[1]

MANIFEST = Path("PATH_TO_FAIRVISION_MANIFEST_CSV")
DATA_ROOT = Path("PATH_TO_FAIRVISION_DATASET")
MODEL_DIR = PROJECT / "datasets/fairvision/results/structural_proxy_seed42"

COMP_DIR = PROJECT / "datasets/fairvision/results/mass99_components"
KG_DIR = PROJECT / "datasets/fairvision/results/patient_specific_kg"
CHECK = PROJECT / "datasets/fairvision/results/kg_construction_check.csv"

COMP_DIR.mkdir(parents=True, exist_ok=True)
KG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiScaleResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights=None)

        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1, f2, f3]


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

tf = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# Teacher
teacher_base = resnet18(weights=ResNet18_Weights.DEFAULT)

teacher = MultiScaleResNet18()

teacher.stem = nn.Sequential(
    teacher_base.conv1,
    teacher_base.bn1,
    teacher_base.relu,
    teacher_base.maxpool,
)
teacher.layer1 = teacher_base.layer1
teacher.layer2 = teacher_base.layer2
teacher.layer3 = teacher_base.layer3

teacher = teacher.to(DEVICE).eval()

for p in teacher.parameters():
    p.requires_grad_(False)


# Student
student = MultiScaleResNet18().to(DEVICE)

ckpt = torch.load(
    MODEL_DIR / "best_student.pt",
    map_location="cpu",
    weights_only=False,
)

student.load_state_dict(ckpt["student"])
student.eval()

for p in student.parameters():
    p.requires_grad_(False)


with open(MODEL_DIR / "threshold.json") as f:
    threshold = float(json.load(f)["threshold"])


@torch.inference_mode()
def anomaly_map(x):
    ft = teacher(x)
    fs = student(x)

    maps = []

    for t, s in zip(ft, fs):
        t = F.normalize(t, p=2, dim=1)
        s = F.normalize(s, p=2, dim=1)

        m = ((t - s) ** 2).mean(
            dim=1,
            keepdim=True,
        )

        m = F.interpolate(
            m,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        maps.append(m)

    return torch.stack(maps, dim=0).mean(dim=0)[:, 0]


@torch.inference_mode()
def infer_volume(volume, batch_size):
    masks = []

    for start in range(0, volume.shape[0], batch_size):
        end = min(start + batch_size, volume.shape[0])

        batch = torch.stack(
            [
                tf(
                    Image.fromarray(volume[z])
                    .convert("RGB")
                )
                for z in range(start, end)
            ]
        ).to(DEVICE)

        score = anomaly_map(batch)

        score = F.interpolate(
            score[:, None],
            size=(200, 200),
            mode="bilinear",
            align_corners=False,
        )[:, 0]

        masks.append(
            (score >= threshold)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

    return np.concatenate(masks, axis=0)


def existing_graph_summary(case_id):
    p = KG_DIR / f"{case_id}_kg.json"

    with open(p) as f:
        g = json.load(f)

    return {
        "case_id": case_id,
        "status": "skipped_existing",
        "lesions": len(g["nodes"]["lesions"]),
        "lesion_slice_edges":
            len(g["edges"]["lesion_to_slice"]),
        "lesion_lesion_edges":
            len(g["edges"]["lesion_to_lesion"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    df = pd.read_csv(MANIFEST)

    # FINAL DEVELOPMENT COHORT ONLY.
    # Preserve original manifest ordering.
    df = (
        df[
            df["split"].isin(["train", "val"])
        ]
        .copy()
        .reset_index(drop=True)
    )

    n_train = int((df["split"] == "train").sum())
    n_val = int((df["split"] == "val").sum())
    n_test = int((df["split"] == "test").sum())

    assert len(df) == 7000, len(df)
    assert n_train == 6000, n_train
    assert n_val == 1000, n_val
    assert n_test == 0, n_test

    print("=" * 90)
    print("FAIRVISION FULL KG CONSTRUCTION")
    print("=" * 90)
    print("Volumes   :", len(df))
    print("Train     :", int((df["split"] == "train").sum()))
    print("Val       :", int((df["split"] == "val").sum()))
    print("Test      :", int((df["split"] == "test").sum()))
    print("Device    :", DEVICE)
    print("Threshold :", threshold)
    print("KG dir    :", KG_DIR)
    print()

    rows = []

    with tempfile.TemporaryDirectory(
        prefix="fairvision_kg_"
    ) as tmpdir:

        tmpdir = Path(tmpdir)

        for i, row in enumerate(
            df.itertuples(index=False),
            start=1,
        ):
            cid = row.case_id

            final_kg = KG_DIR / f"{cid}_kg.json"

            if final_kg.exists():
                rows.append({
                    **existing_graph_summary(cid),
                    "split": row.split,
                    "label": int(row.label),
                    "condition": row.condition,
                })

                if i % 100 == 0:
                    print(
                        f"[{i:05d}/{len(df):05d}] "
                        f"{cid} SKIP"
                    )
                continue

            with np.load(
                DATA_ROOT / Path(row.path),
                allow_pickle=False,
            ) as z:
                volume = z["oct_bscans"]

            if volume.shape != (128, 200, 200):
                raise RuntimeError(
                    f"{cid}: unexpected shape {volume.shape}"
                )

            mask = infer_volume(
                volume,
                args.batch_size,
            )

            fg = int(mask.sum())

            if fg == 0:
                raise RuntimeError(
                    f"{cid}: EMPTY MASK -- "
                    "stopping rather than changing KG definition."
                )

            pred_path = tmpdir / f"{cid}.tif"

            tifffile.imwrite(
                pred_path,
                mask,
                photometric="minisblack",
                compression=None,
            )

            result = build_case(
                pred_path,
                COMP_DIR,
                KG_DIR,
            )

            rows.append({
                **result,
                "status": "built",
                "split": row.split,
                "label": int(row.label),
                "condition": row.condition,
            })

            pred_path.unlink(missing_ok=True)

            if i % 25 == 0 or i == len(df):
                pd.DataFrame(rows).to_csv(
                    CHECK,
                    index=False,
                )

                print(
                    f"[{i:05d}/{len(df):05d}] "
                    f"{cid} | "
                    f"FG={fg:,} | "
                    f"lesions={result['lesions']} | "
                    f"LS={result['lesion_slice_edges']} | "
                    f"LL={result['lesion_lesion_edges']}"
                )

    print()
    print("=" * 90)
    print("FINISHED")
    print("=" * 90)
    print("KG files:",
          len(list(KG_DIR.glob("*_kg.json"))))
    print("Check:", CHECK)


if __name__ == "__main__":
    main()
