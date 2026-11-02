# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.

from pathlib import Path
import re
import json

import numpy as np
import tifffile

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms


ROOT = Path("PATH_TO_NOOR_SOURCE_DATASET")

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "datasets/noor/results/structural_proxy_seed42"

OUT = REPO_ROOT / "datasets/noor/results/structural_evidence_masks"

OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EXTS = {
    ".tif", ".tiff", ".png",
    ".jpg", ".jpeg", ".bmp",
}


def natural_key(x):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", str(x))
    ]


def volume_dirs(cls):
    return sorted(
        [
            p for p in (ROOT / cls).iterdir()
            if p.is_dir()
        ],
        key=natural_key,
    )


def image_files(d):
    return sorted(
        [
            p for p in d.rglob("*")
            if p.is_file()
            and p.suffix.lower() in EXTS
        ],
        key=natural_key,
    )


def case_id(cls, d):
    m = re.search(r"\((\d+)\)", d.name)
    if m is None:
        raise RuntimeError(d)

    return f"NOOR_{cls}_{int(m.group(1)):03d}"


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

tf = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=
            transforms.InterpolationMode.BILINEAR,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


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


# Teacher
from torchvision.models import ResNet18_Weights

teacher_base = resnet18(
    weights=ResNet18_Weights.DEFAULT
)

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
    p.requires_grad = False


# Student
student = MultiScaleResNet18().to(DEVICE)

ckpt = torch.load(
    MODEL_DIR / "best_student.pt",
    map_location="cpu",
    weights_only=False,
)

student.load_state_dict(
    ckpt["student"]
)

student.eval()


with open(
    MODEL_DIR / "threshold.json"
) as f:
    threshold = float(
        json.load(f)["threshold"]
    )


def anomaly_map(x):

    with torch.no_grad():

        ft = teacher(x)
        fs = student(x)

        maps = []

        for t, s in zip(ft, fs):

            t = F.normalize(
                t, p=2, dim=1
            )

            s = F.normalize(
                s, p=2, dim=1
            )

            m = (
                (t - s) ** 2
            ).mean(
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

        return torch.stack(
            maps,
            dim=0,
        ).mean(dim=0)[:, 0]


all_rows = []

for cls in ["AMD", "DME"]:

    for d in volume_dirs(cls):

        cid = case_id(cls, d)

        files = image_files(d)

        masks = []

        for p in files:

            raw = np.asarray(
                Image.open(p).convert("L")
            )

            im = (
                Image.fromarray(raw)
                .convert("RGB")
            )

            x = tf(im)[None].to(DEVICE)

            score = anomaly_map(x)[0]

            score = F.interpolate(
                score[None, None],
                size=raw.shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            score = (
                score.cpu().numpy()
            )

            mask = (
                score >= threshold
            ).astype(np.uint8)

            masks.append(mask)

        vol = np.stack(
            masks,
            axis=0,
        ).astype(np.uint8)

        out_path = (
            OUT / f"{cid}.tif"
        )

        tifffile.imwrite(
            out_path,
            vol,
            photometric="minisblack",
            compression=None,
        )

        fg = int(vol.sum())

        print(
            f"{cid:<18} "
            f"Z={vol.shape[0]:3d} "
            f"foreground={fg}"
        )

        all_rows.append(
            (
                cid,
                cls,
                vol.shape[0],
                fg,
            )
        )


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print("Volumes:", len(all_rows))
print(
    "AMD:",
    sum(r[1] == "AMD" for r in all_rows)
)
print(
    "DME:",
    sum(r[1] == "DME" for r in all_rows)
)
print("Output:", OUT)
