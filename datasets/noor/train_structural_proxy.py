# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.

from pathlib import Path
import re
import random
import json

import numpy as np
import pandas as pd
import tifffile

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)
from torchvision import transforms

import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("PATH_TO_NOOR_SOURCE_DATASET")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "datasets/noor/results/structural_proxy_seed42"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

(OUT / "preview_masks").mkdir(
    exist_ok=True,
)

(OUT / "preview_montages").mkdir(
    exist_ok=True,
)

SEED = 42
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 10
BATCH_SIZE = 32

# Fixed before looking at AMD/DME:
# 0.5% normal-validation pixel false-positive tail.
NORMAL_QUANTILE = 99.5

EXTS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HELPERS
# ============================================================

def natural_key(x):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(
            r"(\d+)",
            str(x),
        )
    ]


def image_files(volume_dir):
    files = [
        p
        for p in volume_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTS
    ]

    return sorted(
        files,
        key=natural_key,
    )


def volume_dirs(cls):
    base = ROOT / cls

    dirs = [
        p
        for p in base.iterdir()
        if p.is_dir()
    ]

    return sorted(
        dirs,
        key=natural_key,
    )


def make_case_id(cls, volume_dir):
    m = re.search(
        r"\((\d+)\)",
        volume_dir.name,
    )

    if m is None:
        raise RuntimeError(
            f"Cannot parse volume ID: {volume_dir}"
        )

    n = int(m.group(1))

    prefix = (
        "NORMAL"
        if cls == "Normal"
        else cls.upper()
    )

    return f"NOOR_{prefix}_{n:03d}"


# ============================================================
# NORMAL VOLUME SPLIT
# ============================================================

normal_dirs = volume_dirs("Normal")

assert len(normal_dirs) == 50, len(normal_dirs)

rng = np.random.default_rng(SEED)

order = rng.permutation(
    len(normal_dirs)
)

train_idx = set(
    order[:40].tolist()
)

rows = []

train_files = []
val_files = []

for i, d in enumerate(normal_dirs):
    cid = make_case_id(
        "Normal",
        d,
    )

    files = image_files(d)

    if not files:
        raise RuntimeError(
            f"No images: {d}"
        )

    split = (
        "train"
        if i in train_idx
        else "val"
    )

    rows.append({
        "case_id": cid,
        "source_dir": str(d),
        "n_slices": len(files),
        "split": split,
    })

    if split == "train":
        train_files.extend(files)
    else:
        val_files.extend(files)


split_df = pd.DataFrame(rows)

split_df.to_csv(
    OUT / "normal_volume_split.csv",
    index=False,
)

assert (
    split_df["split"] == "train"
).sum() == 40

assert (
    split_df["split"] == "val"
).sum() == 10


print("=" * 80)
print("NOOR NORMAL-ONLY ANOMALY LOCALIZATION")
print("=" * 80)
print("Device:", DEVICE)
print(
    "Train normal volumes:",
    int((split_df["split"] == "train").sum()),
)
print(
    "Val normal volumes  :",
    int((split_df["split"] == "val").sum()),
)
print(
    "Train B-scans:",
    len(train_files),
)
print(
    "Val B-scans  :",
    len(val_files),
)


# ============================================================
# DATASET
# ============================================================

imagenet_mean = [
    0.485,
    0.456,
    0.406,
]

imagenet_std = [
    0.229,
    0.224,
    0.225,
]


train_tf = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=
            transforms.InterpolationMode.BILINEAR,
    ),
    transforms.RandomHorizontalFlip(
        p=0.5
    ),
    transforms.RandomAffine(
        degrees=2.0,
        translate=(0.02, 0.02),
        scale=(0.98, 1.02),
    ),
    transforms.ColorJitter(
        brightness=0.05,
        contrast=0.05,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        imagenet_mean,
        imagenet_std,
    ),
])


eval_tf = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=
            transforms.InterpolationMode.BILINEAR,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        imagenet_mean,
        imagenet_std,
    ),
])


class SliceDataset(Dataset):

    def __init__(
        self,
        files,
        transform,
    ):
        self.files = list(files)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(
        self,
        idx,
    ):
        p = self.files[idx]

        im = (
            Image.open(p)
            .convert("L")
            .convert("RGB")
        )

        return self.transform(im)


train_loader = DataLoader(
    SliceDataset(
        train_files,
        train_tf,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)


val_loader = DataLoader(
    SliceDataset(
        val_files,
        eval_tf,
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)


# ============================================================
# TEACHER / STUDENT
# ============================================================

class MultiScaleResNet18(nn.Module):

    def __init__(
        self,
        pretrained,
    ):
        super().__init__()

        if pretrained:
            base = resnet18(
                weights=
                    ResNet18_Weights.DEFAULT
            )
        else:
            base = resnet18(
                weights=None
            )

        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
        )

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

    def forward(
        self,
        x,
    ):
        x = self.stem(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)

        return [
            f1,
            f2,
            f3,
        ]


teacher = MultiScaleResNet18(
    pretrained=True
).to(DEVICE)

student = MultiScaleResNet18(
    pretrained=False
).to(DEVICE)


teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False


optimizer = torch.optim.AdamW(
    student.parameters(),
    lr=1e-4,
    weight_decay=1e-5,
)


def feature_loss(
    ft,
    fs,
):
    loss = 0.0

    for t, s in zip(ft, fs):
        t = F.normalize(
            t,
            p=2,
            dim=1,
        )

        s = F.normalize(
            s,
            p=2,
            dim=1,
        )

        loss = loss + (
            (t - s) ** 2
        ).mean()

    return loss


def anomaly_map(
    x,
):
    with torch.no_grad():

        ft = teacher(x)
        fs = student(x)

        maps = []

        for t, s in zip(ft, fs):

            t = F.normalize(
                t,
                p=2,
                dim=1,
            )

            s = F.normalize(
                s,
                p=2,
                dim=1,
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
        ).mean(
            dim=0
        )[:, 0]


# ============================================================
# TRAIN ON NORMAL ONLY
# ============================================================

best_val = float("inf")

for epoch in range(
    1,
    EPOCHS + 1,
):

    student.train()

    losses = []

    for x in train_loader:

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        with torch.no_grad():
            ft = teacher(x)

        fs = student(x)

        loss = feature_loss(
            ft,
            fs,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        losses.append(
            float(loss.item())
        )

    student.eval()

    val_losses = []

    with torch.no_grad():

        for x in val_loader:

            x = x.to(
                DEVICE,
                non_blocking=True,
            )

            ft = teacher(x)
            fs = student(x)

            val_losses.append(
                float(
                    feature_loss(
                        ft,
                        fs,
                    ).item()
                )
            )

    tr = float(
        np.mean(losses)
    )

    va = float(
        np.mean(val_losses)
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"| train={tr:.6f} "
        f"| normal-val={va:.6f}"
    )

    if va < best_val:

        best_val = va

        torch.save(
            {
                "student":
                    student.state_dict(),

                "seed":
                    SEED,

                "epoch":
                    epoch,

                "normal_val_loss":
                    va,

                "normal_quantile":
                    NORMAL_QUANTILE,
            },
            OUT / "best_student.pt",
        )


# ============================================================
# LOAD BEST
# ============================================================

ckpt = torch.load(
    OUT / "best_student.pt",
    map_location="cpu",
    weights_only=False,
)

student.load_state_dict(
    ckpt["student"]
)

student = student.to(
    DEVICE
).eval()


print(
    "\nBest normal-val epoch:",
    ckpt["epoch"],
)

print(
    "Best normal-val loss :",
    ckpt["normal_val_loss"],
)


# ============================================================
# THRESHOLD FROM NORMAL VALIDATION ONLY
# ============================================================

normal_scores = []

with torch.no_grad():

    for x in val_loader:

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        m = anomaly_map(x)

        # Subsample spatial pixels only for
        # efficient quantile estimation.
        a = (
            m[:, ::4, ::4]
            .cpu()
            .numpy()
            .reshape(-1)
        )

        normal_scores.append(a)


normal_scores = np.concatenate(
    normal_scores
)

threshold = float(
    np.percentile(
        normal_scores,
        NORMAL_QUANTILE,
    )
)


print(
    f"\nNormal-only threshold "
    f"(q={NORMAL_QUANTILE}): "
    f"{threshold:.8f}"
)


with open(
    OUT / "threshold.json",
    "w",
) as f:
    json.dump(
        {
            "threshold": threshold,
            "quantile":
                NORMAL_QUANTILE,
            "threshold_source":
                "10 held-out normal Noor volumes only",
            "AMD_DME_used":
                False,
        },
        f,
        indent=2,
    )


# ============================================================
# SINGLE-SLICE INFERENCE
# ============================================================

def infer_slice(p):

    raw = np.asarray(
        Image.open(p).convert("L")
    )

    im = (
        Image.fromarray(raw)
        .convert("RGB")
    )

    x = eval_tf(im)[None].to(
        DEVICE
    )

    m = anomaly_map(x)[0]

    m = F.interpolate(
        m[None, None],
        size=raw.shape,
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    score = m.cpu().numpy()

    mask = (
        score >= threshold
    ).astype(np.uint8)

    return (
        raw,
        score,
        mask,
    )


# ============================================================
# PREVIEW VOLUMES
# 2 held-out Normal + 3 AMD + 3 DME
# ============================================================

val_case_ids = (
    split_df[
        split_df["split"] == "val"
    ]
    ["case_id"]
    .tolist()
)


case_lookup = {}

for cls in [
    "Normal",
    "AMD",
    "DME",
]:
    for d in volume_dirs(cls):
        cid = make_case_id(
            cls,
            d,
        )

        case_lookup[cid] = (
            cls,
            d,
        )


preview_ids = (
    val_case_ids[:2]
    + [
        make_case_id(
            "AMD",
            d,
        )
        for d in volume_dirs("AMD")[:3]
    ]
    + [
        make_case_id(
            "DME",
            d,
        )
        for d in volume_dirs("DME")[:3]
    ]
)


summary_rows = []


for cid in preview_ids:

    cls, d = case_lookup[cid]

    files = image_files(d)

    masks = []

    mean_scores = []
    max_scores = []

    vis = []

    for z, p in enumerate(files):

        raw, score, mask = (
            infer_slice(p)
        )

        masks.append(mask)

        mean_scores.append(
            float(score.mean())
        )

        max_scores.append(
            float(score.max())
        )

        vis.append(
            (
                z,
                raw,
                score,
                mask,
                float(score.mean()),
            )
        )

    mask_vol = np.stack(
        masks,
        axis=0,
    ).astype(np.uint8)

    tifffile.imwrite(
        OUT
        / "preview_masks"
        / f"{cid}.tif",
        mask_vol,
        photometric="minisblack",
        compression=None,
    )

    mask_fraction = float(
        mask_vol.mean()
    )

    summary_rows.append({
        "case_id": cid,
        "class": cls,
        "n_slices": len(files),
        "mean_anomaly":
            float(
                np.mean(
                    mean_scores
                )
            ),
        "max_anomaly":
            float(
                np.max(
                    max_scores
                )
            ),
        "mask_fraction":
            mask_fraction,
    })


    # Highest mean-anomaly slices.
    vis = sorted(
        vis,
        key=lambda t: t[4],
        reverse=True,
    )[:4]


    fig, axes = plt.subplots(
        len(vis),
        3,
        figsize=(10, 3 * len(vis)),
    )

    if len(vis) == 1:
        axes = axes[None, :]

    for r, (
        z,
        raw,
        score,
        mask,
        _,
    ) in enumerate(vis):

        axes[r, 0].imshow(
            raw,
            cmap="gray",
        )

        axes[r, 0].set_title(
            f"{cid} | slice {z}"
        )

        axes[r, 1].imshow(
            raw,
            cmap="gray",
        )

        axes[r, 1].imshow(
            score,
            cmap="jet",
            alpha=0.45,
        )

        axes[r, 1].set_title(
            "anomaly map"
        )

        axes[r, 2].imshow(
            mask,
            cmap="gray",
        )

        axes[r, 2].set_title(
            "thresholded mask"
        )

        for c in range(3):
            axes[r, c].axis(
                "off"
            )

    fig.tight_layout()

    fig.savefig(
        OUT
        / "preview_montages"
        / f"{cid}.png",
        dpi=160,
    )

    plt.close(fig)

    print(
        f"{cid:<18} "
        f"{cls:<6} "
        f"Z={len(files):3d} "
        f"mask_fraction="
        f"{mask_fraction:.5f}"
    )


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT / "preview_summary.csv",
    index=False,
)


print("\n" + "=" * 80)
print("PREVIEW SUMMARY")
print("=" * 80)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nSaved:",
    OUT,
)
