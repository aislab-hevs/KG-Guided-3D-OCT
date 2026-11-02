# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.

from pathlib import Path
import re, random, json

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


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    "PATH_TO_DUKE_SRINIVASAN_DATASET"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_OUT = REPO_ROOT / "datasets/duke/results/structural_proxy_seed42"

NATIVE_OUT = REPO_ROOT / "datasets/duke/results/structural_evidence_masks"

Z31_OUT = REPO_ROOT / "datasets/duke/results/structural_evidence_masks_z31"

MODEL_OUT.mkdir(parents=True, exist_ok=True)
NATIVE_OUT.mkdir(parents=True, exist_ok=True)
Z31_OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 10
BATCH_SIZE = 32
NORMAL_QUANTILE = 99.5
TARGET_Z = 31

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EXTS = {
    ".tif", ".tiff", ".png",
    ".jpg", ".jpeg", ".bmp",
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# FILE HELPERS
# ============================================================

def natural_key(x):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", str(x))
    ]


def image_files(d):
    return sorted(
        [
            p for p in d.rglob("*")
            if p.is_file()
            and p.suffix.lower() in EXTS
        ],
        key=natural_key,
    )


def find_case_dirs(prefix):
    pat = re.compile(
        rf"^{prefix}\s*(\d+)$",
        re.I,
    )

    found = []

    for d in ROOT.iterdir():
        if not d.is_dir():
            continue

        m = pat.match(d.name.strip())

        if m:
            found.append(
                (int(m.group(1)), d)
            )

    return [
        d for _, d in sorted(found)
    ]


def case_id(cls, d):
    m = re.search(
        r"(\d+)$",
        d.name.strip(),
    )

    if m is None:
        raise RuntimeError(
            f"Cannot parse case: {d}"
        )

    return (
        f"SRIN_{cls}_"
        f"{int(m.group(1)):02d}"
    )


normal_dirs = find_case_dirs("NORMAL")
amd_dirs = find_case_dirs("AMD")
dme_dirs = find_case_dirs("DME")

assert len(normal_dirs) == 15, len(normal_dirs)
assert len(amd_dirs) == 15, len(amd_dirs)
assert len(dme_dirs) == 15, len(dme_dirs)


# ============================================================
# NORMAL-ONLY SPLIT
# ============================================================

rng = np.random.default_rng(SEED)
perm = rng.permutation(15)

train_ids = set(
    perm[:12].tolist()
)

train_files = []
val_files = []
split_rows = []

for i, d in enumerate(normal_dirs):

    files = image_files(d)

    if not files:
        raise RuntimeError(
            f"No B-scans: {d}"
        )

    split = (
        "train"
        if i in train_ids
        else "val"
    )

    split_rows.append({
        "case_id":
            case_id("NORMAL", d),
        "source_dir":
            str(d),
        "n_slices":
            len(files),
        "split":
            split,
    })

    if split == "train":
        train_files.extend(files)
    else:
        val_files.extend(files)


pd.DataFrame(
    split_rows
).to_csv(
    MODEL_OUT / "normal_volume_split.csv",
    index=False,
)

print("=" * 80)
print("SRINIVASAN NORMAL-ONLY PATHOLOGY LOCALIZATION")
print("=" * 80)

print("Device:", DEVICE)
print("Normal train volumes: 12")
print("Normal val volumes  : 3")
print("Train B-scans:", len(train_files))
print("Val B-scans  :", len(val_files))


# ============================================================
# DATA
# ============================================================

imagenet_mean = [
    0.485, 0.456, 0.406
]

imagenet_std = [
    0.229, 0.224, 0.225
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

    def __getitem__(self, idx):

        im = (
            Image.open(
                self.files[idx]
            )
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
)


# ============================================================
# TEACHER / STUDENT
# ============================================================

class MultiScaleResNet18(nn.Module):

    def __init__(
        self,
        pretrained=False,
    ):
        super().__init__()

        base = resnet18(
            weights=(
                ResNet18_Weights.DEFAULT
                if pretrained
                else None
            )
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

    def forward(self, x):

        x = self.stem(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)

        return [f1, f2, f3]


teacher = MultiScaleResNet18(
    pretrained=True
).to(DEVICE).eval()

student = MultiScaleResNet18(
    pretrained=False
).to(DEVICE)

for p in teacher.parameters():
    p.requires_grad = False


optimizer = torch.optim.AdamW(
    student.parameters(),
    lr=1e-4,
    weight_decay=1e-5,
)


def feature_loss(ft, fs):

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

        loss += (
            (t - s) ** 2
        ).mean()

    return loss


def anomaly_map(x):

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
# TRAIN NORMAL ONLY
# ============================================================

best_val = float("inf")

for epoch in range(
    1,
    EPOCHS + 1,
):

    student.train()

    tr_losses = []

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

        tr_losses.append(
            float(loss.item())
        )

    student.eval()

    va_losses = []

    with torch.no_grad():

        for x in val_loader:

            x = x.to(
                DEVICE,
                non_blocking=True,
            )

            va_losses.append(
                float(
                    feature_loss(
                        teacher(x),
                        student(x),
                    ).item()
                )
            )

    tr = float(
        np.mean(tr_losses)
    )

    va = float(
        np.mean(va_losses)
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
                "epoch":
                    epoch,
                "normal_val_loss":
                    va,
                "seed":
                    SEED,
            },
            MODEL_OUT / "best_student.pt",
        )


# ============================================================
# LOAD BEST + NORMAL-ONLY THRESHOLD
# ============================================================

ckpt = torch.load(
    MODEL_OUT / "best_student.pt",
    map_location="cpu",
    weights_only=False,
)

student.load_state_dict(
    ckpt["student"]
)

student = student.to(
    DEVICE
).eval()


normal_scores = []

with torch.no_grad():

    for x in val_loader:

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        m = anomaly_map(x)

        normal_scores.append(
            m[:, ::4, ::4]
            .cpu()
            .numpy()
            .reshape(-1)
        )


normal_scores = np.concatenate(
    normal_scores
)

threshold = float(
    np.percentile(
        normal_scores,
        NORMAL_QUANTILE,
    )
)

print()
print(
    "Best normal-val epoch:",
    ckpt["epoch"],
)

print(
    f"Normal-only threshold "
    f"(q={NORMAL_QUANTILE}): "
    f"{threshold:.8f}"
)


with open(
    MODEL_OUT / "threshold.json",
    "w",
) as f:
    json.dump(
        {
            "threshold":
                threshold,
            "quantile":
                NORMAL_QUANTILE,
            "threshold_source":
                "3 held-out Srinivasan normal volumes",
            "AMD_DME_used":
                False,
        },
        f,
        indent=2,
    )


# ============================================================
# INFERENCE AMD / DME
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

    return mask


index_rows = []
n_done = 0


for cls, dirs in [
    ("AMD", amd_dirs),
    ("DME", dme_dirs),
]:

    for d in dirs:

        cid = case_id(cls, d)

        files = image_files(d)

        masks = [
            infer_slice(p)
            for p in files
        ]

        native = np.stack(
            masks,
            axis=0,
        ).astype(np.uint8)

        native_path = (
            NATIVE_OUT
            / f"{cid}.tif"
        )

        tifffile.imwrite(
            native_path,
            native,
            photometric="minisblack",
            compression=None,
        )

        z = native.shape[0]

        if z < TARGET_Z:
            raise RuntimeError(
                f"{cid}: Z={z} < {TARGET_Z}"
            )

        idx = np.round(
            np.linspace(
                0,
                z - 1,
                TARGET_Z,
            )
        ).astype(int)

        if len(np.unique(idx)) != TARGET_Z:
            raise RuntimeError(
                f"{cid}: duplicate z31 indices"
            )

        z31 = native[idx]

        tifffile.imwrite(
            Z31_OUT / f"{cid}.tif",
            z31,
            photometric="minisblack",
            compression=None,
        )

        for target_z, native_z in enumerate(idx):

            index_rows.append({
                "case_id":
                    cid,
                "target_z":
                    target_z,
                "native_z":
                    int(native_z),
                "native_Z":
                    z,
            })

        n_done += 1

        print(
            f"[{n_done:02d}/30] "
            f"{cid} "
            f"nativeZ={z:3d} -> 31 "
            f"foreground={int(z31.sum())}"
        )


pd.DataFrame(
    index_rows
).to_csv(
    MODEL_OUT / "native_to_z31_indices.csv",
    index=False,
)


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print("AMD/DME volumes:", n_done)
print("Native masks:", NATIVE_OUT)
print("Z31 masks   :", Z31_OUT)
