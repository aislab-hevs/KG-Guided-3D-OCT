# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)
from torchvision import transforms


# ============================================================
# CONFIG
# ============================================================

PROJECT = Path(__file__).resolve().parents[2]

MANIFEST = Path("PATH_TO_FAIRVISION_MANIFEST_CSV")

OUT = PROJECT / "datasets/fairvision/results/structural_proxy_seed42"
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

EPOCHS = 10
BATCH_SIZE = 32

LR = 1e-4
WEIGHT_DECAY = 1e-5

NORMAL_QUANTILE = 99.5

REFERENCE_CONDITION = "no.amd.diagnosis"

BASE_SPLIT = Path("PATH_TO_FAIRVISION_INNER_SPLIT_CSV")
DATA_ROOT = Path("PATH_TO_FAIRVISION_DATASET")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# MANIFEST / REFERENCE-ONLY SPLIT
# ============================================================

df = pd.read_csv(
    MANIFEST
).copy()

df["case_id"] = (
    df["case_id"]
    .astype(str)
)

base = pd.read_csv(BASE_SPLIT).copy()

assert len(base) == 4800
assert base["case_id"].is_unique

base_ids = set(
    base["case_id"].astype(str)
)

# KD reference cohort is restricted to the already frozen
# base4800 development cohort. Therefore neither meta1200
# nor official validation participates in fitting or
# checkpoint/threshold calibration.
reference = (
    df[
        (df["split"] == "train")
        & (df["case_id"].isin(base_ids))
        & (df["label"] == 0)
        & (
            df["condition"]
            == REFERENCE_CONDITION
        )
    ]
    .copy()
    .reset_index(drop=True)
)

assert len(reference) > 0

assert (
    reference["split"]
    == "train"
).all()

assert (
    reference["case_id"]
    .isin(base_ids)
).all()

assert (
    reference["label"]
    == 0
).all()

assert (
    reference["condition"]
    == REFERENCE_CONDITION
).all()

# Explicitly prove that no train case outside base4800
# entered the KD reference cohort.
assert set(reference["case_id"]).issubset(base_ids)

rng = np.random.default_rng(
    SEED
)

order = rng.permutation(
    len(reference)
)

# 80 / 20 volume-disjoint split.
n_train = int(
    np.floor(
        0.80
        * len(reference)
    )
)

train_idx = order[
    :n_train
]

val_idx = order[
    n_train:
]

train_df = (
    reference
    .iloc[train_idx]
    .copy()
    .reset_index(drop=True)
)

val_df = (
    reference
    .iloc[val_idx]
    .copy()
    .reset_index(drop=True)
)

assert len(train_df) + len(val_df) == len(reference)
assert len(train_df) > 0
assert len(val_df) > 0

assert set(
    train_df["case_id"]
).isdisjoint(
    set(
        val_df["case_id"]
    )
)

split_df = pd.concat(
    [
        train_df.assign(
            reference_split="train"
        ),
        val_df.assign(
            reference_split="val"
        ),
    ],
    ignore_index=True,
)

split_df[
    [
        "case_id",
        "path",
        "label",
        "condition",
        "reference_split",
    ]
].to_csv(
    OUT
    / "reference_volume_split.csv",
    index=False,
)


# ============================================================
# TRANSFORMS -- SOURCE-MATCHED
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


# ============================================================
# MODEL -- SOURCE-MATCHED
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
    p.requires_grad_(False)


optimizer = torch.optim.AdamW(
    student.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# LOSS / ANOMALY MAP -- SOURCE-MATCHED
# ============================================================

def feature_loss(
    ft,
    fs,
):
    loss = 0.0

    for t, s in zip(
        ft,
        fs,
    ):
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


@torch.inference_mode()
def anomaly_map(
    x,
):
    ft = teacher(x)
    fs = student(x)

    maps = []

    for t, s in zip(
        ft,
        fs,
    ):
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
# VOLUME HELPERS
# ============================================================

def load_volume(
    path,
):
    path = Path(path)
    if not path.is_absolute():
        path = DATA_ROOT / path
    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        volume = z[
            "oct_bscans"
        ]

    if volume.shape != (
        128,
        200,
        200,
    ):
        raise RuntimeError(
            f"Unexpected shape "
            f"{volume.shape}: "
            f"{path}"
        )

    return volume


def make_batch(
    volume,
    z_indices,
    transform,
):
    tensors = []

    for z in z_indices:
        im = (
            Image.fromarray(
                volume[int(z)]
            )
            .convert("L")
            .convert("RGB")
        )

        tensors.append(
            transform(im)
        )

    return torch.stack(
        tensors,
        dim=0,
    )


# ============================================================
# TRAIN / VALIDATE
# ============================================================

def train_one_epoch(
    epoch,
):
    student.train()

    # Epoch-specific deterministic volume order.
    epoch_rng = np.random.default_rng(
        SEED + epoch
    )

    volume_order = epoch_rng.permutation(
        len(train_df)
    )

    losses = []

    for j, idx in enumerate(
        volume_order,
        start=1,
    ):
        row = train_df.iloc[
            int(idx)
        ]

        volume = load_volume(
            row["path"]
        )

        # Randomize B-scan order inside each volume.
        z_order = epoch_rng.permutation(
            128
        )

        for start in range(
            0,
            128,
            BATCH_SIZE,
        ):
            z_idx = z_order[
                start:
                start + BATCH_SIZE
            ]

            x = make_batch(
                volume,
                z_idx,
                train_tf,
            ).to(
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
                float(
                    loss.item()
                )
            )

        if (
            j % 100 == 0
            or j == len(train_df)
        ):
            print(
                f"  train volumes "
                f"{j:04d}/"
                f"{len(train_df):04d}",
                flush=True,
            )

    return float(
        np.mean(losses)
    )


@torch.inference_mode()
def validate():
    student.eval()

    losses = []

    for j, row in enumerate(
        val_df.itertuples(
            index=False
        ),
        start=1,
    ):
        volume = load_volume(
            row.path
        )

        for start in range(
            0,
            128,
            BATCH_SIZE,
        ):
            z_idx = np.arange(
                start,
                min(
                    start
                    + BATCH_SIZE,
                    128,
                ),
            )

            x = make_batch(
                volume,
                z_idx,
                eval_tf,
            ).to(
                DEVICE,
                non_blocking=True,
            )

            ft = teacher(x)
            fs = student(x)

            loss = feature_loss(
                ft,
                fs,
            )

            losses.append(
                float(
                    loss.item()
                )
            )

        if (
            j % 50 == 0
            or j == len(val_df)
        ):
            print(
                f"  val volumes "
                f"{j:03d}/"
                f"{len(val_df):03d}",
                flush=True,
            )

    return float(
        np.mean(losses)
    )


# ============================================================
# RUN
# ============================================================

print("=" * 90)
print("FAIRVISION EXPLICIT-noAMD REFERENCE KD")
print("=" * 90)

print("Device              :", DEVICE)
print("Base cohort         :", len(base))
print("Reference volumes   :", len(reference))
print("Reference train     :", len(train_df))
print("Reference val       :", len(val_df))
print("Meta1200 used       : NO")
print(
    "Train B-scans       :",
    len(train_df) * 128,
)
print(
    "Val B-scans         :",
    len(val_df) * 128,
)
print("Epochs              :", EPOCHS)
print("Batch size          :", BATCH_SIZE)
print("LR                  :", LR)
print("Weight decay        :", WEIGHT_DECAY)
print("Quantile            :", NORMAL_QUANTILE)
print("AMD used            : NO")
print("Official val used   : NO")


best_val = float(
    "inf"
)

history = []

for epoch in range(
    1,
    EPOCHS + 1,
):
    print(
        "\n"
        + "=" * 90
    )
    print(
        f"EPOCH "
        f"{epoch:02d}/"
        f"{EPOCHS}"
    )
    print(
        "=" * 90
    )

    tr = train_one_epoch(
        epoch
    )

    va = validate()

    history.append({
        "epoch": epoch,
        "train_loss": tr,
        "reference_val_loss": va,
    })

    pd.DataFrame(
        history
    ).to_csv(
        OUT
        / "history.csv",
        index=False,
    )

    print(
        f"\nEpoch "
        f"{epoch:02d}/{EPOCHS} "
        f"| train={tr:.6f} "
        f"| reference-val={va:.6f}"
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
                "reference_val_loss":
                    va,
                "normal_quantile":
                    NORMAL_QUANTILE,
                "reference_condition":
                    REFERENCE_CONDITION,
                "reference_train_n":
                    len(train_df),
                "reference_val_n":
                    len(val_df),
            },
            OUT
            / "best_student.pt",
        )

        print(
            "  -> saved new best checkpoint"
        )


# ============================================================
# LOAD BEST
# ============================================================

ckpt = torch.load(
    OUT
    / "best_student.pt",
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
    "\nBest reference-val epoch:",
    ckpt["epoch"],
)

print(
    "Best reference-val loss :",
    ckpt[
        "reference_val_loss"
    ],
)


# ============================================================
# q99.5 THRESHOLD FROM REFERENCE VAL ONLY
# ============================================================

normal_scores = []

with torch.inference_mode():

    for j, row in enumerate(
        val_df.itertuples(
            index=False
        ),
        start=1,
    ):
        volume = load_volume(
            row.path
        )

        for start in range(
            0,
            128,
            BATCH_SIZE,
        ):
            z_idx = np.arange(
                start,
                min(
                    start
                    + BATCH_SIZE,
                    128,
                ),
            )

            x = make_batch(
                volume,
                z_idx,
                eval_tf,
            ).to(
                DEVICE,
                non_blocking=True,
            )

            m = anomaly_map(
                x
            )

            # EXACT SOURCE CALIBRATION RULE:
            # spatial subsampling only.
            a = (
                m[
                    :,
                    ::4,
                    ::4,
                ]
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
                .reshape(-1)
            )

            normal_scores.append(
                a
            )

        if (
            j % 25 == 0
            or j == len(val_df)
        ):
            print(
                f"Threshold calibration "
                f"{j:03d}/"
                f"{len(val_df):03d}",
                flush=True,
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


with open(
    OUT
    / "threshold.json",
    "w",
) as f:
    json.dump(
        {
            "threshold":
                threshold,
            "quantile":
                NORMAL_QUANTILE,
            "threshold_source":
                (
                    f"{len(val_df)} held-out FairVision "
                    "base4800 no.amd.diagnosis "
                    "reference volumes only"
                ),
            "reference_condition":
                REFERENCE_CONDITION,
            "reference_train_volumes":
                len(train_df),
            "reference_val_volumes":
                len(val_df),
            "AMD_used":
                False,
            "official_validation_used":
                False,
            "best_epoch":
                int(
                    ckpt["epoch"]
                ),
            "best_reference_val_loss":
                float(
                    ckpt[
                        "reference_val_loss"
                    ]
                ),
        },
        f,
        indent=2,
    )


print(
    "\n"
    + "=" * 90
)

print(
    "FAIRVISION REFERENCE MODEL COMPLETE"
)

print(
    "=" * 90
)

print(
    "Best epoch     :",
    ckpt["epoch"],
)

print(
    "Best val loss  :",
    ckpt[
        "reference_val_loss"
    ],
)

print(
    "q99.5 threshold:",
    f"{threshold:.10f}",
)

print(
    "Checkpoint     :",
    OUT
    / "best_student.pt",
)

print(
    "Threshold file :",
    OUT
    / "threshold.json",
)
