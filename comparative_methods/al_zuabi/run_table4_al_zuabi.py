# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.
# Comparator implementation adapted to the common Huang split and validation-selected protocol; see README.md for the source publication.

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import timm

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)


# ============================================================
# AL ZUABI ET AL. COMPARATOR
# HUANG CONTROLLED TRANSFER-LEARNING ADAPTATION
#
# COMMON HUANG SUBJECT-DISJOINT DATA PROTOCOL
# ============================================================

SEED = 42

IMG_SIZE = 224
EPOCHS = 11
CANDIDATE_EPOCHS = {3, 5, 7, 9, 11}
BATCH_SIZE = 32

LR = 3e-4
WEIGHT_DECAY = 1e-4

THRESHOLD = 0.50  # training-time reporting only; final threshold selected on validation

REPO_ROOT = Path(__file__).resolve().parents[2]
META_CSV = REPO_ROOT / "comparative_methods/results/huang_table4_metadata.csv"

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUT_DIR = REPO_ROOT / "comparative_methods/al_zuabi/results" / f"table4_seed42_{stamp}"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

BEST_CKPT = OUT_DIR / "best_val_eye_auroc.pt"

SUPPORTED = {
    ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff",
}

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_all(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_all(SEED)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# FINAL CLEAN MANIFEST
# ============================================================

if not META_CSV.exists():
    raise FileNotFoundError(
        f"Missing manifest: {META_CSV}"
    )

meta = pd.read_csv(META_CSV).copy()

required = {
    "case_id",
    "pid",
    "path",
    "label",
    "split",
}

missing = required - set(meta.columns)

if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )

meta["case_id"] = meta["case_id"].astype(str)
meta["pid"] = meta["pid"].astype(str)
meta["label"] = meta["label"].astype(int)

meta["split"] = (
    meta["split"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# ============================================================
# LOCKED ORIGINAL HUANG / EYECLIP SPLIT
# ============================================================

EXPECTED = {
    "train": (126, 94),
    "val":   (34, 20),
    "test":  (32, 21),
}

for split_name, (n_volumes, n_eyes) in EXPECTED.items():

    d = meta[
        meta["split"] == split_name
    ]

    actual = (
        len(d),
        d["pid"].nunique(),
    )

    expected = (
        n_volumes,
        n_eyes,
    )

    if actual != expected:
        raise RuntimeError(
            f"{split_name}: split mismatch. "
            f"Found {actual}; expected {expected}."
        )


train_eyes = set(
    meta.loc[
        meta["split"] == "train",
        "pid",
    ]
)

val_eyes = set(
    meta.loc[
        meta["split"] == "val",
        "pid",
    ]
)

test_eyes = set(
    meta.loc[
        meta["split"] == "test",
        "pid",
    ]
)

if train_eyes & val_eyes:
    raise RuntimeError("TRAIN/VAL Eye-ID overlap.")

if train_eyes & test_eyes:
    raise RuntimeError("TRAIN/TEST Eye-ID overlap.")

if val_eyes & test_eyes:
    raise RuntimeError("VAL/TEST Eye-ID overlap.")

if (
    meta.groupby("pid")["label"]
    .nunique()
    .max()
    != 1
):
    raise RuntimeError(
        "Diagnosis inconsistency inside Eye ID."
    )

# ============================================================
# OCT IO
# ============================================================

ROOT = REPO_ROOT


def resolve_path(raw):

    p = Path(str(raw)).expanduser()

    if not p.is_absolute():
        p = ROOT / p

    p = p.resolve()

    if not p.exists():
        raise FileNotFoundError(
            f"Missing OCT volume: {p}"
        )

    return p


def directory_images(path):

    files = sorted(
        p
        for p in path.iterdir()
        if (
            p.is_file()
            and
            p.suffix.lower() in SUPPORTED
        )
    )

    if not files:
        raise RuntimeError(
            f"No B-scans in {path}"
        )

    return files


def volume_length(path):

    if path.is_dir():
        return len(
            directory_images(path)
        )

    with Image.open(path) as im:
        return int(
            getattr(im, "n_frames", 1)
        )


def read_frame(path, index):

    if path.is_dir():

        files = directory_images(path)

        with Image.open(
            files[index]
        ) as im:

            return (
                im.convert("RGB")
                .copy()
            )

    with Image.open(path) as im:

        n = int(
            getattr(im, "n_frames", 1)
        )

        if not 0 <= index < n:
            raise IndexError(
                f"{path}: frame {index}/{n}"
            )

        if n > 1:
            im.seek(index)

        return (
            im.convert("RGB")
            .copy()
        )


# ============================================================
# ONE EXACT CENTRAL B-SCAN PER VOLUME
# IDENTICAL POLICY TO FINAL DAVIS
# ============================================================

rows = []

for _, row in meta.iterrows():

    path = resolve_path(
        row["path"]
    )

    n_frames = volume_length(
        path
    )

    if n_frames <= 0:
        raise RuntimeError(
            f"Invalid OCT volume: {path}"
        )

    center = int(
        n_frames // 2
    )

    rows.append(
        {
            "case_id": str(row["case_id"]),
            "pid": str(row["pid"]),
            "path": str(path),
            "label": int(row["label"]),
            "split": str(row["split"]),
            "n_frames": int(n_frames),
            "center_index": center,
        }
    )


resolved = pd.DataFrame(rows)

resolved.to_csv(
    OUT_DIR / "meta_snapshot.csv",
    index=False,
)


# ============================================================
# SAME TRANSFORMS AS FINAL DAVIS
# ============================================================

train_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE),
            interpolation=InterpolationMode.BILINEAR,
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=5,
            interpolation=InterpolationMode.BILINEAR,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


eval_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE),
            interpolation=InterpolationMode.BILINEAR,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


# ============================================================
# ONE DATASET ITEM = ONE VOLUME = ONE CENTRAL B-SCAN
# ============================================================

class CentralDataset(Dataset):

    def __init__(
        self,
        frame,
        transform,
    ):

        self.frame = (
            frame
            .reset_index(drop=True)
            .copy()
        )

        self.transform = transform

    def __len__(self):

        return len(self.frame)

    def __getitem__(self, idx):

        row = self.frame.iloc[idx]

        image = read_frame(
            Path(row["path"]),
            int(row["center_index"]),
        )

        image = self.transform(
            image
        )

        return (
            image,

            torch.tensor(
                int(row["label"]),
                dtype=torch.long,
            ),

            str(row["case_id"]),

            str(row["pid"]),
        )


train_df = (
    resolved[
        resolved["split"] == "train"
    ]
    .reset_index(drop=True)
)

val_df = (
    resolved[
        resolved["split"] == "val"
    ]
    .reset_index(drop=True)
)

test_df = (
    resolved[
        resolved["split"] == "test"
    ]
    .reset_index(drop=True)
)


train_ds = CentralDataset(
    train_df,
    train_transform,
)

val_ds = CentralDataset(
    val_df,
    eval_transform,
)

test_ds = CentralDataset(
    test_df,
    eval_transform,
)


# Absolute no-replication guard.
assert len(train_ds) == 126
assert len(val_ds) == 34
assert len(test_ds) == 32


g = torch.Generator()
g.manual_seed(SEED)


train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
    generator=g,
)


val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
)


test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
)


# ============================================================
# XCEPTION TRANSFER MODEL
#
# KEY DIFFERENCE FROM FAILED RUNS:
#
# - pretrained Xception backbone FROZEN
# - backbone BN running statistics FROZEN
# - only the binary classifier is optimized
#
# This is NOT the previous 20.8M-parameter full fine-tune.
# ============================================================

available = set(
    timm.list_models()
)

if "legacy_xception" in available:
    MODEL_NAME = "legacy_xception"

elif "xception" in available:
    MODEL_NAME = "xception"

else:
    raise RuntimeError(
        "Xception unavailable in timm."
    )


class FrozenXception(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

        feature_dim = (
            self.backbone.num_features
        )

        self.classifier = nn.Linear(
            feature_dim,
            2,
        )

    def train(self, mode=True):

        super().train(mode)

        # Crucial:
        # frozen pretrained feature extractor,
        # including fixed BatchNorm statistics.
        self.backbone.eval()

        self.classifier.train(mode)

        return self

    def forward(self, x):

        # No gradients required through frozen backbone.
        with torch.no_grad():
            features = self.backbone(x)

        return self.classifier(
            features
        )


model = FrozenXception().to(
    DEVICE
)


# ============================================================
# CLASS WEIGHTS: EXACT SAME 44 / 33 TRAIN COHORT
# ============================================================

labels = train_df[
    "label"
].to_numpy(
    dtype=np.int64
)

counts = np.bincount(
    labels,
    minlength=2,
)



weights = (
    len(labels)
    /
    (
        2.0
        *
        counts.astype(np.float64)
    )
)


criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )
)


# ONLY classifier parameters.
optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# EYE-LEVEL EVALUATION
# ============================================================

@torch.inference_mode()
def predict(loader):

    model.eval()

    rows = []

    for (
        images,
        labels,
        case_ids,
        pids,
    ) in loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(images)

        p_dme = (
            torch.softmax(
                logits,
                dim=1,
            )[:, 1]
            .cpu()
            .numpy()
        )

        y = labels.numpy()

        for (
            yy,
            pp,
            case_id,
            pid,
        ) in zip(
            y,
            p_dme,
            case_ids,
            pids,
        ):

            rows.append(
                {
                    "case_id":
                        str(case_id),

                    "pid":
                        str(pid),

                    "y_true":
                        int(yy),

                    "p_dme":
                        float(pp),
                }
            )


    volumes = pd.DataFrame(rows)

    eyes = (
        volumes
        .groupby(
            ["pid", "y_true"],
            as_index=False,
        )["p_dme"]
        .mean()
    )

    return eyes, volumes


def metrics(eyes):

    y = eyes[
        "y_true"
    ].to_numpy(
        dtype=np.int64
    )

    p = eyes[
        "p_dme"
    ].to_numpy(
        dtype=np.float64
    )

    pred = (
        p >= THRESHOLD
    ).astype(np.int64)

    cm = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    sens = (
        tp / (tp + fn)
        if tp + fn
        else float("nan")
    )

    spec = (
        tn / (tn + fp)
        if tn + fp
        else float("nan")
    )

    return {
        "AUROC":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "AUPRC":
            float(
                average_precision_score(
                    y,
                    p,
                )
            ),

        "BalancedAccuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),

        "MacroF1":
            float(
                f1_score(
                    y,
                    pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "Accuracy":
            float(
                accuracy_score(
                    y,
                    pred,
                )
            ),

        "Sensitivity":
            float(sens),

        "Specificity":
            float(spec),
    }


# ============================================================
# VERIFY THIS REALLY IS A DIFFERENT MODEL CONFIGURATION
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print("=" * 78)
print(
    "AL-ZUABI / XCEPTION -- "
    "HUANG ORIGINAL-SPLIT FROZEN-TRANSFER FINAL"
)
print("=" * 78)

print(f"Device               : {DEVICE}")
print(f"Xception             : {MODEL_NAME}")
print("ImageNet pretrained  : YES")

print(
    "Backbone             : "
    "FROZEN pretrained feature extractor"
)

print(
    "Backbone BN stats    : FROZEN"
)

print(
    "Trainable component  : "
    "2-class linear classifier ONLY"
)

print(
    f"Total parameters     : "
    f"{total_params:,}"
)

print(
    f"Trainable parameters : "
    f"{trainable_params:,}"
)

print()

for split_name in (
    "train",
    "val",
    "test",
):

    d = resolved[
        resolved["split"] == split_name
    ]

    print(
        f"{split_name.upper():5s}: "
        f"{len(d):3d} volumes | "
        f"{d['pid'].nunique():3d} eyes | "
        f"{len(d):3d} B-scans"
    )


print()

print(
    "2D adaptation        : "
    "ONE CENTRAL B-SCAN / VOLUME"
)

print(
    "Label replication    : NONE"
)

print(
    f"Independent train units: "
    f"{len(train_ds)}"
)

print(f"Epochs               : {EPOCHS}")
print(f"Batch size           : {BATCH_SIZE}")
print("Optimizer            : AdamW")
print(f"LR                   : {LR}")
print(f"Weight decay         : {WEIGHT_DECAY}")
print("Loss                 : weighted CE")

print(
    f"Class weights        : "
    f"AMD={weights[0]:.6f}, "
    f"DME={weights[1]:.6f}"
)

print(
    f"Threshold            : "
    f"FIXED {THRESHOLD}"
)

print("Threshold tuning     : NONE")
print("Model selection      : VAL eye AUROC only")
print("Test usage           : ONCE after model selection")

print(
    f"Output               : {OUT_DIR}"
)

print("=" * 78)


# ============================================================
# TRAIN
# ============================================================

history = []

best_auc = -np.inf
best_epoch = None


for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    running = 0.0
    n_seen = 0


    for (
        images,
        labels,
        _,
        _,
    ) in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        bs = labels.size(0)

        running += (
            float(loss.item())
            * bs
        )

        n_seen += bs


    train_loss = (
        running
        / n_seen
    )


    val_eyes, _ = predict(
        val_loader
    )

    val_metrics = metrics(
        val_eyes
    )


    print(
        f"EPOCH {epoch:02d}/{EPOCHS} | "
        f"loss={train_loss:.6f} | "
        f"VAL AUROC="
        f"{val_metrics['AUROC']:.4f} | "
        f"AUPRC="
        f"{val_metrics['AUPRC']:.4f} | "
        f"BAcc="
        f"{val_metrics['BalancedAccuracy']:.4f} | "
        f"MacroF1="
        f"{val_metrics['MacroF1']:.4f} | "
        f"Acc="
        f"{val_metrics['Accuracy']:.4f}"
    )


    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            **{
                f"val_{k}": v
                for k, v
                in val_metrics.items()
            },
        }
    )


    if (
        epoch in CANDIDATE_EPOCHS
        and val_metrics["AUROC"] > best_auc
    ):

        best_auc = (
            val_metrics["AUROC"]
        )

        best_epoch = epoch

        torch.save(
            {
                "model_state":
                    model.state_dict(),

                "epoch":
                    epoch,

                "val_metrics":
                    val_metrics,
            },
            BEST_CKPT,
        )


pd.DataFrame(history).to_csv(
    OUT_DIR / "training_history.csv",
    index=False,
)


# ============================================================
# FINAL CHECKPOINT + ONE TEST
# ============================================================

ckpt = torch.load(
    BEST_CKPT,
    map_location=DEVICE,
)

model.load_state_dict(
    ckpt["model_state"]
)


print()
print("=" * 78)
print(
    f"BEST VALIDATION CHECKPOINT: "
    f"epoch {best_epoch} | "
    f"EYE AUROC={best_auc:.4f}"
)
print("=" * 78)


val_eyes, val_volumes = predict(
    val_loader
)


# ============================================================
# VALIDATION-ONLY OPERATING-POINT SELECTION
#
# p_dme is already a softmax probability in [0,1].
# Select threshold exclusively on validation eyes by
# maximizing balanced accuracy, then freeze for test.
# ============================================================

_val_y = val_eyes[
    "y_true"
].to_numpy(
    dtype=np.int64
)

_val_p = val_eyes[
    "p_dme"
].to_numpy(
    dtype=np.float64
)

_unique = np.sort(
    np.unique(
        _val_p
    )
)

_candidates = [0.0]

if len(_unique) > 1:
    _candidates.extend(
        (
            _unique[:-1]
            +
            _unique[1:]
        )
        / 2.0
    )

_candidates.append(1.0)

_best = None

for _t in _candidates:

    _pred = (
        _val_p >= _t
    ).astype(np.int64)

    _bacc = balanced_accuracy_score(
        _val_y,
        _pred,
    )

    # Primary criterion: maximum validation balanced accuracy.
    # Tie-break: threshold closest to 0.5.
    _key = (
        float(_bacc),
        -abs(float(_t) - 0.5),
    )

    if (
        _best is None
        or _key > _best[0]
    ):
        _best = (
            _key,
            float(_t),
        )

THRESHOLD = _best[1]

print()
print("=" * 78)
print("VALIDATION-SELECTED OPERATING POINT")
print("=" * 78)
print(
    f"Selected threshold: "
    f"{THRESHOLD:.8f}"
)

_val_pred = (
    _val_p >= THRESHOLD
).astype(np.int64)

print(
    f"Validation BAcc: "
    f"{balanced_accuracy_score(_val_y, _val_pred):.4f}"
)

print(
    f"Validation AUROC: "
    f"{roc_auc_score(_val_y, _val_p):.4f}"
)

print(
    "Threshold frozen before held-out test evaluation."
)
print("=" * 78)


# Held-out test is evaluated only after threshold is frozen.
test_eyes, test_volumes = predict(
    test_loader
)


val_eyes.to_csv(
    OUT_DIR / "best_val_eye_predictions.csv",
    index=False,
)

val_volumes.to_csv(
    OUT_DIR / "best_val_volume_predictions.csv",
    index=False,
)

test_eyes.to_csv(
    OUT_DIR / "test_eye_predictions.csv",
    index=False,
)

test_volumes.to_csv(
    OUT_DIR / "test_volume_predictions.csv",
    index=False,
)


val_final = metrics(
    val_eyes
)

test_final = metrics(
    test_eyes
)


result = {
    "method":
        "Al Zuabi 2026 - Xception",

    "adaptation":
        "frozen ImageNet Xception feature extractor",

    "data_protocol":
        "locked original Huang eye-disjoint split",

    "train":
        "126 volumes / 94 eyes",

    "validation":
        "34 volumes / 20 eyes",

    "test":
        "32 volumes / 21 eyes",

    "input":
        "one central B-scan per volume",

    "seed":
        SEED,

    "epochs":
        EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "optimizer":
        "AdamW",

    "lr":
        LR,

    "weight_decay":
        WEIGHT_DECAY,

    "loss":
        "weighted cross entropy",

    "threshold":
        THRESHOLD,

    "threshold_tuning":
        True,

    "threshold_selection":
        "validation eye-level maximum balanced accuracy",

    "best_epoch":
        int(best_epoch),

    "best_val_metrics":
        val_final,

    "test_metrics":
        test_final,
}


with open(
    OUT_DIR / "final_result.json",
    "w",
) as f:

    json.dump(
        result,
        f,
        indent=2,
    )


print()
print("=" * 78)
print("FINAL TEST -- EYE LEVEL")
print("=" * 78)

print(
    f"Best epoch          : "
    f"{best_epoch}"
)

print(
    f"Threshold           : "
    f"{THRESHOLD:.2f} FIXED"
)

print(
    f"N test eyes         : "
    f"{len(test_eyes)}"
)

print()

for key in (
    "AUROC",
    "AUPRC",
    "BalancedAccuracy",
    "MacroF1",
    "Accuracy",
    "Sensitivity",
    "Specificity",
):

    print(
        f"{key:19s}: "
        f"{test_final[key]:.4f}"
    )

print("=" * 78)

print(
    f"Saved to: {OUT_DIR}"
)

print("=" * 78)
