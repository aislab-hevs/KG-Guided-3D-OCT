# ISBI 2027 public release.
# Local VLFATRollout reproduction used by run_table4_vlfatrollout.py.
# Set PATH_TO_VLFATROLLOUT_REPOSITORY to the official VLFATRollout source repository.

from __future__ import annotations

import random
import re
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

META_CSV = ROOT / "comparative_methods/results/huang_table4_metadata.csv"
VLFAT_ROOT = Path("PATH_TO_VLFATROLLOUT_REPOSITORY")

OUT_DIR = (
    ROOT
    / "comparative_methods"
    / "vlfatrollout"
    / "results"
    / "local_reference_seed42"
)

sys.path.insert(0, str(VLFAT_ROOT))

from model_zoo.feature_extrc.models import ViT_VaR
from data.preprocessing.preprocess import middle_idxs
import torchvision.transforms as T


# ---------------------------------------------------------------------
# Local reproduction of the released VLFATRollout transform_custom()
#
# Official sequence:
#   ToPILImage
#   optional Grayscale
#   Resize((image_size, image_size))
#   ToTensor
#   ImageNet normalization for 3-channel input
# ---------------------------------------------------------------------

_MEAN_3CHANNEL = [0.485, 0.456, 0.406]
_STD_3CHANNEL = [0.229, 0.224, 0.225]
_MEAN_GRAY = [0.5]
_STD_GRAY = [0.5]


def transform_custom(
    images,
    image_size,
    gray_scale=True,
    resize=True,
):
    imgs = []

    for image in images:
        image = T.ToPILImage()(image)

        if gray_scale:
            image = T.Grayscale()(image)

        if resize:
            image = T.Resize(
                (image_size, image_size)
            )(image)

        image = T.ToTensor()(image)

        if gray_scale:
            image = T.Normalize(
                mean=_MEAN_GRAY,
                std=_STD_GRAY,
            )(image)
        else:
            image = T.Normalize(
                mean=_MEAN_3CHANNEL,
                std=_STD_3CHANNEL,
            )(image)

        imgs.append(image)

    return torch.stack(imgs)




# ---------------------------------------------------------------------
# Fixed experiment configuration
# ---------------------------------------------------------------------

SEED = 42

IMAGE_SIZE = 224
NUM_CLASSES = 2

# Official VLFAT/VLFATRollout variable-resolution set.
FRAME_CHOICES = [5, 10, 15, 20, 25]
INITIAL_FRAMES = 25

# VLFATRollout architecture.
DIM = 768
DEPTH = 12
HEADS = 3
RESERVE_TOKENS = 81
DISCARD_RATIO = 0.9
HEAD_FUSION = "mean"

# Paper training protocol.
MAX_EPOCHS = 50
ROLLOUT_WARMUP_EPOCHS = 10
EARLY_STOP_PATIENCE = 10
LR = 6e-6
WEIGHT_DECAY = 0.02

# Paper batch size is 8. Batch 1 is confirmed on the 4080 SUPER;
# accumulate 8 samples before each optimizer update.
BATCH_SIZE = 1
ACCUM_STEPS = 8

NUM_WORKERS = 4

# Whole-volume test inference: only the independent slice encoder is
# chunked. All slice embeddings are still passed jointly to the VFA.
TEST_SLICE_CHUNK = 16

AUGMENTATION_NAMES = [
    "identity",
    "brightness",
    "blur",
    "salt",
    "pepper",
    "rotate",
    "eraser",
]


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ---------------------------------------------------------------------
# Logging helper required by the official model constructor
# ---------------------------------------------------------------------

class SilentLogger:
    def info(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------
# Stable variable-length PE interpolation
#
# The released ViT_VaR pose_embd() replaces temporal_pos_encodings with
# a NEW nn.Parameter whenever T changes. Once the optimizer has already
# been constructed, that replacement leaves the new Parameter outside
# the optimizer.
#
# Here the original learnable 25-slice PE remains the canonical
# Parameter and every requested resolution is differentiably
# interpolated from it. No architecture, dimensionality, or attention
# operation is changed.
# ---------------------------------------------------------------------

def stable_pose_embd(self, num_patch_new: int):
    pe = self.temporal_pos_encodings

    if pe.shape[1] == num_patch_new:
        return pe

    pe_t = pe.transpose(-2, -1)

    if self.interpolation_type == "linear":
        tmp = F.interpolate(
            pe_t,
            size=num_patch_new,
            mode="linear",
            align_corners=False,
        )
    else:
        tmp = F.interpolate(
            pe_t,
            size=num_patch_new,
            mode=self.interpolation_type,
        )

    return tmp.transpose(-2, -1)


def build_model() -> torch.nn.Module:
    model = ViT_VaR(
        model_type="vit_base_patch16_224",
        pretrained=False,
        num_classes=NUM_CLASSES,
        dim=DIM,
        n_frames=INITIAL_FRAMES,
        logger=SilentLogger(),
        interpolation_type="linear",
        depth=DEPTH,
        heads=HEADS,
        reserve_token_nums=RESERVE_TOKENS,
        discard_ratio=DISCARD_RATIO,
        head_fusion=HEAD_FUSION,
    )

    # Compatibility correction for variable-length training.
    model.pose_embd = types.MethodType(stable_pose_embd, model)

    return model


# ---------------------------------------------------------------------
# File / OCT slice utilities
# ---------------------------------------------------------------------

def natural_key(path: Path):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", path.name)
    ]


def resolve_volume_dir(value) -> Path:
    p = Path(str(value))

    if not p.is_absolute():
        p = ROOT / p

    return p


def list_slice_files(volume_dir) -> list[Path]:
    p = resolve_volume_dir(volume_dir)

    files = []
    for pattern in (
        "*.tif",
        "*.tiff",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.bmp",
    ):
        files.extend(p.glob(pattern))

    files = sorted(set(files), key=natural_key)

    if not files:
        raise RuntimeError(f"No OCT slices found in: {p}")

    return files


def read_slice_u8(path: Path) -> np.ndarray:
    """
    Per-B-scan min-max normalization to uint8 [0,255],
    equivalent to the normalization used by the released loader.
    No OpenCV dependency.
    """
    with Image.open(path) as im:
        arr = np.asarray(im)

    # Convert RGB/RGBA input to grayscale if necessary.
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float32)
        arr = (
            0.299 * arr[..., 0]
            + 0.587 * arr[..., 1]
            + 0.114 * arr[..., 2]
        )

    arr = np.asarray(arr, dtype=np.float32)

    lo = float(arr.min())
    hi = float(arr.max())

    if hi > lo:
        arr = (arr - lo) * (255.0 / (hi - lo))
    else:
        arr = np.zeros_like(arr)

    return np.clip(
        np.rint(arr),
        0,
        255,
    ).astype(np.uint8)


# ---------------------------------------------------------------------
# Dependency-free compatibility reproduction of the augmentation family
# used by the released repository.
#
# The released augmentation module depends on old imgaug/vidaug packages.
# We preserve the same one-augmentation-per-volume policy and the same
# augmentation parameterization where it is explicit in the repository.
# ---------------------------------------------------------------------

def aug_identity(images: list[np.ndarray]) -> list[np.ndarray]:
    return [img.copy() for img in images]


def aug_brightness(images: list[np.ndarray]) -> list[np.ndarray]:
    # Official code samples one factor for the complete volume.
    factor = random.uniform(0.1, 1.9)

    return [
        np.asarray(
            ImageEnhance.Brightness(
                Image.fromarray(img)
            ).enhance(factor),
            dtype=np.uint8,
        )
        for img in images
    ]


def motion_blur_kernel(k: int, angle: float) -> np.ndarray:
    """
    Build a k x k normalized linear motion kernel without OpenCV.
    """
    kernel = np.zeros(
        (k, k),
        dtype=np.float32,
    )

    center = (k - 1) / 2.0
    theta = np.deg2rad(angle)

    # Dense sampling along a line through the kernel center.
    samples = max(32, k * 8)

    for t in np.linspace(
        -center,
        center,
        samples,
    ):
        x = center + t * np.cos(theta)
        y = center + t * np.sin(theta)

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))

        dx = x - x0
        dy = y - y0

        for yy, wy in (
            (y0, 1.0 - dy),
            (y0 + 1, dy),
        ):
            for xx, wx in (
                (x0, 1.0 - dx),
                (x0 + 1, dx),
            ):
                if (
                    0 <= yy < k
                    and 0 <= xx < k
                ):
                    kernel[yy, xx] += (
                        wy * wx
                    )

    total = float(kernel.sum())

    if total <= 0:
        kernel[k // 2, :] = 1.0
        total = float(kernel.sum())

    return kernel / total


def aug_blur(images: list[np.ndarray]) -> list[np.ndarray]:
    """
    Motion blur with k=5 and angle sampled from [-45,45].
    Uses torch.conv2d only; no OpenCV dependency.
    """
    out = []

    for img in images:
        angle = random.uniform(
            -45.0,
            45.0,
        )

        kernel = motion_blur_kernel(
            5,
            angle,
        )

        x = torch.from_numpy(
            img.astype(np.float32)
        )[None, None]

        w = torch.from_numpy(
            kernel
        )[None, None]

        blurred = F.conv2d(
            x,
            w,
            padding=2,
        )[0, 0]

        blurred = (
            blurred
            .clamp(0, 255)
            .round()
            .to(torch.uint8)
            .numpy()
        )

        out.append(blurred)

    return out


def aug_salt(images: list[np.ndarray]) -> list[np.ndarray]:
    # Exact vidaug Salt(ratio=20) logic:
    # pixels for which randint(20) == 0 become 255.
    out = []

    for img in images:
        noise = np.random.randint(
            20,
            size=img.shape,
        )

        x = np.where(
            noise == 0,
            255,
            img,
        ).astype(np.uint8)

        out.append(x)

    return out


def aug_pepper(images: list[np.ndarray]) -> list[np.ndarray]:
    # Exact vidaug Pepper(ratio=20) logic.
    out = []

    for img in images:
        noise = np.random.randint(
            20,
            size=img.shape,
        )

        x = np.where(
            noise == 0,
            0,
            img,
        ).astype(np.uint8)

        out.append(x)

    return out


def aug_rotate(images: list[np.ndarray]) -> list[np.ndarray]:
    # Released code applies a consistent +10 degree rotation to the clip.
    # expand=True mirrors scipy.ndimage.rotate's enlarged canvas behavior.
    out = []

    for img in images:
        rotated = Image.fromarray(img).rotate(
            10.0,
            resample=Image.Resampling.BILINEAR,
            expand=True,
            fillcolor=0,
        )

        out.append(
            np.asarray(rotated, dtype=np.uint8)
        )

    return out


def aug_eraser(images: list[np.ndarray]) -> list[np.ndarray]:
    # Match released eraser defaults.
    s_l, s_h = 0.1, 0.2
    r_1, r_2 = 0.3, 1 / 0.3

    h_img, w_img = images[0].shape[:2]

    while True:
        s = np.random.uniform(
            s_l,
            s_h,
        ) * h_img * w_img

        r = np.random.uniform(
            r_1,
            r_2,
        )

        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))

        left = np.random.randint(0, w_img)
        top = np.random.randint(0, h_img)

        if (
            left + w <= w_img
            and top + h <= h_img
        ):
            break

    value = np.random.uniform(0, 255)

    out = []

    for img in images:
        x = img.copy()
        x[top:top + h, left:left + w] = value
        out.append(
            np.asarray(x, dtype=np.uint8)
        )

    return out


AUGMENTATIONS = {
    "identity": aug_identity,
    "brightness": aug_brightness,
    "blur": aug_blur,
    "salt": aug_salt,
    "pepper": aug_pepper,
    "rotate": aug_rotate,
    "eraser": aug_eraser,
}


def apply_train_augmentation(
    images: list[np.ndarray],
) -> list[np.ndarray]:
    # Released loader chooses ONE augmentation at random per volume.
    name = random.choice(AUGMENTATION_NAMES)
    return AUGMENTATIONS[name](images)


# ---------------------------------------------------------------------
# Official-compatible image preprocessing
# ---------------------------------------------------------------------

def preprocess_files(
    files: list[Path],
    augment: bool,
) -> torch.Tensor:
    """
    Return [T, 3, 224, 224].

    Steps:
      1. per-slice min-max normalization to uint8 [0,255]
      2. optional one augmentation for the training volume
      3. grayscale -> three repeated channels
      4. official transform_custom:
            Resize(224,224), ToTensor, ImageNet normalization
    """
    images = [
        read_slice_u8(path)
        for path in files
    ]

    if augment:
        images = apply_train_augmentation(
            images
        )

    arr = np.stack(images, axis=0)
    arr = np.expand_dims(arr, axis=-1)
    arr = np.repeat(arr, 3, axis=-1)

    x = transform_custom(
        arr,
        IMAGE_SIZE,
        gray_scale=False,
    )

    return x


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class OCTVolumeDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        n_frames: int,
        augment: bool,
    ):
        self.df = (
            df
            .reset_index(drop=True)
            .copy()
        )

        self.n_frames = int(n_frames)
        self.augment = bool(augment)

        # Cache only file names, not image tensors.
        self.volume_files = [
            list_slice_files(row.path)
            for _, row in self.df.iterrows()
        ]

        for files in self.volume_files:
            if len(files) < min(FRAME_CHOICES):
                raise RuntimeError(
                    f"Volume has only {len(files)} slices; "
                    f"VLFATRollout requires at least "
                    f"{min(FRAME_CHOICES)}."
                )

    def __len__(self):
        return len(self.df)

    def set_n_frames(self, n_frames: int):
        n_frames = int(n_frames)

        if n_frames not in FRAME_CHOICES:
            raise ValueError(
                f"n_frames={n_frames} not in "
                f"{FRAME_CHOICES}"
            )

        self.n_frames = n_frames

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        files = self.volume_files[idx]

        if len(files) < self.n_frames:
            raise RuntimeError(
                f"{row.case_id}: only {len(files)} slices "
                f"for n_frames={self.n_frames}"
            )

        # Official random_middle sampling.
        ids = middle_idxs(
            n_selection=self.n_frames,
            len_arr=len(files),
        )

        selected = [
            files[i]
            for i in ids
        ]

        x = preprocess_files(
            selected,
            augment=self.augment,
        )

        return {
            "x": x,
            "label": torch.tensor(
                int(row.label),
                dtype=torch.long,
            ),
            "pid": str(row.pid),
            "case_id": str(row.case_id),
        }


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def subject_table(
    volume_df: pd.DataFrame,
) -> pd.DataFrame:

    assert (
        volume_df
        .groupby("pid")["label"]
        .nunique()
        == 1
    ).all()

    return (
        volume_df
        .groupby(
            "pid",
            as_index=False,
        )
        .agg(
            label=("label", "first"),
            prob_DME=("prob_DME", "mean"),
            n_volumes=("label", "size"),
        )
    )


def calculate_metrics(
    subject_df: pd.DataFrame,
) -> dict:

    y = subject_df["label"].to_numpy()
    p = subject_df["prob_DME"].to_numpy()

    pred = (
        p >= 0.5
    ).astype(int)

    return {
        "AUROC":
            roc_auc_score(y, p),

        "AUPRC":
            average_precision_score(y, p),

        "BalancedAccuracy":
            balanced_accuracy_score(
                y,
                pred,
            ),

        "MacroF1":
            f1_score(
                y,
                pred,
                average="macro",
            ),

        "Accuracy":
            accuracy_score(
                y,
                pred,
            ),
    }


# ---------------------------------------------------------------------
# Validation on the current training resolution
# ---------------------------------------------------------------------

@torch.inference_mode()
def evaluate_sampled(
    model,
    loader,
    device,
    criterion,
):

    model.eval()

    rows = []
    total_loss = 0.0
    total_n = 0

    for batch in loader:

        x = batch["x"].to(
            device,
            non_blocking=True,
        )

        y = batch["label"].to(
            device,
            non_blocking=True,
        )

        logits = model(x)
        loss = criterion(logits, y)

        prob = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        total_loss += (
            float(loss.item())
            * len(y)
        )
        total_n += len(y)

        for i in range(len(y)):
            rows.append({
                "case_id":
                    batch["case_id"][i],

                "pid":
                    batch["pid"][i],

                "label":
                    int(y[i].item()),

                "prob_DME":
                    float(prob[i].item()),
            })

    volume_df = pd.DataFrame(rows)
    subject_df = subject_table(volume_df)

    metrics = calculate_metrics(
        subject_df
    )

    return (
        volume_df,
        subject_df,
        metrics,
        total_loss / total_n,
    )


# ---------------------------------------------------------------------
# Whole-volume inference
# ---------------------------------------------------------------------

@torch.inference_mode()
def forward_whole_volume(
    model,
    files: list[Path],
    device,
    chunk_size: int,
):
    """
    Whole-volume VLFATRollout inference.

    Only the independent 2D slice encoder is chunked for memory.
    The embeddings of ALL B-scans are concatenated and jointly
    processed by the official volume transformer.
    """
    model.eval()

    slice_features = []

    for start in range(
        0,
        len(files),
        chunk_size,
    ):
        chunk_files = files[
            start:start + chunk_size
        ]

        x = preprocess_files(
            chunk_files,
            augment=False,
        ).to(
            device,
            non_blocking=True,
        )

        features = (
            model
            .feature_extractor_spacial(x)
        )

        slice_features.append(
            features
        )

    # [T,768] -> [1,T,768]
    features = torch.cat(
        slice_features,
        dim=0,
    ).unsqueeze(0)

    cls_token = model.temporal_token.expand(
        1,
        -1,
        -1,
    )

    features = torch.cat(
        (
            cls_token,
            features,
        ),
        dim=1,
    )

    features = (
        features
        + model.pose_embd(
            features.shape[1]
        )
    )

    features = model.pos_drop(
        features
    )

    features = (
        model
        .feature_extractor_temporal(
            features
        )
    )

    volume_cls = features[:, 0]

    logits = model.cls_head(
        volume_cls
    )

    return logits


@torch.inference_mode()
def evaluate_test_whole_volume(
    model,
    test_df: pd.DataFrame,
    device,
):

    model.eval()

    rows = []

    print(
        "\nWhole-volume TEST inference..."
    )

    for number, (_, row) in enumerate(
        test_df.iterrows(),
        start=1,
    ):

        files = list_slice_files(
            row.path
        )

        logits = forward_whole_volume(
            model,
            files,
            device,
            chunk_size=TEST_SLICE_CHUNK,
        )

        prob = torch.softmax(
            logits,
            dim=1,
        )[0, 1]

        rows.append({
            "case_id":
                str(row.case_id),

            "pid":
                str(row.pid),

            "label":
                int(row.label),

            "n_slices":
                len(files),

            "prob_DME":
                float(prob.item()),
        })

        print(
            f"  test volume "
            f"{number:02d}/{len(test_df)} | "
            f"slices={len(files):3d}"
        )

    volume_df = pd.DataFrame(rows)
    subject_df = subject_table(volume_df)

    metrics = calculate_metrics(
        subject_df
    )

    return (
        volume_df,
        subject_df,
        metrics,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    seed_everything(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this run."
        )

    device = torch.device("cuda")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("VLFATRollout 2024 — adapted reimplementation")
    print("=" * 78)
    print("Torch :", torch.__version__)
    print("CUDA  :", torch.version.cuda)
    print("GPU   :", torch.cuda.get_device_name(0))
    print("Seed  :", SEED)

    # -------------------------------------------------------------
    # Fixed benchmark split
    # -------------------------------------------------------------

    df = pd.read_csv(META_CSV)

    required = {
        "case_id",
        "pid",
        "path",
        "label",
        "split",
    }

    assert required.issubset(
        df.columns
    )

    df["label"] = (
        df["label"]
        .astype(int)
    )

    assert set(
        df["label"].unique()
    ) == {0, 1}

    assert len(df) == 192
    assert len(df[df.split == "train"]) == 126
    assert len(df[df.split == "val"]) == 34
    assert len(df[df.split == "test"]) == 32

    assert (
        df[df.split == "train"]
        .pid.nunique()
        == 94
    )

    assert (
        df[df.split == "val"]
        .pid.nunique()
        == 20
    )

    assert (
        df[df.split == "test"]
        .pid.nunique()
        == 21
    )

    train_df = (
        df[df.split == "train"]
        .copy()
    )
    val_df = (
        df[df.split == "val"]
        .copy()
    )
    test_df = (
        df[df.split == "test"]
        .copy()
    )

    train_pid = set(train_df.pid)
    val_pid = set(val_df.pid)
    test_pid = set(test_df.pid)

    assert not (train_pid & val_pid)
    assert not (train_pid & test_pid)
    assert not (val_pid & test_pid)

    print("\nDataset:")
    print(
        df.groupby("split").agg(
            volumes=("case_id", "size"),
            subjects=("pid", "nunique"),
        )
    )

    # -------------------------------------------------------------
    # Data
    # -------------------------------------------------------------

    train_ds = OCTVolumeDataset(
        train_df,
        n_frames=INITIAL_FRAMES,
        augment=True,
    )

    val_ds = OCTVolumeDataset(
        val_df,
        n_frames=INITIAL_FRAMES,
        augment=False,
    )

    train_generator = (
        torch.Generator()
        .manual_seed(SEED)
    )

    val_generator = (
        torch.Generator()
        .manual_seed(SEED + 1)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=train_generator,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=val_generator,
        persistent_workers=False,
    )

    # -------------------------------------------------------------
    # Official weighted-loss setting
    # -------------------------------------------------------------

    counts = np.bincount(
        train_df["label"].to_numpy(),
        minlength=2,
    ).astype(np.float64)

    class_weights = (
        len(train_df)
        / (2.0 * counts)
    )

    weight_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    print(
        "\nTrain volume counts:",
        counts.astype(int).tolist()
    )
    print(
        "CE class weights   :",
        class_weights.tolist()
    )

    criterion = (
        torch.nn.CrossEntropyLoss(
            weight=weight_tensor
        )
    )

    # -------------------------------------------------------------
    # Model / optimizer / scheduler
    # -------------------------------------------------------------

    model = build_model().to(
        device
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "\nModel parameters:",
        f"{total_params/1e6:.2f} M"
    )

    print(
        "Canonical temporal PE:",
        tuple(
            model
            .temporal_pos_encodings
            .shape
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=MAX_EPOCHS,
            eta_min=0.0,
        )
    )

    print("\nTraining protocol:")
    print("  epochs             :", MAX_EPOCHS)
    print("  physical batch     :", BATCH_SIZE)
    print("  grad accumulation  :", ACCUM_STEPS)
    print(
        "  effective batch    :",
        BATCH_SIZE * ACCUM_STEPS
    )
    print("  lr                 :", LR)
    print("  weight decay       :", WEIGHT_DECAY)
    print("  precision          : FP32")
    print("  initial slices     :", INITIAL_FRAMES)
    print("  variable slices    :", FRAME_CHOICES)
    print(
        "  rollout starts     : epoch",
        ROLLOUT_WARMUP_EPOCHS + 1,
    )
    print("  retained tokens    :", RESERVE_TOKENS)
    print("  discard ratio      :", DISCARD_RATIO)
    print("  head fusion        :", HEAD_FUSION)
    print(
        "  checkpoint metric  : "
        "subject-level validation AUROC"
    )
    print(
        "  test               : untouched until final checkpoint"
    )

    # Dedicated RNG for the paper's per-epoch resolution schedule.
    schedule_rng = np.random.RandomState(
        SEED
    )

    current_n = INITIAL_FRAMES

    best_auc = -float("inf")
    best_epoch = None
    bad_epochs = 0
    best_path = OUT_DIR / "best_model.pt"

    history = []

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------

    for epoch in range(MAX_EPOCHS):

        epoch_start = time.time()

        train_ds.set_n_frames(
            current_n
        )
        val_ds.set_n_frames(
            current_n
        )

        rollout_active = (
            epoch
            >= ROLLOUT_WARMUP_EPOCHS
        )

        model.feature_extractor_spacial.filter_tokens = (
            rollout_active
        )

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss = 0.0
        running_correct = 0
        running_n = 0

        n_batches = len(train_loader)

        for step, batch in enumerate(
            train_loader
        ):

            # Correct averaging for the final partial accumulation group.
            group_start = (
                step // ACCUM_STEPS
            ) * ACCUM_STEPS

            group_end = min(
                group_start + ACCUM_STEPS,
                n_batches,
            )

            group_size = (
                group_end - group_start
            )

            x = batch["x"].to(
                device,
                non_blocking=True,
            )

            y = batch["label"].to(
                device,
                non_blocking=True,
            )

            # Intentionally FP32:
            # rollout backward was numerically unstable under FP16.
            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at "
                    f"epoch={epoch+1}, step={step+1}"
                )

            (
                loss / group_size
            ).backward()

            running_loss += (
                float(loss.item())
                * len(y)
            )

            pred = logits.argmax(
                dim=1
            )

            running_correct += int(
                (pred == y).sum().item()
            )

            running_n += len(y)

            if (
                step + 1 == group_end
            ):
                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

        train_loss = (
            running_loss / running_n
        )

        train_acc = (
            running_correct / running_n
        )

        # ---------------------------------------------------------
        # Validation only — test is not accessed here.
        # ---------------------------------------------------------

        (
            _,
            val_subject,
            val_metrics,
            val_loss,
        ) = evaluate_sampled(
            model,
            val_loader,
            device,
            criterion,
        )

        lr_now = (
            optimizer
            .param_groups[0]["lr"]
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        print(
            f"\nEpoch {epoch+1:03d}/{MAX_EPOCHS} | "
            f"N={current_n:02d} | "
            f"rollout={int(rollout_active)} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_BAcc={val_metrics['BalancedAccuracy']:.4f} | "
            f"val_AUROC={val_metrics['AUROC']:.4f} | "
            f"val_AUPRC={val_metrics['AUPRC']:.4f} | "
            f"lr={lr_now:.3e} | "
            f"time={elapsed/60:.1f}m"
        )

        # ---------------------------------------------------------
        # Validation-only model selection.
        # Only rollout-active epochs are checkpoint eligible.
        # ---------------------------------------------------------

        should_stop = False

        if rollout_active:
            current_auc = float(
                val_metrics["AUROC"]
            )

            if current_auc > best_auc:
                best_auc = current_auc
                best_epoch = epoch + 1
                bad_epochs = 0

                print(
                    "  -> New best validation checkpoint"
                )

                torch.save(
                    {
                        "method":
                            "VLFATRollout 2024 adapted reimplementation",

                        "epoch":
                            best_epoch,

                        "model":
                            model.state_dict(),

                        "filter_tokens":
                            rollout_active,

                        "n_frames":
                            current_n,

                        "val_metrics":
                            val_metrics,

                        "val_loss":
                            val_loss,

                        "config": {
                            "seed":
                                SEED,

                            "pretrained":
                                False,

                            "image_size":
                                IMAGE_SIZE,

                            "dim":
                                DIM,

                            "depth":
                                DEPTH,

                            "heads":
                                HEADS,

                            "reserve_tokens":
                                RESERVE_TOKENS,

                            "discard_ratio":
                                DISCARD_RATIO,

                            "head_fusion":
                                HEAD_FUSION,

                            "rollout_warmup_epochs":
                                ROLLOUT_WARMUP_EPOCHS,

                            "lr":
                                LR,

                            "weight_decay":
                                WEIGHT_DECAY,

                            "max_epochs":
                                MAX_EPOCHS,

                            "early_stop_patience":
                                EARLY_STOP_PATIENCE,

                            "checkpoint_metric":
                                "subject_AUROC",

                            "effective_batch_size":
                                BATCH_SIZE
                                * ACCUM_STEPS,
                        },
                    },
                    best_path,
                )

            else:
                bad_epochs += 1

            print(
                f"  early-stop counter: "
                f"{bad_epochs}/{EARLY_STOP_PATIENCE}"
            )

            should_stop = (
                bad_epochs >= EARLY_STOP_PATIENCE
            )

        # ---------------------------------------------------------
        # Record history
        # ---------------------------------------------------------

        history.append({
            "epoch":
                epoch + 1,

            "n_frames":
                current_n,

            "rollout_active":
                int(rollout_active),

            "lr":
                lr_now,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_acc,

            "val_loss":
                val_loss,

            **{
                f"val_{k}": v
                for k, v
                in val_metrics.items()
            },
        })

        pd.DataFrame(
            history
        ).to_csv(
            OUT_DIR
            / "training_history.csv",
            index=False,
        )

        scheduler.step()

        if should_stop:
            print(
                f"\nEarly stopping at epoch {epoch+1}: "
                f"no validation AUROC improvement for "
                f"{EARLY_STOP_PATIENCE} rollout-active epochs."
            )
            break

        # Adapted protocol: after each training epoch, randomly
        # sample a new volume resolution from {5,10,15,20,25}.
        next_n = int(
            schedule_rng.choice(
                FRAME_CHOICES
            )
        )

        print(
            f"  next training resolution: "
            f"{next_n} B-scans"
        )

        current_n = next_n

    # -------------------------------------------------------------
    # Load validation-selected checkpoint
    # -------------------------------------------------------------

    print(
        f"\nBest validation epoch: "
        f"{best_epoch}"
    )

    if best_epoch is None or not best_path.exists():
        raise RuntimeError(
            "No rollout-active validation checkpoint was saved."
        )

    print(
        "Best subject-level "
        f"validation AUROC: "
        f"{best_auc:.4f}"
    )

    checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.feature_extractor_spacial.filter_tokens = (
        bool(
            checkpoint[
                "filter_tokens"
            ]
        )
    )

    print(
        "Best checkpoint rollout mode:",
        model
        .feature_extractor_spacial
        .filter_tokens
    )

    # -------------------------------------------------------------
    # TEST EXACTLY ONCE — whole volume
    # -------------------------------------------------------------

    print(
        "\nEvaluating untouched TEST set "
        "for the first and only time."
    )

    (
        test_volume,
        test_subject,
        test_metrics,
    ) = evaluate_test_whole_volume(
        model,
        test_df,
        device,
    )

    test_volume.to_csv(
        OUT_DIR
        / "test_volume_predictions.csv",
        index=False,
    )

    test_subject.to_csv(
        OUT_DIR
        / "test_subject_predictions.csv",
        index=False,
    )

    pd.DataFrame([{
        "Method":
            "VLFATRollout 2024 adapted reimplementation",

        **test_metrics,

        "best_epoch":
            best_epoch,

        "best_val_AUROC":
            best_auc,

        "best_checkpoint_rollout":
            int(
                checkpoint[
                    "filter_tokens"
                ]
            ),
    }]).to_csv(
        OUT_DIR
        / "test_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 92)
    print(
        "FINAL TEST — SUBJECT LEVEL"
    )
    print("=" * 92)

    print(
        f"{'Method':<30}"
        f"{'AUROC':>10}"
        f"{'AUPRC':>10}"
        f"{'Bal. Acc.':>12}"
        f"{'Macro F1':>12}"
        f"{'Accuracy':>12}"
    )

    print("-" * 92)

    print(
        f"{'VLFATRollout 2024':<30}"
        f"{test_metrics['AUROC']:>10.4f}"
        f"{test_metrics['AUPRC']:>10.4f}"
        f"{test_metrics['BalancedAccuracy']:>12.4f}"
        f"{test_metrics['MacroF1']:>12.4f}"
        f"{test_metrics['Accuracy']:>12.4f}"
    )

    print("=" * 92)


if __name__ == "__main__":
    main()
