# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.
# Comparator implementation adapted to the common Huang split and validation-selected protocol; see README.md for the source publication.

from pathlib import Path
from datetime import datetime
import json
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "comparative_methods/results/huang_table4_metadata.csv"

SEED = 42

IMG_SIZE = 224

EPOCHS = 11
CANDIDATE_EPOCHS = {3, 5, 7, 9, 11}

BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4

BOOTSTRAP_N = 10000

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

OUT_DIR = (
    ROOT
    / "comparative_methods/dinov2/results"
    / f"table4_seed42_{STAMP}"
)
OUT_DIR.mkdir(parents=True, exist_ok=False)

BEST_CKPT = OUT_DIR / "best_val_eye_auroc.pt"

SUPPORTED = {
    ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff"
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# METADATA
# ============================================================

if not META.exists():
    raise FileNotFoundError(META)

meta = pd.read_csv(META)

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
        f"Missing metadata columns: {sorted(missing)}"
    )


def normalize_label(x):
    if isinstance(x, str):
        z = x.strip().upper()

        if z == "AMD":
            return 0
        if z == "DME":
            return 1

    return int(x)


meta["label"] = meta["label"].map(normalize_label)
meta["split"] = meta["split"].astype(str).str.lower()

train_df = meta[meta["split"] == "train"].copy()
val_df   = meta[meta["split"] == "val"].copy()
test_df  = meta[meta["split"] == "test"].copy()


def split_summary(df):
    return len(df), df["pid"].nunique()


print("=" * 92)
print("DINOv2 ViT-B/14 — HUANG — FROZEN LINEAR PROBE")
print("COMMON TABLE-4 VALIDATION-SELECTION HORIZON")
print("=" * 92)

for name, df in [
    ("TRAIN", train_df),
    ("VAL", val_df),
    ("TEST", test_df),
]:
    nv, ne = split_summary(df)
    print(
        f"{name:<5}: "
        f"{nv:3d} volumes | "
        f"{ne:3d} eyes"
    )

print()

expected = {
    "train": (126, 94),
    "val":   (34, 20),
    "test":  (32, 21),
}

for name, df in [
    ("train", train_df),
    ("val", val_df),
    ("test", test_df),
]:
    got = split_summary(df)

    if got != expected[name]:
        raise RuntimeError(
            f"{name} split mismatch: "
            f"expected {expected[name]}, got {got}"
        )


# ============================================================
# CENTRAL B-SCAN
# ============================================================

def resolve_path(raw):
    p = Path(str(raw))

    if not p.is_absolute():
        p = ROOT / p

    return p


def load_central_bscan(raw_path):
    p = resolve_path(raw_path)

    if not p.exists():
        raise FileNotFoundError(p)

    # Directory containing B-scans.
    if p.is_dir():
        files = sorted(
            x for x in p.rglob("*")
            if (
                x.is_file()
                and x.suffix.lower() in SUPPORTED
            )
        )

        if not files:
            raise RuntimeError(
                f"No B-scans found in {p}"
            )

        central = files[len(files) // 2]

        with Image.open(central) as im:
            return im.convert("L").copy()

    # Multipage image / TIFF or single image.
    with Image.open(p) as im:
        n_frames = getattr(im, "n_frames", 1)

        if n_frames > 1:
            im.seek(n_frames // 2)

        return im.convert("L").copy()


transform = transforms.Compose(
    [
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE),
            antialias=True,
        ),
        transforms.Grayscale(
            num_output_channels=3
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================
# OFFICIAL DINOv2 ViT-B/14
# ============================================================

print()
print("Loading official DINOv2 ViT-B/14...")

encoder = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitb14",
)

encoder = encoder.to(DEVICE)
encoder.eval()

for p in encoder.parameters():
    p.requires_grad = False

total_encoder_params = sum(
    p.numel() for p in encoder.parameters()
)

trainable_encoder_params = sum(
    p.numel()
    for p in encoder.parameters()
    if p.requires_grad
)

print(
    f"Encoder parameters : "
    f"{total_encoder_params:,}"
)
print(
    f"Encoder trainable  : "
    f"{trainable_encoder_params:,}"
)
print("Encoder            : FROZEN + eval")
print("Input              : central B-scan")
print("Input size         : 224 x 224")
print("Grayscale          : replicated to RGB")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.inference_mode()
def extract_features(df, split_name):
    feats = []
    labels = []
    pids = []
    case_ids = []

    print()
    print(
        f"Extracting frozen DINOv2 features: "
        f"{split_name}"
    )

    for i, row in enumerate(
        df.itertuples(index=False),
        start=1,
    ):
        image = load_central_bscan(row.path)
        x = transform(image).unsqueeze(0).to(DEVICE)

        z = encoder(x)

        if isinstance(z, dict):
            if "x_norm_clstoken" in z:
                z = z["x_norm_clstoken"]
            else:
                raise RuntimeError(
                    "Unexpected DINOv2 dictionary output"
                )

        z = z.squeeze(0).float().cpu()

        feats.append(z)
        labels.append(int(row.label))
        pids.append(str(row.pid))
        case_ids.append(str(row.case_id))

        print(
            f"  {split_name} "
            f"{i:03d}/{len(df):03d}",
            end="\r",
        )

    print()

    return {
        "x": torch.stack(feats),
        "y": torch.tensor(
            labels,
            dtype=torch.long,
        ),
        "pid": np.asarray(pids),
        "case_id": np.asarray(case_ids),
    }


# IMPORTANT:
# TEST is deliberately NOT opened here.

train_data = extract_features(
    train_df,
    "train",
)

val_data = extract_features(
    val_df,
    "validation",
)


# ============================================================
# CLASS WEIGHTS
# 2026 ophthalmology benchmark-style logarithmic weighting
# ============================================================

counts = np.bincount(
    train_data["y"].numpy(),
    minlength=2,
).astype(float)

weights = 1.0 / np.log(counts + 1.0)
weights = weights / weights.mean()

class_weights = torch.tensor(
    weights,
    dtype=torch.float32,
    device=DEVICE,
)

print()
print(
    "Train volume counts :",
    counts.astype(int).tolist(),
)
print(
    "Class weights       :",
    weights.tolist(),
)


# ============================================================
# HEAD
# ============================================================

FEATURE_DIM = int(
    train_data["x"].shape[1]
)

head = nn.Linear(
    FEATURE_DIM,
    2,
).to(DEVICE)

print()
print("Downstream head")
print(
    f"  input dim         : "
    f"{FEATURE_DIM}"
)
print(
    f"  trainable params  : "
    f"{sum(p.numel() for p in head.parameters()):,}"
)


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    head.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    TensorDataset(
        train_data["x"],
        train_data["y"],
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
)


# ============================================================
# EYE-LEVEL HELPERS
# ============================================================

@torch.inference_mode()
def volume_probabilities(
    model,
    data,
):
    model.eval()

    x = data["x"].to(DEVICE)

    logits = model(x)

    prob = torch.softmax(
        logits,
        dim=1,
    )[:, 1]

    return prob.cpu().numpy()


def aggregate_eye(
    data,
    prob,
):
    tmp = pd.DataFrame(
        {
            "pid": data["pid"],
            "label": data["y"].numpy(),
            "prob": prob,
        }
    )

    # Label must be unique within an eye.
    nlabels = (
        tmp.groupby("pid")["label"]
        .nunique()
    )

    if (nlabels != 1).any():
        raise RuntimeError(
            "Inconsistent labels within eye."
        )

    eye = (
        tmp.groupby("pid", as_index=False)
        .agg(
            label=("label", "first"),
            prob=("prob", "mean"),
        )
    )

    return eye


def ranking_metrics(eye):
    y = eye["label"].to_numpy()
    p = eye["prob"].to_numpy()

    return {
        "AUROC": float(
            roc_auc_score(y, p)
        ),
        "AUPRC": float(
            average_precision_score(y, p)
        ),
    }


# ============================================================
# TRAIN
# common candidate horizon = {3,5,7,9,11}
# selection = validation eye AUROC
# ============================================================

best_auc = -np.inf
best_epoch = None
history = []

print()
print("=" * 92)
print("TRAINING PROTOCOL")
print("=" * 92)
print(
    "epochs                :",
    EPOCHS,
)
print(
    "candidate epochs      :",
    sorted(CANDIDATE_EPOCHS),
)
print(
    "optimizer             : AdamW"
)
print(
    "lr                    :",
    LR,
)
print(
    "weight decay          :",
    WEIGHT_DECAY,
)
print(
    "backbone              : FROZEN"
)
print(
    "trainable component   : linear head ONLY"
)
print(
    "checkpoint selection  : validation eye AUROC"
)
print(
    "test                  : untouched until selection"
)
print("=" * 92)


for epoch in range(
    1,
    EPOCHS + 1,
):
    head.train()

    running = 0.0
    n_seen = 0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = head(xb)

        loss = criterion(
            logits,
            yb,
        )

        loss.backward()
        optimizer.step()

        running += (
            float(loss.item())
            * len(yb)
        )
        n_seen += len(yb)

    train_loss = running / n_seen

    val_prob = volume_probabilities(
        head,
        val_data,
    )

    val_eye = aggregate_eye(
        val_data,
        val_prob,
    )

    val_rank = ranking_metrics(
        val_eye
    )

    val_auc = val_rank["AUROC"]

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"loss={train_loss:.6f} | "
        f"val_AUROC={val_auc:.4f} | "
        f"val_AUPRC="
        f"{val_rank['AUPRC']:.4f}"
    )

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_eye_AUROC":
                val_rank["AUROC"],
            "val_eye_AUPRC":
                val_rank["AUPRC"],
            "eligible":
                epoch in CANDIDATE_EPOCHS,
        }
    )

    if (
        epoch in CANDIDATE_EPOCHS
        and val_auc > best_auc
    ):
        best_auc = val_auc
        best_epoch = epoch

        torch.save(
            {
                "head":
                    head.state_dict(),
                "epoch":
                    best_epoch,
                "val_eye_AUROC":
                    best_auc,
            },
            BEST_CKPT,
        )

        print(
            "  -> NEW BEST ELIGIBLE "
            "VALIDATION CHECKPOINT"
        )


pd.DataFrame(
    history
).to_csv(
    OUT_DIR / "training_history.csv",
    index=False,
)


# ============================================================
# LOAD BEST VALIDATION CHECKPOINT
# ============================================================

ckpt = torch.load(
    BEST_CKPT,
    map_location=DEVICE,
)

head.load_state_dict(
    ckpt["head"]
)

head.eval()

print()
print("=" * 92)
print(
    "BEST VALIDATION CHECKPOINT"
)
print("=" * 92)
print(
    f"Selected epoch     : "
    f"{best_epoch}"
)
print(
    f"Best val eye AUROC : "
    f"{best_auc:.4f}"
)
print("=" * 92)


# ============================================================
# VALIDATION OPERATING POINT
# existing Table-4 comparator rule:
# max balanced accuracy on validation eyes
# NOT ROC-closest.
# ============================================================

val_prob = volume_probabilities(
    head,
    val_data,
)

val_eye = aggregate_eye(
    val_data,
    val_prob,
)

y_val = val_eye["label"].to_numpy()
p_val = val_eye["prob"].to_numpy()

fpr, tpr, thresholds = roc_curve(
    y_val,
    p_val,
)

baccs = (
    tpr + (1.0 - fpr)
) / 2.0

idx = int(np.argmax(baccs))
threshold = float(thresholds[idx])

val_pred = (
    p_val >= threshold
).astype(int)

val_bacc = float(
    balanced_accuracy_score(
        y_val,
        val_pred,
    )
)

val_eye["pred"] = val_pred

val_eye.to_csv(
    OUT_DIR
    / "best_val_eye_predictions.csv",
    index=False,
)

print()
print("=" * 92)
print(
    "VALIDATION-SELECTION COMPLETE"
)
print("=" * 92)
print(
    f"Threshold          : "
    f"{threshold:.8f}"
)
print(
    f"Validation AUROC   : "
    f"{roc_auc_score(y_val, p_val):.4f}"
)
print(
    f"Validation BAcc    : "
    f"{val_bacc:.4f}"
)
print(
    "Threshold frozen."
)
print(
    "Opening TEST for first and only time."
)
print("=" * 92)


# ============================================================
# TEST — only now
# ============================================================

test_data = extract_features(
    test_df,
    "test",
)

test_prob = volume_probabilities(
    head,
    test_data,
)

test_eye = aggregate_eye(
    test_data,
    test_prob,
)

y = test_eye["label"].to_numpy()
p = test_eye["prob"].to_numpy()

pred = (
    p >= threshold
).astype(int)

test_eye["pred"] = pred

test_eye.to_csv(
    OUT_DIR
    / "test_eye_predictions.csv",
    index=False,
)


def full_metrics(
    y,
    p,
    pred,
):
    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            pred,
            labels=[0, 1],
        )
        .ravel()
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn)
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else np.nan
    )

    return {
        "AUROC":
            float(
                roc_auc_score(y, p)
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
            float(sensitivity),
        "Specificity":
            float(specificity),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


test_metrics = full_metrics(
    y,
    p,
    pred,
)


# ============================================================
# ORDINARY EYE-LEVEL PERCENTILE BOOTSTRAP
# ============================================================

rng = np.random.default_rng(SEED)

boot = {
    k: []
    for k in [
        "AUROC",
        "AUPRC",
        "BalancedAccuracy",
        "MacroF1",
        "Accuracy",
        "Sensitivity",
        "Specificity",
    ]
}

n = len(y)
valid = 0

for _ in range(BOOTSTRAP_N):
    idxs = rng.integers(
        0,
        n,
        size=n,
    )

    yb = y[idxs]
    pb = p[idxs]

    if len(np.unique(yb)) < 2:
        continue

    predb = (
        pb >= threshold
    ).astype(int)

    mb = full_metrics(
        yb,
        pb,
        predb,
    )

    for k in boot:
        boot[k].append(
            mb[k]
        )

    valid += 1


ci = {}

for k, values in boot.items():
    arr = np.asarray(
        values,
        dtype=float,
    )

    ci[k] = [
        float(
            np.nanpercentile(
                arr,
                2.5,
            )
        ),
        float(
            np.nanpercentile(
                arr,
                97.5,
            )
        ),
    ]


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 92)
print(
    "FINAL TEST — HUANG — EYE LEVEL"
)
print(
    "DINOv2 ViT-B/14 FROZEN "
    "LINEAR PROBE"
)
print("=" * 92)

print(
    f"Best epoch         : "
    f"{best_epoch}"
)

print(
    f"Best val AUROC     : "
    f"{best_auc:.4f}"
)

print(
    f"Threshold          : "
    f"{threshold:.8f}"
)

for k in [
    "AUROC",
    "AUPRC",
    "BalancedAccuracy",
    "MacroF1",
    "Accuracy",
    "Sensitivity",
    "Specificity",
]:
    lo, hi = ci[k]

    print(
        f"{k:<18}: "
        f"{test_metrics[k]:.4f} "
        f"[{lo:.4f}, {hi:.4f}]"
    )

print(
    "TN / FP / FN / TP : "
    f"{test_metrics['TN']} / "
    f"{test_metrics['FP']} / "
    f"{test_metrics['FN']} / "
    f"{test_metrics['TP']}"
)

print(
    f"Valid bootstrap    : "
    f"{valid}/{BOOTSTRAP_N}"
)

print()
print(
    f"Saved: "
    f"{OUT_DIR / 'FINAL_RESULT.json'}"
)
print("=" * 92)


result = {
    "method":
        "DINOv2 ViT-B/14 frozen linear probe",
    "dataset":
        "Huang",
    "seed":
        SEED,
    "input":
        "one central B-scan per volume",
    "encoder":
        "facebookresearch/dinov2 dinov2_vitb14",
    "encoder_frozen":
        True,
    "feature_dim":
        FEATURE_DIM,
    "epochs":
        EPOCHS,
    "candidate_epochs":
        sorted(CANDIDATE_EPOCHS),
    "checkpoint_selection":
        "validation eye AUROC",
    "best_epoch":
        int(best_epoch),
    "best_val_AUROC":
        float(best_auc),
    "threshold_selection":
        "validation eye maximum balanced accuracy",
    "selected_threshold":
        float(threshold),
    "test_metrics":
        test_metrics,
    "bootstrap_95ci":
        ci,
    "bootstrap_valid":
        valid,
}

with open(
    OUT_DIR / "FINAL_RESULT.json",
    "w",
) as f:
    json.dump(
        result,
        f,
        indent=2,
    )
