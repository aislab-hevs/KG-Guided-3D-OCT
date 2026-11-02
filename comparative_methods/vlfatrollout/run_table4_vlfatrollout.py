# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.
# Comparator implementation adapted to the common Huang split and validation-selected protocol; see README.md for the source publication.

from __future__ import annotations

import os
import json
import random
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)


# ================================================================
# FIXED FAIR DOWNSTREAM PROTOCOL
# ================================================================

SEED = 42

EPOCHS = 11
CANDIDATE_EPOCHS = {3, 5, 7, 9, 11}

PHYSICAL_BATCH = 1
ACCUM_STEPS = 4
EFFECTIVE_BATCH = PHYSICAL_BATCH * ACCUM_STEPS

LR = 3e-4
WEIGHT_DECAY = 1e-4

N_BOOT = 10_000

FIXED_VAL_SLICES = 25

TRAIN_SLICE_CHOICES = [
    5,
    10,
    15,
    20,
    25,
]


# ================================================================
# PATHS
# ================================================================

ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT
    / "comparative_methods"
    / "vlfatrollout"
    / "vlfatrollout_model.py"
)

if not SOURCE.exists():
    raise RuntimeError(
        f"Missing source script: {SOURCE}"
    )

duke_ckpt_env = os.environ.get(
    "VLFAT_DUKE_CKPT"
)

if not duke_ckpt_env:
    raise RuntimeError(
        "VLFAT_DUKE_CKPT environment variable is missing."
    )

DUKE_CKPT = Path(
    duke_ckpt_env
).resolve()

if not DUKE_CKPT.exists():
    raise RuntimeError(
        f"Duke checkpoint does not exist: {DUKE_CKPT}"
    )


STAMP = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

OUT_DIR = (
    ROOT
    / "comparative_methods"
    / "vlfatrollout"
    / "results"
    / f"table4_seed42_{STAMP}"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=False,
)


# ================================================================
# IMPORT CURRENT LOCAL VLFAT IMPLEMENTATION
# ================================================================

spec = importlib.util.spec_from_file_location(
    "vlfat_source",
    SOURCE,
)

m = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(m)


# ================================================================
# REPRODUCIBILITY
# ================================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required."
    )

device = torch.device("cuda")


# ================================================================
# CHECKPOINT EXTRACTION
# ================================================================

def tensor_state_dict(
    obj,
    depth=0,
):
    """
    Find the tensor state_dict inside common checkpoint layouts.
    """

    if depth > 5:
        return None

    if isinstance(obj, dict):

        tensors = {
            k: v
            for k, v in obj.items()
            if isinstance(v, torch.Tensor)
        }

        if len(tensors) >= 10:
            return tensors

        preferred = [
            "model",
            "state_dict",
            "model_state_dict",
            "network",
            "net",
            "weights",
        ]

        for key in preferred:
            if key in obj:
                result = tensor_state_dict(
                    obj[key],
                    depth + 1,
                )
                if result is not None:
                    return result

        for value in obj.values():
            if isinstance(value, dict):
                result = tensor_state_dict(
                    value,
                    depth + 1,
                )
                if result is not None:
                    return result

    return None


def normalize_key(key):
    key = str(key)

    changed = True

    while changed:
        changed = False

        for prefix in (
            "module.",
            "model.",
            "network.",
            "net.",
        ):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True

    return key


def load_official_duke_spatial(
    model,
    checkpoint_path,
):
    print()
    print("=" * 92)
    print("OFFICIAL DUKE SPATIAL INITIALIZATION")
    print("=" * 92)
    print("Checkpoint:", checkpoint_path)

    try:
        raw = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

    except Exception:
        # Official checkpoint from the authors' GitHub release.
        raw = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    source_state = tensor_state_dict(
        raw
    )

    if source_state is None:
        raise RuntimeError(
            "Could not locate tensor state_dict "
            "inside official checkpoint."
        )

    source_state = {
        normalize_key(k): v
        for k, v
        in source_state.items()
    }

    target_state = model.state_dict()

    spatial_prefix = (
        "feature_extractor_spacial."
    )

    target_spatial = {
        k: v
        for k, v
        in target_state.items()
        if k.startswith(spatial_prefix)
    }

    matched = {}

    for target_key, target_value in target_spatial.items():

        short_key = target_key[
            len(spatial_prefix):
        ]

        candidates = [
            target_key,
            short_key,
            "feature_extractor_spacial."
            + short_key,
            "feature_extractor_spatial."
            + short_key,
        ]

        found = None

        for candidate in candidates:
            if candidate not in source_state:
                continue

            source_value = source_state[
                candidate
            ]

            if (
                tuple(source_value.shape)
                ==
                tuple(target_value.shape)
            ):
                found = source_value
                break

        if found is None:

            # Final robust suffix match.
            suffix_hits = [
                value
                for key, value
                in source_state.items()
                if (
                    key.endswith(short_key)
                    and
                    tuple(value.shape)
                    ==
                    tuple(target_value.shape)
                )
            ]

            if len(suffix_hits) == 1:
                found = suffix_hits[0]

        if found is not None:
            matched[target_key] = found


    target_numel = sum(
        v.numel()
        for v
        in target_spatial.values()
    )

    matched_numel = sum(
        target_spatial[k].numel()
        for k
        in matched
    )

    coverage = (
        matched_numel
        / target_numel
        if target_numel
        else 0.0
    )

    print(
        f"Spatial tensors matched : "
        f"{len(matched)}/"
        f"{len(target_spatial)}"
    )

    print(
        f"Spatial parameter cover : "
        f"{100.0 * coverage:.2f}%"
    )

    # CRITICAL SAFETY:
    # never silently continue with random spatial encoder.
    if coverage < 0.80:
        raise RuntimeError(
            "Official Duke spatial weight coverage "
            f"is only {100*coverage:.2f}%. "
            "Training aborted before touching Huang."
        )

    updated = dict(
        target_state
    )

    updated.update(
        matched
    )

    model.load_state_dict(
        updated,
        strict=True,
    )

    print(
        "Official Duke SFE successfully loaded."
    )

    return coverage


# ================================================================
# DATA
# ================================================================

df = pd.read_csv(
    m.META_CSV
)

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

assert len(df) == 192

train_df = (
    df[df.split == "train"]
    .copy()
    .reset_index(drop=True)
)

val_df = (
    df[df.split == "val"]
    .copy()
    .reset_index(drop=True)
)

test_df = (
    df[df.split == "test"]
    .copy()
    .reset_index(drop=True)
)


assert len(train_df) == 126
assert len(val_df) == 34
assert len(test_df) == 32

assert train_df.pid.nunique() == 94
assert val_df.pid.nunique() == 20
assert test_df.pid.nunique() == 21

assert not (
    set(train_df.pid)
    & set(val_df.pid)
)

assert not (
    set(train_df.pid)
    & set(test_df.pid)
)

assert not (
    set(val_df.pid)
    & set(test_df.pid)
)


print("=" * 92)
print(
    "VLFATRollout — HUANG — "
    "EYECLIP-MATCHED ADAPTATION"
)
print("=" * 92)

print()
print("Split:")
print(
    df.groupby("split").agg(
        volumes=("case_id", "size"),
        eyes=("pid", "nunique"),
    )
)


# ================================================================
# DATASETS
# ================================================================

train_ds = m.OCTVolumeDataset(
    train_df,
    n_frames=25,
    augment=True,
)

val_ds = m.OCTVolumeDataset(
    val_df,
    n_frames=FIXED_VAL_SLICES,
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
    batch_size=PHYSICAL_BATCH,
    shuffle=True,
    num_workers=m.NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
    worker_init_fn=m.seed_worker,
    generator=train_generator,
    persistent_workers=False,
)


val_loader = DataLoader(
    val_ds,
    batch_size=PHYSICAL_BATCH,
    shuffle=False,
    num_workers=m.NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
    worker_init_fn=m.seed_worker,
    generator=val_generator,
    persistent_workers=False,
)


# ================================================================
# MODEL
# ================================================================

model = m.build_model().to(
    device
)


coverage = load_official_duke_spatial(
    model,
    DUKE_CKPT,
)


# ------------------------------------------------
# Freeze ONLY slice/spatial representation encoder.
#
# Duke temporal/VFA/head weights are NOT loaded.
# Those layers remain Huang-initialized and trainable.
# ------------------------------------------------

for param in (
    model
    .feature_extractor_spacial
    .parameters()
):
    param.requires_grad = False


# We are adapting a pretrained Rollout SFE.
# Rollout stays active throughout the 7-epoch downstream run.
model.feature_extractor_spacial.filter_tokens = True


total_params = sum(
    p.numel()
    for p
    in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p
    in model.parameters()
    if p.requires_grad
)

frozen_params = (
    total_params
    - trainable_params
)


print()
print("Parameter budget:")

print(
    f"  total       : "
    f"{total_params/1e6:.2f} M"
)

print(
    f"  trainable   : "
    f"{trainable_params/1e6:.2f} M"
)

print(
    f"  frozen SFE  : "
    f"{frozen_params/1e6:.2f} M"
)


# ================================================================
# FAIR DOWNSTREAM OPTIMIZATION
# ================================================================

# Match EyeCLIP downstream supervision:
# no class-weight manipulation.
criterion = torch.nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(
    [
        p
        for p in model.parameters()
        if p.requires_grad
    ],
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


print()
print("Training protocol:")
print("  epochs                :", EPOCHS)
print("  physical batch        :", PHYSICAL_BATCH)
print("  grad accumulation     :", ACCUM_STEPS)
print("  effective batch       :", EFFECTIVE_BATCH)
print("  optimizer             : AdamW")
print("  lr                    :", LR)
print("  weight decay          :", WEIGHT_DECAY)
print("  loss                  : unweighted CE")
print("  Duke weights          : spatial encoder ONLY")
print("  spatial encoder       : FROZEN + eval")
print("  temporal VFA/head     : trainable")
print("  rollout               : active from epoch 1")
print("  checkpoint            : validation eye-AUROC")
print("  validation resolution :", FIXED_VAL_SLICES)
print("  test                  : untouched until selection")
print()


# ================================================================
# TRAIN
# ================================================================

best_val_auroc = -float(
    "inf"
)

best_epoch = None

BEST_PATH = (
    OUT_DIR
    / "best_model.pt"
)

history = []

schedule_rng = (
    np.random.RandomState(
        SEED
    )
)

current_n = 25


for epoch in range(
    EPOCHS
):

    train_ds.set_n_frames(
        current_n
    )

    # Keep validation operating point fixed
    # across epochs.
    val_ds.set_n_frames(
        FIXED_VAL_SLICES
    )

    model.train()

    # Frozen encoder should behave like
    # a genuine frozen representation model:
    # no stochastic train-mode behaviour.
    model.feature_extractor_spacial.eval()

    model.feature_extractor_spacial.filter_tokens = True

    optimizer.zero_grad(
        set_to_none=True
    )

    running_loss = 0.0
    running_correct = 0
    running_n = 0

    n_batches = len(
        train_loader
    )


    for step, batch in enumerate(
        train_loader
    ):

        x = batch["x"].to(
            device,
            non_blocking=True,
        )

        y = batch["label"].to(
            device,
            non_blocking=True,
        )

        logits = model(
            x
        )

        loss = criterion(
            logits,
            y,
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Non-finite training loss."
            )

        group_start = (
            step // ACCUM_STEPS
        ) * ACCUM_STEPS

        group_end = min(
            group_start
            + ACCUM_STEPS,
            n_batches,
        )

        group_size = (
            group_end
            - group_start
        )

        (
            loss
            / group_size
        ).backward()

        running_loss += (
            float(loss.item())
            * len(y)
        )

        pred = logits.argmax(
            dim=1
        )

        running_correct += int(
            (pred == y)
            .sum()
            .item()
        )

        running_n += len(
            y
        )

        if (
            step + 1
            == group_end
        ):
            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )


    train_loss = (
        running_loss
        / running_n
    )

    train_acc = (
        running_correct
        / running_n
    )


    # ------------------------------------------------------------
    # VALIDATION — NO TEST
    # ------------------------------------------------------------

    model.feature_extractor_spacial.filter_tokens = True

    (
        _,
        val_subject,
        val_metrics,
        val_loss,
    ) = m.evaluate_sampled(
        model,
        val_loader,
        device,
        criterion,
    )


    val_auroc = float(
        val_metrics[
            "AUROC"
        ]
    )

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"N={current_n:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"train_acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_AUROC={val_auroc:.4f} | "
        f"val_AUPRC="
        f"{val_metrics['AUPRC']:.4f}"
    )


    history.append(
        {
            "epoch":
                epoch + 1,

            "train_slices":
                current_n,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_acc,

            "val_loss":
                float(val_loss),

            "val_AUROC":
                val_auroc,

            "val_AUPRC":
                float(
                    val_metrics[
                        "AUPRC"
                    ]
                ),
        }
    )


    if (
        (epoch + 1) in CANDIDATE_EPOCHS
        and val_auroc > best_val_auroc
    ):
        best_val_auroc = (
            val_auroc
        )

        best_epoch = (
            epoch + 1
        )

        torch.save(
            {
                "model":
                    model.state_dict(),

                "epoch":
                    best_epoch,

                "val_auroc":
                    best_val_auroc,

                "official_duke_checkpoint":
                    str(
                        DUKE_CKPT
                    ),

                "duke_spatial_coverage":
                    coverage,

                "spatial_encoder_frozen":
                    True,

                "duke_temporal_weights_loaded":
                    False,

                "config":
                    {
                        "epochs":
                            EPOCHS,

                        "effective_batch":
                            EFFECTIVE_BATCH,

                        "lr":
                            LR,

                        "weight_decay":
                            WEIGHT_DECAY,

                        "seed":
                            SEED,
                    },
            },
            BEST_PATH,
        )

        print(
            "  -> NEW BEST "
            "VALIDATION AUROC CHECKPOINT"
        )


    current_n = int(
        schedule_rng.choice(
            TRAIN_SLICE_CHOICES
        )
    )


pd.DataFrame(
    history
).to_csv(
    OUT_DIR
    / "training_history.csv",
    index=False,
)


print()
print(
    "Best validation epoch:",
    best_epoch,
)

print(
    "Best validation AUROC:",
    f"{best_val_auroc:.4f}",
)


# ================================================================
# LOAD BEST VALIDATION CHECKPOINT
# ================================================================

checkpoint = torch.load(
    BEST_PATH,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model"]
)

model.feature_extractor_spacial.filter_tokens = True

model.eval()


# ================================================================
# WHOLE-VOLUME INFERENCE
# ================================================================

@torch.inference_mode()
def infer_whole_volume(
    data_df,
    name,
):

    rows = []

    print()
    print(
        f"Whole-volume {name} inference..."
    )

    for index, row in enumerate(
        data_df.itertuples(
            index=False
        ),
        start=1,
    ):

        files = m.list_slice_files(
            row.path
        )

        logits = m.forward_whole_volume(
            model,
            files,
            device,
            chunk_size=m.TEST_SLICE_CHUNK,
        )

        prob = (
            torch.softmax(
                logits,
                dim=1,
            )[0, 1]
            .detach()
            .cpu()
            .item()
        )

        rows.append(
            {
                "case_id":
                    str(
                        row.case_id
                    ),

                "pid":
                    str(
                        row.pid
                    ),

                "label":
                    int(
                        row.label
                    ),

                "n_slices":
                    len(
                        files
                    ),

                "prob_DME":
                    float(
                        prob
                    ),
            }
        )

        print(
            f"  {name.lower()} "
            f"{index:02d}/"
            f"{len(data_df)} | "
            f"slices={len(files)}"
        )


    volume_df = pd.DataFrame(
        rows
    )

    subject_df = m.subject_table(
        volume_df
    )

    return (
        volume_df,
        subject_df,
    )


# ================================================================
# VALIDATION-ONLY THRESHOLD
# ================================================================

def select_threshold(
    subject_df,
):

    y = (
        subject_df[
            "label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    p = (
        subject_df[
            "prob_DME"
        ]
        .to_numpy(
            dtype=float
        )
    )

    unique = np.unique(
        p
    )

    if len(unique) < 2:
        raise RuntimeError(
            "Validation probabilities collapsed."
        )

    candidates = np.concatenate(
        [
            [0.0],
            (
                unique[:-1]
                + unique[1:]
            ) / 2.0,
            [1.0],
        ]
    )


    best = None

    for threshold in candidates:

        pred = (
            p >= threshold
        ).astype(int)

        bacc = (
            balanced_accuracy_score(
                y,
                pred,
            )
        )

        mf1 = (
            f1_score(
                y,
                pred,
                average="macro",
            )
        )

        key = (
            float(bacc),
            float(mf1),
            -abs(
                float(threshold)
                - 0.5
            ),
        )

        if (
            best is None
            or key > best[0]
        ):
            best = (
                key,
                float(
                    threshold
                ),
            )


    return best[1]


# ================================================================
# FINAL METRICS
# ================================================================

def metrics_at_threshold(
    subject_df,
    threshold,
):

    y = (
        subject_df[
            "label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    p = (
        subject_df[
            "prob_DME"
        ]
        .to_numpy(
            dtype=float
        )
    )

    pred = (
        p >= threshold
    ).astype(int)


    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            pred,
            labels=[0, 1],
        ).ravel()
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
            float(
                tp
                / (tp + fn)
            ),

        "Specificity":
            float(
                tn
                / (tn + fp)
            ),

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp),
    }


# ================================================================
# WHOLE-VOLUME VALIDATION FIRST
# ================================================================

val_volume, val_subject = (
    infer_whole_volume(
        val_df,
        "VALIDATION",
    )
)


threshold = select_threshold(
    val_subject
)


val_final = (
    metrics_at_threshold(
        val_subject,
        threshold,
    )
)


print()
print("=" * 92)
print(
    "VALIDATION — VALIDATION-SELECTED OPERATING POINT"
)
print("=" * 92)

print(
    f"Threshold : "
    f"{threshold:.8f}"
)

print(
    f"AUROC     : "
    f"{val_final['AUROC']:.4f}"
)

print(
    f"BAcc      : "
    f"{val_final['BalancedAccuracy']:.4f}"
)


# ================================================================
# TEST ONCE
# ================================================================

print()
print(
    "Validation selection complete. "
    "Opening TEST for the first and only time."
)


test_volume, test_subject = (
    infer_whole_volume(
        test_df,
        "TEST",
    )
)


test_final = (
    metrics_at_threshold(
        test_subject,
        threshold,
    )
)


# ================================================================
# ORDINARY EYE-LEVEL BOOTSTRAP
# ================================================================

y = (
    test_subject[
        "label"
    ]
    .to_numpy(
        dtype=np.int64
    )
)

p = (
    test_subject[
        "prob_DME"
    ]
    .to_numpy(
        dtype=float
    )
)

pred = (
    p >= threshold
).astype(int)


rng = (
    np.random.default_rng(
        SEED
    )
)


boot = {
    "AUROC": [],
    "AUPRC": [],
    "BalancedAccuracy": [],
    "MacroF1": [],
    "Accuracy": [],
    "Sensitivity": [],
    "Specificity": [],
}


for _ in range(
    N_BOOT
):

    idx = rng.integers(
        0,
        len(y),
        size=len(y),
    )

    yy = y[idx]

    if (
        np.unique(yy).size
        < 2
    ):
        continue

    score = p[idx]
    pr = pred[idx]

    tn, fp, fn, tp = (
        confusion_matrix(
            yy,
            pr,
            labels=[0, 1],
        ).ravel()
    )

    boot[
        "AUROC"
    ].append(
        roc_auc_score(
            yy,
            score,
        )
    )

    boot[
        "AUPRC"
    ].append(
        average_precision_score(
            yy,
            score,
        )
    )

    boot[
        "BalancedAccuracy"
    ].append(
        balanced_accuracy_score(
            yy,
            pr,
        )
    )

    boot[
        "MacroF1"
    ].append(
        f1_score(
            yy,
            pr,
            average="macro",
        )
    )

    boot[
        "Accuracy"
    ].append(
        accuracy_score(
            yy,
            pr,
        )
    )

    boot[
        "Sensitivity"
    ].append(
        tp / (tp + fn)
    )

    boot[
        "Specificity"
    ].append(
        tn / (tn + fp)
    )


ci = {}

for name, values in boot.items():

    lo, hi = np.percentile(
        np.asarray(
            values
        ),
        [
            2.5,
            97.5,
        ],
    )

    ci[name] = [
        float(lo),
        float(hi),
    ]


# ================================================================
# SAVE
# ================================================================

val_volume.to_csv(
    OUT_DIR
    / "val_volume_predictions.csv",
    index=False,
)

val_subject.to_csv(
    OUT_DIR
    / "val_eye_predictions.csv",
    index=False,
)

test_volume.to_csv(
    OUT_DIR
    / "test_volume_predictions.csv",
    index=False,
)

test_subject[
    "prediction"
] = (
    test_subject[
        "prob_DME"
    ]
    >= threshold
).astype(int)

test_subject.to_csv(
    OUT_DIR
    / "test_eye_predictions.csv",
    index=False,
)


result = {
    "method":
        "VLFATRollout EyeCLIP-matched",

    "best_epoch":
        best_epoch,

    "best_val_AUROC":
        best_val_auroc,

    "selected_threshold":
        threshold,

    "threshold_source":
        "validation eyes only",

    "duke_checkpoint":
        str(
            DUKE_CKPT
        ),

    "duke_spatial_coverage":
        coverage,

    "duke_weights_loaded":
        "spatial encoder only",

    "spatial_encoder":
        "frozen",

    "temporal_vfa_head":
        "Huang-trained",

    "test":
        test_final,

    "CI95":
        ci,

    "bootstrap_valid":
        len(
            boot["AUROC"]
        ),

    "bootstrap_requested":
        N_BOOT,
}


RESULT_PATH = (
    OUT_DIR
    / "FINAL_RESULT.json"
)

RESULT_PATH.write_text(
    json.dumps(
        result,
        indent=2,
    )
)


# ================================================================
# FINAL PRINT
# ================================================================

print()
print("=" * 92)
print(
    "FINAL TEST — EYE LEVEL"
)
print(
    "VLFATRollout — "
    "OFFICIAL DUKE SFE / EYECLIP-MATCHED"
)
print("=" * 92)

print(
    f"Best epoch         : "
    f"{best_epoch}"
)

print(
    f"Best val AUROC     : "
    f"{best_val_auroc:.4f}"
)

print(
    f"Threshold          : "
    f"{threshold:.8f}"
)


for name in [
    "AUROC",
    "AUPRC",
    "BalancedAccuracy",
    "MacroF1",
    "Accuracy",
    "Sensitivity",
    "Specificity",
]:

    value = (
        test_final[
            name
        ]
    )

    lo, hi = (
        ci[name]
    )

    print(
        f"{name:18s}: "
        f"{value:.4f} "
        f"[{lo:.4f}, {hi:.4f}]"
    )


print(
    "TN / FP / FN / TP : "
    f"{test_final['TN']} / "
    f"{test_final['FP']} / "
    f"{test_final['FN']} / "
    f"{test_final['TP']}"
)

print(
    f"Valid bootstrap    : "
    f"{len(boot['AUROC'])}/"
    f"{N_BOOT}"
)

print()
print(
    "Saved:",
    RESULT_PATH,
)

print("=" * 92)

