import argparse
import os
import sys
import time

import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from tqdm import tqdm

from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from model import build_model
from dataloaders import get_loaders
from splits import TRAIN_PATIENTS, VAL_PATIENTS

# ------------------
# Args
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, default="baseline",
                    choices=["baseline", "wavelet_a", "wavelet_ml"],
                    help="Model variant to train")
parser.add_argument("--wavelet", type=str, default="haar",
                    choices=["haar", "db2", "sym4"],
                    help="Wavelet family — only used with --variant wavelet_ml")
parser.add_argument("--levels", type=int, default=1,
                    choices=[1, 2, 3],
                    help="Decomposition levels — only used with --variant wavelet_ml")
parser.add_argument("--multimodal", action="store_true",
                    help="Use FLAIR + T1 + T2 as input (3 channels)")
parser.add_argument("--use_v2", action="store_true",
                    help="Use SwinUNETR-V2 (residual conv blocks in each Swin stage)")
parser.add_argument("--aug", type=str, default="none",
                    choices=["none", "image", "coeff", "both"],
                    help="Augmentation strategy: none (current baseline), "
                         "image (intensity transforms in data pipeline), "
                         "coeff (frequency-domain perturbations inside wavelet embed), "
                         "both (image + coeff)")
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed (weights init, crops, augmentation); appended to the run name")
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--patience", type=int, default=20,
                    help="Early stopping patience on validation loss")
parser.add_argument("--out_dir", type=str, default="checkpoints",
                    help="Where checkpoints, history and curves are written")
parser.add_argument("--smoke", action="store_true",
                    help="1 epoch, 2 train batches, 2 val cases — checks the pipeline end to end")
args = parser.parse_args()

VARIANT     = args.variant
WAVELET     = args.wavelet
LEVELS      = args.levels
MULTIMODAL  = args.multimodal
USE_V2      = args.use_v2
AUG         = args.aug
IN_CHANNELS = 3 if MULTIMODAL else 1

INTENSITY_AUG = AUG in ("image", "both")
COEFF_AUG     = AUG in ("coeff", "both")

# Run name encodes variant + wavelet family + level + modality + architecture + aug
if VARIANT == "wavelet_ml":
    RUN_NAME = f"wavelet_ml_{WAVELET}_l{LEVELS}"
else:
    RUN_NAME = VARIANT
if MULTIMODAL:
    RUN_NAME += "_mm"
if USE_V2:
    RUN_NAME += "_v2"
if AUG != "none":
    RUN_NAME += f"_aug_{AUG}"
RUN_NAME += f"_s{args.seed}"

# ------------------
# Config
# ------------------
ROOT = "MSLesSeg_Dataset"
DEVICE = "cuda"
EPOCHS = 1 if args.smoke else args.epochs
LR = 1e-4
WEIGHT_DECAY = 1e-5
ROI_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 2

# Early stopping
PATIENCE = args.patience
MIN_DELTA = 1e-4

# Run-aware output paths
OUT_DIR = os.path.join(args.out_dir, "smoke") if args.smoke else args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)
BEST_MODEL_PATH    = os.path.join(OUT_DIR, f"best_{RUN_NAME}.pth")
HISTORY_PATH       = os.path.join(OUT_DIR, f"training_history_{RUN_NAME}.pth")
LOSS_CURVE_PATH    = os.path.join(OUT_DIR, f"loss_curves_{RUN_NAME}.png")
DICE_CURVE_PATH    = os.path.join(OUT_DIR, f"dice_curves_{RUN_NAME}.png")

# The history file is written only when a run finishes, so it marks a
# completed run — lets an interrupted training queue be restarted safely.
if os.path.exists(HISTORY_PATH) and not args.smoke:
    print(f"[{RUN_NAME}] already finished ({HISTORY_PATH} exists) — skipping")
    sys.exit(0)

print(f"Variant    : {VARIANT}")
if VARIANT == "wavelet_ml":
    print(f"Wavelet    : {WAVELET}  |  Levels: {LEVELS}")
print(f"Multimodal : {MULTIMODAL}  |  SwinV2: {USE_V2}  |  in_channels: {IN_CHANNELS}")
print(f"Aug        : {AUG}  |  intensity_aug={INTENSITY_AUG}  coeff_aug={COEFF_AUG}")
print(f"Run name   : {RUN_NAME}  |  seed: {args.seed}  |  epochs: {EPOCHS}  patience: {PATIENCE}")
print(f"Best model will be saved to: {BEST_MODEL_PATH}")

set_determinism(seed=args.seed)

# ------------------
# Data
# ------------------
train_loader, val_loader = get_loaders(
    ROOT, TRAIN_PATIENTS, VAL_PATIENTS,
    multimodal=MULTIMODAL, intensity_aug=INTENSITY_AUG, seed=args.seed,
)
N_TRAIN = 2 if args.smoke else len(train_loader)
N_VAL   = 2 if args.smoke else len(val_loader)

# ------------------
# Model
# ------------------
model = build_model(VARIANT, in_channels=IN_CHANNELS, use_checkpoint=True,
                    wavelet=WAVELET, levels=LEVELS, use_v2=USE_V2,
                    coeff_aug=COEFF_AUG).to(DEVICE)

# ------------------
# Loss / Optim / Metrics
# ------------------
loss_fn = DiceLoss(sigmoid=True)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Dice on binary masks (probability > 0.5). Passing raw probabilities to
# DiceMetric does NOT threshold them for a 1-channel output in MONAI 1.5.
dice_metric = DiceMetric(include_background=False, reduction="mean")

# ------------------
# State
# ------------------
best_dice = 0.0
best_epoch = 0
best_val_loss = float("inf")
early_stop_counter = 0

# ------------------
# History
# ------------------
train_loss_history = []
val_loss_history = []
train_dice_history = []
val_dice_history = []

# ------------------
# Training loop
# ------------------
for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # ========= TRAIN =========
    model.train()
    dice_metric.reset()

    train_loss = 0.0
    steps = 0

    train_iter = iter(train_loader)
    for _ in tqdm(range(N_TRAIN), desc=f"Epoch {epoch} [train]"):
        try:
            batch = next(train_iter)
        except RuntimeError as e:
            # MONAI wraps the original CUDA error, so walk the full chain
            exc, chain = e, ""
            while exc is not None:
                chain += str(exc)
                exc = getattr(exc, "__cause__", None)
            if "INTERNAL ASSERT" in chain:
                print(f"\n[warn] skipping bad batch: {e}")
                continue
            raise

        # batch is list of dicts (multi-patch from RandCropByPosNegLabeld)
        for sample in batch:
            x = sample["image"].to(DEVICE)
            y = sample["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            steps += 1

            preds = (torch.sigmoid(logits) > 0.5).float()
            dice_metric(preds, y)

    train_loss /= steps
    train_dice = dice_metric.aggregate().item()

    train_loss_history.append(train_loss)
    train_dice_history.append(train_dice)

    # ========= VALIDATION =========
    model.eval()
    dice_metric.reset()

    val_loss = 0.0
    val_steps = 0

    with torch.no_grad():
        val_iter = iter(val_loader)
        for _ in tqdm(range(N_VAL), desc=f"Epoch {epoch} [val]"):
            try:
                batch = next(val_iter)
            except RuntimeError as e:
                exc, chain = e, ""
                while exc is not None:
                    chain += str(exc)
                    exc = getattr(exc, "__cause__", None)
                if "INTERNAL ASSERT" in chain:
                    print(f"\n[warn] skipping bad val batch: {e}")
                    continue
                raise

            x = batch["image"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            preds = sliding_window_inference(
                x, ROI_SIZE, SW_BATCH_SIZE, model
            )

            loss = loss_fn(preds, y)
            val_loss += loss.item()
            val_steps += 1

            preds = (torch.sigmoid(preds) > 0.5).float()
            dice_metric(preds, y)

    val_loss = val_loss / val_steps if val_steps > 0 else float("inf")
    val_dice = dice_metric.aggregate().item() if val_steps > 0 else 0.0

    val_loss_history.append(val_loss)
    val_dice_history.append(val_dice)

    # ========= LOG =========
    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Dice: {train_dice:.4f} | "
        f"Val Dice: {val_dice:.4f} | "
        f"{time.time() - epoch_start:.0f}s"
    )

    # ========= CHECKPOINT =========
    if val_dice > best_dice or best_epoch == 0:
        best_dice = val_dice
        best_epoch = epoch
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"New best model saved (Val Dice={best_dice:.4f})")

    # ========= EARLY STOPPING =========
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"EarlyStopping {early_stop_counter}/{PATIENCE}")

    if early_stop_counter >= PATIENCE:
        print("Early stopping triggered")
        break

# ------------------
# Save history
# ------------------
torch.save(
    {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "train_dice": train_dice_history,
        "val_dice": val_dice_history,
        "best_val_dice": best_dice,
        "best_epoch": best_epoch,
        "epochs_trained": len(val_dice_history),
        "args": vars(args),
    },
    HISTORY_PATH,
)

# ------------------
# Plot: Loss
# ------------------
plt.figure(figsize=(8, 5))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Dice Loss")
plt.title(f"Training / Validation Loss [{RUN_NAME}]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(LOSS_CURVE_PATH, dpi=300)
plt.close()

# ------------------
# Plot: Dice
# ------------------
plt.figure(figsize=(8, 5))
plt.plot(train_dice_history, label="Train Dice")
plt.plot(val_dice_history, label="Val Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title(f"Training / Validation Dice [{RUN_NAME}]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(DICE_CURVE_PATH, dpi=300)
plt.close()

print(f"[{RUN_NAME}] best val Dice {best_dice:.4f} at epoch {best_epoch} "
      f"of {len(val_dice_history)}")
print(f"Training history saved to {HISTORY_PATH}")
print(f"Loss curves saved to {LOSS_CURVE_PATH}")
print(f"Dice curves saved to {DICE_CURVE_PATH}")
