import argparse
import atexit
import os
import sys
import time

import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from tqdm import tqdm

from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from model import build_model
from wavelet import WaveletHFLoss
from dataloaders import get_loaders
from splits import TRAIN_PATIENTS, VAL_PATIENTS

# ------------------
# Args
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, default="baseline",
                    choices=["baseline", "wavelet_a", "wavelet_ml", "detail_skip_plain", "detail_skip_haar"],
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
parser.add_argument("--loss", type=str, default="dice", choices=["dice", "dicebce"],
                    help="dice: Dice loss (Phase 0 / Exp. 2); dicebce: Dice + binary cross-entropy")
parser.add_argument("--hf_weight", type=float, default=0.0,
                    help="Weight of the wavelet high-frequency loss (0 = off)")
parser.add_argument("--hf_levels", type=int, default=3,
                    help="Haar decomposition levels used by the high-frequency loss")
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
if args.loss != "dice":
    RUN_NAME += f"_{args.loss}"
if args.hf_weight > 0:
    RUN_NAME += f"_hf{args.hf_weight:g}"
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
# Resume point, rewritten after every epoch and deleted when the run finishes
RESUME_PATH        = os.path.join(OUT_DIR, f"resume_{RUN_NAME}.pth")

# The history file is written only when a run finishes, so it marks a
# completed run — lets an interrupted training queue be restarted safely.
if os.path.exists(HISTORY_PATH) and not args.smoke:
    print(f"[{RUN_NAME}] already finished ({HISTORY_PATH} exists) — skipping")
    sys.exit(0)


# Lock so that several queue workers (one per GPU) never train the same run.
# A lock whose owner process no longer exists (e.g. after a server restart)
# is stale and gets taken over.
LOCK_PATH = os.path.join(OUT_DIR, f"lock_{RUN_NAME}")


def _lock_owner_alive():
    """True if the lock's PID is a train.py process with exactly our arguments."""
    try:
        pid = int(open(LOCK_PATH).read().strip())
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = [c for c in f.read().decode(errors="ignore").split("\0") if c]
    except (OSError, ValueError):
        return False
    script = next((i for i, c in enumerate(cmd) if c.endswith("train.py")), None)
    return script is not None and cmd[script + 1:] == sys.argv[1:]


def _acquire_lock():
    for _ in range(2):
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _lock_owner_alive():
                return False
            os.remove(LOCK_PATH)          # stale lock — take it over
            continue
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        atexit.register(lambda: os.path.exists(LOCK_PATH) and os.remove(LOCK_PATH))
        return True
    return False


if not args.smoke and not _acquire_lock():
    print(f"[{RUN_NAME}] is being trained by another worker — skipping")
    sys.exit(0)

print(f"Variant    : {VARIANT}")
if VARIANT == "wavelet_ml":
    print(f"Wavelet    : {WAVELET}  |  Levels: {LEVELS}")
print(f"Multimodal : {MULTIMODAL}  |  SwinV2: {USE_V2}  |  in_channels: {IN_CHANNELS}")
print(f"Aug        : {AUG}  |  intensity_aug={INTENSITY_AUG}  coeff_aug={COEFF_AUG}")
print(f"Run name   : {RUN_NAME}  |  seed: {args.seed}  |  epochs: {EPOCHS}  patience: {PATIENCE}")
print(f"Loss       : {args.loss}" + (f" + {args.hf_weight:g} x wavelet HF ({args.hf_levels} levels)"
                                     if args.hf_weight > 0 else ""))
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
_seg_loss = DiceLoss(sigmoid=True) if args.loss == "dice" else DiceCELoss(sigmoid=True)
_hf_loss = WaveletHFLoss(levels=args.hf_levels).to(DEVICE) if args.hf_weight > 0 else None


def loss_fn(logits, y):
    loss = _seg_loss(logits, y)
    if _hf_loss is not None:
        loss = loss + args.hf_weight * _hf_loss(logits, y)
    return loss
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
# Resume an interrupted run
# ------------------
start_epoch = 1
if os.path.exists(RESUME_PATH) and not args.smoke:
    ckpt = torch.load(RESUME_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    best_dice, best_epoch = ckpt["best_dice"], ckpt["best_epoch"]
    best_val_loss, early_stop_counter = ckpt["best_val_loss"], ckpt["early_stop_counter"]
    train_loss_history, val_loss_history = ckpt["train_loss"], ckpt["val_loss"]
    train_dice_history, val_dice_history = ckpt["train_dice"], ckpt["val_dice"]
    start_epoch = ckpt["epoch"] + 1
    # Fresh (but seed-determined) augmentation stream for the remaining epochs
    train_loader.dataset.transform.set_random_state(seed=args.seed * 1000 + start_epoch)
    print(f"[{RUN_NAME}] resuming from epoch {start_epoch} "
          f"(best val Dice so far {best_dice:.4f} at epoch {best_epoch})")

# ------------------
# Training loop
# ------------------
for epoch in range(start_epoch, EPOCHS + 1):
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

    # ========= RESUME POINT =========
    if not args.smoke:
        torch.save({
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "best_dice": best_dice, "best_epoch": best_epoch,
            "best_val_loss": best_val_loss, "early_stop_counter": early_stop_counter,
            "train_loss": train_loss_history, "val_loss": val_loss_history,
            "train_dice": train_dice_history, "val_dice": val_dice_history,
        }, RESUME_PATH + ".tmp")
        os.replace(RESUME_PATH + ".tmp", RESUME_PATH)   # atomic: never a half-written file

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

if os.path.exists(RESUME_PATH):
    os.remove(RESUME_PATH)
print(f"[{RUN_NAME}] best val Dice {best_dice:.4f} at epoch {best_epoch} "
      f"of {len(val_dice_history)}")
print(f"Training history saved to {HISTORY_PATH}")
print(f"Loss curves saved to {LOSS_CURVE_PATH}")
print(f"Dice curves saved to {DICE_CURVE_PATH}")
