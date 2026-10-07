#!/usr/bin/env bash
# Sequential multi-seed training queue (Phase 0 re-runs).
#
# Resumable: after an interruption just start the same command again.
# train.py skips finished runs (history file exists) and continues an
# interrupted run from its last completed epoch (resume_<run>.pth).
#
# Usage (GPU server):
#   CUDA_VISIBLE_DEVICES=2 nohup bash run_queue.sh >> logs/queue.log 2>&1 &
#   SEEDS="1" bash run_queue.sh          # only seed 1
#   QUEUE=exp2 ... bash run_queue.sh     # Experiment 2 configs
#
# Seeds are the outer loop: after the first pass every config has one run.

set -u
cd "$(dirname "$0")"
mkdir -p logs

SEEDS="${SEEDS:-1 2 3}"
QUEUE="${QUEUE:-phase0}"
case "$QUEUE" in
    phase0) CONFIGS=(
        "--variant baseline"
        "--variant wavelet_ml --wavelet sym4 --levels 1"
        "--variant wavelet_ml --wavelet haar --levels 2"
        "--variant wavelet_ml --wavelet haar --levels 3"
        "--variant wavelet_ml --wavelet db2 --levels 2"
    ) ;;
    exp2) CONFIGS=(                       # stage-1 detail skip (baseline = phase0 runs)
        "--variant detail_skip_plain"
        "--variant detail_skip_haar"
    ) ;;
    exp3) CONFIGS=(                       # training objective (baseline arch.)
        "--variant baseline --loss dicebce"
        "--variant baseline --loss dicebce --hf_weight 1"
    ) ;;
    r2) CONFIGS=(                         # improved recipe (cosine LR, lesion-balanced crops, aug)
        "--variant baseline --recipe r2"
        "--variant baseline --recipe r2 --hf_weight 1"
    ) ;;
    exp5) CONFIGS=(                       # wavelet-synthesis decoder, recipe r2 (baseline = baseline_r2)
        "--variant waveup_haar --recipe r2"
        "--variant waveup_haar_learn --recipe r2"
        "--variant waveup_rand_learn --recipe r2"
    ) ;;
    exp5c) CONFIGS=(                      # confirmation of the Haar-decoder lead (use SEEDS="4 5 6")
        "--variant baseline --recipe r2"
        "--variant waveup_haar --recipe r2"
        "--variant waveup_rand_learn --recipe r2"
    ) ;;
    *) echo "unknown QUEUE=$QUEUE"; exit 1 ;;
esac

for seed in $SEEDS; do
    for cfg in "${CONFIGS[@]}"; do
        tag="$(echo "$cfg" | sed -e 's/--variant //' -e 's/ --wavelet /_/' -e 's/ --levels /_l/'                                  -e 's/ --recipe /_/' -e 's/ --loss /_/' -e 's/ --hf_weight /_hf/')_s${seed}"
        echo "[$(date '+%F %T')] START $tag"
        if python train.py $cfg --seed "$seed" >> "logs/train_${tag}.log" 2>&1; then
            echo "[$(date '+%F %T')] DONE  $tag  ($(grep -h 'best val Dice\|skipping' "logs/train_${tag}.log" | tail -1))"
        else
            echo "[$(date '+%F %T')] FAILED $tag — see logs/train_${tag}.log"
        fi
    done
done
echo "[$(date '+%F %T')] QUEUE FINISHED"
