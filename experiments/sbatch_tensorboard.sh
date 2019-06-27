#!/usr/bin/env bash
sbatch --time=48:00:00 \
    --job-name="tensorboard" \
    --partition=cpu \
    --mail-type=ALL \
    --mail-user=bstriner@cs.cmu.edu \
    /data/VOL3/bstriner/aae_losses/experiments/tensorboard.sh
