#!/usr/bin/env bash
source /data/VOL3/bstriner/pyvenv/bin/activate
python3.6 -m tensorboard.main --logdir=/data/VOL3/bstriner/aae_losses/output > /data/VOL3/bstriner/aae_losses/logs/tb.out 2>&1
