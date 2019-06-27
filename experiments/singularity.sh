#!/usr/bin/env bash
singularity exec --nv \
    /data/VOL3/bstriner/singularity/images/10.0-tf-nightly.simg \
    /data/VOL3/bstriner/aae_losses/experiments/run.sh \
    "$@"
