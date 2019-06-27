#!/usr/bin/env bash
export PYTHONPATH=/data/VOL3/bstriner/aae_loses:$PYTHONPATH
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
cd /data/VOL3/bstriner/aae_loses/experiments
OUTPUT_FILE="$1"
shift
python3 train.py "$@" > "${OUTPUT_FILE}" 2>&1
