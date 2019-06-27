#!/usr/bin/env bash
mkdir -p /data/VOL3/bstriner/aae_losses/logs

JOB_NAME="aae-gan"
/data/VOL3/bstriner/aae_losses/experiments/sbatch.sh \
    "${JOB_NAME}" \
    /data/VOL3/bstriner/aae_losses/logs/${JOB_NAME}.txt \
    --config="conf/gan.json" \
    --model_dir="../output/${JOB_NAME}" \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --save_summary_steps=200  \
    --save_checkpoints_steps=5000 \
    --max_steps=200000

JOB_NAME="aae-wgan"
/data/VOL3/bstriner/aae_losses/experiments/sbatch.sh \
    "${JOB_NAME}" \
    /data/VOL3/bstriner/aae_losses/logs/${JOB_NAME}.txt \
    --config="conf/wgan.json" \
    --model_dir="../output/${JOB_NAME}" \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --save_summary_steps=200  \
    --save_checkpoints_steps=5000 \
    --max_steps=200000

JOB_NAME="aae-mgan"
/data/VOL3/bstriner/aae_losses/experiments/sbatch.sh \
    "${JOB_NAME}" \
    /data/VOL3/bstriner/aae_losses/logs/${JOB_NAME}.txt \
    --config="conf/mgan.json" \
    --model_dir="../output/${JOB_NAME}" \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --save_summary_steps=200  \
    --save_checkpoints_steps=5000 \
    --max_steps=200000
