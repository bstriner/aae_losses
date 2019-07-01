#!/usr/bin/env bash
mkdir -p /data/VOL3/bstriner/aae_losses/logs
/data/VOL3/bstriner/aae_losses/experiments/sbatch_tensorboard.sh

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

JOB_NAME="aae-gan-weight-100"
/data/VOL3/bstriner/aae_losses/experiments/sbatch.sh \
    "${JOB_NAME}" \
    /data/VOL3/bstriner/aae_losses/logs/${JOB_NAME}.txt \
    --config="conf/gan.json" \
    --model_dir="../output/${JOB_NAME}" \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --save_summary_steps=200  \
    --save_checkpoints_steps=5000 \
    --hparams="gan_weight=100" \
    --max_steps=200000


JOB_NAME="aae-gan-weight-1000"
/data/VOL3/bstriner/aae_losses/experiments/sbatch.sh \
    "${JOB_NAME}" \
    /data/VOL3/bstriner/aae_losses/logs/${JOB_NAME}.txt \
    --config="conf/gan.json" \
    --model_dir="../output/${JOB_NAME}" \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --save_summary_steps=200  \
    --save_checkpoints_steps=5000 \
    --hparams="gan_weight=1000" \
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

JOB_NAME="aae-wgan-weight-100"
/data/VOL3/bstriner/aae_losses/experiments/sbatch.sh \
    "${JOB_NAME}" \
    /data/VOL3/bstriner/aae_losses/logs/${JOB_NAME}.txt \
    --config="conf/wgan.json" \
    --model_dir="../output/${JOB_NAME}" \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --save_summary_steps=200  \
    --save_checkpoints_steps=5000 \
    --hparams="gan_weight=100" \
    --max_steps=200000

JOB_NAME="aae-wgan-weight-1000"
/data/VOL3/bstriner/aae_losses/experiments/sbatch.sh \
    "${JOB_NAME}" \
    /data/VOL3/bstriner/aae_losses/logs/${JOB_NAME}.txt \
    --config="conf/wgan.json" \
    --model_dir="../output/${JOB_NAME}" \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --save_summary_steps=200  \
    --save_checkpoints_steps=5000 \
    --hparams="gan_weight=1000" \
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

JOB_NAME="aae-mgan-weight-100"
/data/VOL3/bstriner/aae_losses/experiments/sbatch.sh \
    "${JOB_NAME}" \
    /data/VOL3/bstriner/aae_losses/logs/${JOB_NAME}.txt \
    --config="conf/mgan.json" \
    --model_dir="../output/${JOB_NAME}" \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --save_summary_steps=200  \
    --save_checkpoints_steps=5000 \
    --hparams="gan_weight=100" \
    --max_steps=200000


JOB_NAME="aae-gan-stoch"
/data/VOL3/bstriner/aae_losses/experiments/sbatch.sh \
    "${JOB_NAME}" \
    /data/VOL3/bstriner/aae_losses/logs/${JOB_NAME}.txt \
    --config="conf/gan.json" \
    --model_dir="../output/${JOB_NAME}" \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --save_summary_steps=200  \
    --save_checkpoints_steps=5000 \
    --hparams="stochastic=true" \
    --max_steps=200000

JOB_NAME="aae-gan-stoch-100"
/data/VOL3/bstriner/aae_losses/experiments/sbatch.sh \
    "${JOB_NAME}" \
    /data/VOL3/bstriner/aae_losses/logs/${JOB_NAME}.txt \
    --config="conf/gan.json" \
    --model_dir="../output/${JOB_NAME}" \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --save_summary_steps=200  \
    --save_checkpoints_steps=5000 \
    --hparams="stochastic=true,gan_weight=100" \
    --max_steps=200000

JOB_NAME="aae-gan-stoch-1000"
/data/VOL3/bstriner/aae_losses/experiments/sbatch.sh \
    "${JOB_NAME}" \
    /data/VOL3/bstriner/aae_losses/logs/${JOB_NAME}.txt \
    --config="conf/gan.json" \
    --model_dir="../output/${JOB_NAME}" \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --save_summary_steps=200  \
    --save_checkpoints_steps=5000 \
    --hparams="stochastic=true,gan_weight=1000" \
    --max_steps=200000

JOB_NAME="aae-wgan-stoch"
/data/VOL3/bstriner/aae_losses/experiments/sbatch.sh \
    "${JOB_NAME}" \
    /data/VOL3/bstriner/aae_losses/logs/${JOB_NAME}.txt \
    --config="conf/wgan.json" \
    --model_dir="../output/${JOB_NAME}" \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --save_summary_steps=200  \
    --save_checkpoints_steps=5000 \
    --hparams="stochastic=true" \
    --max_steps=200000

JOB_NAME="aae-wgan-stoch-100"
/data/VOL3/bstriner/aae_losses/experiments/sbatch.sh \
    "${JOB_NAME}" \
    /data/VOL3/bstriner/aae_losses/logs/${JOB_NAME}.txt \
    --config="conf/wgan.json" \
    --model_dir="../output/${JOB_NAME}" \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --save_summary_steps=200  \
    --save_checkpoints_steps=5000 \
    --hparams="stochastic=true,gan_weight=100" \
    --max_steps=200000

JOB_NAME="aae-wgan-stoch-1000"
/data/VOL3/bstriner/aae_losses/experiments/sbatch.sh \
    "${JOB_NAME}" \
    /data/VOL3/bstriner/aae_losses/logs/${JOB_NAME}.txt \
    --config="conf/wgan.json" \
    --model_dir="../output/${JOB_NAME}" \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --save_summary_steps=200  \
    --save_checkpoints_steps=5000 \
    --hparams="stochastic=true,gan_weight=1000" \
    --max_steps=200000

JOB_NAME="aae-wgan-stoch-1000-v2"
/data/VOL3/bstriner/aae_losses/experiments/sbatch.sh \
    "${JOB_NAME}" \
    /data/VOL3/bstriner/aae_losses/logs/${JOB_NAME}.txt \
    --config="conf/wgan.json" \
    --model_dir="../output/${JOB_NAME}" \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --save_summary_steps=200  \
    --save_checkpoints_steps=5000 \
    --hparams="stochastic=true,gan_weight=1000,penalty_weight=1000" \
    --max_steps=200000

JOB_NAME="aae-wgan-stoch-1000-v3"
/data/VOL3/bstriner/aae_losses/experiments/sbatch.sh \
    "${JOB_NAME}" \
    /data/VOL3/bstriner/aae_losses/logs/${JOB_NAME}.txt \
    --config="conf/wgan.json" \
    --model_dir="../output/${JOB_NAME}" \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --save_summary_steps=200  \
    --save_checkpoints_steps=5000 \
    --hparams="stochastic=true,gan_weight=100,penalty_weight=100,reconstruction_weight=0.001,noise_dim=256" \
    --max_steps=200000

JOB_NAME="aae-mgan-stoch"
/data/VOL3/bstriner/aae_losses/experiments/sbatch.sh \
    "${JOB_NAME}" \
    /data/VOL3/bstriner/aae_losses/logs/${JOB_NAME}.txt \
    --config="conf/mgan.json" \
    --model_dir="../output/${JOB_NAME}" \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --save_summary_steps=200  \
    --save_checkpoints_steps=5000 \
    --hparams="stochastic=true" \
    --max_steps=200000
