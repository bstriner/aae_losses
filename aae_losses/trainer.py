import os

import tensorflow as tf
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.training import train_and_evaluate

from .data.mnist import make_input_fns
from .hparams import get_hparams
from .model import model_fn

DEFAULT_HPARAMS = {
    'loss': 'wgan',
    "gen_lr": 3e-5,
    "dis_lr": 3e-4,
    "encoder_dim": 320,
    "encoder_depth": 3,
    "decoder_dim": 320,
    "decoder_depth": 3,
    "discriminator_dim": 512,
    "discriminator_depth": 5,
    "latent_dim": 128,
    "penalty_weight": 100.0,
    "reconstruction_weight": 1e-1,
    "gan_weight": 10.0,
    "discriminator_steps": 10,
    "stochastic": False,
    "noise_dim": 128
}
FLAGS = tf.app.flags.FLAGS


def train():
    model_dir = FLAGS.model_dir
    os.makedirs(model_dir, exist_ok=True)

    hparams = get_hparams(
        model_dir,
        DEFAULT_HPARAMS,
        hparams_file=FLAGS.config,
        hparams_str=FLAGS.hparams,
        validate=True
    )
    run_config = RunConfig(
        model_dir=model_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        save_summary_steps=FLAGS.save_summary_steps)

    train_input_fn, eval_input_fn = make_input_fns()

    # Model
    estimator = Estimator(
        model_fn=model_fn,
        config=run_config,
        params=hparams)
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=FLAGS.max_steps)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=FLAGS.eval_steps,
        throttle_secs=0)
    train_and_evaluate(
        eval_spec=eval_spec,
        train_spec=train_spec,
        estimator=estimator
    )


def train_flags(
        config,
        model_dir,
        batch_size,
        save_checkpoints_steps=2000,
        save_summary_steps=100,
        save_summary_steps_slow=400,
        max_steps=200000):
    tf.app.flags.DEFINE_string('config', config, 'config file')
    tf.app.flags.DEFINE_string('hparams', '', 'hparam keys/values')
    tf.app.flags.DEFINE_string('model_dir', model_dir, 'Model directory')
    tf.app.flags.DEFINE_integer('batch_size', batch_size, 'Batch size')
    tf.app.flags.DEFINE_integer('save_checkpoints_steps', save_checkpoints_steps, 'save_checkpoints_steps')
    tf.app.flags.DEFINE_integer('max_steps', max_steps, 'max_steps')
    tf.app.flags.DEFINE_integer('min_steps', 40000, 'Batch size')
    tf.app.flags.DEFINE_integer('max_steps_without_decrease', 8000, 'Batch size')
    tf.app.flags.DEFINE_integer('eval_steps', 200, 'max_steps')
    tf.app.flags.DEFINE_integer('save_summary_steps', save_summary_steps, 'max_steps')
    tf.app.flags.DEFINE_integer('save_summary_steps_slow', save_summary_steps_slow, 'max_steps')
    tf.app.flags.DEFINE_integer('shuffle_buffer_size', 1000, 'shuffle_buffer_size')
    tf.app.flags.DEFINE_integer('prefetch_buffer_size', 100, 'prefetch_buffer_size')
    tf.app.flags.DEFINE_integer('num_parallel_calls', 4, 'num_parallel_calls')
