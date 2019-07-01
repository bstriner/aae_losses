import tensorflow as tf

from aae_losses.trainer import train, train_flags


def main(_):
    train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    train_flags(
        config='conf/gan.json',
        model_dir='../output/gan/v2',
        batch_size=32,
        save_summary_steps=100,
        save_summary_steps_slow=400,
        save_checkpoints_steps=2000
    )
    tf.app.run()
