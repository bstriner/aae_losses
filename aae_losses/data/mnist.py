import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.datasets import mnist


def load_dataset():
    train, test = mnist.load_data()
    return train, test


def data_preproc(x):
    x = (x.astype(np.float32) * 2. / 255.) - 1.
    x = np.expand_dims(x, axis=-1)
    print("X Range: {} to {}".format(np.min(x), np.max(x)))
    print("X shape: {}".format(x.shape))
    return x


def make_input_fn(data, num_epochs=None):
    x, y = data
    x = data_preproc(x)
    fn = tf.estimator.inputs.numpy_input_fn(
        {'x': x}, y,
        batch_size=tf.flags.FLAGS.batch_size,
        num_epochs=num_epochs,
        shuffle=True
    )
    return fn


def make_input_fns():
    train, test = load_dataset()
    return make_input_fn(train), make_input_fn(test)


if __name__ == '__main__':
    train, test = load_dataset()
    x = data_preproc(train[0])
    print(x.shape)
    print(np.min(x))
    print(np.max(x))
