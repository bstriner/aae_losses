import tensorflow as tf
from tensorflow.contrib import slim


def decoder_net(
        z,
        params
):
    assert z.shape.ndims == 2
    h = z
    for i in range(params.encoder_depth):
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.decoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='mlp_{}'.format(i)
        )
    h = slim.fully_connected(
        inputs=h,
        num_outputs=28*28,
        activation_fn=tf.tanh,
        scope='mlp_output'
    )
    h = tf.reshape(h, (-1, 28, 28, 1))
    return h
