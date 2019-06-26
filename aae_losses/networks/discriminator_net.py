import tensorflow as tf
from tensorflow.contrib import slim


def discriminator_net(
        z,
        params
):
    assert z.shape.ndims == 2
    h = z
    for i in range(params.discriminator_depth):
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.discriminator_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='mlp_{}'.format(i)
        )
    h = slim.fully_connected(
        inputs=h,
        num_outputs=1,
        activation_fn=None,
        scope='mlp_output'
    )
    h = tf.squeeze(h, axis=-1)
    return h
