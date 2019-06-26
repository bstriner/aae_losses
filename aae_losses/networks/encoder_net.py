import tensorflow as tf
from tensorflow.contrib import slim


def encoder_net(
        x,
        params
):
    assert x.shape.ndims == 4
    h = tf.reshape(x, (-1, 28*28))
    for i in range(params.encoder_depth):
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='mlp_{}'.format(i)
        )
    mu = slim.fully_connected(
        inputs=h,
        num_outputs=params.latent_dim,
        activation_fn=None,
        scope='mu'
    )
    logsigmasq = slim.fully_connected(
        inputs=h,
        num_outputs=params.latent_dim,
        activation_fn=None,
        scope='logsigmasq'
    )
    return mu, logsigmasq
