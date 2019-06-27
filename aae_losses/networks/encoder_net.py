import tensorflow as tf
from tensorflow.contrib import slim


def sample(mu, logsigmasq):
    noise = tf.random.normal(shape=tf.shape(mu))
    sigma = tf.exp(logsigmasq / 2.0)
    return mu + (noise * sigma)


def encoder_net(
        x,
        params
):
    assert x.shape.ndims == 4
    h = tf.reshape(x, (-1, 28 * 28))
    if params.stochastic:
        noise = tf.random.normal(shape=(tf.shape(h)[0], params.noise_dim))
        h = tf.concat([h, noise], axis=-1)
    for i in range(params.encoder_depth):
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='mlp_{}'.format(i)
        )
    if params.stochastic:
        latent = slim.fully_connected(
            inputs=h,
            num_outputs=params.latent_dim,
            activation_fn=None,
            scope='latent'
        )
    else:
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
        latent = sample(mu=mu, logsigmasq=logsigmasq)
    return latent
