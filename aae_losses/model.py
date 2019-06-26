import tensorflow as tf
from tensorflow.contrib.gan.python.train import RunTrainOpsHook

from .losses import wgan_losses, wgan_penalty_loss
from .networks.decoder_net import decoder_net
from .networks.discriminator_net import discriminator_net
from .networks.encoder_net import encoder_net
from .train_op import get_total_loss, make_train_op


def sample(mu, logsigmasq):
    noise = tf.random.normal(shape=tf.shape(mu))
    sigma = tf.exp(logsigmasq / 2.0)
    return mu + (noise * sigma)


def image_grid_summary(name, x):
    assert x.shame.ndims == 4
    img = x[:25]
    img = tf.reshape(img, (5,5,28,28))
    img = tf.transpose(img, (0,2,1,3))
    img = tf.reshape(img, (1, 28*5, 28*5, 1))
    tf.summary.image(name, img)


def model_fn(features, labels, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    x = features['x']
    n = tf.shape(x)[0]
    with tf.variable_scope('autoencoder') as autoencoder_scope:
        with tf.variable_scope('encoder'):
            mu, logsigmasq = encoder_net(
                x=x,
                params=params
            )
            latent = sample(mu=mu, logsigmasq=logsigmasq)
            latent_prior = tf.random.normal(shape=tf.shape(latent))
        with tf.variable_scope('decoder', reuse=False) as decoder_scope:
            x_autoencoded = decoder_net(
                z=latent,
                params=params
            )
        with tf.variable_scope(decoder_scope, reuse=True):
            x_generated = decoder_net(
                z=latent_prior,
                params=params
            )
        assert x_generated.shape.ndims == x.shape.ndims
        reconstruction_loss = tf.reduce_sum(
            tf.square(x - x_autoencoded)
        )
        tf.summary.scalar("reconstruction_loss", reconstruction_loss)
        tf.losses.add_loss(reconstruction_loss)
    with tf.variable_scope('discriminator', reuse=False) as discriminator_scope:
        with tf.name_scope('discriminator/real/'):
            y_real = discriminator_net(
                z=latent_prior,
                params=params
            )
    with tf.variable_scope(discriminator_scope, reuse=True):
        with tf.name_scope('discriminator/fake/'):
            y_fake = discriminator_net(
                z=latent,
                params=params
            )
    with tf.variable_scope(discriminator_scope, reuse=True):
        with tf.name_scope('discriminator/interp/'):
            alpha = tf.random.uniform(shape=[n, 1], dtype=tf.float32)
            z_interp = (alpha * latent) + ((1.0 - alpha) * latent_prior)
            y_interp = discriminator_net(
                z=z_interp,
                params=params
            )
            wgan_penalty_loss(
                z_interp=z_interp,
                y_interp=y_interp,
                penalty_weight=params.penalty_weight
            )
    wgan_losses(
        generator_scope=autoencoder_scope.name,
        discriminator_scope=discriminator_scope.name,
        y_real=y_real,
        y_fake=y_fake,
        gan_weight=params.gan_weight
    )

    dis_total_loss = get_total_loss(discriminator_scope.name)
    gen_total_loss = get_total_loss(autoencoder_scope.name)
    dis_op = make_train_op(
        scope=discriminator_scope.name,
        optimizer=tf.train.AdamOptimizer(3e-4),
        global_step=None,
        total_loss=dis_total_loss,
        clip_gradient_norm=0,
        clip_gradient_value=0,
        summarize_gradients=False
    )
    gen_op = make_train_op(
        scope=autoencoder_scope.name,
        optimizer=tf.train.AdamOptimizer(1e-4),
        global_step=tf.train.get_or_create_global_step(),
        total_loss=gen_total_loss,
        clip_gradient_norm=0,
        clip_gradient_value=0,
        summarize_gradients=False
    )
    dis_train_hook = RunTrainOpsHook(
        train_ops=[dis_op],
        train_steps=params.discriminator_steps
    )
    tf.summary.image("generated", x_generated)
    tf.summary.image("autoencoded", x_autoencoded)
    tf.summary.image("original", x)
    image_grid_summary("generated_grid", x_generated)
    image_grid_summary("autoencoded_grid", x_autoencoded)
    image_grid_summary("original_grid", x)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=gen_total_loss,
        eval_metric_ops={},
        evaluation_hooks=[],
        train_op=gen_op,
        predictions={},
        training_hooks=[dis_train_hook])
