import tensorflow as tf


def wgan_penalty_loss(
        z_interp,
        y_interp,
        penalty_weight
):
    grads = tf.gradients(tf.reduce_sum(y_interp), z_interp)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=-1))
    penalty = penalty_weight * tf.reduce_sum(tf.square(grad_norm - 1.0))
    tf.losses.add_loss(penalty)
    tf.summary.scalar('grad_penalty', penalty)


def wgan_losses(
        generator_scope,
        discriminator_scope,
        y_real,
        y_fake,
        gan_weight
):
    y_real_mean = tf.reduce_mean(y_real)
    y_fake_mean = tf.reduce_mean(y_fake)
    with tf.name_scope(generator_scope + "/"):
        gen_loss = tf.identity(-y_fake_mean * gan_weight, "gen_loss_calc")
        tf.losses.add_loss(gen_loss)
        tf.summary.scalar('gen_loss', gen_loss)
    with tf.name_scope(discriminator_scope + "/"):
        dis_loss = tf.identity((y_fake_mean - y_real_mean) * gan_weight, "dis_loss_calc")
        tf.losses.add_loss(dis_loss)
        tf.summary.scalar('dis_loss', dis_loss)


def gan_losses(
        generator_scope,
        discriminator_scope,
        y_real,
        y_fake,
        gan_weight
):
    with tf.name_scope(generator_scope + "/"):
        gen_fake = tf.reduce_mean(tf.nn.softplus(-y_fake))
        gen_loss = tf.identity(gen_fake * gan_weight, "gen_loss_calc")
        tf.losses.add_loss(gen_loss)
        tf.summary.scalar('gen_loss', gen_loss)
    with tf.name_scope(discriminator_scope + "/"):
        dis_real = tf.reduce_mean(tf.nn.softplus(-y_real))
        dis_fake = tf.reduce_mean(tf.nn.softplus(y_fake))
        dis_loss = tf.identity((dis_real + dis_fake) * 0.5 * gan_weight, "dis_loss_calc")
        tf.losses.add_loss(dis_loss)
        tf.summary.scalar('dis_loss', dis_loss)


def mgan_losses(
        generator_scope,
        discriminator_scope,
        y_real,
        y_fake,
        gan_weight
):
    with tf.name_scope(generator_scope + "/"):
        gen_fake = tf.reduce_mean(-tf.nn.softplus(y_fake))
        gen_loss = tf.identity(gen_fake * gan_weight, "gen_loss_calc")
        tf.losses.add_loss(gen_loss)
        tf.summary.scalar('gen_loss', gen_loss)
    with tf.name_scope(discriminator_scope + "/"):
        dis_real = tf.reduce_mean(tf.nn.softplus(-y_real))
        dis_fake = tf.reduce_mean(tf.nn.softplus(y_fake))
        dis_loss = tf.identity((dis_real + dis_fake) * 0.5 * gan_weight, "dis_loss_calc")
        tf.losses.add_loss(dis_loss)
        tf.summary.scalar('dis_loss', dis_loss)


def aae_losses(

        generator_scope,
        discriminator_scope,
        y_real,
        y_fake,
        params
):
    if params.loss == 'wgan':
        return wgan_losses(
            generator_scope=generator_scope,
            discriminator_scope=discriminator_scope,
            y_real=y_real,
            y_fake=y_fake,
            gan_weight=params.gan_weight)
    elif params.loss == 'gan':
        return gan_losses(
            generator_scope=generator_scope,
            discriminator_scope=discriminator_scope,
            y_real=y_real,
            y_fake=y_fake,
            gan_weight=params.gan_weight)
    elif params.loss == 'mgan':
        return mgan_losses(
            generator_scope=generator_scope,
            discriminator_scope=discriminator_scope,
            y_real=y_real,
            y_fake=y_fake,
            gan_weight=params.gan_weight)
    else:
        raise NotImplementedError()
