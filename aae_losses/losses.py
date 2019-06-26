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
