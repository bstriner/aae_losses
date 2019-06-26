import tensorflow as tf
from tensorflow.contrib.slim.python.slim.learning import clip_gradient_norms
from tensorflow.contrib.training.python.training.training import create_train_op


def clip_gradient_values(gradients_to_variables, min_value, max_value):
    clipped_grads_and_vars = []
    for grad, var in gradients_to_variables:
        if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
                tmp = tf.clip_by_value(grad.values, min_value, max_value)
                grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
            else:
                grad = tf.clip_by_value(grad, min_value, max_value)
        clipped_grads_and_vars.append((grad, var))
    return clipped_grads_and_vars


def make_transform_grads_fn(clip_gradient_norm=0, clip_gradient_value=0):
    def transform_grads_fn(grads):
        # Clip gradients.
        if clip_gradient_norm > 0 and clip_gradient_value > 0:
            raise ValueError("Only one of clip_gradient_norm or clip_gradient_value should be set")
        if clip_gradient_norm > 0:
            with tf.name_scope('clip_grads'):
                grads = clip_gradient_norms(grads, clip_gradient_norm)
        if clip_gradient_value > 0:
            with tf.name_scope('clip_grads'):
                grads = clip_gradient_values(grads, -clip_gradient_value, clip_gradient_value)
        return grads

    return transform_grads_fn


def get_total_loss(scope):
    with tf.name_scope(scope + "/"):
        losses = tf.losses.get_losses(scope=scope)
        print("{} losses: {}".format(scope, losses))
        losses += tf.losses.get_regularization_losses(scope=scope)
        total_loss = tf.add_n(losses)
        return total_loss


def make_train_op(
        scope, optimizer, global_step, total_loss,
        clip_gradient_norm=0, clip_gradient_value=0, summarize_gradients=False):
    transform_grads_fn = make_transform_grads_fn(
        clip_gradient_norm=clip_gradient_norm,
        clip_gradient_value=clip_gradient_value)
    variables = tf.trainable_variables(scope=scope)
    updates = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS, scope=scope)
    train_op = create_train_op(
        total_loss=total_loss,
        optimizer=optimizer,
        update_ops=updates,
        variables_to_train=variables,
        transform_grads_fn=transform_grads_fn,
        summarize_gradients=summarize_gradients,
        global_step=global_step)
    return train_op
