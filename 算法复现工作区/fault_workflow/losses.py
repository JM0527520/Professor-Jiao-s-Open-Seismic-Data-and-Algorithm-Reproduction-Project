"""Checkpoint-metadata loss, algebraically equivalent to pinned unet3.py."""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="fault_workflow")
def cross_entropy_balanced(y_true, y_pred):
    """Batch-balanced BCE on probabilities, including the author's empty guard.

    Author: mean(weighted_logit_BCE * (1-beta)), beta=n_neg/n_total.
    Expanding the two terms avoids division by zero for empty/full targets.
    Original probability clipping and batch-wide counts are retained.
    """
    probability = tf.convert_to_tensor(y_pred)
    labels = tf.cast(y_true, probability.dtype)
    epsilon = tf.cast(tf.keras.backend.epsilon(), probability.dtype)
    probability = tf.clip_by_value(probability, epsilon, 1 - epsilon)
    count_positive = tf.reduce_sum(labels)
    count_negative = tf.reduce_sum(1 - labels)
    beta = tf.math.divide_no_nan(count_negative, count_negative + count_positive)
    cost = -tf.reduce_mean(beta * labels * tf.math.log(probability)
                           + (1 - beta) * (1 - labels) * tf.math.log1p(-probability))
    return tf.where(tf.equal(count_positive, 0), tf.zeros_like(cost), cost)
