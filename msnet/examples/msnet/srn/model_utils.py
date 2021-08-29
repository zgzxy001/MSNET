import tensorflow as tf


def cal_feature_dist_s(x_1, x_2, axis=0):
    x_1 = tf.reshape(x_1, [-1])
    x_2 = tf.reshape(x_2, [-1])
    normalize_a = tf.nn.l2_normalize(x_1, axis)
    normalize_b = tf.nn.l2_normalize(x_2, axis)
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis)
    return 1-cos_similarity


def cal_feature_dist(x_1, x_2, axis=0):
    """

    :param x_1: HxWx1024
    :param x_2: HxWx1024
    :param axis: 0
    :return:
    """

    x_shape = tf.shape(x_1)
    x_1 = tf.reshape(x_1, [-1, x_shape[-1]])
    x_2 = tf.reshape(x_2, [-1, x_shape[-1]])
    x_1_2 = [x_1, x_2]
    final_result = tf.map_fn(lambda x: cal_feature_dist_s(x[0], x[1]), x_1_2,  dtype=tf.float32)
    return tf.reduce_sum(final_result) / tf.multiply(tf.cast(x_shape[-3], tf.float32), tf.cast(x_shape[-2], tf.float32))


def custom_hinge_loss(x_pred_2, pos_pred_2, neg_pred_2,
                      x_pred_3, pos_pred_3, neg_pred_3,
                      x_pred_4, pos_pred_4, neg_pred_4,
                      x_pred_5, pos_pred_5, neg_pred_5,
                      M=1):
    dist_2 = cal_feature_dist(x_pred_2, pos_pred_2) \
           - cal_feature_dist(x_pred_2, neg_pred_2) + M
    max_value_2 = tf.maximum(dist_2, 0)

    dist_3 = cal_feature_dist(x_pred_3, pos_pred_3) \
             - cal_feature_dist(x_pred_3, neg_pred_3) + M
    max_value_3 = tf.maximum(dist_3, 0)

    dist_4 = cal_feature_dist(x_pred_4, pos_pred_4) \
             - cal_feature_dist(x_pred_4, neg_pred_4) + M
    max_value_4 = tf.maximum(dist_4, 0)

    dist_5 = cal_feature_dist(x_pred_5, pos_pred_5) \
             - cal_feature_dist(x_pred_5, neg_pred_5) + M
    max_value_5 = tf.maximum(dist_5, 0)

    max_value = max_value_2 + max_value_3 + max_value_4 + max_value_5
    return max_value



