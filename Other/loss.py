import tensorflow as tf

# the embedding features loss

# the keypoints loss

# the regression loss


def neg_loss(preds, gt):
    _, h, w, c = preds.get_shape().as_list()
    num_pos = tf.reduce_sum(tf.cast(gt==1, tf.float32))

    neg_weights = tf.pow(1 - gt, 4)
    pos_weights = tf.ones_like(preds, dtype=tf.float32)
    weights = tf.where(gt==1, pos_weights, neg_weights)
    inverse_preds = tf.where(gt==1, preds, 1-preds)

    loss = tf.log(inverse_preds+0.0001) * tf.pow(1-inverse_preds, 2) * weights
    loss = tf.reduce_mean(loss)
    loss = -loss/(num_pos+1)
    return loss


def ae_loss(tl_tag, br_tag, tl_tag_mark, br_tag_mark, tag_mask):
    num = tf.reduce_sum(tag_mask, axis=1, keep_dims=True)
    # get the group points
    tl_tag_mark = tf.cast(tl_tag_mark, tf.int64)
    br_tag_mark = tf.cast(br_tag_mark, tf.int64)
    tl_tag = tf.gather(tl_tag, tl_tag_mark, axis=1, batch_dims=1)
    br_tag = tf.gather(br_tag, br_tag_mark, axis=1, batch_dims=1)

    tag_mean = (tl_tag + br_tag)/2
    pull = tf.squeeze(tl_tag - tag_mean) + tf.square(br_tag - tag_mean)
    pull = pull * tag_mask / (num+1e-4)
    pull = tf.reduce_mean(pull)

    tag_mean1 = tf.expand_dims(tag_mean, 1)
    tag_mean2 = tf.expand_dims(tag_mean, 2)
    tag_mask = tf.expand_dims(tag_mask, 1) * tf.expand_dims(tag_mask, 2)

    num = tf.expand_dims(num, 2)
    dist = 1 - tf.abs(tag_mean1 -tag_mean2)
    num2 = (num - 1) * num
    dist = tf.maximum(tf.zeros_like(dist), dist)
    dist = dist / (num2 + 1e-4)
    dist = dist * tag_mask
    push = tf.reduce_mean(dist)

    return push + pull


def regr_loss(regr, gt_regr, mask):
    mask = tf.expand_dims(mask, -1)
    regr_loss = tf.square(gt_regr - regr) * mask
    regr_loss = tf.reduce_sum(regr_loss)/(tf.reduce_sum(mask)+1e-4)
    return regr_loss
