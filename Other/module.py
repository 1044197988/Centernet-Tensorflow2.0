import tensorflow as tf
from tensorflow.keras import layers

"""
Inspired by this https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md, 
so we use block by this order relu-->bn-->conv in the backbone and conv-->bn-->relu in others layers.
"""


def convblock(inputs, outchannels, kernel_size):

    x = layers.Conv2D(outchannels, kernel_size, padding='same', use_bias=False)(inputs)
    x = layers.ReLU(max_value=6)(x)
    out = layers.BatchNormalization()(x)
    return out


def top_pool(inputs):
    _, h, w, c = inputs.get_shape().as_list()
    x = layers.Lambda(lambda x: tf.reverse(x,[1]))(inputs)
    x = layers.Lambda(lambda x: tf.pad(x, tf.constant([[0, 0], [h-1, 0], [0, 0], [0, 0]]), constant_values=0))(x)
    x = layers.Lambda(lambda x: tf.nn.max_pool(x, (h, 1), (1,1), padding="VALID"))(x)
    out = layers.Lambda(lambda x: tf.reverse(x, [1]))(x)
    return out


def left_pool(inputs):
    _, h, w, c = inputs.get_shape().as_list()
    x = layers.Lambda(lambda x: tf.reverse(x, [2]))(inputs)
    x = layers.Lambda(lambda x: tf.pad(x, tf.constant([[0, 0], [0, 0], [w-1, 0], [0, 0]]), constant_values=0))(x)
    x = layers.Lambda(lambda x:tf.nn.max_pool(x, (1, w), (1, 1), padding="VALID"))(x)
    out = layers.Lambda(lambda x: tf.reverse(x, [2]))(x)
    return out


def bottom_pool(inputs):
    _, h, w, c = inputs.get_shape().as_list()
    x = layers.Lambda(lambda x: tf.pad(inputs, tf.constant([[0, 0], [h - 1, 0], [0, 0], [0, 0]]), constant_values=0))(inputs)
    out = layers.Lambda(lambda x: tf.nn.max_pool(x, (h, 1), (1, 1), padding="VALID"))(x)
    return out


def right_pool(inputs):
    _, h, w, c = inputs.get_shape().as_list()
    x = layers.Lambda(lambda x: tf.pad(inputs, tf.constant([[0, 0], [0, 0], [w - 1, 0], [0, 0]]), constant_values=0))(inputs)
    out = layers.Lambda(lambda x: tf.nn.max_pool(x, (1, w), (1, 1), padding="VALID"))(x)
    return out


def center_pool(inputs, outchannels):
    x = convblock(inputs, outchannels, 3)
    poolw = right_pool(left_pool(x))
    y = convblock(inputs, outchannels, 3)
    poolh = bottom_pool(top_pool(y))
    pool = layers.Add()([poolw, poolh])
    features = layers.Conv2D(outchannels, 3, padding='same')(pool)
    features = layers.BatchNormalization()(features)
    skip_x = layers.Conv2D(outchannels, 1, padding='same')(inputs)
    skip_x = layers.BatchNormalization()(skip_x)
    out = layers.Add()([skip_x, features])
    out = layers.ReLU(max_value=6)(out)
    out = convblock(out, outchannels, 3)
    return out


def pool(inputs, outchannels, pool1, pool2,):
    x1 = convblock(inputs, outchannels, 3)
    x2 = convblock(inputs, outchannels, 3)
    x2 = pool2(x2)
    x = layers.Add()([x1, x2])
    x = layers.Conv2D(outchannels, 3, padding='same')(x)
    x = pool1(x)

    y1 = convblock(inputs, outchannels, 3)
    y2 = convblock(inputs, outchannels, 3)
    y2 = pool1(y2)
    y = layers.Add()([y1, y2])
    y = layers.Conv2D(outchannels, 3, padding='same')(y)
    y = pool2(y)
    feat = layers.Add()([x, y])
    feat = layers.Conv2D(outchannels, 3, padding='same')(feat)
    feat = layers.BatchNormalization()(feat)
    skip_x = layers.Conv2D(outchannels, 1, padding='same')(inputs)
    skip_x = layers.BatchNormalization()(skip_x)
    out = layers.Add()([skip_x, feat])
    out = layers.ReLU(max_value=6)(out)
    out = convblock(out, outchannels, 3)
    return out


def cascade_tl_pool(inputs, outchannels):
    out = pool(inputs, outchannels, top_pool, left_pool)
    return out


def cascade_br_pool(inputs, outchannels):
    out = pool(inputs, outchannels, bottom_pool, right_pool)
    return out


def res_layer0(inputs, outchannels):
    x = convblock(inputs, int(outchannels/2), 1)
    x = convblock(x, int(outchannels / 2), 3)
    out = convblock(x, outchannels, 1)
    return out


def res_layer1(inputs, outchannels):
    x = convblock(inputs, int(outchannels/2), 1)
    x = convblock(x, int(outchannels / 2), 3)
    out = convblock(x, outchannels, 1)
    skip_x = convblock(inputs, outchannels, 1)
    out = layers.Add()([out, skip_x])
    return out


def upsample_module(inputs, out1, out2):
    left, right = inputs

    xl = res_layer0(left,out2)
    xl = res_layer0(xl, out2)

    xr = convblock(right, out1, 3)
    xr = convblock(xr, out2, 3)
    xr = layers.UpSampling2D()(xr)
    out = layers.Add()([xl, xr])
    return out


def down_module(inputs, out1, out2):
    x = layers.MaxPool2D(2, 2)(inputs)
    x = res_layer0(x, out1)
    out = res_layer1(x, out2)
    return out


def hourglass_module(inputs, outchans, finout):
    x1 = down_module(inputs, outchans[0], outchans[1])
    x2 = down_module(x1, outchans[1], outchans[2])
    x3 = down_module(x2, outchans[2], outchans[3])
    x4 = down_module(x3, outchans[3], outchans[4])
    x5 = down_module(x4, outchans[4], outchans[5])

    x6 = res_layer0(x5, outchans[5])
    p5 = res_layer0(x6, outchans[5])

    p4 = upsample_module([x4, p5], outchans[5], outchans[4])
    p3 = upsample_module([x3, p4], outchans[4], outchans[3])
    p2 = upsample_module([x2, p3], outchans[3], outchans[2])
    p1 = upsample_module([x1, p2], outchans[2], outchans[1])
    heatmaps = upsample_module([inputs, p1], outchans[1], outchans[0])
    heatmaps = convblock(heatmaps, finout, 3)
    return heatmaps


def base_module(inputs, outchannels):
    x = layers.Conv2D(128, 7, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(2, strides=2)(x)
    out = res_layer1(x, outchannels)
    return out


def heat_layer(inputs, outchannels, nbclass):
    x = convblock(inputs, outchannels, 3)
    out = layers.Conv2D(nbclass, 1, padding='same')(x)
    return out



