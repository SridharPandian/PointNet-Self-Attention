import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        assert(K==3)
        weights = tf.get_variable('weights', [256, 3*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [3*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, 3, K])
    return transform


def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value

    net = tf_util.conv2d(inputs, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform

def combine_heads(x):
    x = tf.transpose(x, [0, 2, 1, 3])
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)

    return ret

def split_heads(x, n):
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return tf.transpose(ret, [0, 2, 1, 3])

def self_attention_transform_net(ind, input_tensor,
                dk,
                dv,
                num_heads,
                attention_dropout,
                is_training,
                residual_on=True):
    q = tf_util.conv2d(
        tf.expand_dims(input_tensor, 2),
        dk, [1, 1],
        padding='VALID',
        stride=[1, 1],
        is_training=is_training,
        bn=True,
        activation_fn=None,
        scope=str(ind) + '_transformer_q')
    k = tf_util.conv2d(
        tf.expand_dims(input_tensor, 2),
        dk, [1, 1],
        padding='VALID',
        stride=[1, 1],
        is_training=is_training,
        bn=True,
        activation_fn=None,
        scope=str(ind) + '_transformer_k')
    v = tf_util.conv2d(
        tf.expand_dims(input_tensor, 2),
        dv, [1, 1],
        padding='VALID',
        stride=[1, 1],
        is_training=is_training,
        bn=True,
        activation_fn=None,
        scope=str(ind) + '_transformer_v')
    bias = None

    q = tf.squeeze(q, 2)
    k = tf.squeeze(k, 2)
    v = tf.squeeze(v, 2)
    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    key_depth_per_head = dk // num_heads
    q *= key_depth_per_head ** -0.5

    logits = tf.matmul(q, k, transpose_b=True)

    if bias is not None:
        logits += bias

    weights = tf.nn.softmax(logits,
                            name=str(ind) + '_attention_weights')

    weights = tf_util.dropout(weights,
                              keep_prob=1.0 - attention_dropout,
                              is_training=is_training,
                              scope=str(ind) + '_attn_dp')

    v = split_heads(v, num_heads)
    x = tf.matmul(weights, v)
    x = combine_heads(x)

    output_depth = dv
    x = tf_util.conv2d(
        tf.expand_dims(x, 2),
        output_depth, [1, 1],
        padding='VALID',
        stride=[1, 1],
        is_training=is_training,
        bn=True,
        scope=str(ind) + '_transformer_output')

    x = tf.squeeze(x)

    if residual_on:
        return combine_heads(v) + x
    else:
        return x
