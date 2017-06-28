from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

VGG_MEAN = [103.939, 116.779, 123.68]

class FCN:

    def __init__(self):
        self.wd = 5e-4

    def build(self, rgb, net_type='fcn_32s', train=False, num_classes=20, 
            random_init_fc8=False, debug=False):
        """
        Build the VGG model using loaded weights
        Parameters
        ----------
        rgb: image batch tensor
            Image in rgb shap. Scaled to Intervall [0, 255]
        net:type: Network type [fcn_32s, fcn_16s, fcn_8s, deconvNet]
        train: bool
            Whether to build train or inference graph
        num_classes: int
            How many classes should be predicted (by fc8)
        random_init_fc8 : bool
            Whether to initialize fc8 layer randomly.
            Finetuning is required in this case.
        debug: bool
            Whether to print additional Debug Information.
        """
        
        # Convert RGB to BGR
        with tf.name_scope('Processing'):

            red, green, blue = tf.split(rgb, 3, 3)
            # assert red.get_shape().as_list()[1:] == [224, 224, 1]
            # assert green.get_shape().as_list()[1:] == [224, 224, 1]
            # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], 3)

            if debug:
                bgr = tf.Print(bgr, [tf.shape(bgr)],
                               message='Shape of input image: ',
                               summarize=4, first_n=1)

        with tf.variable_scope('vgg_16', values=[bgr]) as sc:
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):

                self.conv1_2 = slim.repeat(bgr, 2, slim.conv2d, 64, 
                        [3,3], scope='conv1')
                self.pool1 = slim.max_pool2d(self.conv1_2, [2,2], scope='pool1')

                self.conv2_2 = slim.repeat(self.pool1, 2, slim.conv2d, 
                        128, [3,3], scope='conv2')
                self.pool2 = slim.max_pool2d(self.conv2_2, [2,2], scope='pool2')

                self.conv3_3 = slim.repeat(self.pool2, 3, slim.conv2d, 
                        256, [3,3], scope='conv3')
                self.pool3 = slim.max_pool2d(self.conv3_3, [2,2], scope='pool3')

                self.conv4_3 = slim.repeat(self.pool3, 3, slim.conv2d, 
                        512, [3,3], scope='conv4')
                self.pool4 = slim.max_pool2d(self.conv4_3, [2,2], scope='pool4')

                self.conv5_3 = slim.repeat(self.pool4, 3, slim.conv2d, 
                        512, [3,3], scope='conv5')
                self.pool5 = slim.max_pool2d(self.conv5_3, [2,2], scope='pool5')

                self.fc6 = slim.conv2d(self.pool5, 4096, [7,7], padding='SAME', scope='fc6')
                self.fc6 = slim.dropout(self.fc6, 0.5, is_training=train, scope='dropout6')

                self.fc7 = slim.conv2d(self.fc6, 4096, [1,1], padding='SAME', scope='fc7')
                self.fc7 = slim.dropout(self.fc7, 0.5, is_training=train, scope='dropout7')

                self.score_fr = slim.conv2d(self.fc7, num_classes, [1,1], padding='SAME',
                        activation_fn=None, normalizer_fn=None, scope='score_fr')

        self.pred = tf.argmax(self.score_fr, dimension=3)
        if net_type == 'fcn_32s':
            self.upscore = self._upscore_layer(self.score_fr, 
                    output_shape=tf.shape(bgr),
                    out_dims=num_classes,
                    debug=debug,
                    name='up', ksize=64, stride=32)

        elif net_type == 'fcn_16s':
            # TODO: implement fcn_16s
            TODO = True
            self.upscore2 = self._upscore_layer(self.score_fr,
                    output_shape = tf.shape(self.pool4),
                    out_dims = num_classes,
                    debug = debug, name = 'upscore2',
                    ksize = 4, stride = 2)
            self.score_pool4 = slim.conv2d(self.pool4, num_classes, [1,1], padding='SAME',
                    activation_fn=None, normalizer_fn=None, scope='score_pool4')
            self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)
            self.upscore = self._upscore_layer(self.fuse_pool4,
                    output_shape = tf.shape(bgr),
                    out_dims = num_classes,
                    debug = debug, name = 'upscore32',
                    ksize = 32, stride = 16)
        elif net_type == 'fcn_8s':
            # TODO: implement fcn_8s
            TODO = True
        else:
            # TODO: implement deconvNet
            TODO = True

        self.pred_up = tf.argmax(self.upscore, dimension=3)

    def _get_bilinear_filter(self, f_shape):
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="up_filter", initializer=init,
                               shape=weights.shape)
    def _upscore_layer(self, bottom, output_shape,
                       out_dims, name, debug,
                       ksize=4, stride=2):

        # inp_shape = (batch, height, width, in_imds)
        with tf.variable_scope(name):

            # Compute parameters for deconvolution 
            # Obtain bilinear filter weight (height, width, out_dims, in_dims) 
            in_dims = bottom.get_shape()[3].value
            f_shape = [ksize, ksize, out_dims, in_dims]
            weights = self._get_bilinear_filter(f_shape)

            # Obtain output shape
            out_shape = tf.stack([output_shape[0], output_shape[1], output_shape[2], out_dims])
            # Obtain stride parameter
            strides = [1, stride, stride, 1]
            deconv = tf.nn.conv2d_transpose(bottom, weights, out_shape,
                    strides, padding='SAME')

            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)

        _activation_summary(deconv)
        return deconv

    def _get_deconv_layer(self, x, w_shape, b_shape, name, padding='SAME'):
        w = self._get_weight_variable(w_shape)
        b = self._get_bias_variable([b_shape])

        x_shape = tf.shape(x)
        out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], w_shape[2]])

        deconv = tf.nn.conv2d_transpose(x, w, out_shape, [1,1,1,1], padding=padding)
        deconv_bias = tf.nn.bias_add(deconv, b)

        return deconv_bias

    def _unravel_argmax(self, argmax, shape):
        output_list = []
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
        return tf.pack(output_list)

    def _get_unpool_layer2x2(self, x, raveled_argmax, out_shape):
        argmax = self.unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
        output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

        height = tf.shape(output)[0]
        width = tf.shape(output)[1]
        channels = tf.shape(output)[2]

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

        t2 = tf.squeeze(argmax)
        t2 = tf.pack((t2[0], t2[1]), axis=0)
        t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

        t = tf.concat(3, [t2, t1])
        indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

        x1 = tf.squeeze(x)
        x1 = tf.reshape(x1, [-1, channels])
        x1 = tf.transpose(x1, perm=[1, 0])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)

def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
