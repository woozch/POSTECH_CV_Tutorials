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
                # TODO 3: define vgg-16 network with fully convolutional layers

                self.score_fr = slim.conv2d(self.fc7, num_classes, [1,1], padding='SAME',
                        activation_fn=None, normalizer_fn=None, scope='score_fr')

        self.pred = tf.argmax(self.score_fr, dimension=3)
        if net_type == 'fcn_32s':
            self.upscore = self._upscore_layer(self.score_fr, 
                    output_shape=tf.shape(bgr),
                    out_dims=num_classes,
                    debug=debug,
                    name='up', factor=32)

        elif net_type == 'fcn_16s':
            # TODO 4: implement fcn_16s
            TODO = True
        elif net_type == 'fcn_8s':
            # TODO 4: implement fcn_8s
            TODO = True
        else:
            # TODO 5: implement deconvNet
            TODO = True

        self.pred_up = tf.argmax(self.upscore, dimension=3)

    def _get_bilinear_filter(self, size):
        """
        Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
        """
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

    def _get_bilinear_weights(self, factor, num_classes):
        """
        Create weights matrix for transposed convolution with bilinear filter
        initialization.
        """
        filter_size = 2 * factor - factor % 2

        weights = np.zeros((filter_size,
                            filter_size,
                            num_classes,
                            num_classes), dtype=np.float32)

        bilinear = self._get_bilinear_filter(filter_size)
        for i in range(num_classes):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="up_filter", initializer=init,
                               shape=weights.shape)

    def _upscore_layer(self, bottom, output_shape,
                       out_dims, name, debug, factor=2):

        # inp_shape = (batch, height, width, in_dims)
        with tf.variable_scope(name):

            # Compute parameters for deconvolution 
            # Obtain bilinear filter weight (filter_size, filter_size, out_dims, out_dims)
            weights = self._get_bilinear_weights(factor, out_dims)

            # Obtain output shape
            out_shape = tf.stack([output_shape[0], output_shape[1], 
                output_shape[2], out_dims])
            # Obtain stride parameter
            strides = [1, factor, factor, 1]
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

    # Below functions are for unpooling layer
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

    def unpool_layer2x2_batch(self, x, argmax):
        '''
        Args:
            x: 4D tensor of shape [batch_size x height x width x channels]
            argmax: A Tensor of type Targmax. 4-D. The flattened indices of the max
            values chosen for each output.
        Return:
            4D output tensor of shape [batch_size x 2*height x 2*width x channels]
        '''
        x_shape = tf.shape(x)
        out_shape = [x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]]

        batch_size = out_shape[0]
        height = out_shape[1]
        width = out_shape[2]
        channels = out_shape[3]

        argmax_shape = tf.to_int64([batch_size, height, width, channels])
        argmax = unravel_argmax(argmax, argmax_shape)

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [batch_size*(width//2)*(height//2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, batch_size, height//2, width//2, 1])
        t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

        t2 = tf.to_int64(tf.range(batch_size))
        t2 = tf.tile(t2, [channels*(width//2)*(height//2)])
        t2 = tf.reshape(t2, [-1, batch_size])
        t2 = tf.transpose(t2, perm=[1, 0])
        t2 = tf.reshape(t2, [batch_size, channels, height//2, width//2, 1])

        t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

        t = tf.concat(4, [t2, t3, t1])
        indices = tf.reshape(t, [(height//2)*(width//2)*channels*batch_size, 4])

        x1 = tf.transpose(x, perm=[0, 3, 1, 2])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(out_shape))
        return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))


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
