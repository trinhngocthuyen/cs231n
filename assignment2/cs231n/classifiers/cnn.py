from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


def calc_conv_out_hw(h, w, num_filters, h_filter=1, w_filter=1, stride=1, pad=0):
    h_out = 1 + (h + 2*pad - h_filter) // stride
    w_out = 1 + (w + 2*pad - w_filter) // stride
    return h_out, w_out


def calc_pool_out_hw(h, w, h_pool=1, w_pool=1, stride=1):
    h_out = 1 + (h - h_pool) // stride
    w_out = 1 + (w - w_pool) // stride
    return h_out, w_out


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        C, H, W = input_dim
        h_conv_out, w_conv_out = calc_conv_out_hw(H, W, num_filters=num_filters,
                                                  h_filter=filter_size, w_filter=filter_size,
                                                  pad=(filter_size - 1) // 2)
        h_pool_out, w_pool_out = calc_pool_out_hw(h_conv_out, w_conv_out,
                                                  h_pool=2, w_pool=2, stride=2)

        self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = np.random.normal(0, weight_scale, (num_filters * h_pool_out * w_pool_out, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        ############################################################################
        out, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param=conv_param, pool_param=pool_param)
        out, cache2 = affine_relu_forward(out, W2, b2)
        out, cache3 = affine_forward(out, W3, b3)
        scores = out
        ############################################################################

        if y is None:
            return scores

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 3))
        ############################################################################
        df, dW3, db3 = affine_backward(dscores, cache3)
        df, dW2, db2 = affine_relu_backward(df, cache2)
        df, dW1, db1 = conv_relu_pool_backward(df, cache1)

        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3

        grads = {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2,
            'W3': dW3, 'b3': db3,
        }
        ############################################################################

        return loss, grads
