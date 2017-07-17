from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


def array_to_dict(arr, prefix):
    dict = {}
    for i in range(len(arr)):
        dict['{}{}'.format(prefix, i)] = arr[i]
    return dict


def extract_array(dict, prefix, n_items):
    arr = []
    for i in range(n_items):
        arr.append(dict['{}{}'.format(prefix, i)])
    return arr


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        ############################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        out1, cache1 = affine_relu_forward(X, W1, b1)
        out2, cache2 = affine_forward(out1, W2, b2)
        scores = out2
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        dout1, dW2, db2 = affine_backward(dscores, cache2)
        _, dW1, db1 = affine_relu_backward(dout1, cache1)

        dW1 += self.reg * W1
        dW2 += self.reg * W2

        grads = {
            'W1': dW1,
            'W2': dW2,
            'b1': db1,
            'b2': db2
        }
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        dims = [input_dim] + hidden_dims + [num_classes]
        weights, biases = [], []
        for l in range(self.num_layers):
            w = np.random.normal(0, weight_scale, (dims[l], dims[l + 1]))
            b = np.zeros(dims[l + 1])
            weights.append(w)
            biases.append(b)

        self.params.update(array_to_dict(weights, prefix='W'))
        self.params.update(array_to_dict(biases, prefix='b'))

        # tnthuyen: Another approach to store params is to put all weights into an array:
        #   self.params['weights'] = weights # where weights = [W1, W2, ...]
        # However, since we cast the params to dtype, we could not store it as an primitive array like this.
        # Also, we could also ravel all the weights and later shape them based on hidden_dims. I dont prefer this way.
        # P/s: I dont think casting parameters like in the end of this function is a good idea :)
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
            gammas = [np.random.randn(dims[l + 1]) for l in range(self.num_layers - 1)]
            betas = [np.random.randn(dims[l + 1]) for l in range(self.num_layers - 1)]

            self.params.update(array_to_dict(gammas, prefix='gamma'))
            self.params.update(array_to_dict(betas, prefix='beta'))

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def extract_bn_param(self, l):
        if not self.use_batchnorm:
            return None, None, None
        gamma = self.params['gamma{}'.format(l)]
        beta = self.params['beta{}'.format(l)]
        bn_param = self.bn_params[l]
        return gamma, beta, bn_param


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        ############################################################################
        weights = extract_array(self.params, prefix='W', n_items=self.num_layers)
        biases = extract_array(self.params, prefix='b', n_items=self.num_layers)

        # { FC - [BN] - ReLU - [Dropdout] } x (L-1)
        outs = [X]
        caches = []
        for l in range(self.num_layers - 1):
            gamma, beta, bn_param = self.extract_bn_param(l)
            out, cache = affine_bn_relu_forward(outs[-1], weights[l], biases[l], gamma=gamma, beta=beta, bn_param=bn_param)
            outs.append(out)
            caches.append(cache)

        outs = outs[1:]  # Remove the first one: X

        # Last layer: FC
        scores, cache = affine_forward(outs[-1], weights[-1], biases[-1])
        outs.append(scores)
        caches.append(cache)
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum([np.sum(w ** 2) for w in weights])

        grads = {}
        grads_weights, grads_biases = [], []
        grads_gammas, grads_betas = [], []
        ############################################################################
        # Last layer: FC
        dout, dW, db = affine_backward(dscores, caches[-1])
        grads_weights.append(dW + self.reg * weights[-1])
        grads_biases.append(db)

        for l in range(self.num_layers-1)[::-1]:
            dout, dW, db, dgamma, dbeta= affine_bn_relu_backward(dout, caches[l], use_batchnorm=self.use_batchnorm)
            grads_weights.append(dW + self.reg * weights[l])
            grads_biases.append(db)
            if self.use_batchnorm:
                grads_gammas.append(dgamma)
                grads_betas.append(dbeta)
        ############################################################################
        grads.update(array_to_dict(grads_weights[::-1], prefix='W'))
        grads.update(array_to_dict(grads_biases[::-1], prefix='b'))

        if self.use_batchnorm:
            grads.update(array_to_dict(grads_gammas[::-1], prefix='gamma'))
            grads.update(array_to_dict(grads_betas[::-1], prefix='beta'))

        return loss, grads
