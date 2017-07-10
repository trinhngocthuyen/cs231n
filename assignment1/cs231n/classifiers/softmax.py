import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax(Z):
    EZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return EZ / np.sum(EZ, axis=1).reshape((-1, 1))


def softmax_gradient(ins=None, outs=None):
    assert(ins is not None or outs is not None)
    if outs is None:
        outs = softmax(ins)
    return outs * (1 - outs)


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    num_train = X.shape[0]
    num_class = W.shape[1]

    Z = X.dot(W)
    Z = Z - np.max(Z, axis=1, keepdims=True)
    EZ = np.exp(Z)
    AZ = EZ / np.sum(EZ, axis=1, keepdims=True)

    for n in xrange(num_train):
        # Note: loss_i = Y_i * log(A_i) + ...
        # Since we use one-hot-coding for Y_i (ex. label 2 --> [0, 1, 0, ..., 0]),
        # We only care the k-th factor (if it has label k), and Y_k = 1
        for j in xrange(num_class):
            # Update formula: dW[:, j] += ((AZ[n, j] - Y[n, j]) * X[n, :])
            if j == y[n]:
                loss += -np.log(AZ[n, j])
                dW[:, j] += (AZ[n, j] - 1) * X[n, :]   # Y[n, j] = 1
            else:
                dW[:, j] += AZ[n, j] * X[n, :]         # Y[n, j] = 0

    loss /= num_train
    dW /= num_train

    # Regularization
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    num_train = X.shape[0]
    num_class = W.shape[1]

    # One-hot-coding matrix for y
    Y = np.zeros((num_train, num_class))
    Y[xrange(num_train), y] = 1

    Z = X.dot(W)
    Z = Z - np.max(Z, axis=1, keepdims=True)
    EZ = np.exp(Z)
    AZ = EZ / np.sum(EZ, axis=1, keepdims=True)

    loss_each = -np.log(AZ[xrange(num_train), y])
    loss = np.sum(loss_each) / num_train
    dW = np.dot(X.T, (AZ - Y)) / num_train

    # Regularization
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    #############################################################################

    return loss, dW
