import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros_like(W)  # initialize the gradient as zero
    delta = 1

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + delta
            if margin > 0:
                loss += margin
                dW[:, j] += X[i, :]
                dW[:, y[i]] -= X[i, :]


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W


    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    delta = 1

    #############################################################################
    num_train = X.shape[0]

    scores = X.dot(W)
    correct_class_score = scores[xrange(num_train), y].reshape((-1, 1))

    # Note: For convenient, we temporarily treat the margins of correct classes the same as others
    # --> end up with the loss at the indices of correct classes: delta (= max(0, s_i - s_yi + delta))
    # We will reset the loss at the indices of correct classes to zero
    margins = np.maximum(scores - correct_class_score + delta, 0)
    margins[xrange(num_train), y] = 0
    loss = np.mean(np.sum(margins, axis=1))

    # Add regularization to the loss
    loss += reg * np.sum(W * W)
    #############################################################################


    #############################################################################
    binary = np.double(margins > 0)
    row_sum = np.sum(binary, axis=1)
    binary[xrange(num_train), y] = -row_sum.T

    dW = np.dot(X.T, binary) / num_train
    dW += 2 * reg * W
    #############################################################################

    return loss, dW
