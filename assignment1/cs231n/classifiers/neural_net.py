from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange
from cs231n.classifiers.softmax import softmax, softmax_gradient


def reLU(Z):
    return np.maximum(Z, 0)


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def forward(self, X, need_values_on_paths=False):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        Z1 = np.dot(X, W1) + b1
        A1 = reLU(Z1)
        scores = np.dot(A1, W2) + b2
        softmax_scores = softmax(scores)
        if need_values_on_paths:
            return Z1, A1, scores, softmax_scores
        return softmax_scores

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        num_train, num_feature = X.shape
        num_class = W2.shape[1]

        # Compute the forward pass
        #############################################################################
        Z1, A1, scores, softmax_scores = self.forward(X, need_values_on_paths=True)
        #############################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        #############################################################################
        loss = np.sum(-np.log(softmax_scores[xrange(num_train), y] + 1e-9)) / num_train # Add 1e-9 to prevent log(0)
        loss += reg * np.sum(W1 * W1) + reg * np.sum(W2 * W2)
        #############################################################################

        # Backward pass: compute gradients
        #############################################################################
        Y = np.zeros((num_train, num_class))
        Y[xrange(num_train), y] = 1

        softmax_loss_grad = lambda Y_hat, Y_truth: (Y_hat - Y_truth) / Y_hat.shape[0]
        relu_grad = lambda ins: ins > 0

        # scores = softmax(Z2). loss = cross_entropy(scores)
        # some maths transformation --> dscores = (Y_hat - Y_truth) / N
        dscores = softmax_loss_grad(softmax_scores, Y)

        # scores = A1.W2 + b2
        dW2 = np.dot(A1.T, dscores) + 2 * reg * W2
        db2 = np.sum(dscores, axis=0)
        dA1 = np.dot(dscores, W2.T)

        # A1 = reLU(Z1)
        dZ1 = dA1 * relu_grad(Z1)

        # Z1 = X.W1 + b1
        dW1 = np.dot(X.T, dZ1) + 2 * reg * W1
        db1 = np.sum(dZ1, axis=0)

        # -----------------------------------------------
        # SHORT FORMULAS
        # e2 = softmax_loss_grad(softmax_scores, Y)
        # dW2 = np.dot(A1.T, e2)
        # db2 = np.sum(e2, axis=0)
        #
        # e1 = np.dot(e2, W2.T) * relu_grad(Z1)
        # dW1 = np.dot(X.T, e1)
        # db1 = np.sum(e1, axis=0)
        # ------------------------------------------------

        grads = {
            'W1': dW1,
            'W2': dW2,
            'b1': db1,
            'b2': db2,
        }
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in xrange(num_iters):
            idxs = np.random.permutation(num_train)
            idxs_batch = np.random.choice(idxs, min(num_train, batch_size), replace=False)

            #########################################################################
            X_batch = X[idxs_batch, :]
            y_batch = y[idxs_batch]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            for param_name in grads:
                param_grad_num = grads[param_name]
                self.params[param_name] -= learning_rate * param_grad_num
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        ###########################################################################
        class_scores = self.forward(X)
        return np.argmax(class_scores, axis=1)
        ###########################################################################
