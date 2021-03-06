from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    ###########################################################################
    N = x.shape[0]
    input_size = np.prod(x.shape[1:])
    x_unrolled = x.reshape(N, input_size)
    out = np.dot(x_unrolled, w) + b
    cache = (x, w, b)
    ###########################################################################
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    ###########################################################################
    N = x.shape[0]
    input_size = np.prod(x.shape[1:])    # = D
    x_unrolled = x.reshape(N, input_size)

    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x_unrolled.T, dout)
    db = np.sum(dout, axis=0)
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    ###########################################################################
    out = np.maximum(x, 0)
    cache = x
    ###########################################################################
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache
    ###########################################################################
    dx = (x > 0) * dout
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    running_std_inv = bn_param.get('running_std_inv', np.zeros(D, dtype=x.dtype))

    #######################################################################
    xmean = x.mean(axis=0)
    xvar = ((x - xmean) ** 2).mean(axis=0)    # x_var = x.std(axis=0) ** 2
    xstd_inv = 1 / np.sqrt(xvar + eps)

    if mode == 'train':
        xhat = (x - xmean) * xstd_inv
        running_mean = momentum * running_mean + (1 - momentum) * xmean
        running_var = momentum * running_var + (1 - momentum) * xvar
        running_std_inv = 1 / np.sqrt(running_var + eps)

    elif mode == 'test':
        xhat = (x - running_mean) * running_std_inv
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    out = gamma * xhat + beta
    cache = (gamma, beta, bn_param, xhat, x, xmean, xvar, xstd_inv)
    #######################################################################

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    bn_param['running_std_inv'] = running_std_inv

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    gamma, beta, bn_param, xhat, x, xmean, xvar, xstd_inv = cache
    N = x.shape[0]
    eps = bn_param.get('eps', 1e-5)

    ###########################################################################
    # out = gamma * xhat + beta
    dgamma = np.sum(dout * xhat, axis=0)
    dbeta = np.sum(dout, axis=0)
    dxhat = dout * gamma

    # xhat = (x - xmean) * xstd_inv
    dxstd_inv = np.sum(dxhat * (x - xmean), axis=0)
    dxmean1 = np.sum(-dxhat * xstd_inv, axis=0)
    dx1 = dxhat * xstd_inv

    # xstd_inv = 1 / np.sqrt(xvar + eps)
    dxvar = dxstd_inv * -0.5 * (xstd_inv ** 3)

    # xvar = ((x - xmean) ** 2).mean(axis=0)  # x_var = x.std(axis=0) ** 2
    dxmean2 = np.mean(dxvar * 2. * (xmean - x), axis=0)
    dx2 = dxvar * 2 * (x - xmean) / N

    # xmean = x.mean(axis=0)
    dxmean = dxmean1 + dxmean2
    dx3 = np.tile(dxmean / N, (N, 1))
    dx = dx1 + dx2 + dx3

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.assignment2/FullyConnectedNets.ipynb

    Inputs / outputs: Same as batchnorm_backward
    """
    gamma, beta, bn_param, xhat, x, xmean, xvar, xstd_inv = cache
    ###########################################################################
    # Ref: Check out the original paper
    N = x.shape[0]
    dgamma = np.sum(dout * xhat, axis=0)
    dbeta = np.sum(dout, axis=0)
    dxhat = dout * gamma

    dxvar = np.sum(dxhat * (x - xmean) * -0.5 * (xstd_inv ** 3), axis=0)
    dxmean = np.sum(-dxhat * xstd_inv + dxvar * -2. * (x - xmean), axis=0)

    dx1 = dxhat * xstd_inv
    dx2 = dxvar * 2 * (x - xmean) / N
    dx3 = dxmean / N
    dx = dx1 + dx2 + dx3
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = np.random.rand(*x.shape) < p

    if mode == 'train':
        #######################################################################
        out = mask * x / p
        #######################################################################
    elif mode == 'test':
        #######################################################################
        out = x
        #######################################################################
    else:
        raise ValueError('Invalid forward dropout mode "%s"' % mode)

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        p = dropout_param['p']
        dx = dout * mask / p
        #######################################################################
    elif mode == 'test':
        dx = dout
    else:
        raise ValueError('Invalid backward dropout mode "%s"' % mode)

    return dx


# tnthuyen: Added for convenience
def pad_edges(x, pad_width=0):
    """
    Return data with zero padding to height and width. x has shape (N, C, H, W)
    """
    return np.pad(x, [(0, 0), (0, 0), (pad_width, pad_width), (pad_width, pad_width)],
                  mode='constant', constant_values=0)


# tnthuyen: Added for convenience
def extract_from_pad(x_pad, pad_width):
    """
    Return original data before padding. x_pad has shape (N, C, HH, WW)
    """
    return x_pad[:, :, pad_width:-pad_width, pad_width:-pad_width] if pad_width > 0 else x_pad


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    ###########################################################################
    n_inputs, n_channels, h_in, w_in = x.shape
    n_filters, _, h_filter, w_filter = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    h_out = int(1 + (h_in + 2 * pad - h_filter) / stride)
    w_out = int(1 + (w_in + 2 * pad - w_filter) / stride)
    out = np.zeros((n_inputs, n_filters, h_out, w_out))

    x_pad = pad_edges(x, pad)

    for n in range(n_inputs):
        for idf in range(n_filters):
            for idh in range(h_out):
                h_prev = idh * stride
                for idw in range(w_out):
                    w_prev = idw * stride
                    block = x_pad[n, :, h_prev:h_prev+h_filter, w_prev:w_prev+w_filter]
                    out[n, idf, idh, idw] = np.sum(block * w[idf]) + b[idf]

    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache
    stride, pad = conv_param['stride'], conv_param['pad']
    x_pad = pad_edges(x, pad)

    n_inputs, n_channels, h_in, w_in = x.shape
    n_filters, _, h_filter, w_filter = w.shape
    _, _, h_out, w_out = dout.shape

    ###########################################################################
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.sum(dout, axis=(0, 2, 3))   # sum by N, HH, WW

    for n in range(n_inputs):
        for idf in range(n_filters):
            for idh in range(h_out):
                h_prev = idh * stride
                for idw in range(w_out):
                    w_prev = idw * stride
                    block = x_pad[n, :, h_prev:h_prev+h_filter, w_prev:w_prev+w_filter]
                    dx_pad[n, :, h_prev:h_prev+h_filter, w_prev:w_prev+w_filter] += w[idf] * dout[n, idf, idh, idw]
                    dw[idf] += block * dout[n, idf, idh, idw]

    dx = extract_from_pad(dx_pad, pad)
    ###########################################################################

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    ###########################################################################
    n_inputs, n_channels, h_in, w_in = x.shape
    h_pool, w_pool, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    h_out = int(1 + (h_in - h_pool) / stride)
    w_out = int(1 + (w_in - w_pool) / stride)
    out = np.zeros((n_inputs, n_channels, h_out, w_out))

    for n in range(n_inputs):
        for idc in range(n_channels):
            for idh in range(h_out):
                h_prev = idh * stride
                for idw in range(w_out):
                    w_prev = idw * stride
                    block = x[n, idc, idh*stride:idh*stride+h_pool, idw*stride:idw*stride+w_pool]
                    out[n, idc, idh, idw] = np.max(block)

    ###########################################################################
    cache = (x, out, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, out, pool_param = cache
    h_pool, w_pool, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    n_inputs, n_channels, _, _ = x.shape
    _, _ , h_out, w_out = dout.shape

    dx = np.zeros_like(x)
    ###########################################################################
    for n in range(n_inputs):
        for idc in range(n_channels):
            for idh in range(h_out):
                h_prev = idh * stride
                for idw in range(w_out):
                    w_prev = idw * stride
                    mask = x[n, idc, h_prev:h_prev+h_pool, w_prev:w_prev+h_pool] == out[n, idc, idh, idw]
                    dx[n, idc, h_prev:h_prev+h_pool, w_prev:w_prev+h_pool] += mask * dout[n, idc, idh, idw]
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    ###########################################################################
    N, C, H, W = x.shape
    x_trans = np.transpose(x, axes=(0, 2, 3, 1)).reshape(-1, C) # Shape: (N, H, W, C)
    x_trans_normed, cache = batchnorm_forward(x_trans, gamma=gamma, beta=beta, bn_param=bn_param)
    out = np.transpose(x_trans_normed.reshape(N, H, W, C), axes=(0, 3, 1, 2))
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """

    ###########################################################################
    N, C, H, W = dout.shape
    dout_trans = np.transpose(dout, axes=(0, 2, 3, 1)).reshape(-1, C)   # Shape: (N, H, W, C)
    dx_trans, dgamma, dbeta = batchnorm_backward_alt(dout_trans, cache)
    dx = np.transpose(dx_trans.reshape(N, H, W, C), axes=(0, 3, 1, 2))
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
