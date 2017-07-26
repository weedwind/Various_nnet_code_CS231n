from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


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
                 use_batchnorm = False, dtype=np.float32):
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
        self.use_batchnorm = use_batchnorm

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        pool_out_h = (input_dim[1] - 2) / 2 + 1
        pool_out_w = (input_dim[2] - 2) / 2 + 1

        self.params['W1'] = weight_scale * np.random.randn(num_filters, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * np.random.randn(num_filters * pool_out_h * pool_out_w, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        if self.use_batchnorm:
            self.params['gamma1'] = np.ones(num_filters)
            self.params['beta1'] = np.zeros(num_filters)
            self.params['gamma2'] = np.ones(hidden_dim)
            self.params['beta2'] = np.zeros(hidden_dim)
        
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'}, {'mode': 'train'}]

        ############################################################################
        #                             END OF YOUR CODE                             #
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
        
        if self.use_batchnorm:
            gamma1, beta1 = self.params['gamma1'], self.params['beta1']
            gamma2, beta2 = self.params['gamma2'], self.params['beta2']

        mode = 'test' if y is None else 'train'
        
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        if not self.use_batchnorm:
           out, cache0 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
           out, cache1 = affine_relu_forward(out, W2, b2)
        else:
           out, cache0 = conv_bn_relu_pool_forward(X, W1, b1, gamma1, beta1, conv_param, self.bn_params[0], pool_param)
           out, cache1 = affine_bn_relu_forward(out, W2, b2, gamma2, beta2, self.bn_params[1])
        scores, cache2 = affine_forward(out, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        dout, grads['W3'], grads['b3'] = affine_backward(dout, cache2)
        
        if not self.use_batchnorm:
           dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, cache1)
           _, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, cache0)
        else:
           dout, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] = affine_bn_relu_backward(dout, cache1)
           _, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = conv_bn_relu_pool_backward(dout, cache0)

        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        grads['W3'] += self.reg * W3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
