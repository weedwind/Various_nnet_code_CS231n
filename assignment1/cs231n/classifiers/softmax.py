import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_train = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
     f = np.dot(X[i], W)        # f is the raw score
     f -= np.max(f)             # deal with numerical overflow
     prob = np.exp(f) / np.sum(np.exp(f))        # predicted probabilities
     loss += -np.log(prob[y[i]])         # loss
     y_bin = np.zeros(num_class)
     y_bin[y[i]] = 1                            # convert the class label to 1-out-of-K format
     for j in xrange(num_class):
        dW[:,j] += (prob[j] - y_bin[j]) * X[i]    # derivative of the loss function with respect to the kth input of the softmax function is prob[k] - target[k]
  loss /= num_train
  loss += reg * np.sum(W * W)      # regularization
  dW /= num_train
  dW += reg * 2 * W                # regularization
     
  #############################################################################
  #                          END OF YOUR CODE                                 #
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
  num_train = X.shape[0]
  num_class = W.shape[1]
  F = np.dot(X, W)        # raw score matrix
  F -= np.max(F, axis = 1).reshape(num_train,1)     # deal with numerical overflow
  P = np.exp(F) / np.sum(np.exp(F), axis = 1).reshape(num_train,1)     # probability matrix
  loss += np.sum(-np.log(P[range(num_train), y]))                      # loss
  Tg = np.zeros((num_train, num_class))
  Tg[range(num_train), y] = 1                                           # the target class labels are converted to 1-out-of-k format in Tg
  dW += np.dot(X.T, P - Tg)
  loss /= num_train
  loss += reg * np.sum(W * W)    # regularization
  dW /= num_train
  dW += reg * 2 * W              # regularization

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

