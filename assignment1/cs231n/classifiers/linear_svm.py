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
  dW = np.zeros(W.shape) # initialize the gradient as zero

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
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]           # the change in W will affect scores[j] of a wrong class        
        dW[:,y[i]] += (-X[i])     # the change in W will also affect scores[y[i]] of the correct class

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train         # average the gradient

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W             # gradient of the regularization 

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  
  """
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  scores = np.dot(X, W)  # scores is a matrix now
  correct_class_scores = scores[range(num_train), y].reshape(num_train,1)   # correct_class_scores is a column vector now
  margin = np.maximum(0, scores - correct_class_scores + 1)
  margin[range(num_train),y] = 0         # set the margin of the correct class to be 0
  loss = np.sum(margin)
  loss /= num_train                      # average loss
  loss += reg * np.sum(W * W)            # add regularization

  margin_mask = (margin > 0).astype("int")
  row_sum = np.sum(margin_mask, axis = 1)   # how many classes have nonzero margin for each data.
  margin_mask[range(num_train), y] = -row_sum   # the vector X[i] needs to be subtracted multiple times for the gradient of the correct class weights
  dW += np.dot(X.T, margin_mask)                # this computes the gradient, which is a vecterized form
  dW /= num_train     # average gradient
  dW += reg * 2 * W      # add gradient of the regularizor
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
