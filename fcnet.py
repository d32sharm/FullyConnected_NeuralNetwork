import numpy as np
from past.builtins import xrange

""" 
    Helper Function 
"""

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
    out = None
    
    # reshape the input into (N, d_1 *...* d_k)
    input_shape = x.shape
    prod = 1
    for i in range(1,len(input_shape)):
        prod *= input_shape[i]

    a = x.reshape(x.shape[0],prod)
    out = np.dot(a,w) + b
    
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
   
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    
    input_shape = x.shape
    prod = 1
    for i in range(1,len(input_shape)):
        prod *= input_shape[i]

    x_reshaped = x.reshape(x.shape[0], prod)
    dw = (x_reshaped.T).dot(dout)

    db = np.sum(dout,axis=0)
       
    return dx, dw, db


def sigmoid_forward(x):
    """
    Computes the forward pass for a layer of logistic sigmoid.

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """

    out = 1/(1+np.exp(-x))

    cache = x
    return out, cache

def sigmoid_backward(dout, cache):
    """
    Computes the backward pass for a layer of logistic sigmoid function.

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    f = lambda x: 1/(1 + np.exp(-x)) # activation function (sigmoid)

    fun = f(x)

    dx = np.multiply(fun, (1-fun))
    dx = np.multiply(dx,dout)

    return dx

def affine_sigmoid_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a sigmoid

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the sigmoid
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, sigmoid_cache = sigmoid_forward(a)
    cache = (fc_cache, sigmoid_cache)
    return out, cache


def affine_sigmoid_backward(dout, cache):
    """
    Backward pass for the affine-sigmoid convenience layer
    """

    # fc_cache contains w (weights), x(input) , b(bias)
    # sigmoid cache contains the combination wx+b
    fc_cache, sigmoid_cache = cache
    da = sigmoid_backward(dout, sigmoid_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

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
    # softmax
    num = np.exp(x)
    den = np.sum(num, axis=1)
    softmax = num/den[:, None]
    N = x.shape[0]

    # compute the los per class
    loss = softmax[np.arange(N), y]
    loss = -np.log(loss)

    # sum all the losses and divide by number of class
    # Also add the regularization loss term
    loss = np.sum(loss)/N 
        
    dscores = softmax
    dscores[np.arange(N), y] -= 1
    dscores /= N

    return loss, dscores

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    Sigmoid nonlinearities, and a softmax loss function. For a network with L layers,
    the architecture will be

    {affine - sigmoid } x (L - 1) - affine - softmax

    where {...} block is repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and are learned
    in the train function
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 reg=0.0, weight_scale=1e-2, dtype=np.float32):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        """
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # Initialize the network parameters with different weights and biases for  #
        # network layers                                                           #
        ############################################################################
        
        key = ['W' + str(1), 'b' + str(1)]
        self.params[key[0]] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
        self.params[key[1]] = np.zeros(hidden_dims[0])
        
        for i in range(1, len(hidden_dims)):
            key = ['W' + str(i+1), 'b' + str(i+1)]
                
            self.params[key[0]] = weight_scale * np.random.randn(hidden_dims[i-1], hidden_dims[i])
            self.params[key[1]] = np.zeros(hidden_dims[i])

        key = ['W' + str(self.num_layers), 'b' + str(self.num_layers)]
        self.params[key[0]] = weight_scale * np.random.randn(hidden_dims[len(hidden_dims)-1], num_classes)
        self.params[key[1]] = np.zeros(num_classes)


        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        scores = None
        ############################################################################
        # Implementing the forward pass for the fully-connected net, computing     #
        # the class scores for X and storing them in the scores variable.          #
        ############################################################################

        l_input = X.copy()
        out = []
        cache = []
        for i in range(self.num_layers - 1):
            # layerwise compute the forward pass and store outputs in out list
            key = ['W' + str(i+1), 'b' + str(i+1)]
            lout, lcache = affine_sigmoid_forward(l_input, self.params[key[0]], self.params[key[1]])
            out.append(lout)
            cache.append(lcache)
            l_input = lout

        key = ['W' + str(self.num_layers), 'b' + str(self.num_layers)]
        scores, lcache = affine_forward(out[self.num_layers - 2], self.params[key[0]], self.params[key[1]])
        cache.append(lcache)
         
        # regularization parameter compute by summing square of all weight vectors
        R = 0
        for i in range(1, self.num_layers + 1):
            key = 'W' + str(i)
            R += np.sum(np.power(self.params[key], 2))

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        ########################
        # Backward pass to compute the loss and gradients
        ########################

        loss, dscore = softmax_loss(scores, y)
        # Apply regularization of the loss 
        loss = loss + 0.5 * self.reg * R

        key = ['W' + str(self.num_layers), 'b' + str(self.num_layers)]
        dx, grads[key[0]], grads[key[1]] = affine_backward(dscore, cache[self.num_layers - 1])
        grads[key[0]] += self.reg * self.params[key[0]] 

        for i in range(self.num_layers - 1, 0, -1):
            key = ['W' + str(i), 'b' + str(i)]
            dx, grads[key[0]], grads[key[1]] = affine_sigmoid_backward(dx, cache[i-1])
            # Apply regularization to the gradients
            grads[key[0]] += self.reg * self.params[key[0]]

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

        indices = np.array(range(num_train))

        for it in xrange(num_iters):
          # randomly select number of indices equal batch size to train on 
          batch_indices = np.random.choice(indices, size=batch_size)

          # minibatch of training data
          X_batch = X[batch_indices]
          y_batch = y[batch_indices]


          loss, grads = self.loss(X_batch, y=y_batch)
          loss_history.append(loss)

          # Stochastic Gradient Descent
          for i in range(1, self.num_layers+1):
            key = ['W' + str(i), 'b' + str(i)]
            self.params[key[0]] += -learning_rate * grads[key[0]]
            self.params[key[1]] += -learning_rate * grads[key[1]]

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
        y_pred = None

        scores = self.loss(X)

        # Apply softmax activation
        num = np.exp(scores)
        den = np.sum(num, axis=1)
        softmax = num/den[:, None]

        y_pred = np.argmax(softmax, axis=1)

        return y_pred

