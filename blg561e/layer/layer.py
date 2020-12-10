import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    '''
        Abstract layer class which implements forward and backward methods
    '''

    def __init__(self):
        self.x = None

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('Abstract class!')

    @abstractmethod
    def backward(self, x):
        raise NotImplementedError('Abstract class!')

    def __repr__(self):
        return 'Abstract layer class'


class LayerWithWeights(Layer):
    '''
        Abstract class for layer with weights(CNN, Affine etc...)
    '''

    def __init__(self, input_size, output_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W = np.random.rand(input_size, output_size)
        self.b = np.zeros(output_size)
        self.x = None
        self.db = np.zeros_like(self.b)
        self.dW = np.zeros_like(self.W)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('Abstract class!')

    @abstractmethod
    def backward(self, x):
        raise NotImplementedError('Abstract class!')

    def __repr__(self):
        return 'Abstract layer class'


class ReLU(Layer):

    def __init__(self):
        # Dont forget to save x or relumask for using in backward pass
        self.x = None

    def forward(self, x):
        '''
            Forward pass for ReLU
            :param x: outputs of previous layer
            :return: ReLU activation
        '''
        # Do not forget to copy the output to object to use it in backward pass
        self.x = x.copy()
        # This is used for avoiding the issues related to mutability of python arrays
        
        x = np.maximum(0, x)     #only takes positive values as it is, negatives will be 0
        
        # Implement relu activation

        return x

    def backward(self, dprev):
        '''
            Backward pass of ReLU
            :param dprev: gradient of previos layer:
            :return: upstream gradient
        '''
        dx = None
        # Your implementation starts        
        dx = (self.x > 0 ) * 1.    # Only positive values' derivatives are going to be 1, others zero.
        dx = dx * dprev            # From chain rule only positive x's are mupltiplied with dprev
        # End of your implementation
        return dx


class YourActivation(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x):
        '''
            :param x: outputs of previous layer
            :return: output of activation
        '''
        # Lets have an activation of X^2
        # TODO: CHANGE IT
        self.x = x.copy()
        x = np.maximum(0, x)   #get positive values
        out = x ** 2           #and squared them
        return out

    def backward(self, dprev):
        '''
            :param dprev: gradient of next layer:
            :return: downstream gradient
        '''
        # TODO: CHANGE IT
        # Example: derivate of X^2 is 2X
        dx = (self.x > 0 ) * 1.   #positive samples' derivative is 1, others zero
        dx = dprev * dx * 2       #x squared's derivative is 2x  and mltiplied by dprev from chain rule
        return dx


class Softmax(Layer):
    def __init__(self):
        self.probs = None

    def forward(self, x):
        '''
            Softmax function
            :param x: Input for classification (Likelihoods)
            :return: Class Probabilities
        '''
        # Normalize the class scores (i.e output of affine linear layers)
        # In order to avoid numerical unstability.
        # Do not forget to copy the output to object to use it in backward pass
        probs = None
       
        # Your implementation starts
        self.x = x.copy()
        x = x - np.max(x, axis = 1).reshape(-1, 1)    #shift values so max value will be zero to provide the stability *for all examples
        probs = np.exp(x) / np.sum(np.exp(x), axis = 1).reshape(-1, 1) #normalized class probabilities *for each example.
        self.probs = probs
        # End of your implementation

        return probs

    def backward(self, y):
        '''
            Implement the backward pass w.r.t. softmax loss
            -----------------------------------------------
            :param y: class labels. (as an array, [1,0,1, ...]) Not as one-hot encoded
            :return: upstream derivate

        '''
        dx = None
        # Your implementation starts
        p = self.probs.copy()
        p[np.arange(len(y)), y] -= 1 #https://www.youtube.com/watch?v=1N837i4s1T8&list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH&index=9
        dx = p / len(y)              #probability of real classes are subtracted by 1 and divided by number of samples
        # End of your implementation

        return dx


def loss(probs, y):
    '''
        Calculate the softmax loss
        --------------------------
        :param probs: softmax probabilities
        :param y: correct labels
        :return: loss
    '''
    loss = None
    # Your implementation starts
    loss = np.sum(-np.log(probs[np.arange(len(y)), y] + 1e-10)) / len(y)    #negative log of correct class probs are summed and divided by N
    #https://www.youtube.com/watch?v=PpFTODTztsU&list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH&index=8
    # End of your implementation         
    return loss


class AffineLayer(LayerWithWeights):
    def __init__(self, input_size, output_size, seed=None):
        super(AffineLayer, self).__init__(input_size, output_size, seed=seed)

    def forward(self, x):
        '''
            :param x: activations/inputs from previous layer
            :return: output of affine layer
        '''
        out = None
        # Vectorize the input to [batchsize, others] array
        batch_size = x.shape[0]
        
        # Do the affine transform
        
        X = x.reshape(batch_size, -1)  #input reshaped to (Number of example to, Dimension)
        out = X.dot(self.W) +self.b    #out = X.W + b
                
        # Save x for using in backward pass
        self.x = x.copy()

        return out

    def backward(self, dprev):
        '''
            :param dprev: gradient of next layer:
            :return: downstream gradient
        '''

        batch_size = self.x.shape[0]
        # Vectorize the input to a 1D ndarray
        x_vectorized = None
        dx, dw, db = None, None, None

        # YOUR CODE STARTS
            
        x = self.x.reshape(batch_size, -1) #input reshaped to (Number of example to, Dimension)
        dx = dprev.dot(self.W.T)           #out = X.W + b    dx = dprev.W  (dprev is coming from chain rule, derivative of next layer)
        dw = x.T.dot(dprev)                #dw = x.dprev
        db = dprev.sum(axis=0)             #db = 1 * dprev
        
        dx = dx.reshape(self.x.shape)      # X reshaped to the original one
        
        # YOUR CODE ENDS

        # Save them for backward pass
        self.db = db.copy()
        self.dW = dw.copy()
        return dx, dw, db

    def __repr__(self):
        return 'Affine layer'
 
    
class Model(Layer):
    def __init__(self, model=None):
        self.layers = model
        self.y = None

    def __call__(self, moduleList):
        for module in moduleList:
            if not isinstance(module, Layer):
                raise TypeError(
                    'All modules in list should be derived from Layer class!')

        self.layers = moduleList

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y):
        self.y = y.copy()
        dprev = y.copy()
        dprev = self.layers[-1].backward(y)
        for layer in reversed(self.layers[:-1]):
            if isinstance(layer, LayerWithWeights):
                dprev = layer.backward(dprev)[0]
            else:
                dprev = layer.backward(dprev)
        return dprev

    def __iter__(self):
        return iter(self.layers)

    def __repr__(self):
        return 'Model consisting of {}'.format('/n -- /t'.join(self.layers))

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

class VanillaSDGOptimizer(object):
    def __init__(self, model, lr=1e-3, regularization_str=1e-4):
        self.reg = regularization_str
        self.model = model
        self.lr = lr

    def optimize(self):
        for m in self.model:
            if isinstance(m, LayerWithWeights):
                self._optimize(m)

    def _optimize(self, m):
        '''
            Optimizer for SGDMomentum
            Do not forget to add L2 regularization!
            :param m: module with weights to optimize
        '''
        # Your implementation starts
        m.W += - self.lr * (m.dW + self.reg * m.W)   #update W  W(t+1) = W(t) - lr * dW      [dW = derivarive of L2 (reg * W(t)) + W(t)]
        m.b += - self.lr * m.db                      #update b  b(t+1) = b(t) - lr * dW
        # End of your implementation
       
class SGDWithMomentum(VanillaSDGOptimizer):
    def __init__(self, model, lr=1e-3, regularization_str=1e-4, mu=.5):
        self.reg = regularization_str
        self.model = model
        self.lr = lr
        self.mu = mu
        # Save velocities for each model in a dict and use them when needed.
        # Modules can be hashed
        self.velocities = {m: 0 for m in model}

    def _optimize(self, m):
        '''
            Optimizer for SGDMomentum
            Do not forget to add L2 regularization!
            :param m: module with weights to optimize
        '''
        # Your implementation starts
        self.velocities[m] = self.mu * self.velocities[m] - self.lr * (m.dW + self.reg * m.W)  #momentum update integrete velocity with hyperparameter mu
        #https://cs231n.github.io/neural-networks-3/
        m.W += self.velocities[m]       #update W  W(t+1) = W(t) + Velocity
        m.b += - self.lr * m.db         #update b  b(t+1) = b(t) - lr * dW
        # End of your implementation
