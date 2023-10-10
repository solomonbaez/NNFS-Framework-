import numpy as np

# basic layer for data storage
class InputLayer:
    def forward(self, inputs, training):
        self.output = inputs


# standard layer for non-convolutional
class DenseLayer:
    # weight initializer utilized in weight distribution modification
    # used when model will not learn in accordance with learning rate adjustments
    def __init__(self, n_in, n_neurons, init_w=0.01, l1_w=0.0, l2_w=0.0, l1_b=0.0, l2_b=0.0):
        # initialize input size
        self.weights = init_w * np.random.randn(n_in, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # regularization strength
        self.l1_w = l1_w
        self.l2_w = l2_w
        self.l1_b = l1_b
        self.l2_b = l2_b

    # set instance parameters
    def set(self, weights, biases):
        self.weights = weights
        self.biases = biases

    # return instance parameters
    def get(self):
        return self.weights, self.biases

    def forward(self, inputs, training=False):
        # perform the output calculation and store
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # backpropogation gradient production
        # weight gradient component
        self.dweights = np.dot(self.inputs.T, dvalues)
        # bias gradient component
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # L1 and L2 backpropogation on weights
        if self.l1_w > 0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights < 0] = -1
            self.dweights += self.l1_w * dl1

        if self.l2_w > 0:
            self.dweights += 2 * self.l2_w * self.weights

        # L1 and L2 backpropogation on biases
        if self.l1_b > 0:
            dl1 = np.ones_like(self.biases)
            dl1[self.biases < 0] = -1
            self.dbiases += self.l1_b * dl1

        if self.l2_b > 0:
            self.dbiases += 2 * self.l2_b * self.biases

        # input gradient component
        self.dinputs = np.dot(dvalues, self.weights.T)


# Dropout Layer class used exclusively in training
# randomly disable neurons (sets outputs to zero) at a given rate per forward pass
# forces more neurons to learn the data
# increases the likelyhood of understanding the underlying function in a dataset
class DropoutLayer:
    def __init__(self, rate):
        # rate is inverted and stored
        self.rate = 1 - rate

    # Bernoulli disribution filter
    # P (r = 1) = p, P (r = 0) = q
    # where q = 1 - p and q == ratio of neurons to disable
    # importantly, output values must be scaled to match training/prediction states
    def forward(self, inputs, training):
        self.inputs = inputs

        # determine if the model is training
        if not training:
            self.output = inputs.copy()
            return

        # generate a scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / \
                           self.rate
        # apply mask to inputs
        self.output = inputs * self.binary_mask

    # partial derivative of Bernoulli distribution
    # f'(r = 0) = 0, f'(r > 0) = (r)(1-q)**(-1)
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
