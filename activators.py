import numpy as np


# Rectified Linear Unit activation function
class ReLU:
    # calculate predictions for model outputs
    def predict(self, outputs):
        return outputs

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # copy the input gradient values and structure
        self.dinputs = dvalues.copy()
        # produce a zero gradient where input values were invalid
        self.dinputs[self.inputs <= 0] = 0


# SoftMax activation function
class SoftMax:
    # calculate predictions for model outputs
    def predict(self, outputs):
        return np.argmax(outputs, axis=1)

    def forward(self, inputs):
        self.inputs = inputs

        # exponential function application and normalization by the maximum input
        exponentials = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalization per softmax methodology
        self.output = exponentials/np.sum(exponentials, axis=1, keepdims=True)


# Sigmoid activation function
# first component for binary logistic regression
# single neurons can represent two classes
class Sigmoid:
    # calculate predictions for model outputs
    def predict(self, outputs):
        return np.argmax(outputs, axis=1)

    def forward(self, inputs):
        self.inputs = inputs

        # Sigmoid activation
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        # Partial derivative of the sigmoid activation function
        # s' = s(1 - s)
        self.dinputs = dvalues * (1 - self.output) * self.output


# Linear activation
class Linear:
    # return predictions for model outputs
    def predict(self, outputs):
        return outputs

    # store inputs
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        # linear derivative is 1
        self.dinputs = dvalues.copy()