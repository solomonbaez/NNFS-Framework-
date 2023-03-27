import numpy as np


# common accuracy class
class Accuracy:

    # calculate accuracy given predictions and ground truths
    def calculate(self, inputs, targets):
        # compare inputs and targets
        comparisons = self.compare(inputs, targets)

        # return accuracy
        return np.mean(comparisons)


# accuracy calculation for regression models
class RegressionAccuracy(Accuracy):
    def __init__(self):
        # initialize the precision property
        self.precision = None

    # calculate precision based on ground truths
    def initialize(self, targets, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(targets) / 250

    # compare predictions and ground truths
    def compare(self, inputs, targets):
        # return simulated accuracy
        return np.absolute(inputs - targets) < self.precision


# accuracy calculation for categorical models
class CategoricalAccuracy(Accuracy):
    def __init__(self, *, binary=False):
        # initialize whether binary logistic regression is used
        self.binary = binary

    # initialization is required per the model module but not required for this object
    def initialize(self, inputs):
        pass

    # compare predictions to ground truths
    def compare(self, inputs, targets):
        if not self.binary and len(targets.shape) == 2:
            targets = np.argmax(targets, axis=1)

        return inputs == targets