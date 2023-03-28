import numpy as np


# common loss class
class Loss:
    # store trainable layers
    def store_trainable_layers(self, trainable):
        self.trainable = trainable

    # calculate regularization loss
    def regularization(self):
        # initialize regularization loss
        loss_r = 0

        # calculate regulariation loss per layer
        for layer in self.trainable:

            # add l1 and l2 regularization to the temporary loss
            if layer.l1_w > 0:
                loss_r += layer.l1_w * np.sum(np.abs(layer.weights))

            if layer.l2_w > 0:
                loss_r += layer.l2_w * np.sum(layer.weights * layer.weights)

            if layer.l1_b > 0:
                loss_r += layer.l1_b * np.sum(np.abs(layer.biases))

            if layer.l2_b > 0:
                loss_r += layer.l2_b * np.sum(layer.biases * layer.biases)

        return loss_r

    # calculate data loss and return losses
    def calculate(self, inputs, targets, regularization=False, accumulating=False):
        sample_loss = self.forward(inputs, targets)

        data_loss = np.mean(sample_loss)

        if accumulating:
            # account for accumulated losses and sums
            self.accumulated_sum += np.sum(sample_loss)
            self.accumulated_count += len(sample_loss)

        if not regularization: return data_loss

        return data_loss, self.regularization()

    # calculate accumulated loss
    def accumulate(self, *, regularization=False):

        # calculate mean loss:
        data_loss = self.accumulated_sum / self.accumulated_count

        # return data loss if regularization is disabled
        if not regularization: return data_loss

        return data_loss, self.regularization()

    # reset accumulated loss
    def reset(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


# Categorical Cross Entropy loss function
class LossCCE(Loss):
    # SoftMax processed data is passed as the input, y as the target
    def forward(self, inputs, targets):
        samples = len(inputs)

        # prevent /0 process death without impacting the mean
        clipped_inputs = np.clip(inputs, 1e-7, 1 - 1e-7)

        # ensure processing of both scalar and one-hot encoded inputs
        if len(targets.shape) == 1:
            confidences = clipped_inputs[range(samples), targets]
        elif len(targets.shape) == 2:
            confidences = np.sum(clipped_inputs * targets, axis=1)

        # calculate and return CCE data loss
        losses = -np.log(confidences)
        return losses


# Binary Cross-Entropy loss function
# second component of binary logistic regression
class BinaryCE(Loss):
    # class values are either 0 or 1
    # thus, the incorrect class = 1 - correct class
    def forward(self, inputs, targets):
        clipped_inputs = np.clip(inputs, 1e-7, 1 - 1e-7)

        # loss calculated on a single ouput is a vector of losses
        # sample loss will be the mean of losses from a single sample
        # sample loss = ((current output)**-1) * sum(loss)
        sample_losses = -(targets * np.log(clipped_inputs) + (1 - targets)
                          * np.log(1 - clipped_inputs))

        # return data loss
        return np.mean(sample_losses, axis=-1)

    def backward(self, dvalues, targets):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        # prevent /0 process death without impacting the mean
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # calculate and normalize the gradient
        self.dinputs = -(targets / clipped_dvalues -
                         (1 - targets) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples


# Mean Squared Error loss function
class MSE(Loss):
    def forward(self, inputs, targets):
        # calculate mse
        sample_losses = (targets - inputs) ** 2

        # return data loss
        return np.mean(sample_losses, axis=-1)

    def backward(self, dvalues, targets):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        # calculate and normalize gradient
        self.dinputs = -2 * (targets - dvalues) / outputs / samples


# Mean Absolute Error loss function
class MAE(Loss):
    def forward(self, inputs, targets):
        # calculate mae
        sample_losses = np.abs(targets - inputs)

        # return data loss
        return np.mean(sample_losses, axis=-1)

    def backward(self, dvalues, targets):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        # calculate and normalize gradient
        self.dinputs = np.sign(targets - dvalues) / outputs / samples