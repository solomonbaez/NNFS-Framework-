from layers import InputLayer
from activators import SoftMax, SoftMaxCCE
from loss import LossCCE

# Neural Network Model class
class Model:
    def __init__(self):
        # store network objects
        self.layers = []
        # initialize combined loss and activation
        self.combined_loss_activation = None

    # add objects to the model network
    def add(self, layer):
        self.layers.append(layer)

    # set loss method and optimizer type
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # finalize the model
    def finalize(self):
        # create and set the input layer
        self.layer_input = InputLayer()

        # object count
        layer_count = len(self.layers)

        # initialize a set of trainable layers
        self.trainable = []

        # iterate through the network objects
        # construct a linked list of layers
        for i in range(layer_count):
            # initialize the first layer using the input layer
            if i == 0:
                self.layers[i].prev = self.layer_input
                self.layers[i].next = self.layers[i+1]

            # iterate through the network objects
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.activation = self.layers[i]

            # determine if network object is trainable
            if hasattr(self.layers[i], "weights"):
                self.trainable.append(self.layers[i])

        # load trainable layers into the loss object
        self.loss.store_trainable_layers(self.trainable)

        # determine if SoftMaxCCE is utilized
        if isinstance(self.layers[-1], SoftMax) and \
           isinstance(self.loss, LossCCE):
            self.combined_loss_activation = SoftMaxCCE()

    # train the model
    def train(self, X, y, *, epochs=1, report=1, batches=None, validation=None):
        # initialize the accuracy object
        self.accuracy.initialize(y)

        # pre-set epoch steps
        steps = 1

        if validation:
            steps_val = 1
            X_val, y_val = validation

        # if data is sliced, determine epoch steps
        if batches:
            steps = len(X) // batches
            print(f"batch_steps: {steps}")
            # check for overflow
            if steps * batches < len(X):
                steps += 1

            if validation:
                steps_val = len(X_val) // batches
                print(f"val_steps: {steps_val}")
                # check for overflow
                if steps_val * batches < len(X_val):
                    steps_val += 1

        # training loop
        for epoch in range(1, epochs + 1):
            # report epoch
            print(f"epoch: {epoch}")

            # reset accumulation
            self.loss.reset()
            self.accuracy.reset()

            for step in range(steps):
                if not batches:
                    X_batch = X
                    y_batch = y
                else:
                    X_batch = X[step*batches:(step+1)*batches]
                    y_batch = y[step*batches:(step+1)*batches]

                # forward pass
                output = self.forward(X_batch, training=True)

                # calculate data and regularization losses if applicable
                data_loss, reg_loss = self.loss.calculate(output, y_batch,
                                                          regularization=True,
                                                          accumulating=True)

                # calculate overall loss
                loss = data_loss + reg_loss

                # calculate accuracy
                predictions = self.activation.predict(output)
                accuracy = self.accuracy.calculate(predictions, y_batch,
                                                   accumulating=True)

                # backpropogate the model
                self.backward(output, y_batch)

                # update model parameters
                self.optimizer.pre_update()
                for layer in self.trainable:
                    self.optimizer.update(layer)
                self.optimizer.post_update()

                # report batch performance
                if not step % report or step == steps - 1:
                    print(f"step: {step}, " +
                          f"accuracy: {accuracy:.3f}, " +
                          f"loss: {loss:.3f}, " +
                          f"data_loss: {data_loss:.3f}, " +
                          f"regularization_loss: {reg_loss:.3f}, " +
                          f"lr: {self.optimizer.current_lr}")

            # report model performance
            data_loss_epoch, reg_loss_epoch = self.loss.accumulate(regularization=True)
            loss_epoch = data_loss_epoch + reg_loss_epoch
            accuracy_epoch = self.accuracy.accumulate()
            print(f"training, " +
                  f"accuracy: {accuracy_epoch:.3f}, " +
                  f"loss: {loss_epoch:.3f}, " +
                  f"data_loss: {data_loss_epoch:.3f}, " +
                  f"regularization_loss: {reg_loss_epoch:.3f}, " +
                  f"lr: {self.optimizer.current_lr}")

            # batch validate if needed
            if validation:

                # reset accumulation
                self.loss.reset()
                self.accuracy.reset()

                for step in range(steps_val):
                    if not step % report or step == step - 1:
                        print(f"batch validation step: {step}")
                    if batches:
                        X_batch = X_val
                        y_batch = y_val
                    else:
                        X_batch = X_val[step*batches:(step+1)*batches]
                        y_batch = y_val[step*batches:(step+1)*batches]

                    # forward pass
                    output_val = self.forward(X_batch, training=False)

                    # calculate loss
                    self.loss.calculate(output_val, y_batch, accumulating=True)

                    # calculate accuracy
                    predictions_val = self.activation.predict(output_val)
                    self.accuracy.calculate(predictions_val, y_batch, accumulating=True)

                # report validation performance
                loss_val = self.loss.accumulate()
                accuracy_val = self.accuracy.accumulate()

                print(f"validation, " +
                      f"accuracy: {accuracy_val:.3f}, " +
                      f"loss: {loss_val:.3f}")

    # forward pass
    def forward(self, X, training):

        # begin the linked list of trainable layers
        # push data into the input layer
        self.layer_input.forward(X, training)

        # continue pushing data through the linked list
        # outputs from previous layers are inputs into the next
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # return model results
        return layer.output

    # backward pass
    def backward(self, inputs, targets):

        # determine if combined activation and loss is utilized
        if self.combined_loss_activation:
            self.combined_loss_activation.backward(inputs, targets)

            # manipulate final layer gradient to account for combined functionality
            self.layers[-1].dinputs = self.combined_loss_activation.dinputs

            # reverse the linked list of trainable layers and backpropogate the model
            # exclude the last layer
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        # begin reversing the linked list of trainable layers
        self.loss.backward(inputs, targets)

        # reverse the linked list to backpropogate the model
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
