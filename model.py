from layers import InputLayer


# Neural Network Model class
class Model:
    def __init__(self):
        # store network objects
        self.layers = []

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

    # train the model
    def train(self, X, y, *, epochs=1, report=1, validation=None):
        # initialize the accuracy object
        self.accuracy.initialize(y)

        # training loop
        for epoch in range(1, epochs + 1):

            # forward pass
            output = self.forward(X)

            # calculate data and regularization losses if applicable
            data_loss, reg_loss = self.loss.calculate(output, y, regularization=True)

            # calculate overall loss
            loss = data_loss + reg_loss

            # calculate accuracy
            predictions = self.activation.predict(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # backpropogate the model
            self.backward(output, y)

            # update model parameters
            self.optimizer.pre_update()
            for layer in self.trainable:
                self.optimizer.update(layer)
            self.optimizer.post_update()

            # report model performance
            if not epoch % report:
                print(f"epoch: {epoch}, " +
                      f"accuracy: {accuracy:.3f}, " +
                      f"loss: {loss:.3f}, " +
                      f"data_loss: {data_loss:.3f}, " +
                      f"regularization_loss: {reg_loss:.3f}, " +
                      f"lr: {self.optimizer.current_lr:.3f}")

        # validate the model if validation data is supplied
        if validation:
            X_val, y_val = validation

            # forward pass
            output_val = self.forward(X_val)

            # calculate loss
            loss_val = self.loss.calculate(output_val, y_val)

            # calculate accuracy
            predictions_val = self.activation.predict(output_val)
            accuracy_val = self.accuracy.calculate(predictions_val, y_val)

            # report validation performance
            print(f"validation, " +
                  f"accuracy: {accuracy_val:.3f}, " +
                  f"loss: {loss_val:.3f}")

    # forward pass
    def forward(self, X):

        # begin the linked list of trainable layers
        # push data into the input layer
        self.layer_input.forward(X)

        # continue pushing data through the linked list
        # outputs from previous layers are inputs into the next
        for layer in self.layers:
            layer.forward(layer.prev.output)

        # return model results
        return layer.output

    # backward pass
    def backward(self, inputs, targets):

        # begin reversing the linked list of trainable layers
        self.loss.backward(inputs, targets)

        # reverse the linked list to backpropogate the model
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
