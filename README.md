# Koios-Framework
Koios is a Python Neural Network Fremework built using NumPy that allows you to create, customize, and train neural network models for image analysis. Currently, categorical and regression analysis is supported. This framework provides flexibility and ease of use for both beginners and experienced deep learning practitioners interested in building on a streamlined paradigm.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Importing Components](#importing-components)
  - [Initializing the Dataset](#initializing-the-dataset)
  - [Creating and Configuring the Model](#creating-and-configuring-the-model)
  - [Training the Model](#training-the-model)
- [Contributing](#contributing)
- [License](#license)

## Installation

You can install Koios by cloning the GitHub repository:

```bash
git clone https://github.com/solomonbaez/koios-framework
```

## Usage
### Importing Components

Koios is designed to be modular, with a common Model class as the organizational center. 
To get started, import the necessary components of the framework:

```python
from model import Model
from layers import InputLayer, DeepLayer
from activators import ReLU, Sigmoid
from optimizers import OptimizerAdaM
from loss import *
from accuracy import *
```

### Initializing the Dataset

You can use a wide variety of test libraries to initalize datasets, in this example Keras MNIST data is utilized:

```python
from keras.datasets import mnist
(X, y), (X_valid, y_valid) = mnist.load_data()
```

### Creating and Configuring the Model

Next, instantiate and configure your neural network model. Here's an example:

```python
# Instantiate the model
model = Model()

# Add layers
model.add(InputLayer(2))
model.add(DeepLayer(64, activation=ReLU(), l2_w=5e-4, l2_b=5e-4))
model.add(DeepLayer(1, activation=Sigmoid()))

# Set loss, optimizer, and accuracy objects
model.set(
    loss=BinaryCrossEntropy(),
    optimizer=OptimizerAdaM(decay=5e-7),
    accuracy=BinaryAccuracy()
)

model.finalize()
```

### Training the Model

You can train the model with your dataset using the train method:

```python

model.train(X, y, validation=(X_test, y_test), epochs=10000, report=100)
```

This code trains the model on your dataset for a specified number of epochs, reporting progress every 100 epochs.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository on GitHub.
2. Clone your forked repository to your local machine.
3. Create a new branch for your feature or bug fix.
4. Make your changes and commit them.
5. Push your changes to your fork on GitHub.
6. Open a pull request to the main repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
