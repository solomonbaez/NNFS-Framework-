# import model and associated subcomponent modules
from model import Model
from layers import InputLayer, DeepLayer
from activators import ReLU, Sigmoid
from optimizers import OptimizerAdaM
from loss import *
from accuracy import *

# import and initialize the dataset
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# create train and test datasets
X, y = spiral_data(samples=100, classes=2)
X_test, y_test = spiral_data(samples=100, classes=2)

# reshape labels to be sublists
# inner lists are binary outputs
y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# instantiate the model
# model type: Binary Logistic Regression
model = Model()

# add layers
model.add(DeepLayer(2, 64, l2_w=5e-4, l2_b=5e-4))
model.add(ReLU())
model.add(DeepLayer(64, 1))
model.add(Sigmoid())

# set loss, optimizer, and accuracy objects
model.set(
    loss=BinaryCE(),
    optimizer=OptimizerAdaM(decay=5e-7),
    accuracy=CategoricalAccuracy(binary=True)
)

model.finalize()

model.train(X, y, validation=(X_test, y_test), epochs=10000, report=100)