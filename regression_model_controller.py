# import model and associated subcomponent modules
from model import Model
from layers import InputLayer, DeepLayer
from activators import ReLU, Linear
from optimizers import OptimizerAdaM
from loss import *
from accuracy import *

# import and initialize the dataset
import nnfs
from nnfs.datasets import sine_data
nnfs.init()

# create sine-wave dataset
X, y = sine_data()

# instantiate the model
regression_model = Model()

# add layers
regression_model.add(DeepLayer(1, 64))
regression_model.add(ReLU())
regression_model.add(DeepLayer(64, 64))
regression_model.add(ReLU())
regression_model.add(DeepLayer(64, 1))
regression_model.add(Linear())

# set loss, optimizer, and accuracy objects
regression_model.set(
    loss=MSE(),
    optimizer=OptimizerAdaM(learning_rate=0.005, decay=1e-3),
    accuracy=RegressionAccuracy()
)

# finalize and train the model
regression_model.finalize()
regression_model.train(X, y, epochs=10000, report=100)
