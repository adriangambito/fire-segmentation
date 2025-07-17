"""Fire-Segmentation: A Flower / TensorFlow app."""
import random
import numpy
import tensorflow as tf

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from fire_segmentation.task import load_data, load_model

SEED = 42  # scegli un numero fisso

# Imposta il seed per Python, NumPy e TensorFlow
#os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
numpy.random.seed(SEED)
tf.random.set_seed(SEED)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, steps_per_epoch, verbose
    ):
        self.model = model
        self.x_train, self.y_train = data
        self.epochs = epochs
        #self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbose

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            verbose=self.verbose,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = load_model()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)
    epochs = context.run_config["local-epochs"]
    #batch_size = context.run_config["batch-size"]
    steps_per_epoch= context.run_config["steps-per-epoch"]
    verbose = context.run_config.get("verbose")

    # Return Client instance
    return FlowerClient(
        net, data, epochs, steps_per_epoch, verbose
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
