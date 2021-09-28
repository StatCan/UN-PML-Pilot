import os
import sys
import timeit
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torchvision

import har

USE_FEDBN: bool = True

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Flower Client... see flower examples
class HARClient(fl.client.NumPyClient):
    """Flower client implementing for HAR data using PyTorch."""

    def __init__(
        self,
        model: har.NeuralNetwork,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self) -> List[np.ndarray]:
        self.model.train()
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        har.train(self.model, self.trainloader, epochs=10, device=DEVICE)
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[int, float, float]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = har.test(self.model, self.testloader, device=DEVICE)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}


def main() -> None:
    """Load data, start CifarClient."""
    # Paths to data
    TEST_PATH = os.environ["TEST_PATH"]
    TRAIN_PATH = os.environ["TRAIN_PATH"]
    # Load data
    trainloader, testloader = har.load_data(test_path=TEST_PATH, train_path=TRAIN_PATH)

    # Load model
    model = har.NeuralNetwork().to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))[0].to(DEVICE))

    # get hostname and port from env
    servername = ''
    try:
        servername = os.environ['HAR_SERVER']
        print('servername:port: ', servername)
        # Start client
        client = HARClient(model, trainloader, testloader)
        fl.client.start_numpy_client(servername, client)
    except KeyError as err:
        print(f"No hostname specified - {err}")


if __name__ == "__main__":
    main()



