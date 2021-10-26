#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""harclient.py script runs a Federated Learning client using flower.

The script can be run in command line:

Examples:
---------
    $./harclient.py -s localhost:8080
        -T../../OUTPUT/2\ -\ ONS/train/2_ALL_train.csv
        -t../../OUTPUT/2\ -\ ONS/test/2_ALL_test.csv
or
    $python harclient.py -s localhost:8080
        -T../../OUTPUT/2\ -\ ONS/train/2_ALL_train.csv
        -t../../OUTPUT/2\ -\ ONS/test/2_ALL_test.csv
using environment variable HAR_SERVER
    $HAR_SERVER=[::]:8080 python harclient.py
            -T../../OUTPUT/2\ -\ ONS/train/2_ALL_train.csv
            -t../../OUTPUT/2\ -\ ONS/test/2_ALL_test.csv

Attributes:
-----------
DEVICE : str
    The device for training ``cuda`` or ``cpu``

"""
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import click
import flwr as fl
import numpy as np
from click import secho
import har

# USE_FEDBN: bool = True

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HARClient(fl.client.NumPyClient):
    """Flower client implementing for HAR data using PyTorch.

    Client implementation.
    """

    def __init__(self,
                 model: object,  # har.NeuralNetwork,
                 trainloader: torch.utils.data.DataLoader,
                 testloader: torch.utils.data.DataLoader,
                 debug: bool = False) -> None:
        """Set model and train-test data loaders.

        Parameters:
        ----------
        model
            torch model
        trainloader
            DataLoader for train dataset
        testloader
            DataLoader for test dataset
        debug : bool
            Flag for trigger some debug messages
        """
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        # HACK this specifies if False is the first communication between
        # server and client in a round of training
        self.train_state = False
        self.debug = debug

    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters.

        Returns:
        --------
        list
            model parameters as a list of NumPy ndarrays, excluding
            parameters of BN layers when using FedBN
        """
        # this code is commented for use in a future version
        # self.model.train()
        # if USE_FEDBN:
        #     return [
        #         val.cpu().numpy()
        #         for name, val in self.model.state_dict().items()
        #         if "bn" not in name
        #     ]
        # else:
        #     # Return model parameters as a list of NumPy ndarrays
        #     return [val.cpu().numpy() for _, val
        #             in self.model.state_dict().items()]
        # raise Exception("Not implemented (server-side parameter init)")
        # if self.train_state:
        return [val.cpu().numpy() for _, val in
                self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy ndarrays.

        Parameters:
        -----------
        parameters
            model parameters as a list of NumPy ndarrays, excluding
            parameters of BN layers when using FedBN
        """
        # self.model.train()
        # if USE_FEDBN:
        #     keys = [k for k in self.model.state_dict().keys()
        #             if "bn" not in k]
        #     params_dict = zip(keys, parameters)
        #     state_dict = OrderedDict({k: torch.Tensor(v)
        #                              for k, v in params_dict})
        #     self.model.load_state_dict(state_dict, strict=False)
        # else:
        #     params_dict = zip(self.model.state_dict().keys(), parameters)
        #     state_dict = OrderedDict({k: torch.Tensor(v)
        #                              for k, v in params_dict})
        #     self.model.load_state_dict(state_dict, strict=True)
        #
        # params_dict = zip(self.model.state_dict().keys(), parameters)
        # state_dict = OrderedDict(
        #     {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        # )
        # self.model.load_state_dict(state_dict, strict=True)
        # if not self.train_state:
        keys = [k for k in self.model.state_dict().keys()]  # if "bn" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters: List[np.ndarray],
            config: Dict[str, str]) -> Tuple[List[np.ndarray], int]:
        """Set model parameters, train model, return updated model parameters.

        Parameters:
        -----------
        parameters
            model parameters as a list of NumPy ndarrays, excluding
            parameters of BN layers when using FedBN

        config
            complete.

        Returns:
        --------
        tuple
            updated parameters, size of train dataset, None
        """
        #print(parameters)
        if not self.train_state:
            secho("Model from config(server)",
                  bg='cyan', fg='white', bold=True)
            # secho(f"CONFIG - {config}", bg='cyan', fg='white')
            # Load model
            model = har.NeuralNetwork().to(DEVICE).train()

            secho(f"Voiding modules {model.linear_relu_stack}", fg='yellow')
            model.linear_relu_stack_dict, model.linear_relu_stack = None, None
            secho(f"Voided {model.linear_relu_stack}", fg='yellow')

            secho(f"Setting model from config {config['config']}", fg='yellow')
            model.from_string(config['config'])
            self.model = model
            # Perform a single forward pass to properly initialize BatchNorm
            # _ = model(next(iter(trainloader))[0].to(DEVICE))
            secho(f"New model is {self.model}", fg='green', bold=True)
            # we are fetching parameters from server
            self.set_parameters(parameters)
            if self.debug:
                print("First state\n",
                      self.model.state_dict()
                      ['linear_relu_stack.Linear3.weight'])
            self.train_state = True
        else:
            if self.debug:
                print("Actual state\n",
                      self.model.state_dict()
                      ['linear_relu_stack.Linear3.weight'])
            self.set_parameters(parameters)
            if self.debug:
                print("New state\n",
                      self.model.state_dict()
                      ['linear_relu_stack.Linear3.weight'])

        har.train(self.model, self.trainloader, epochs=int(config['epochs']),
                  device=DEVICE)
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters: List[np.ndarray],
                 config: Dict[str, str]) -> Tuple[int, float, float]:
        """Set model parameters, evaluate model on local test dataset,
        and return result.

        Parameters:
        -----------
        parameters
            model parameters as a list of NumPy ndarrays, excluding
            parameters of BN layers when using FedBN

        config
            complete.

        Returns:
        --------
        tuple
            loss, size, and accuracy
        """
        self.set_parameters(parameters)
        loss, accuracy = har.test(self.model, self.testloader, device=DEVICE)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}


@click.command()
@click.option('-s', '--servername', prompt=False,
              default=lambda: os.environ.get('HAR_SERVER', ''))
@click.option('-T', '--training_set', prompt=False,
              default=lambda: os.environ.get('TRAIN_PATH', ''))
@click.option('-t', '--test_set', prompt=False,
              default=lambda: os.environ.get('TEST_PATH', ''))
@click.option('-d', '--debug', prompt=False,
              default=False)
def main(servername: str, training_set: str, test_set: str,
         debug: bool) -> None:
    """Run a Federated Learning client using flower.

    Parameters
    ----------
    servername
        The FQDN or IP address with port, ex. mydomain.com:8080
    training_set
        training dataset path
    test_set
        test dataset path
    debug
        Flag for trigger some debug strings
    """
    print()
    secho('Running a Federated Learning Client using Flower',
          bg='magenta', fg='white')
    print()
    secho('Using servername:port = {}'.format(servername), fg='green')

    # Load data
    trainloader, testloader = har.load_data(test_set, training_set)

    # Note that no model is loaded on client side at this point

    try:
        # Start client / no model
        client = HARClient(None, trainloader, testloader, debug)
        secho('CONNECTING', blink=True, bold=True)
        fl.client.start_numpy_client(servername, client)
        print()
        secho("Training done!", bg='green', fg='white')
        print()
    except KeyError as err:
        secho(f"No hostname specified - {err}", bg='red', fg='white')


if __name__ == "__main__":
    main()
