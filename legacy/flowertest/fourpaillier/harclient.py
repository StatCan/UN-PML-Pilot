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
    $HAR_SERVER=[::]:8080 python har-client.py
            -T../../OUTPUT/2\ -\ ONS/train/2_ALL_train.csv
            -t../../OUTPUT/2\ -\ ONS/test/2_ALL_test.csv

Attributes:
-----------
DEVICE : str
    The device for training ``cuda`` or ``cpu``

"""
import os
import enum
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import click
import flwr as fl
import numpy as np
from click import secho
import matplotlib.pyplot as plt

from har import har
import simplephe.simplephe as sp

USE_FEDBN: bool = False

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NSO(enum.Enum):
    """Identify each NSO."""

    CBS = 0
    ISTAT = 1
    ONS = 2
    STATCAN = 3


class HARClient(fl.client.NumPyClient):
    """Flower client implementing for HAR data using PyTorch.

    Client implementation.
    """

    def __init__(
        self,
        model: object,  # har.NeuralNetwork,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        debug: bool = False,
        test_set_name: str = None,
    ) -> None:
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
        self.first_evaluation_call = True
        self.debug = debug
        self.keygen = sp.KeyGenerator.load()
        self.test_set_name =\
            NSO(int(str(os.path.basename(test_set_name)).split("_")[0]))

    def get_parameters(self) -> List[np.ndarray]:
        """Get parameters."""
        self.model.train()
        secho("Calling get_parameters", fg="yellow")
        encrypted_parameters = []
        for n, (name, val) in enumerate(self.model.state_dict().items()):
            if n % 2 != 0 or n == 4:
                secho(f"Encrypting {name} {val.cpu().numpy().shape}",
                      fg="yellow")
                enc_ndarray = (
                    sp.EncArray(val.cpu().numpy())
                    .encrypt(self.keygen.public_key)
                    .serialize_ndarray()
                )
                print(f"Encrypted ndarray shape {enc_ndarray.shape}")
                encrypted_parameters.append(enc_ndarray)
            else:
                secho(f"Not encrypted {name} {val.cpu().numpy().shape}",
                      fg="blue")
                encrypted_parameters.append(val.cpu().numpy())
        return encrypted_parameters

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set parameters."""
        secho(f"Setting {len(parameters)} parameters", fg="white", bg="green")
        parameters_clear = []
        # iterate in list of arrays from each client
        for n, e in enumerate(parameters):
            if e.flatten().dtype.type is np.str_:
                secho(
                    f"Deserialiazing and decrypting {e.shape} elements ",
                    fg="yellow",
                    nl=False,
                )
                # HACK exponent has changed from 32 to 47
                enc_array = sp.EncArray.deserialize_ndarray(
                    e, self.keygen.public_key, -47
                )
                secho(f"with shape {enc_array.shape}", fg="yellow")
                parameters_clear.append(enc_array.decrypt(self.keygen.private_key))
            else:
                # this one is on the clear
                parameters_clear.append(e)
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        secho("Updating model dict", fg="white", bg="green")
        params_dict = zip(self.model.state_dict().keys(), parameters_clear)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int]:
        """Set model parameters, train model, return updated model parameters.

        Parameters:
        -----------
        parameters
            model parameters as a list of NumPy ndarrays, excluding
            parameters of BN layers when using FedBN
        config
            complete

        Returns:
        --------
        tuple
            updated parameters, size of train dataset, None
        """
        secho(f"Calling fit {config}", fg="magenta")
        # get current round
        self.round = int(config["round"])
        # create plots only at first evaluation or maybe round=0
        if self.first_evaluation_call:
            self.ax, self.lines = set_plot(config["num_rounds"],
                                           self.test_set_name)
            self.first_evaluation_call = False
        for n, e in enumerate(parameters):
            # if need to check for ciphertexts
            if isinstance(e[0], np.str_):
                secho("Found encrypted layers", fg="green")
                break
        self.set_parameters(parameters)
        har.train(self.model, self.trainloader, epochs=25, device=DEVICE)

        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[int, float, float]:
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
        secho("Calling evaluate", fg="green")

        for n, e in enumerate(parameters):
            if isinstance(e[0], np.str_):
                # if need to check for ciphertexts
                secho("Found encrypted layers", fg="green")
                break
        self.set_parameters(parameters)
        loss, accuracy = har.test(self.model, self.testloader, device=DEVICE)
        # plot something
        # self.round_counter += 1
        if self.lines:
            self.lines[0].set_data(
                np.append(self.lines[0].get_xdata(), self.round),
                np.append(self.lines[0].get_ydata(), accuracy),
            )
            self.lines[0].figure.canvas.flush_events()
            self.lines[1].set_data(
                np.append(self.lines[1].get_xdata(), self.round),
                np.append(self.lines[1].get_ydata(), loss),
            )
            self.lines[1].figure.canvas.flush_events()
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}


@click.command()
@click.option(
    "-s", "--servername", prompt=False,
    default=lambda: os.environ.get("HAR_SERVER", "")
)
@click.option(
    "-T",
    "--training_set",
    prompt=False,
    default=lambda: os.environ.get("TRAIN_PATH", ""),
)
@click.option(
    "-t", "--test_set", prompt=False,
    default=lambda: os.environ.get("TEST_PATH", "")
)
@click.option("-d", "--debug", prompt=False, default=False)
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
    secho("Running a Federated Learning Client using Flower",
          bg="magenta", fg="white")
    print()
    secho("Using servername:port = {}".format(servername), fg="green")

    # ax, lines = [], []  # set_plot(10, test_set)

    # Load data
    trainloader, testloader = har.load_data(test_set, training_set)

    # Note that no model is loaded on client side at this point
    # Load model
    model = har.NeuralNetwork().to(DEVICE).train()
    while True:
        try:
            # Start client / no model
            client = HARClient(model, trainloader, testloader, debug, test_set)
            secho("CONNECTING", blink=True, bold=True)
            fl.client.start_numpy_client(servername, client)
            print()
            secho("Training done!", bg="green", fg="white")
            print()
        except KeyError as err:
            secho(f"No hostname specified - {err}", bg="red", fg="white")
        if debug:
            input("Press Enter to continue...")
        break


def set_plot(number_of_rounds, title):
    """Set simple plot for accuracy vs round."""
    plt.ion()
    fig, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel(r"Round #")
    ax1.set_ylabel(r"Accuracy %")
    (lines1,) = ax1.plot([None, None], "-bo")
    # tweak axis
    ax1.set_xlim(1, number_of_rounds)
    ax1.xaxis.get_major_locator().set_params(integer=True)
    # second axis
    ax2 = ax1.twinx()
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylim(0, 2)
    ax2.set_ylabel(r"Loss")
    (lines2,) = ax2.plot([0, 0], "-ro")
    return ([ax1, ax2], [lines1, lines2])


if __name__ == "__main__":
    main()
