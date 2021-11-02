#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""har-server.py script runs a Federated Learning server using flower.

The script can be run in command line:

Examples:
---------
    $./har-server.py - s 0.0.0.0:8080
or
    $python har-server.py -s [::]:8080
using environment variable HAR_SERVER
    $HAR_SERVER=[::]:8080 python har-server.py -m 2 -M 2 -r3

Attributes:
-----------
Model_dict : str
    String that contains the ``OrderedDict`` with the layers.
Notes:
------
This is module contains some HACKs, and it can be / should be improved.

"""
import os
from typing import Dict, Optional, Tuple, Callable
from collections import OrderedDict  # from Python v3.7 dict are ordered

import click
import flwr as fl
from click import secho
import torch
from torch import nn
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
import har
import time
from logger import accuracy_logger, time_logger


def run_server(servername: str, min_fit_clients: int,
               min_available_clients: int,
               number_of_rounds: int,
               number_of_epochs: int,
               training_set: str = None,
               test_set: str = None,
               debug: bool = False) -> None:
    """Run a Federated Learning server using flower.

    Parameters
    ----------
    servername
        The FQDN or IP address with port, ex. mydomain.com:8080
    min_fit_clients
        Minimum number of clients to be sampled for the next round of training
    min_available_clients
        Minimum number of clients that need to be connected to the server
        before a training round
    number_of_rounds
        Number of training rounds
    number_of_epochs
        Number of epochs in a given round
    training_set
        training dataset path
    test_set
        test dataset path
    debug
        Flag for trigger some debug strings
    """

    global Model_dict
    # small toy model for testing if clients get the correct model
    # min_dict = OrderedDict([("Linear1", nn.Linear(561, 10)),
    #                  ("ReLU1", nn.ReLU()),
    #                  ("Linear2", nn.Linear(10, 10)),
    #                  ("ReLU2", nn.ReLU()),
    #                  ("Linear3", nn.Linear(10, 6))])
    # model = har.NeuralNetwork(min_dict)
    # large model
    global NUMBER_OF_EPOCHS
    global NUMBER_OF_ROUNDS
    NUMBER_OF_ROUNDS = number_of_rounds
    NUMBER_OF_EPOCHS = number_of_epochs
    model = har.NeuralNetwork()
    # get str for default model
    Model_dict = model.to_string()
    if debug:
        print(model.state_dict()['linear_relu_stack.Linear3.weight'])
    try:
        if test_set and training_set:
            # Load data some data
            trainloader, testloader = har.load_data(test_set, training_set)
        else:
            trainloader, testloader = None, None
        # defines the strategy
        weights = [val.cpu().numpy() for _, val in
                   model.state_dict().items()]
        strategy = fl.server.strategy.FedAvg(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=fit_config,
            initial_parameters=fl.common.weights_to_parameters(weights),
            eval_fn=get_eval_fn(model, trainloader, testloader, debug),
            # on_evaluate_config=evaluate_config,
            fraction_fit=1,  # 100% client availability
        )
        while True:
            # start server and listen using FedAvg strategy
            print()
            secho('Running a Federated Learning Server using Flower',
                  bg='blue', fg='white')
            print()
            secho('Using servername:port = {}'.format(servername),
                  fg='green')
            secho(
                'Minimum number of clients for each round = {}'
                .format(min_fit_clients), fg='green')
            secho(
                'Minimum number of clients to start learning = {}'
                .format(min_available_clients), fg='green')
            secho('Number of training rounds = {}'
                  .format(number_of_rounds), fg='green')
            print()
            secho('LISTENING', blink=True, bold=True)
            start_training = time.perf_counter()
            fl.server.start_server(servername,
                                   config={"num_rounds": number_of_rounds},
                                   strategy=strategy)
            print()
            secho("Training completed!", bg='green', fg='white')
            end_of_training = time.perf_counter()
            time_logger.info(f"{number_of_rounds};{number_of_epochs};{end_of_training - start_training}")
            print()
            break
    except KeyError as err:
        secho(f"No hostname specified - {err}", bg='red', fg='white')


Model_dict = None

def fit_config(rnd: int) -> Dict[str, str]:
    """Set config dictionary.

    Parameters:
    -----------
    rnd
        number of rounds (not used)
    Returns:
    --------
    config
        model configuration, including ``Model_dict`` string
    """
    global Model_dict
    config: Dict[str, fl.common.Scalar] = {
        #"epoch_global": str(rnd),
        "epochs": NUMBER_OF_EPOCHS,
        #"batch_size": str(32),
        "config": Model_dict
    }
    return config


def set_weights(model: torch.nn.ModuleList,
                weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays.

    Parameters:
    model
        The pytorch model
    weights
        list of numpy ndarrays
    """
    keys = [k for k in model.state_dict().keys()]  # if "bn" not in k]
    params_dict = zip(keys, weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)


round_counter = 0


def get_eval_fn(model: object,
                trainloader: torch.utils.data.DataLoader,
                testloader: torch.utils.data.DataLoader,
                debug: bool
                ) -> Callable[[fl.common.Weights], None]:
    """Return an evaluation function for server-side evaluation.

    Parameters:
    -----------
    model
    """
    global round_counter
    round_counter = 0
    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Tuple[float, float]:
        global round_counter
        if debug:
            print(Tensor(weights[4]))
        set_weights(model, weights)
        print()
        secho("Testing model on server side", bg="yellow", fg="black")
        test_loss, accuracy = har.test(model, testloader, "cpu")
        round_counter += 1
        if round_counter == NUMBER_OF_ROUNDS + 1:
            final_accuracy = accuracy
            accuracy_logger.info(f"{NUMBER_OF_ROUNDS};{NUMBER_OF_EPOCHS};{final_accuracy}")
    # save the model
    torch.save(model.state_dict(), "model.pt")
    return evaluate


if __name__ == "__main__":
    # prepare for a simple plot
    global lines
    global ax
    plt.ion()

    fig, ax = plt.subplots()
    ax.set_ylim(0, 100)
    ax.set_xlabel(r"Round #")
    ax.set_ylabel(r"Accuracy %")
    lines, = ax.plot([0, 0], "-bo")

    run_server()
