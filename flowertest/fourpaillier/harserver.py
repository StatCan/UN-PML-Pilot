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
import numpy as np
import simplephe.simplephe as sp


# we need a global, fit_config only configures number of rounds
Config = {}


@click.command()
@click.option(
    "-s", "--servername", prompt=False, default=lambda: os.environ.get("HAR_SERVER", "")
)
@click.option("-m", "--min_fit_clients", prompt=False, default=4, type=int)
@click.option(
    "-M",
    "--min_available_clients",
    prompt=False,
    default=4,
    show_default=True,
    type=int,
)
@click.option(
    "-r", "--number_of_rounds", prompt=False, default=3, show_default=True, type=int
)
@click.option(
    "-T",
    "--training_set",
    prompt=False,
    default=lambda: os.environ.get("TRAIN_PATH", ""),
)
@click.option(
    "-t", "--test_set", prompt=False, default=lambda: os.environ.get("TEST_PATH", "")
)
@click.option("-d", "--debug", prompt=False, default=False)
def run_server(
    servername: str,
    min_fit_clients: int,
    min_available_clients: int,
    number_of_rounds: int,
    training_set: str = None,
    test_set: str = None,
    debug: bool = False,
) -> None:
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
    training_set
        training dataset path
    test_set
        test dataset path
    debug
        Flag for trigger some debug strings
    """
    try:
        strategy = sp.SimplePaillierAvg(
            on_fit_config_fn=fit_config,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            min_eval_clients=min_available_clients,  # Evaluate all
            fraction_fit=1,  # 100% client availability
        )
        # start server and listen using FedAvg strategy
        print()
        secho(
            "Running a Federated Learning Server using Flower "
            + "with Paillier Homomorphic Encryption Aggregation",
            bg="blue",
            fg="white",
        )
        print()
        secho("Using servername:port = {}".format(servername), fg="green")
        secho(
            "Minimum number of clients for each round = {}".format(min_fit_clients),
            fg="green",
        )
        secho(
            "Minimum number of clients to start learning = {}".format(
                min_available_clients
            ),
            fg="green",
        )
        secho("Number of training rounds = {}".format(number_of_rounds), fg="green")
        print()
        secho("LISTENING", blink=True, bold=True)
        global Config
        Config["num_rounds"] = number_of_rounds
        fl.server.start_server(servername, config=Config, strategy=strategy)
        print()
        secho("Training completed!", bg="green", fg="white")
        print()

    except KeyError as err:
        secho(f"No hostname specified - {err}", bg="red", fg="white")


def fit_config(rnd: int) -> Dict[str, str]:
    """Set config dictionary.

    Parameters:
    -----------
    rnd
        number of rounds
    Returns:
    --------
    config
        we should specify server/strategy configuration here
    """
    global Config
    # config: Dict[str, fl.common.Scalar] = {
    #    "round": str(rnd)
    # }
    Config["round"] = str(rnd)
    return Config


if __name__ == "__main__":
    # call run_server
    run_server()
