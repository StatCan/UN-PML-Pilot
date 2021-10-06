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

"""
import os
from typing import Dict, Optional, Tuple
from collections import OrderedDict  # from Python v3.7 dict are ordered

import click
import flwr as fl
from click import secho
from torch import nn
from torch import Tensor

import har

@click.command()
@click.option('-s', '--servername', prompt=False,
              default=lambda: os.environ.get('HAR_SERVER', ''))
@click.option('-m', '--min_fit_clients', prompt=False,
              default=4, type=int)
@click.option('-M', '--min_available_clients', prompt=False,
              default=4, show_default=True, type=int)
@click.option('-r', '--number_of_rounds', prompt=False,
              default=3, show_default=True, type=int)
def run_server(servername: str, min_fit_clients: int,
               min_available_clients: int,
               number_of_rounds: int):
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
    """
    global Model_dict
    # small toy model
    # min_dict = OrderedDict([("Linear1", nn.Linear(561, 10)),
    #                  ("ReLU1", nn.ReLU()),
    #                  ("Linear2", nn.Linear(10, 10)),
    #                  ("ReLU2", nn.ReLU()),
    #                  ("Linear3", nn.Linear(10, 6))])
    # model = har.NeuralNetwork(min_dict)

    # large model
    model = har.NeuralNetwork()
    Model_dict = model.get_model_string()
    print(Model_dict)
    print(model.state_dict()['linear_relu_stack.Linear3.weight'])
    try:
        # defines the strategy
        weights = [val.cpu().numpy() for _, val in
                   model.state_dict().items()]
        strategy = fl.server.strategy.FedAvg(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=fit_config,
            initial_parameters=fl.common.weights_to_parameters(weights),
            eval_fn=get_eval_fn(model),
            #on_evaluate_config=evaluate_config,
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
            fl.server.start_server(servername,
                                   config={"num_rounds": number_of_rounds},
                                   strategy=strategy)
            print()
            secho("Aggregation completed!", bg='green', fg='white')
            print()
    except KeyError as err:
        secho(f"No hostname specified - {err}", bg='red', fg='white')


Model_dict = None


def fit_config(rnd: int):
    global Model_dict
    config: Dict[str, fl.common.Scalar] = {
        #"epoch_global": str(rnd),
        "epochs": "10",
        #"batch_size": str(32),
        "config": Model_dict
    }
    return config

def get_eval_fn(model: object):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model 
    #(x_train, y_train), _ = ...

    # validation set
    #x_val, y_val = ...

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        print(Tensor(weights[4]))
        # Update model with the latest parameters
        # model.set_weights(weights)
        #loss, accuracy = model.evaluate(x_val, y_val)
        #return loss, {"accuracy": accuracy}

    return evaluate

if __name__ == "__main__":
    run_server()
