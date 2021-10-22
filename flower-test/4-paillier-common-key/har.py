# -*- coding: utf-8 -*-
"""Inital model for HAR dataset.

This module defines the neural model and data loaders

Attributes:
-----------
DATA_ROOT : str
    Default data directory
device : str
    The device for training ``cuda`` or ``cpu``
DATA_URL : str
    Default data url for download
model : object
    Model instance

Notes:
------
This is module contains some HACKs, and it can be / should be improved.
For instance importing and exporting layers could use a ``JSON`` object.
For this test we stick to python ``dict``.
"""

import os
import re
import ast
import urllib
import zipfile
from collections import OrderedDict  # from Python v3.7 dict are ordered
from typing import Tuple, Dict

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import Tensor

DATA_ROOT = "../../OUTPUT/"

# set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# reproducibility - REMOVE
torch.manual_seed(0)

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
)


def parse_module(module: OrderedDict) -> Tuple[str, str]:
    """Parse layer description.

    PyTorch shows a layer description like this:
    'Linear(in_features=561, out_features=512, bias=True)'
    This module parse this string to get the name of the layer (Linear)
    and the named arguments for this layer with correct types
    (in_features=561, out_features=512, bias=True)

    Parameters:
    -----------
    module
        Dictionary with layer or activation description

    Returns:
    --------
    layer_act
        Name of the layer or activation
    kwrds
        keyword arguments dictionary, useful to recreate the layer

    Notes:
    ------
    This is a HACK, and it can be / should be improved.
    """
    # use regex to keep layer/activation in first group and keyword
    # arguments in second group
    m = re.search(r"(\w+)\((.*?)\)", module)
    layer_act = m.group(1)
    if m.group(2):
        # split arguments
        mods = m.group(2).split(",")
        # put keyword and argument in a dictionary
        kwrds = {m.split("=")[0].strip(): m.split("=")[1].strip() for m in mods}
    else:
        kwrds = {}

    for k, v in kwrds.items():
        try:
            # get the correct type for argument and updates the dictionary
            kwrds[k] = ast.literal_eval(v)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as e:
            print(f"Error infering {v} type for key {k}")
            print(e)

    return layer_act, kwrds


class NeuralNetwork(nn.Module):
    """Define the Neural Network Model.

    The model is defined/loaded here.
    """

    def __init__(self, model_dict: OrderedDict = None) -> None:
        """Call base `init` and `set_nn` to construct the network.

        Parameters:
        -----------
        model_dict
            Dictionary with modules parameters
        """
        super(NeuralNetwork, self).__init__()
        self.set_nn(model_dict)

    def set_nn(self, model_dict: OrderedDict = None) -> None:
        """Set the network architecture from model_dict or defaults.

        Parameters:
        -----------
        model_dict
            Dictionary with modules parameters
        """
        if not model_dict:
            self.linear_relu_stack_dict = OrderedDict(
                [
                    ("Linear1", nn.Linear(561, 512)),
                    ("ReLU1", nn.ReLU()),
                    ("Linear2", nn.Linear(512, 512)),
                    ("ReLU2", nn.ReLU()),
                    ("Linear3", nn.Linear(512, 6)),
                ]
            )
        else:
            self.linear_relu_stack_dict = model_dict
        # let's work with sequential (connected) layers
        self.linear_relu_stack = nn.Sequential(self.linear_relu_stack_dict)

    def forward(self, x: Tensor) -> Tensor:
        """Do a forward pass.

        Parameters:
        -----------
        x
            tensor
        Returns:
        --------
        tensor
            logits
        """
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def to_dict(self) -> Dict:
        """Return a dictionary with network layers.

        Returns:
        --------
        stack_odict
            Ordered dictionary with layer configuration

        Notes:
        ------
        This is part of a HACK, and it can be / should be improved.
        """
        stack_odict = OrderedDict()
        # get first key only
        first_key = next(iter(self._modules))
        # it should be a Sequential object
        seq = self._modules[first_key]

        stack_odict = {}
        for key, module in seq._modules.items():
            # get string representation of module
            mod_str = repr(module)
            stack_odict[key] = str(mod_str)

        return stack_odict

    def from_dict(self, model_dict: OrderedDict) -> None:
        """Set network layers from dictionary.

        Parameters:
        --------
        model_dict
            Ordered dictionary with layer configuration

        Notes:
        ------
        This is part of a HACK, and it can be / should be improved.
        """
        self.linear_relu_stack_dict = OrderedDict()
        # go through the input dict
        for key, module in model_dict.items():
            name, kwrds = parse_module(module)
            if "linear" in name.lower():
                self.linear_relu_stack_dict[key] = nn.Linear(**kwrds)
            if "relu" in name.lower():
                self.linear_relu_stack_dict[key] = nn.ReLU(**kwrds)
            if "bn" in name.lower():
                self.linear_relu_stack_dict[key] = nn.BatchNorm1d(**kwrds)
        # let's build the model from specifications above
        self.linear_relu_stack = nn.Sequential(self.linear_relu_stack_dict)

    def to_string(self) -> str:
        """Return a string with a dictionary with network layers.

        Returns:
        --------
        str
            String eith Ordered dictionary with layer configuration

        Notes:
        ------
        This is part of a HACK, and it can be / should be improved.
        We could redifine the ``__repr__`` instead.
        """
        return str(self.to_dict())

    def from_string(self, model_str: str) -> None:
        """Set network layers from a string of python dictionary.

        Parameters:
        --------
        model_str
            Ordered dictionary with layer configuration

        Notes:
        ------
        This is part of a HACK, and it can be / should be improved.
        """
        try:
            # use ast literal eval to build dict from string
            self.from_dict(ast.literal_eval(model_str))
        except Exception as e:
            print("Error loading model, ", e)


model = NeuralNetwork().to(device)


def load_data(
    test_path: str = None, train_path: str = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Load datasets.

    Parameters:
    ----------
    test_path
        Path for test dataset

    train_path
        Path for train dataset

    Returns:
    --------
    tuple
        train DataLoader, test DataLoader
    """
    if test_path and train_path:
        trainingset = pd.read_csv(train_path, delimiter=";")
        testset = pd.read_csv(test_path, delimiter=";")

        x_train = pd.concat([trainingset[str(i)] for i in range(561)], axis=1)
        y_train = trainingset["Y"] - 1

        x_test = pd.concat([testset[str(i)] for i in range(561)], axis=1)
        y_test = testset["Y"] - 1
    else:
        if not os.path.isdir("data"):
            # check if file was deflated
            if not os.path.isfile("har-data.zip"):
                # we have to download the data
                urllib.request.urlretrieve(DATA_URL, filename="har-data.zip")

            # unzip it
            with zipfile.ZipFile("har-data.zip", "r") as zip_ref:
                zip_ref.extractall(
                    ".",
                    members=[
                        "UCI HAR Dataset/train/X_train.txt",
                        "UCI HAR Dataset/train/y_train.txt",
                        "UCI HAR Dataset/test/X_test.txt",
                        "UCI HAR Dataset/test/y_test.txt",
                    ],
                )

            # rename dir
            os.rename("UCI HAR Dataset", "data")

        x_train = pd.read_csv(
            "data/train/X_train.txt",
            delim_whitespace=True,
            names=["F" + str(i) for i in range(1, 562)],
        )
        y_train = pd.read_csv(
            "data/train/y_train.txt", delim_whitespace=True, names=["label"]
        )
        y_train["label"] = y_train["label"] - 1

        x_test = pd.read_csv(
            "data/test/X_test.txt",
            delim_whitespace=True,
            names=["F" + str(i) for i in range(1, 562)],
        )
        y_test = pd.read_csv(
            "data/test/y_test.txt", delim_whitespace=True, names=["label"]
        )
        y_test["label"] = y_test["label"] - 1

    training_data = torch.utils.data.TensorDataset(
        torch.tensor(x_train.values).float(), torch.as_tensor(y_train.values).squeeze()
    )
    test_data = torch.utils.data.TensorDataset(
        torch.tensor(x_test.values).float(), torch.as_tensor(y_test.values).squeeze()
    )

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


def train(model, dataloader, epochs, device) -> None:
    """Train model.

    Parameters:
    ----------
    model
        the model
    dataloader
        Data Loader
    epochs
        number of epochs
    device
        CPU or GPU
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    size = len(dataloader.dataset)
    print(f"Training {epochs} epoch(s) w/ {size} batches each")
    for epoch in range(epochs):
        model.to(device)
        model.train()
        print("Epoch: ", epoch)
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            pred = model(x)
            # print("pred:",pred.shape, pred.dtype," y:" ,y.shape,y.dtype)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(model, dataloader, device) -> Tuple[float, float]:
    """Test the model.

    Parameters:
    ----------
    model
        pytorch model

    dataloader
        test DataLoader

    device
        CPU or GPU

    Returns:
    --------
    tuple
        test loss, accuracy
    """
    loss_fn = nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%"
        f", Avg loss: {test_loss:>8f} \n"
    )
    return test_loss, 100 * correct
