"""
Inital model for HAR
"""

import os
import urllib
import zipfile

from typing import Tuple

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torch import Tensor

DATA_ROOT = "../../OUTPUT/"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    """ Neural Network
    """
    def __init__(self) -> None:
        super(NeuralNetwork, self).__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(561, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
        )

    def forward(self, x: Tensor) -> Tensor:
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


def load_data() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    #try:
    #    print('Data dir: ', os.environ['HAR_DATA_DIR'])
    #    har_data_dir = DATA_ROOT+"{}".format(os.environ['HAR_DATA_DIR'])
    #except KeyError as err:
    #    print(f"No data dir specified - {err}")
    
    if not os.path.isdir('data'):
        if not os.path.isfile('har-data.zip'):
            # we have to download the data
            urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip", filename="har-data.zip")

        # unzip it
        with zipfile.ZipFile("har-data.zip","r") as zip_ref:
            zip_ref.extractall('.', members=['UCI HAR Dataset/train/X_train.txt',
                                             'UCI HAR Dataset/train/y_train.txt',
                                             'UCI HAR Dataset/test/X_test.txt',
                                             'UCI HAR Dataset/test/y_test.txt',])

        # rename dir
        os.rename('UCI HAR Dataset', 'data')

    X_train=pd.read_csv("data/train/X_train.txt", delim_whitespace=True, names=["F"+str(i) for i in range(1, 562)])
    y_train=pd.read_csv("data/train/y_train.txt", delim_whitespace=True, names=["label"])
    y_train['label']=y_train['label']-1

    X_test=pd.read_csv("data/test/X_test.txt", delim_whitespace=True, names=["F"+str(i) for i in range(1, 562)])
    y_test=pd.read_csv("data/test/y_test.txt", delim_whitespace=True, names=["label"])
    y_test['label']=y_test['label']-1

    training_data = torch.utils.data.TensorDataset(torch.tensor(X_train.values).float(), torch.as_tensor(y_train.values).squeeze())
    test_data=torch.utils.data.TensorDataset(torch.tensor(X_test.values).float(), torch.as_tensor(y_test.values).squeeze())
    
    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    return train_dataloader, test_dataloader

def train(model, dataloader, epochs, device) -> None:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    size = len(dataloader.dataset)
    print(f"Training {epochs} epoch(s) w/ {size} batches each")
    for epoch in range(epochs):
        model.to(device)
        model.train()
        print("Epoch: ", epoch)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            #print("pred:",pred.shape, pred.dtype," y:" ,y.shape,y.dtype)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    
def test(model, dataloader, device) -> Tuple[float, float]:
    loss_fn = nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, 100*correct
    
            
