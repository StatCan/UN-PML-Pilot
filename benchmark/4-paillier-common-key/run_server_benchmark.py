#!/usr/bin/env python

#Run from command line with e.g.
# ./run_server_benchmark.py -s [::]:8080 -m 2 -M 2 -T ../../../OUTPUT/3\ -\ STATCAN/train/3_ALL_train.csv -t../../OUTPUT/3\ -\ STATCAN/test/3_ALL_test.csv
#Then run run_client_benchmark.py 
from har_server import run_server
import click
import os

@click.command()
@click.option('-s', '--servername', prompt=False,
              default=lambda: os.environ.get('HAR_SERVER', ''))
@click.option('-m', '--min_fit_clients', prompt=False,
              default=4, type=int)
@click.option('-M', '--min_available_clients', prompt=False,
              default=4, show_default=True, type=int)
@click.option('-T', '--training_set', prompt=False,
              default=lambda: os.environ.get('TRAIN_PATH', ''))
@click.option('-t', '--test_set', prompt=False,
              default=lambda: os.environ.get('TEST_PATH', ''))
@click.option('-d', '--debug', prompt=False,
              default=False)
@click.option('--min_epochs', prompt=False, 
              default=10)
@click.option('--max_epochs', prompt=False, 
              default=30)
@click.option('--epoch_interval', prompt=False,
              default=5)
@click.option('--min_rounds', prompt=False, 
              default=1)
@click.option('--max_rounds', prompt=False, 
              default=10)
@click.option('--round_interval', prompt=False, 
              default=4)
def benchmark_server(servername: str,
                     min_fit_clients: int,
                     min_available_clients: int,
                     training_set: str = None,
                     test_set: str = None,
                     debug: bool = False,
                     min_epochs: int = 5,
                     max_epochs: int = 15,
                     epoch_interval: int = 3,
                     min_rounds: int = 5,
                     max_rounds: int = 15,
                     round_interval: int = 3):
    no_of_epochs = min_epochs
    while True:
        no_of_rounds = min_rounds    
        while True:
            print(f"Running server with {no_of_rounds} rounds and {no_of_epochs} epochs")
            global round_counter
            round_counter = 0
            run_server(servername = servername,
                       min_fit_clients = min_fit_clients, 
                       min_available_clients = min_available_clients,
                       number_of_rounds = no_of_rounds,
                       number_of_epochs = no_of_epochs,
                       training_set = training_set,
                       test_set = test_set,
                       debug = False)
            no_of_rounds += round_interval
            if no_of_rounds > max_rounds:
                break
        no_of_epochs += epoch_interval
        if no_of_epochs > max_epochs:
            break

if __name__ == '__main__':
    benchmark_server()
