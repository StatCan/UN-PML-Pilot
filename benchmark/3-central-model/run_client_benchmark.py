#!/usr/bin/env python

#Run from command line after running benchmark_server.py using for example
# ./run_client_benchmark.py -T../../OUTPUT/3\ -\ STATCAN/train/3_ALL_train.csv -t../../OUTPUT/3\ -\ STATCAN/test/3_ALL_test.csv -s localhost:8080

from har_client import run_client
import click
import os
import time

@click.command()
@click.option('-s', '--servername', prompt=False,
              default=lambda: os.environ.get('HAR_SERVER', ''))
@click.option('-T', '--training_set', prompt=False,
              default=lambda: os.environ.get('TRAIN_PATH', ''))
@click.option('-t', '--test_set', prompt=False,
              default=lambda: os.environ.get('TEST_PATH', ''))
@click.option('-d', '--debug', prompt=False,
              default=False)
@click.option('--min_epochs', prompt=False, 
              default=5)
@click.option('--max_epochs', prompt=False, 
              default=15)
@click.option('--epoch_interval', prompt=False,
              default=1)
@click.option('--min_rounds', prompt=False, 
              default=1)
@click.option('--max_rounds', prompt=False, 
              default=10)
@click.option('--round_interval', prompt=False, 
              default=2)
def benchmark_client(servername: str,
                     training_set: str,
                     test_set: str,
                     debug: str,
                     min_epochs: int,
                     max_epochs: int,
                     epoch_interval: int,
                     min_rounds: int,
                     max_rounds: int,
                     round_interval: int):

    no_of_epochs = min_epochs # just for matching number of iterations with server
    while True:
        no_of_rounds = min_rounds  # just for matching number of iterations with server  
        while True:
            print(f"Running client with {no_of_rounds} rounds and {no_of_epochs}")
            run_client(servername = servername,
                       training_set = training_set,
                       test_set = test_set,
                       debug = False)
            time.sleep(2)
            no_of_rounds += round_interval
            if no_of_rounds > max_rounds:
                break
        no_of_epochs += epoch_interval
        if no_of_epochs > max_epochs:
            break

if __name__ =='__main__':
    benchmark_client()

