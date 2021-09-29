import os

import flwr as fl
import click

@click.command()
@click.option('-s', '--servername', prompt=False,
              default=lambda: os.environ.get('HAR_SERVER', ''))
@click.option('-m', '--min_fit_clients', prompt=False,
              default=4, type=int)
@click.option('-M', '--min_available_clients', prompt=False,
              default=4, show_default=True, type=int)
@click.option('-r', '--number_of_rounds', prompt=False,
              default=3, show_default=True, type=int)
def run_server(servername: str, min_fit_clients: int, min_available_clients: int, number_of_rounds: int):
        # get hostname and port from env
    try:
        # defines the strategy
        strategy = fl.server.strategy.FedAvg(
            #fraction_fit=1,  # 100% client availability
            min_fit_clients=min_fit_clients,  # Minimum number of clients to be sampled for the next round
            min_available_clients=min_available_clients,  # Minimum number of clients that need to be connected to the server before a training round
        )     
        while True:
            # start server and listen using FedAvg strategy
            print()
            click.echo(click.style('Running a Federated Learning Server using Flower', bg='blue', fg='white'))
            print()
            click.echo(click.style('Using servername:port = {}'.format(servername), fg='green'))
            click.echo(click.style('Minimum number of clients for each round = {}'.format(min_fit_clients), fg='green'))
            click.echo(click.style('Minimum number of clients to start learning = {}'.format(min_available_clients), fg='green'))
            click.echo(click.style('Number of training rounds = {}'.format(number_of_rounds), fg='green'))
            print()
            click.echo(click.style('LISTENING', blink=True, bold=True))
            fl.server.start_server(servername, config={"num_rounds": number_of_rounds}, strategy=strategy)
            print()
            click.echo(click.style("Aggregation completed!", bg='green', fg='white'))
            print()
    except KeyError as err:
        click.echo(click.style(f"No hostname specified - {err}", bg='red', fg='white'))

if __name__ == "__main__":
    run_server()
