import os

import flwr as fl

if __name__ == "__main__":

    # get hostname and port from env
    servername = ''
    try:
        servername = os.environ['HAR_SERVER']
        print('servername:port: ', servername)
        # defines the strategy
        strategy = fl.server.strategy.FedAvg(
            #fraction_fit=1,  # 100% client availability
            min_fit_clients=2,  # Minimum number of clients to be sampled for the next round
            min_available_clients=2,  # Minimum number of clients that need to be connected to the server before a training round
        )
    
        # start server and listen to 8080 using FedAvg strategy
        fl.server.start_server(servername, config={"num_rounds": 3}, strategy=strategy)
    except KeyError as err:
        print(f"No hostname specified - {err}")