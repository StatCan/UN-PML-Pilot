import flwr as fl

if __name__ == "__main__":
    # defines the strategy
    strategy = fl.server.strategy.FedAvg(                                                                                     
        #fraction_fit=1,  # 100% client availability
        min_fit_clients=2,  # Minimum number of clients to be sampled for the next round
        min_available_clients=2,  # Minimum number of clients that need to be connected to the server before a training round
    )
    
    # start server and listen to 8080 using FedAvg strategy
    fl.server.start_server("[::]:8080", config={"num_rounds": 3}, strategy=strategy)
