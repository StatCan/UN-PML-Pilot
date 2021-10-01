# Flower test

## Plain FedAvg

### Second test
- Central Authority (server)
- 4 NSOs
- FedAvg
- Data splitted
- Save updated model (to do)
<img src="flower-4-clients.gif" width="800" />

## Server instructions
Please make sure that conda and git are installed.
1. Create the environment and activate it

        conda create --name=flower-test
        conda activate flower-test
2. Clone this project (please set SSH-Keys if available, or token authentication)

        git clone https://github.com/StatCan/UN-PML-Pilot.git
3. Install the libraries

        pip install flwr torch torchvision click
4. Change dir

        cd flower-test/2-four-clients
4. Export environment variable for servername and port and launch server 

        export HAR_SERVER=[::]:8080 python server.py

- Or using command line options:

        Usage: har-server.py [OPTIONS]

        Options:
          -s, --servername TEXT
          -m, --min_fit_clients INTEGER
          -M, --min_available_clients INTEGER
                                          [default: 4]
          -r, --number_of_rounds INTEGER  [default: 3]
          --help                          Show this message and exit.
Example:

        python har-server.py -s localhost:8080 -m 2 -M 2 -r3
<img src="flower-server.png" width="720" />        

## Client instructions
Please make sure that conda and git are installed.

1. Create the environment and activate it
        
        conda create --name=flower-test
        conda activate flower-test
2. Clone this project (please set SSH-Keys if available, or token authentication)
        
        git clone https://github.com/StatCan/UN-PML-Pilot.git
3. Install the libraries
        
        pip install flwr torch torchvision click
4. Change dir 
        
        cd flower-test/2-four-clients
4. Export environment variable for servername and port and launch client 
        
        HAR_SERVER=localhost:8080 TEST_PATH=path_to_test_dataset TRAIN_PATH=path_to_train_dataset python har-client.py
- Example:
        
        TEST_PATH=../../OUTPUT/3\ -\ STATCAN/test/3_ALL_test.csv TRAIN_PATH=../../OUTPUT/3\ -\ STATCAN/train/3_ALL_train.csv HAR_SERVER=localhost:8080 python har-client.py
        
- Or using command line options:

        Usage: har-client.py [OPTIONS]

        Load data, start HAR Client.

        Options:
          -s, --servername TEXT
          -T, --training_set TEXT
          -t, --test_set TEXT
          --help                   Show this message and exit.

- Example:

        python har-client.py -s localhost:8080 -T../../OUTPUT/3\ -\ STATCAN/train/3_ALL_train.csv -t../../OUTPUT/3\ -\ STATCAN/test/3_ALL_test.csv
<img src="flower-client.png" width="720" />
6. Because we need another client to start and finish the training, repeat steps 4-5 in another shell.






