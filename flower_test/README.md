# Flower test

## Plain FedAvg

### [First test](1-plain-avg/README.md)
- Central Authority (server)
- 2 NSOs (clients)
- FedAvg
- Same datasets

### [Second test](2-four-clients/README.md)
- Central Authority (server)
- 4 NSOs
- FedAvg
- Data splitted
- Save updated model

### [Third test](3-central-model/README.md)
- Same as 2nd test
- The model is sent by the CA

### [Fourth test](4-paillier-common-key/README.md)
- This setting is similar to the 2nd test, the first client to connect sends the initial weights but in this case they are encrypted
- Encrypted FedAvg using Paillier




