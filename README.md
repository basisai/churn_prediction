# churn_prediction
[![Build Status](https://travis-ci.com/basisai/churn_prediction.svg?branch=master)](https://travis-ci.com/basisai/churn_prediction)
[![Python version](https://img.shields.io/badge/python-3.7-blue.svg)](https://shields.io/)

Preprocess raw data, train and serve ML model for customer churn data.

### Data exploration and Model prototyping
See [notebook](./doc/churn_prediction.ipynb)

### Data processing and model training flowchart
![flowchart](./doc/flow.png)

### Test your server
```
pip3 install -r requirements-serve.txt

# generate the grpc stubs
python3 -m grpc_tools.protoc -I ./protos --python_out=. --grpc_python_out=. serve.proto

# run the client
python3 client.py -e <Your endpoint>
```