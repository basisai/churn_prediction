# churn_prediction
[![Build Status](https://travis-ci.com/basisai/churn_prediction.svg?branch=master)](https://travis-ci.com/basisai/churn_prediction)
[![Python version](https://img.shields.io/badge/python-3.7-blue.svg)](https://shields.io/)

This example is an end-to-end Bedrock tutorial on AWS. It demonstrates how to use Bedrock to preprocess data, train a model and create a HTTP endpoint for serving. This example complements [Bedrock documentation](https://docs.basis-ai.com/guides/quickstart).

### Data exploration and Model prototyping
See [notebook](./doc/churn_prediction.ipynb)

### Data processing and model training flowchart
![flowchart](./doc/flow.png)

## Goals
At the end of the tutorial, the user will be able to
- set up a Bedrock training pipeline
- monitor the training
- deploy a model endpoint in HTTPS
- query the endpoint
- monitor the endpoint API metrics

## Run on Bedrock
Just follow the Bedrock [quickstart guide](https://docs.basis-ai.com/guides/quickstart). We have already prepared the data on S3 so that you do not have to upload any.

## Test your endpoint
After deploying your model as an endpoint, you can test it with the query below
```
curl -X POST \
  <MODEL_ENDPOINT_URL> \
  -H 'Content-Type: application/json' \
  -H 'X-Bedrock-Api-Token: <MODEL_ENDPOINT_TOKEN>' \
  -d '{"State": "ME", "Area_Code": 408, "Intl_Plan": 1, "VMail_Plan": 1, "VMail_Message": 21, "CustServ_Calls": 4, "Day_Mins": 156.5, "Day_Calls": 122, "Eve_Mins": 209.2, "Eve_Calls": 125, "Night_Mins": 158.7, "Night_Calls": 81, "Intl_Mins": 11.1, "Intl_Calls": 3}'
```
