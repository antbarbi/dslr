# DSLR

42 - DataScience x Logistic Regression

## Introduction
This project consists of implementing linear classification model called logistic regression. It will introduce us to different ways of data visualization as well as different algorithms that is used to train the data. The goal of this project is to predict the house where each Hogwarts student belong to.

## Setup:
```
git clone git@github.com:antbarbi/dslr.git
cd dslr
pip3 install -r requirements.txt
```

## Project

### Stats Metrics
Display information for all numerical features:
```bash
python3 describe.py dataset_train.csv
```

### Training Phase
Train dataset with different algorithms:
```bash
python3 logreg_train.py dataset_train.csv # gradient descent
```
```bash
python3 logreg_train.py dataset_train.csv -c # display cost history
```
```bash
python3 logreg_train.py dataset_train.csv -sgd # stochastic gradient descent
```
```bash
python3 logreg_train.py dataset_train.csv -mbgd # stochastic gradient descent
```
As a result, we will obtain a `weights.json` file that will be used in the prediction phase.

### Prediction Phase
Predict the house of each student:
```bash
python3 logreg_predict.py dataset_test.csv weights.json
```

As a result, we will obtain a `houses.csv` file that will be compared to `dataset_truth.csv`:
```bash
python3 evaluate.py houses.csv dataset_truth.csv
```
