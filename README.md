# Recommender system note & exp
This repo will record the note about recommender system and some experiment code and result

### prerequisite
```
pip install -r requirement.txt
```

### Usage
```
cd exp/
python3 run_exp.py --dataset movielens --size 100k
```
## Experiment environment

### fold structure
* dataset
* preprocessing
* model
* evaulation
* utils


#### dataset

This folder contains the dataset preparing which including download the dataset and unzip the data set and load in pandas dataframe

#### preprocessing

This folder contains some utils to convert the dataset into format which model can accept and run

#### model

This folder collects all kinds of models which can fit the dataset and predict the result to evaluate

#### evaulation

This folder has some metric functions to evaluate the model result which including the rating metric and ranking metric

#### util

This is foldeer provide some utils for some exp to use.

