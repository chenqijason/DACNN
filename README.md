# DACNN-PyTorch


# Requirements
* Python 3.8
* [PyTorch v1.8+](https://github.com/pytorch/pytorch)
* Numpy
* SciPy


# Configurations

#### Data

- PhoenixVisRecord.csv is raw data

- Datasets are organized into 2 separate files: **_train.txt_** and **_test.txt_**

- Same to other data format for recommendation, each file contains a collection of triplets:

  > user item rating

  The only difference is the triplets are organized in chronological order.



# Usage
1. Install required packages.
2. prepross the raw data
3. run <code>python dataprocess.py<code>
4. train the model and evalute the performance
5. run <code>python train_caser.py<code>
