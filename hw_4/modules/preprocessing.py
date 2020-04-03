'''Filling missing values and so on '''

import pandas as pd
from utils.constants_and_functions import time_format, prepare_dataset, ultimate_dataset

train = pd.read_csv('../dataset/dataset_raw_train.csv', sep=';')
test = pd.read_csv('../dataset/dataset_raw_test.csv', sep=';')
print(train.shape, test.shape)


prepare_dataset(dataset=train, dataset_type='train')
prepare_dataset(dataset=test, dataset_type='test')

train_new = pd.read_csv('../dataset/dataset_train.csv', sep=';')
test_new = pd.read_csv('../dataset/dataset_test.csv', sep=';')

print(train_new.info())
print(test_new.info())
