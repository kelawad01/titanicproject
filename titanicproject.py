import sklearn
import pandas as pd 
import numpy as np 
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

traindata = pd.read_csv("train.csv")
testdata = pd.read_csv("test.csv")

print(traindata.head())
print(testdata.head())