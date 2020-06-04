# Machine Learning Project: Predicting Deaths on the Titanic
# Restructuring and Exploratory analysis of training data

import sklearn
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import metrics
import utils

traindf = pd.read_csv("train.csv")
utils.clean_data(traindf)
testdf = pd.read_csv("test.csv")
utils.clean_data(testdf)

print(traindf.head())
print(traindf.shape)
print(traindf.describe())

print(traindf['Survived'].value_counts())
print(testdf.isna().sum())

# Exploratory Figures Part 1
# fig = plt.figure(figsize=(18,6))

# plt.subplot2grid((2,3), (0,0))
# traindf.Survived.value_counts(normalize = True).plot(kind = "bar", alpha = 0.5)
# plt.title("Survivor rate")

# plt.subplot2grid((2,3), (0,1))
# plt.scatter(traindf.Survived, traindf.Age, alpha = 0.1)
# plt.title("Survivors by Age")

# plt.subplot2grid((2,3), (0,2))
# traindf.Pclass.value_counts(normalize = True).plot(kind = "bar", alpha = 0.5)
# plt.title("Passenger Class")

# plt.subplot2grid((2,3), (1,0), colspan=2)
# for x in [1,2,3]:
#     traindf.Age[traindf.Pclass == x].plot(kind = "kde")
# plt.title("Passenger Class: Age Density") 
# plt.legend(("First Class", "Second Class", "Third Class")) 

# plt.subplot2grid((2,3), (1,2))
# traindf.Embarked.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
# plt.title("Embarked")

# # Exploratory Figures Part 2
# fig2 = plt.figure(figsize=(18,6))

# plt.subplot2grid((3,4), (0,0))
# traindf.Survived.value_counts(normalize = True).plot(kind = "bar", alpha = 0.5)
# plt.title("Survivor Counts")

# plt.subplot2grid((3,4), (0,1))
# traindf.Survived[traindf.Sex == "male"].value_counts(normalize = True).plot(kind = "bar", alpha = 0.5)
# plt.title("Male Survivors")

# plt.subplot2grid((3,4), (0,2))
# traindf.Survived[traindf.Sex == "female"].value_counts(normalize = True).plot(kind = "bar", alpha = 0.5)
# plt.title("Female Survivors")

# plt.subplot2grid((3,4), (1,0), colspan=3)
# for x in [1,2,3]:
#     traindf.Survived[traindf.Pclass == x].plot(kind = "kde")
# plt.title("Passenger Class: Survivor Density")
# plt.legend(("First", "Second", "Third"))

# plt.subplot2grid((3,4), (2, 0))
# traindf.Survived[(traindf.Sex == "male") & (traindf.Pclass == 1)].value_counts(normalize =True).plot(kind = "bar", alpha = 0.5)
# plt.title("First Class Male Survival")

# plt.subplot2grid((3,4), (2, 1))
# traindf.Survived[(traindf.Sex == "male") & (traindf.Pclass == 3)].value_counts(normalize =True).plot(kind = "bar", alpha = 0.5)
# plt.title("Third Class Male Survival")

# plt.subplot2grid((3,4), (2,2))
# traindf.Survived[(traindf.Sex == "female") & (traindf.Pclass == 1)].value_counts(normalize = True).plot(kind = "bar", alpha = 0.5)
# plt.title("First Class Female Survival")

# plt.subplot2grid((3,4), (2,3))
# traindf.Survived[(traindf.Sex == "female") & (traindf.Pclass == 3)].value_counts(normalize = True).plot(kind = "bar", alpha = 0.5)
# plt.title("Third Class Female Survival")

# plt.show()