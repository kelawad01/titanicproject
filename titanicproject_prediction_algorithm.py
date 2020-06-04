import sklearn
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import tree, model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import utils

traindf = pd.read_csv("train.csv")
utils.clean_data(traindf)

testdf = pd.read_csv("test.csv")
utils.clean_data(testdf)

target_train = traindf["Survived"].values
features_train = traindf[
    ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title"]
    ].values

features_test = testdf[
    ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title"]
    ].values

decision_tree = tree.DecisionTreeClassifier(
    random_state = 1
    )

decision_tree_fit = decision_tree.fit(features_train, target_train)

scores_tree = model_selection.cross_val_score(
    decision_tree, 
    features_train, 
    target_train, 
    scoring = "accuracy",
    cv = 50
    )

print(scores_tree)
print("Decision Tree Training Accuracy: ", scores_tree.mean())

rf = RandomForestClassifier(
    n_estimators = 10, 
    criterion = "entropy", 
    random_state = 0
    )

rf_fit = rf.fit(features_train, target_train)

scores_forest = model_selection.cross_val_score(
    rf,
    features_train,
    target_train,
    scoring = "accuracy",
    cv = 50
)

print(scores_forest)
print("Random Forest Training Accuracy: ", scores_forest.mean())

importances = rf.feature_importances_

print(importances)
rf_rank = np.argsort(importances)[::-1]
print(rf_rank)

tree_predict = decision_tree.predict(features_test)
rf_predict = rf.predict(features_test)