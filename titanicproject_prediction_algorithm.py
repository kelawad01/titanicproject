import sklearn
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import tree, model_selection
from sklearn.ensemble import RandomForestClassifier

import utils

traindf = pd.read_csv("train.csv")
utils.clean_data(traindf)

testdf = pd.read_csv("test.csv")
utils.clean_data(testdf)

target = traindf["Survived"].values
features = traindf[
    ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title"]
    ].values

decision_tree = tree.DecisionTreeClassifier(
    random_state = 1
    )

decision_tree_fit = decision_tree.fit(features, target)

scores_tree = model_selection.cross_val_score(
    decision_tree, 
    features, 
    target, 
    scoring = "accuracy",
    cv = 50
    )

print(scores_tree)
print(scores_tree.mean())

ran_forest = RandomForestClassifier(
    n_estimators = 10, 
    criterion = "entropy", 
    random_state = 0
    )

ran_forest.fit(features, target)

scores_forest = model_selection.cross_val_score(
    ran_forest,
    features,
    target,
    scoring = "accuracy",
    cv = 50
)

print(scores_forest)
print(scores_forest.mean())

importances = ran_forest.feature_importances_

print(importances)
ran_forest_rank = np.argsort(importances)[::-1]
print(ran_forest_rank)