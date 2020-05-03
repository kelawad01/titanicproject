import sklearn
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import tree, model_selection
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

import utils

traindf = pd.read_csv("train.csv")
utils.clean_data(traindf)


testdf = pd.read_csv("test.csv")
utils.clean_data(testdf)

target = traindf["Survived"].values
features = traindf[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].values

decision_tree = tree.DecisionTreeClassifier(
    random_state = 1
    )

decision_tree_fit = decision_tree.fit(features, target)

scores = model_selection.cross_val_score(
    decision_tree, 
    features, 
    target, 
    scoring = "accuracy",
    cv = 50)

# kfold = model_selection.KFold(
#     n_splits=10, 
#     shuffle = True, 
#     random_state = 0
#     )

print(scores)
print(scores.mean())

# clf = tree.DecisionTreeClassifier(
#     random_state = 1
#     )

# clf_fit = clf.fit(features, target)
# clf_predict = clf.predict(testdf)

# predictions = pd.DataFrame({
#     "PassengerId": test["PassengerId"],
#     "Survived": clf_predict
# })

# predictions.head()