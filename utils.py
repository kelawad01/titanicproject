# Clean Data Function - Ju Liu from Predicting Titanic Survivors with Machine Learning Lecture

# This function replaces missing "Fare" data with the median fare value from the data set
# The function replaces missing "Age" data with the median age of all passengers in the data set
# This function changes categorical "Sex" and "Embarked" features into dummy numerics.

def clean_data(data):
    data["Fare"] = data ["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())

    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1

    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2

def transpose_matrix(data)