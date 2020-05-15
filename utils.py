# Clean Data Function - Ju Liu from Predicting Titanic Survivors with Machine Learning Lecture

# This function replaces missing "Fare" data with the median fare value from the data set
# The function replaces missing "Age" data with the median age of all passengers in the data set
# This function changes categorical "Sex" and "Embarked" features into dummy numerics.

def clean_data(data):
    data.drop(["Cabin"], axis = 1, inplace = True)
    data["Title"] = data["Name"].str.extract(' ([A-Za-z]+)\.', expand = False)

    title_mapping = {
        "Mr":0,
        "Miss":1,
        "Mrs":2,
        "Master":3,
        "Dr":4,
        "Rev":5,
        "Col":6,
        "Mlle":7,
        "Major":8,
        "Ms":9,
        "Countess":10,
        "Mme":11,
        "Capt":12,
        "Don":13,
        "Lady":14,
        "Sir":15,
        "Jonkheer":16
    }

    data["Title"] = data["Title"].map(title_mapping)
    
    data.drop(["Name"], axis = 1, inplace = True)
    
    data["Age"].fillna(data.groupby("Title")["Age"].transform("median"), inplace = True)

    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1

    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2