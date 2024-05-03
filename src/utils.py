import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def prepare_data():
    train = pd.read_csv("data/train.csv")

    train["Age"] = train["Age"].fillna(0)
    train.loc[train["Embarked"].isnull(), "Embarked"] = "S"

    train['Age'] = train['Age'].astype(int)
    train['Pclass'] = train['Pclass'].astype(object)
    train['child'] = (train["Age"] <= 16).astype(int)

    train = train.drop(columns=["Cabin", 'Ticket', 'Fare', 'Name', 'PassengerId'])
    train["family_size"] = train["SibSp"] + train["Parch"] + 1
    train = train.drop(columns=["SibSp", "Parch"])

    train = pd.get_dummies(train, drop_first=False)

    return train


def train_model(train):
    X, y = train.drop("Survived", axis=1), train['Survived']

    pipeline = Pipeline([("rf_class", RandomForestClassifier(random_state=42))])

    param_dist = {
        "rf_class__n_estimators": np.arange(200, 2001, 200),
        "rf_class__max_features": ["auto", "sqrt", "log2"],
        "rf_class__max_depth": list(np.arange(10, 101, 10)) + [None],
        "rf_class__min_samples_split": [2, 5, 10],
        "rf_class__min_samples_leaf": [1, 2, 4, 8],
        "rf_class__bootstrap": [True, False]
    }

    rf = RandomizedSearchCV(
        pipeline,
        param_dist,
        cv=5,
        n_iter=50,
        n_jobs=4,
        verbose=1,
    )
    rf.fit(X, y)

    with open('rf_fitted.pkl', 'wb') as file:
        pickle.dump(rf, file)


def read_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not exists")

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model

