import pandas as pd
import pickle

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------------
# Load Dataset
# -------------------------------------

df = pd.read_csv(
    "data/heart.csv"
)

# -------------------------------------
# Features and target
# -------------------------------------

X=df.drop(
    "target",
    axis=1
)

y=df["target"]

# -------------------------------------
# Split
# -------------------------------------

X_train,X_test,y_train,y_test=(
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

# -------------------------------------
# Hyperparameters
# -------------------------------------

params={

    "n_estimators":[
        50,
        100,
        200,
        300
    ],

    "max_depth":[
        5,
        10,
        15,
        None
    ],

    "min_samples_split":[
        2,
        5,
        10
    ],

    "min_samples_leaf":[
        1,
        2,
        4
    ],

    "max_features":[
        "sqrt",
        "log2"
    ]
}

rf=RandomForestClassifier()

search=RandomizedSearchCV(

    estimator=rf,

    param_distributions=params,

    n_iter=10,

    cv=5,

    scoring="accuracy",

    random_state=42,

    n_jobs=-1

)

search.fit(
    X_train,
    y_train
)

best_model=search.best_estimator_

# -------------------------------------
# Evaluate
# -------------------------------------

pred=best_model.predict(
    X_test
)

accuracy=accuracy_score(
    y_test,
    pred
)

print(
    "Accuracy:",
    accuracy
)

print(
    search.best_params_
)

# -------------------------------------
# Save model
# -------------------------------------

pickle.dump(

    best_model,

    open(
        "models/random_forest_model.pkl",
        "wb"
    )
)

pickle.dump(

    search.best_params_,

    open(
        "models/best_params.pkl",
        "wb"
    )
)

pickle.dump(

    {
        "accuracy":accuracy
    },

    open(
        "models/model_info.pkl",
        "wb"
    )
)

print(
    "Saved Successfully"
)