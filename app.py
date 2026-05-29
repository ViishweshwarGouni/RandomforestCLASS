import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide"
)

# -------------------------------
# Load files
# -------------------------------

model=pickle.load(

    open(
        "models/random_forest_model.pkl",
        "rb"
    )
)

params=pickle.load(

    open(
        "models/best_params.pkl",
        "rb"
    )
)

info=pickle.load(

    open(
        "models/model_info.pkl",
        "rb"
    )
)

df=pd.read_csv(
    "data/heart.csv"
)

X=df.drop(
    "target",
    axis=1
)

# -------------------------------
# Title
# -------------------------------

st.title(
    "❤️ Heart Disease Prediction"
)

st.write(
    "Random Forest Classifier"
)

# -------------------------------
# Metrics
# -------------------------------

c1,c2=st.columns(2)

c1.metric(
    "Accuracy",
    f"{info['accuracy']:.2%}"
)

c2.metric(
    "Trees",
    params["n_estimators"]
)

# -------------------------------
# Hyperparameters
# -------------------------------

st.subheader(
    "Best Hyperparameters"
)

st.json(
    params
)

# -------------------------------
# User Input
# -------------------------------

st.subheader(
    "Enter Patient Details"
)

input_data={}

for col in X.columns:

    value=st.number_input(

        col,

        value=float(
            X[col].mean()
        )
    )

    input_data[col]=value

# -------------------------------
# Prediction
# -------------------------------

if st.button(
    "Predict"
):

    input_df=pd.DataFrame(
        [input_data]
    )

    prediction=model.predict(
        input_df
    )

    probability=model.predict_proba(
        input_df
    )

    if prediction[0]==1:

        st.error(
            "⚠️ High Risk of Heart Disease"
        )

    else:

        st.success(
            "✅ Low Risk of Heart Disease"
        )

    st.write(

        f"Confidence: {max(probability[0])*100:.2f}%"

    )