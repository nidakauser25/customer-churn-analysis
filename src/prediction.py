import joblib
import pandas as pd

model = joblib.load("models/churn_model.pkl")

def predict_churn(input_df):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    return prediction, probability