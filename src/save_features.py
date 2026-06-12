import pandas as pd
import joblib

df = pd.read_csv("data/cleaned_customer_churn.csv")

df = df.drop("Churn", axis=1)

joblib.dump(list(df.columns), "models/features.pkl")

print("Features saved successfully!")