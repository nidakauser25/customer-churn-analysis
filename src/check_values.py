import pandas as pd

df = pd.read_csv("data/cleaned_customer_churn.csv")

print(df["gender"].unique())
print(df["Partner"].unique())
print(df["Dependents"].unique())
print(df["Contract"].unique())
print(df["InternetService"].unique())