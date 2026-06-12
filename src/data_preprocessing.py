import pandas as pd

# Load dataset
df = pd.read_csv("data/customer_churn.csv")

print("Original Shape:", df.shape)

# Remove customerID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

# Save cleaned dataset
df.to_csv("data/cleaned_customer_churn.csv", index=False)

print("\nCleaned dataset saved successfully!")