import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/cleaned_customer_churn.csv")

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

# Store encoders
encoders = {}

# Encode categorical columns
for column in df.columns:
    if not pd.api.types.is_numeric_dtype(df[column]):
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        encoders[column] = le

# Save encoders
joblib.dump(encoders, "models/encoders.pkl")

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

best_model = None
best_accuracy = 0
best_model_name = ""

print("\nMODEL COMPARISON")
print("-" * 40)

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"{name}: {accuracy * 100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print("\nBEST MODEL")
print("-" * 40)
print(f"Model: {best_model_name}")
print(f"Accuracy: {best_accuracy * 100:.2f}%")

# Save best model
joblib.dump(best_model, "models/churn_model.pkl")

print("\nBest model saved successfully!")

# Feature Importance
if hasattr(best_model, "feature_importances_"):

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTOP 10 IMPORTANT FEATURES")
    print("-" * 40)

    print(importance_df.head(10))

    importance_df.to_csv(
        "models/feature_importance.csv",
        index=False
    )

    print("\nFeature importance saved!")