import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load the correct balanced dataset
df = pd.read_csv("data/balanced_finance_dataset.csv")

# ✅ Define the 11 important input features
features = [
    "Income", "Age", "Dependents", "Occupation", "City_Tier",
    "Rent", "Groceries", "Transport", "Entertainment", "Utilities", "Healthcare"
]

# Set up input X and label y
X = df[features]
y = df["label"]  # This should be Saver / Balanced / Overspending

# Define which columns are categorical
categorical_cols = ["Occupation", "City_Tier"]

# Use one-hot encoding for categorical and passthrough for numeric
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder='passthrough')

# Build the pipeline: preprocessing + RandomForest
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=150, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the full pipeline
pipeline.fit(X_train, y_train)

# Evaluate accuracy
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and feature names
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/finance_model.pkl")
joblib.dump(features, "model/feature_names.pkl")
print("✅ Model and features saved to 'model/' folder.")
