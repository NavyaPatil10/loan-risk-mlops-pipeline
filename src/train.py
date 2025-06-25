import pandas as pd
import pickle
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

# Load data
df = pd.read_csv("data/processed/processed_data.csv")

# Assume binary classification and last column is the target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"]
)

# Train model
model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"]
)
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model training complete.")
