import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("data/processed/processed_data.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict
y_pred = model.predict(X)

# Evaluate
accuracy = accuracy_score(y, y_pred)

# Save metrics
with open("metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)

print(f"✅ Model evaluation complete. Accuracy: {accuracy:.2f}")
