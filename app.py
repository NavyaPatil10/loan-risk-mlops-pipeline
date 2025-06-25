import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Define the request schema
class InputData(BaseModel):
    features: list  # e.g., [1, 0, 35, 5000]

# Load the trained model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.get("/")
def read_root():
    return {"msg": "Loan Risk Model is Live 🎯"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.features])
    prediction = model.predict(df)[0]
    return {"prediction": str(prediction)}
