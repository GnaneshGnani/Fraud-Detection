import torch
import joblib
import numpy as np

from fastapi import FastAPI
from model import FraudDetectionModel

app = FastAPI()

model = FraudDetectionModel()
model.load_state_dict(torch.load("fraud_model.pth", map_location = torch.device('cpu')))
model.eval()

@app.get("/")
def home():
    return "Welcome to Fraud Detection"

@app.post("/predict")
def predict(data: dict):
    # print(data)
    # return data

    min_max_scaler = joblib.load('min_max_scaler.pkl')

    features = np.array(data["features"]).reshape(1, -1)
    features[0][-1] = min_max_scaler.transform(features[0][-1].reshape(1, -1))
    features = torch.tensor(features, dtype = torch.float32)

    prediction = model(features).item()
    
    return {"fraud_probability": prediction}