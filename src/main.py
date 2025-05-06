import os
import torch
import joblib
import numpy as np

from fastapi import FastAPI
from src.models.model import FraudDetectionModel

app = FastAPI()

model_params = joblib.load('artifacts/model_params.pkl')
model = FraudDetectionModel(model_params["input_size"], model_params["hidden_size"], model_params["output_size"])
latest_model_number = len(os.listdir("artifacts/model"))
model.load_state_dict(torch.load("artifacts/model/fraud_model_" + str(latest_model_number) + ".pth", map_location = torch.device('cpu'), weights_only = True))
model.eval()

@app.get("/")
def home():
    return "Welcome to Fraud Detection"

@app.post("/predict")
def predict(data: dict):
    min_max_scaler = joblib.load('artifacts/min_max_scaler.pkl')

    features = np.array(data["features"]).reshape(1, -1)
    features[0][-1] = min_max_scaler.transform(features[0][-1].reshape(1, -1))
    features = torch.tensor(features, dtype = torch.float32)

    prediction = model(features).item()
    
    return {"fraud_probability": prediction}