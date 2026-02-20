from fastapi import FastAPI
import joblib
from src.predict import predict

app = FastAPI()

model = joblib.load("models/credit_risk_model.pkl")


@app.get("/")
def home():
    return {"message": "Credit Risk Model Running"}
from src.schema import CreditInput


@app.post("/predict")
@app.post("/predict")
def make_prediction(data: CreditInput):

    result = predict(data.dict(), model)
    return result

