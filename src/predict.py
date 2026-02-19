import pandas as pd
import joblib

# Load model + config
model = joblib.load("models/credit_risk_model.pkl")
config = joblib.load("models/model_config.pkl")

threshold = config["threshold"]

# Example prediction
sample = pd.read_csv("data/GermanCredit.csv") \
            .drop("Risk", axis=1) \
            .iloc[[0]]

probability = model.predict_proba(sample)[:, 1][0]

prediction = int(probability >= threshold)

print("Risk Probability:", round(probability, 3))
print("Decision:", "Bad Risk" if prediction else "Good Risk")
