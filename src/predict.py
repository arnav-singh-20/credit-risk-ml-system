import pandas as pd


def predict(input_data: dict, model):

    # Convert input JSON → DataFrame
    df = pd.DataFrame([input_data])

    # Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    result = {
        "prediction": int(prediction),
        "risk_probability": float(probability)
    }

    return result
