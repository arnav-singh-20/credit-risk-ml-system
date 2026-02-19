import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load data
df = pd.read_csv("data/GermanCredit.csv")

X = df.drop("Risk", axis=1)
y = df["Risk"].map({"Good": 0, "Bad": 1})

# Load model
model = joblib.load("models/credit_risk_model.pkl")

# Predictions
y_pred = model.predict(X)

# Confusion matrix
cm = confusion_matrix(y, y_pred)

ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()
