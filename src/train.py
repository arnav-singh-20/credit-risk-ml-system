import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor


# Load dataset
df = pd.read_csv("data/GermanCredit.csv")

X = df.drop("Risk", axis=1)
y = df["Risk"].map({"Good": 0, "Bad": 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Build preprocessing
preprocessor = build_preprocessor(X_train)

# Pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    "model__max_depth": [3, 5, 7, 10],
    "model__min_samples_leaf": [1, 5, 10]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="recall",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Save trained model
joblib.dump(best_model, "models/credit_risk_model.pkl")

# Save threshold config
config = {"threshold": 0.35}
joblib.dump(config, "models/model_config.pkl")

print("Training completed and model saved.")
