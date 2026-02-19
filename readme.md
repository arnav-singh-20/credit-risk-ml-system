# 🏦 Credit Risk Prediction System  
## End-to-End Machine Learning & MLOps Project

---

<p align="center">
Building a production-style Machine Learning system that predicts credit risk while aligning model decisions with real business objectives.
</p>

---

## 🚀 Overview

Financial institutions must decide whether a loan applicant is likely to default.  
While many ML projects focus only on model accuracy, real-world systems must balance:

- financial risk
- customer experience
- decision stability
- deployment reliability

This project builds a **complete ML pipeline** that moves beyond model training into **ML Engineering and MLOps thinking**.

The system predicts credit risk and demonstrates how machine learning integrates with real decision-making workflows.

---

## 🎯 Problem Statement

Given applicant financial and behavioral data, predict:

> **Will this customer become a credit risk?**

Business reality:

| Error Type | Impact |
|------------|--------|
| ❌ False Negative | Risky borrower approved → financial loss |
| ⚠ False Positive | Safe borrower rejected → customer dissatisfaction |

Therefore, the goal is **not maximum accuracy**, but:

> Detect risky customers reliably while maintaining operational balance.

---

## 🧠 Key Concepts Demonstrated

- Feature Engineering Pipelines
- Logistic Regression vs Decision Tree comparison
- Hyperparameter tuning (GridSearchCV)
- Recall-focused optimization
- Threshold tuning (Model vs Business Policy separation)
- Deployment-ready ML artifacts
- Reproducible ML workflow

---

## 🏗️ System Architecture

Raw Applicant Data
↓
Feature Engineering Pipeline
↓
Machine Learning Model
↓
Risk Probability Output
↓
Decision Layer (Threshold Policy)
↓
Loan Approval Decision


---

## 📊 Dataset

**German Credit Risk Dataset**

Each record represents a loan applicant including:

- checking account status
- credit history
- savings behavior
- employment duration
- loan amount & duration
- demographic indicators

Target Variable:
Good → Safe borrower (0)
Bad → Risky borrower (1)

---
## ⚙️ Feature Engineering Strategy

Feature preprocessing is implemented using **scikit-learn Pipelines** to guarantee consistency between training and inference.

### Numeric Features
- Median Imputation (robust for financial skew)
- Standard Scaling (optimization stability)

### Categorical Features
- Most Frequent Imputation
- One-Hot Encoding
- `handle_unknown="ignore"` for production safety

### Why Pipelines?

✔ Prevent data leakage  
✔ Ensure reproducibility  
✔ Enable deployment without manual preprocessing  
✔ Maintain identical transformations in production

---

## 🤖 Models Implemented
| Model | Purpose |
|------|---------|
| Logistic Regression | Stable interpretable baseline |
| Decision Tree | Captures nonlinear relationships |

---

## 🔍 Hyperparameter Optimization

Performed using:
GridSearchCV (5-Fold Cross Validation)

Optimized parameters:
- max_depth
- min_samples_leaf

Optimization metric:
Recall
Reason:
> Missing risky borrowers is more costly than rejecting safe ones.

---
## 🎚️ Threshold Optimization (Real Industry Practice)

Machine learning models output probabilities — not decisions.

Default threshold = `0.5` is arbitrary.

This project introduces a **Decision Layer**:

Model → Probability
Business Policy → Threshold

Benefits:

- adjust business strategy without retraining
- adapt to economic conditions
- separate ML from operational policy

---

## 📈 Evaluation Metrics

Accuracy alone is misleading for credit risk.

Primary metrics used:

- **Recall** → detect risky customers
- **Precision** → avoid excessive rejections
- **F1 Score** → balanced performance
- **Confusion Matrix** → business impact visualization

---

## 💾 Deployment-Ready Design

Saved artifacts:

models/
├── credit_risk_model.pkl → preprocessing + model pipeline
└── model_config.pkl → decision threshold

Model and business logic are intentionally separated.

---

## 📂 Project Structure

credit-risk-ml-system/
│
├── src/
│ ├── preprocess.py
│ ├── train.py
│ ├── evaluate.py
│ └── predict.py
│
├── data/
├── models/
├── reports/
├── requirements.txt
└── README.md

---

## ▶️ How to Run Locally

Install dependencies:
bash
pip install -r requirements.txt

Train model:
python src/train.py

Evaluate model:
python src/evaluate.py

Run prediction:
python src/predict.py

🧩 Engineering Insights

This project emphasizes how ML behaves in production systems:
Model performance must align with business cost.
Feature pipelines are mandatory for reproducibility.
Threshold tuning separates prediction from decision-making.
Stability often matters more than raw accuracy.

🚧 Future Improvements
Gradient Boosting / XGBoost comparison
Probability calibration
Data drift monitoring
FastAPI inference API
Automated retraining workflow

👨‍💻 Author
Arnav Singh
Aspiring ML Engineer focused on:
Machine Learning Systems
MLOps
Production-ready AI
Applied ML Engineering

⭐ If you found this project interesting, consider starring the repository!

