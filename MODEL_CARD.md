# 📄 Model Card — Credit Risk Prediction Model

---

## 1. Model Overview

**Model Name:** Credit Risk Classification Model  
**Version:** v1.0  
**Author:** Arnav Singh  
**Task Type:** Binary Classification  
**Domain:** Financial Risk Assessment / Credit Scoring  

This model predicts whether a loan applicant represents a **credit risk** (high probability of default) or **non-risk** based on financial and demographic attributes.

The system is designed using **MLOps-oriented architecture**, separating preprocessing, modeling, evaluation, and inference pipelines.

---

## 2. Intended Use

### ✅ Primary Use Cases
- Loan approval decision support
- Risk flagging for banking systems
- Credit monitoring pipelines
- Financial risk analytics

### ❌ Out-of-Scope Uses
- Fully automated loan approval decisions without human review
- Legal or regulatory decision-making without auditing
- Use on populations significantly different from training data

---

## 3. Model Architecture

Pipeline:
Raw Data → Preprocessing → Feature Encoding → Model → Probability → Threshold Decision

### Algorithms Evaluated
- Linear Regression (baseline comparison)
- Logistic Regression
- Decision Tree Classifier

Final deployment model selected based on:
- Recall performance
- Calibration stability
- Generalization capability

---

## 4. Training Data

Dataset: **German Credit Risk Dataset**

### Features Include:
- Account status
- Credit history
- Loan duration
- Savings account balance
- Employment duration
- Loan amount
- Age
- Housing status

### Target Variable
Risk = 1 → High Credit Risk
Risk = 0 → Low Credit Risk

---

## 5. Data Preprocessing

Steps applied:

- Missing value handling
- Ordinal encoding for ordered categories
- One-hot encoding for nominal variables
- Feature scaling (where required)
- Pipeline encapsulation using sklearn

Preprocessing is saved inside the model pipeline to prevent **training-serving skew**.

---

## 6. Evaluation Metrics

Because credit risk is a **high-cost prediction problem**, accuracy alone is insufficient.

Primary metrics:

| Metric | Purpose |
|------|------|
| Recall | Minimize missed risky customers (False Negatives) |
| Precision | Control unnecessary loan rejection |
| ROC-AUC | Overall ranking ability |
| PR-AUC | Performance under class imbalance |

### Business Priority
Reduce False Negatives → Avoid granting loans to defaulters

---
## 7. Decision Threshold Strategy

The default probability threshold (0.5) was NOT used.

A custom threshold was selected to align with business risk tolerance:
Higher Threshold → Conservative lending
Lower Threshold → Aggressive lending
Threshold stored separately in: model_config.pkl
This enables business tuning without retraining the model.

---

## 8. Model Performance (Example)

| Model | Recall | Precision | ROC-AUC |
|------|------|------|------|
| Logistic Regression | High | Stable | Strong |
| Decision Tree | Very High Train Accuracy | Higher Variance | Moderate |
| Final Model | Balanced | Stable | Production-ready |

*(Exact values depend on retraining run)*

---

## 9. Ethical Considerations

Potential risks:

- Historical financial bias in dataset
- Socioeconomic proxy variables
- Over-rejection of certain applicant groups

Mitigation strategies:
- Model monitoring
- Threshold auditing
- Human-in-loop review

---

## 10. Limitations

- Dataset size relatively small
- Static training data
- No temporal drift modeling yet
- Requires periodic retraining

---

## 11. Monitoring Recommendations (MLOps)

Production monitoring should include:

- Prediction distribution drift
- Feature drift detection
- Recall degradation tracking
- Calibration monitoring
- False negative rate alerts

---

## 12. Reproducibility

Training can be reproduced using:

```bash
python src/train.py
Evaluation: python src/evaluate.py
Prediction:python src/predict.py


