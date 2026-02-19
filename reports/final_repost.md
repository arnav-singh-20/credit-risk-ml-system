# Final Project Report

## Objective
Predict credit risk using machine learning aligned with banking decisions.

## Pipeline
- Median imputation
- One-hot encoding
- Standard scaling
- Decision Tree model

## Optimization
GridSearchCV with Recall scoring.

## Business Goal
Minimize False Negatives (missed risky borrowers).

## Threshold Strategy
Threshold chosen separately from model to allow policy adjustment without retraining.

## Deployment Design
Model and preprocessing stored as single pipeline artifact.
