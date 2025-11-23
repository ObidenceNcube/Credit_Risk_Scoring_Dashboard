# Credit_Risk_Scoring_Dashboard
This project demonstrates an end-to-end workflow for building an interpretable ML model to predict loan default risk
It includes:

✔️ Data Simulation

Generates synthetic loan applicant data (age, income, credit score, loan amount, etc.)

Introduces realistic default risk patterns.

✔️ Model Training (XGBoost)

Trains an XGBClassifier on the simulated dataset.

Computes performance metrics including ROC AUC, accuracy, and confusion matrix.

✔️ Global Interpretability

Uses SHAP summary plots to visualise global feature importance and direction of influence.

✔️ Local Interpretability & Applicant Scoring

Interactive sidebar for entering a new applicant profile.

Predicts default probability + provides approval/rejection suggestion.

Generates local SHAP waterfall plots to explain the prediction.

**Live App Experience**

After running the app, you can explore:

- Model Performance Overview

- Default rate

- ROC AUC Score

- Model accuracy

- Confusion matrix

- Global Model Interpretability (SHAP)

- Feature impact visualised across the entire dataset

- Individual Applicant Scoring

- Adjustable sliders for applicant features

- Risk prediction with probability

- Approval/Reject recommendation

- Local SHAP explanation for transparency
