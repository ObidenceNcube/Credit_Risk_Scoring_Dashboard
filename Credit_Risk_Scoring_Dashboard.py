import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
import shap
import matplotlib.pyplot as plt

# 1. CONFIGURATION AND INITIAL SETUP

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Credit Risk Dashboard", page_icon="🏦")

# Global variables for model and explainer (cached)
@st.cache_resource
def load_data_and_train_model():
    """Simulate data and train XGBoost model."""
    st.info("Simulating data and training the XGBoost model. This runs once.", icon="⚙️")

    # Seed for reproducibility
    np.random.seed(42)

    # Data Simulation
    n_samples = 10000
    data = {
        'Age': np.random.randint(20, 70, n_samples),
        'Income_k': np.random.randint(20, 200, n_samples),
        'Credit_Score': np.random.randint(500, 850, n_samples),
        'Loan_Amount_k': np.random.randint(5, 100, n_samples),
        'Term_Months': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'Employment_Years': np.random.randint(0, 30, n_samples),
        # Target variable (0: low risk/paid, 1: high risk/default)
        # Create risk based on low Credit Score, low Income, and high Loan Amount
        'Default': np.zeros(n_samples)
    }

    df = pd.DataFrame(data)

    # Introduce synthetic default risk (higher risk if CScore is low AND Loan is high)
    risk_condition = (df['Credit_Score'] < 600) | (df['Income_k'] < 40) | (df['Loan_Amount_k'] > 70)
    # Set approximately 15% of the high-risk group to default
    default_indices = df[risk_condition].sample(frac=0.3, random_state=42).index
    df.loc[default_indices, 'Default'] = 1

    # Split data
    X = df.drop('Default', axis=1)
    y = df['Default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Model Training (XGBoost)
    model = XGBClassifier(
        objective='binary:logistic',
        #use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    # SHAP Explainer Setup
    # Use the Test set background data for SHAP Explainer
    explainer = shap.TreeExplainer(model, data=X_test)

    return model, explainer, X_train, X_test, y_test

# Load resources
model, explainer, X_train, X_test, y_test = load_data_and_train_model()
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 2. MODEL PERFORMANCE METRICS

def display_model_metrics():
    """Calculate and display key model performance metrics."""
    st.header("1. Model Performance Overview")
    col1, col2, col3, col4 = st.columns(4)

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, model.predict(X_test)).ravel()
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    default_rate = y_test.mean() * 100

    col1.metric("Dataset Default Rate", f"{default_rate:.2f}%")
    col2.metric("ROC AUC Score (Test Set)", f"{auc:.4f}", delta="Higher is Better")
    col3.metric("Model Accuracy (Test Set)", f"{accuracy:.2f}")

    with col4:
        st.write("**Confusion Matrix**")
        st.markdown(f"""
        | Pred \\ True | Low Risk (0) | High Risk (1) |
        | :---: | :---: | :---: |
        | **Low Risk (0)** | {tn} (TN) | {fn} (FN) |
        | **High Risk (1)** | {fp} (FP) | {tp} (TP) |
        """)
        st.caption("A well-performing model minimizes FN (missed defaults).")


# 3. GLOBAL INTERPRETABILITY (SHAP Summary)

def display_global_interpretability(explainer, X_train):
    """Calculate and display global feature importance using SHAP."""
    st.header("2. Global Model Interpretability (SHAP Summary)")
    st.markdown("""
    This plot shows the *global* impact of each feature on the model's output.
    
    - **Position on X-axis:** Magnitude of the feature's impact on the prediction (SHAP value).
    - **Color:** Feature value (Red = High, Blue = Low).
    """)

    # Calculate SHAP values on the training set (a small sample is sufficient for speed)
    X_sample = X_train.sample(n=500, random_state=42)
    #shap_values = explainer.shap_values(X_sample)
    shap_values = explainer(X_sample)

    # Create the SHAP summary plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    ax.set_title("Feature Impact on Default Risk (Global)")
    plt.tight_layout()
    st.pyplot(fig)


# 4. INDIVIDUAL APPLICANT SCORING AND LOCAL INTERPRETABILITY

def display_local_scoring(model, explainer, feature_names):
    """Create input UI, score, and display local SHAP explanation."""
    st.header("3. Individual Applicant Scoring & Local Interpretation")

    # User Input Sidebar
    st.sidebar.header("Applicant Profile Input")
    st.sidebar.markdown("Use the sliders to define a new loan applicant's profile.")

    # Input Sliders
    age = st.sidebar.slider("Age (Years)", 20, 70, 35)
    income = st.sidebar.slider("Annual Income (K ZiG)", 20, 200, 75, step=5)
    credit_score = st.sidebar.slider("Credit Score (FICO-like)", 500, 850, 720)
    loan_amount = st.sidebar.slider("Loan Amount (K ZiG)", 5, 100, 25, step=5)
    term_months = st.sidebar.select_slider("Loan Term (Months)", options=[12, 24, 36, 48, 60], value=36)
    employment_years = st.sidebar.slider("Years Employed", 0, 30, 5)

    # Create a DataFrame for the new applicant
    applicant_data = pd.DataFrame({
        'Age': [age],
        'Income_k': [income],
        'Credit_Score': [credit_score],
        'Loan_Amount_k': [loan_amount],
        'Term_Months': [term_months],
        'Employment_Years': [employment_years] 
    })

    # Prediction
    prediction_proba = model.predict_proba(applicant_data)[0, 1]
    prediction = model.predict(applicant_data)[0]

    # Display Score
    st.subheader(f"Credit Risk Prediction for Applicant:")
    colA, colB = st.columns([1, 2])

    with colA:
        if prediction == 1:
            st.error(f"High Risk of Default ({prediction_proba:.2%})", icon="⚠️")
        else:
            st.success(f"Low Risk of Default ({prediction_proba:.2%})", icon="✅")

        # Threshold based recommendation
        threshold = 0.20 # Example threshold
        if prediction_proba > threshold:
            st.warning(f"Recommendation: **REJECT** (Risk > {threshold:.0%})")
        else:
            st.info(f"Recommendation: **APPROVE** (Risk <= {threshold:.0%})")


    # Local SHAP Interpretation
    with colB:
        st.subheader("Why this score? (Local SHAP Explanation)")
        st.markdown(
            "This plot shows how *each specific feature value* contributed to the final prediction."
        )

        # Calculate SHAP values for the single applicant
        #shap_values_local = explainer.shap_values(applicant_data)
        shap_values_local = explainer(applicant_data)

        # Force Plot / Waterfall Plot (Waterfall is generally better for single observations)
        # Note: shap.force_plot is complex to render in Streamlit. Using waterfall plot.
        fig_waterfall = plt.figure(figsize=(10, 6))
        # The base value (E[f(X)]) is the average model output across the background dataset
        # The output value (f(X)) is the log-odds of the predicted probability
        #shap.waterfall_plot(
            #shap.Explanation(
                #values=shap_values_local[0],
                #base_values=explainer.expected_value,
                #data=applicant_data.iloc[0].values,
                #feature_names=applicant_data.columns.tolist()
            #),
            #max_display=10,
            #show=False
        #)
        shap.plots.waterfall(shap_values_local[0], max_display=10, show=False)
        #st.pyplot(fig_waterfall, bbox_inches='tight')
        st.pyplot(fig_waterfall)
        st.caption("Features pushing the prediction higher (towards default) are red; lower (away from default) are blue.")


# 5. MAIN EXECUTION

def main():
    """Main function to orchestrate the dashboard."""
    st.title("🏦 Credit Risk Scoring Dashboard")
    st.markdown("---")

    # 1. Model Performance
    with st.container():
        display_model_metrics()
        st.markdown("---")

    # 2. Global Interpretability
    with st.container():
        display_global_interpretability(explainer, X_train)
        st.markdown("---")

    # 3. Local Scoring and Interpretation
    with st.container():
        display_local_scoring(model, explainer, X_train.columns.tolist())

# Run the app
if __name__ == "__main__":
    main()