import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load Model and Label Encoder
model = joblib.load('fraud_xgb_model.pkl')
le = joblib.load('label_encoder.pkl')

# Feature Engineering Function 
def add_engineered_features(df):
    df = df.copy()
    required_cols = ['oldbalanceOrg', 'newbalanceOrig', 'newbalanceDest', 'oldbalanceDest']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    df['diffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['diffDest'] = df['newbalanceDest'] - df['oldbalanceDest']
    return df

# App Layout
st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("💳 Intelligent Fraud Detection System")
st.markdown("Use this app to detect whether a financial transaction is likely **fraudulent** or **legitimate**, and understand the reason behind the prediction.")

# Input Form 
with st.form("input_form"):
    st.markdown("### Input Transaction Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        step = st.number_input("Step (Hour)", min_value=1, max_value=744, value=10)
        amount = st.number_input("Amount", min_value=0.0, value=1000.0)
    with col2:
        oldbalanceOrg = st.number_input("Old Balance (Origin)", value=5000.0)
        newbalanceOrig = st.number_input("New Balance (Origin)", value=4000.0)
    with col3:
        oldbalanceDest = st.number_input("Old Balance (Destination)", value=2000.0)
        newbalanceDest = st.number_input("New Balance (Destination)", value=3000.0)

    type_str = st.selectbox("Transaction Type", le.classes_)
    submitted = st.form_submit_button("Predict")

# Prediction Logic 
if submitted:
    type_encoded = le.transform([type_str])[0]

    input_df = pd.DataFrame([{
        'step': step,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'type': type_encoded
    }])

    input_df = add_engineered_features(input_df)
    input_df = input_df[model.get_booster().feature_names]

    prediction = model.predict(input_df)[0]
    fraud_prob = model.predict_proba(input_df)[0][1]

    tab1, tab2 = st.tabs(["📊 Prediction Result", "🧠 Interpretation"])

    # Tab 1: Result 
    with tab1:
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown("### 📌 Prediction Summary")
            if prediction == 1:
                st.error(f"This transaction is likely **FRAUDULENT** (Confidence: {fraud_prob:.2%})")
            else:
                st.success(f"This transaction is likely **LEGITIMATE** (Confidence: {1 - fraud_prob:.2%})")

        with col2:
            st.markdown("### 📈 Fraud Probability Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fraud_prob * 100,
                number={'suffix': "%", 'font': {'size': 28}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if prediction == 1 else "green"},
                    'steps': [
                        {'range': [0, 50], 'color': "#c8e6c9"},
                        {'range': [50, 80], 'color': "#fff9c4"},
                        {'range': [80, 100], 'color': "#ffcdd2"}
                    ],
                },
                title={'text': "Fraud Probability", 'font': {'size': 14}}
            ))
            fig_gauge.update_layout(width=320, height=200, margin=dict(t=10, b=0, l=0, r=0))
            st.plotly_chart(fig_gauge)

    # Tab 2: Explanation 
    with tab2:
        st.markdown("###  Feature Contribution")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        with st.expander("📉 Waterfall Plot"):
            fig_waterfall, ax = plt.subplots(figsize=(3.5, 2.5))  
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value,
                shap_values[0],
                input_df.iloc[0],
                max_display=10
            )
            st.pyplot(fig_waterfall)

        st.markdown("###  Top Influencing Features")
        top_features = sorted(
            zip(input_df.columns, shap_values[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]

        for i, (feature, value) in enumerate(top_features, 1):
            direction = "↑" if value > 0 else "↓"
            st.write(f"{i}. **{feature}** ({direction} impact: {value:.2f})")