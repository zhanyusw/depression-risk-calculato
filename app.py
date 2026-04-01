import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Precision Mental Health Calculator", layout="wide")

st.title("🧠 Precision Risk Calculator for Depressive Trajectories")
st.markdown("**Powered by Elastic Net and SHAP explainable AI.** This clinical tool provides individualized probability predictions and risk attribution for late-onset depressive trajectories in older adults with chronic conditions.")

@st.cache_resource
def load_assets():
    model = joblib.load('elastic_net_model.pkl') 
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.sidebar.header("📋 Patient Baseline Characteristics")

age = st.sidebar.number_input("Age", value=68, min_value=45, max_value=120)
gender = st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female (0)" if x==0 else "Male (1)")
srh = st.sidebar.slider("Self-rated health (1-5)", 1, 5, 3)
ADLS = st.sidebar.slider("ADLs Impairment (0-6)", 0, 6, 0)
IADLS = st.sidebar.slider("IADLs Impairment (0-8)", 0, 8, 0)
mobility = st.sidebar.slider("Physical mobility impairment (0-5)", 0, 5, 0)
slfmem = st.sidebar.slider("Self-reported memory (1-5)", 1, 5, 3)
income = st.sidebar.number_input("Total household income", value=30000)
ftrhlp = st.sidebar.selectbox("Expected care availability", options=[0, 1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
satlife = st.sidebar.slider("Life satisfaction (1-5)", 1, 5, 3)

# 核心数据矩阵
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'srh': [srh],
    'ADLS': [ADLS],
    'IADLS': [IADLS],
    'mobility': [mobility],
    'slfmem': [slfmem],
    'income': [income],
    'ftrhlp': [ftrhlp],
    'satlife': [satlife]
})

if st.button("🚀 Generate Individualized Risk Report"):
    input_scaled = scaler.transform(input_data)
    probas = model.predict_proba(input_scaled)[0] 
    
    st.subheader("📊 Trajectory Probabilities")
    col1, col2, col3 = st.columns(3)
    col1.metric("Class 1: Chronically High", f"{probas[0]*100:.1f}%")
    col2.metric("Class 2: Consistently Low", f"{probas[1]*100:.1f}%")
    col3.metric("Class 3: Rapidly Escalating Risk", f"{probas[2]*100:.1f}%", 
                delta="High Risk Warning" if probas[2]>0.3 else None, delta_color="inverse")
    
    st.progress(float(probas[2]), text="Class 3: Rapidly Escalating Risk Indicator")

    st.subheader("🔍 Individualized Risk Attribution (SHAP Waterfall Plot)")
    st.markdown("*Note: **Red bars** indicate factors driving the risk higher for the Rapidly Escalating trajectory (Class 3), while **blue bars** indicate protective factors lowering the risk.*")
    
    explainer = shap.LinearExplainer(model, scaler.mean_.reshape(1, -1))
    shap_values = explainer(input_scaled)
    
    feature_name_mapping = {
        'age': 'Age',
        'gender': 'Gender',
        'srh': 'Self-rated health',
        'ADLS': 'ADLs',
        'IADLS': 'IADLs',
        'mobility': 'Physical mobility',
        'slfmem': 'Self-reported memory',
        'income': 'Total household income',
        'ftrhlp': 'Expected care availability',
        'satlife': 'Life satisfaction'
    }
    
    shap_values.feature_names = [feature_name_mapping[col] for col in input_data.columns]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0, :, 2], show=False) 
    st.pyplot(fig)