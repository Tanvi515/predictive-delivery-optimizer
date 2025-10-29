# app.py - Streamlit dashboard
import streamlit as st
import pandas as pd
import altair as alt
import joblib
from data import load_all
from features import merge_datasets, engineer_features
from utils import compute_kpis
from model import train_model, load_model, select_features

st.set_page_config(layout="wide", page_title="Predictive Delivery Optimizer")
st.title("🚚 Predictive Delivery Optimizer — NexGen Logistics")

DATA_DIR = "Case study internship data"

@st.cache_data
def load_data():
    return load_all(DATA_DIR)

data = load_data()
merged = merge_datasets(data)
df = engineer_features(merged)
kpis = compute_kpis(df)

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Orders", kpis['total_orders'])
c2.metric("On-time %", f"{kpis['on_time_pct']:.1f}%" if kpis['on_time_pct'] else "N/A")
c3.metric("Avg Delivery Cost", f"{kpis['avg_delivery_cost']:.2f}" if kpis['avg_delivery_cost'] else "N/A")
c4.metric("Avg Distance (km)", f"{kpis['avg_distance_km']:.1f}" if kpis['avg_distance_km'] else "N/A")

st.markdown("---")
st.header("📊 Delay Rate by Weekday")
if 'order_weekday' in df.columns:
    delay = df.groupby('order_weekday')['delay_flag'].mean().reset_index()
    chart = alt.Chart(delay).mark_bar().encode(x='order_weekday', y='delay_flag')
    st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.header("🧠 Model Training")
if st.button("Train baseline model"):
    try:
        metrics = train_model(df)
        st.success(f"Model trained! Accuracy: {metrics['accuracy']:.2f}")
    except Exception as e:
        st.error(f"Training failed: {e}")

st.markdown("---")
st.subheader("🔍 Predict Delay for a Single Order")
model = load_model()
order_id = st.text_input("Enter order_id:")
if order_id and model is not None:
    row = df[df['order_id'].astype(str) == str(order_id)]
    if len(row):
        X = row[select_features(row)].fillna(0)
        prob = model.predict_proba(X)[0][1]
        st.metric("Predicted Delay Probability", f"{prob:.2f}")
        if prob > 0.5:
            st.warning("⚠️ High chance of delay — suggest reassigning route.")
