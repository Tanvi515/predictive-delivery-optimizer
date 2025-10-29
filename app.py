# app.py - Streamlit dashboard (Final Enhanced Version)
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import joblib
from data import load_all
from features import merge_datasets, engineer_features
from utils import compute_kpis
from model import train_model, load_model, select_features

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(layout="wide", page_title="Predictive Delivery Optimizer")
st.title("🚚 Predictive Delivery Optimizer — NexGen Logistics")

DATA_DIR = "Case study internship data"

# -------------------------------
# Load and Prepare Data
# -------------------------------
@st.cache_data
def load_data():
    return load_all(DATA_DIR)

data = load_data()
merged = merge_datasets(data)
df = engineer_features(merged)
kpis = compute_kpis(df)

# -------------------------------
# KPI Summary
# -------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Orders", kpis['total_orders'])
c2.metric("On-time %", f"{kpis['on_time_pct']:.1f}%" if kpis['on_time_pct'] else "N/A")
c3.metric("Avg Delivery Cost", f"{kpis['avg_delivery_cost']:.2f}" if kpis['avg_delivery_cost'] else "N/A")
c4.metric("Avg Distance (km)", f"{kpis['avg_distance_km']:.1f}" if kpis['avg_distance_km'] else "N/A")

# -------------------------------
# Visualization 1 — Delay Rate by Weekday
# -------------------------------
st.markdown("---")
st.header("📊 Delay Rate by Weekday")
if 'order_weekday' in df.columns:
    delay = df.groupby('order_weekday')['delay_flag'].mean().reset_index()
    chart = alt.Chart(delay).mark_bar(color='#1f77b4').encode(
        x=alt.X('order_weekday', sort=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']),
        y=alt.Y('delay_flag', title='Average Delay Rate'),
        tooltip=['order_weekday', alt.Tooltip('delay_flag', format='.2f')]
    )
    st.altair_chart(chart, use_container_width=True)

# -------------------------------
# Visualization 2 — Delivery Cost vs Distance
# -------------------------------
st.markdown("---")
st.header("📈 Delivery Cost vs Distance (Colored by Delay)")
if 'distance_km' in df.columns and 'delivery_cost' in df.columns and 'delay_flag' in df.columns:
    fig1 = px.scatter(
        df, x='distance_km', y='delivery_cost', color='delay_flag',
        title="Delivery Cost vs Distance",
        labels={'distance_km': 'Distance (km)', 'delivery_cost': 'Delivery Cost (INR)', 'delay_flag': 'Delayed (1=Yes, 0=No)'},
        opacity=0.7
    )
    st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# Visualization 3 — Delivery Delay Distribution
# -------------------------------
st.markdown("---")
st.header("⏱️ Delivery Delay Distribution (Days)")
if 'eta_gap_days' in df.columns:
    fig2 = px.histogram(df, x='eta_gap_days', nbins=15,
                        title="Distribution of Delivery Delays",
                        labels={'eta_gap_days': 'Delay in Days'})
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Model Training Section
# -------------------------------
st.markdown("---")
st.header("🧠 Model Training")
if st.button("Train baseline model"):
    try:
        metrics = train_model(df)
        st.success(f"Model trained successfully! Accuracy: {metrics['accuracy']:.2f}")
    except Exception as e:
        st.error(f"Training failed: {e}")

# -------------------------------
# Prediction Section
# -------------------------------
st.markdown("---")
st.subheader("🔍 Predict Delay for a Single Order")

model = load_model()
order_id = st.text_input("Enter Order ID:")

if order_id and model is not None:
    row = df[df['order_id'].astype(str) == str(order_id)]
    if len(row):
        X = row[select_features(row)].fillna(0)
        prob = model.predict_proba(X)[0][1]
        st.metric("Predicted Delay Probability", f"{prob:.2f}")

        if prob > 0.5:
            st.warning("High chance of delay detected.")
        else:
            st.success("On-time delivery predicted.")

        # -------------------------------
        # Corrective Action Recommendation
        # -------------------------------
        st.markdown("### 🧩 Recommended Corrective Actions")

        def suggest_action(row):
            actions = []
            if 'distance_km' in row and 'traffic_delay' in row:
                if row['distance_km'].values[0] > 250 and row['traffic_delay'].values[0] > 60:
                    actions.append("Reassign to a faster carrier or alternate route.")
            if 'priority' in row and 'delivery_cost' in row:
                if str(row['priority'].values[0]).lower() == 'low' and row['delivery_cost'].values[0] > 5000:
                    actions.append("Group this with other low-priority deliveries to reduce costs.")
            if 'eta_gap_days' in row and row['eta_gap_days'].values[0] > 3:
                actions.append("Notify customer of delay and offer rescheduling options.")
            if not actions:
                actions.append("No corrective action required — delivery on track.")
            return actions

        recs = suggest_action(row)
        for a in recs:
            st.write("✅", a)

    else:
        st.error("Order ID not found in dataset.")
