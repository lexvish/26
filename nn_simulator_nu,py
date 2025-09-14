import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# --- SETTINGS ---
num_countries = 20
countries = [f"Country {i+1}" for i in range(num_countries)]

# --- INPUTS ---
st.title("Neural Network Simulator (20 Countries)")
st.write("Adjust weights and inputs, then see contributions.")

inputs = [st.slider(f"{c} Input", 0.0, 1.0, 0.5, 0.01) for c in countries]
weights = [st.slider(f"{c} Weight", -2.0, 2.0, 1.0, 0.01) for c in countries]

# --- COMPUTATION ---
inputs = np.array(inputs)
weights = np.array(weights)
weighted_inputs = inputs * weights
total_output = np.sum(weighted_inputs)

# Transfer function (sigmoid for demo)
output = 1 / (1 + np.exp(-total_output))

st.subheader("🔹 Output")
st.metric("Final Value (Sigmoid)", round(output, 4))

# --- VISUALIZATION DATA ---
df = pd.DataFrame({
    "Country": countries,
    "Input": inputs,
    "Weight": weights,
    "Contribution": weighted_inputs
})

# --- BAR CHART: Contributions ---
st.subheader("📊 Country Contributions (Bar Chart)")
fig_bar = px.bar(df, x="Country", y="Contribution", color="Contribution",
                 color_continuous_scale="RdBu", title="Contribution by Country")
st.plotly_chart(fig_bar, use_container_width=True)

# --- RADAR CHART: Inputs & Weights ---
st.subheader("🕸️ Radar Chart (Inputs vs Weights)")
radar_df = pd.DataFrame({
    "Country": countries,
    "Input": inputs,
    "Weight": weights
})

fig_radar = px.line_polar(radar_df, r="Input", theta="Country", line_close=True, name="Input")
fig_radar.add_scatterpolar(r=radar_df["Weight"], theta=radar_df["Country"], line_close=True, name="Weight")
fig_radar.update_traces(fill="toself")
st.plotly_chart(fig_radar, use_container_width=True)
