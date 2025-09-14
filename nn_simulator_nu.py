# nn_simulator_nu.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="International Security NN Simulator", layout="wide")
st.title("🌍 Neural Network Simulator for 20 Countries")

# -------------------------------
# 1. Define countries and initial weights
# -------------------------------
countries = [
    "USA", "China", "Germany", "Russia", "UK", "France",
    "Japan", "India", "Canada", "Brazil", "Italy", "South Korea",
    "Australia", "Spain", "Mexico", "Netherlands", "Turkey", "Saudi Arabia",
    "Sweden", "Switzerland"
]

# Example GDP-based weight coefficients (normalized)
weights = np.array([
    0.25, 0.22, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02,
    0.015, 0.015, 0.01, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004
])

# -------------------------------
# 2. Country flags dictionary (PNG links)
# -------------------------------
flags = {
    "USA": "https://countryflagsapi.com/png/us",
    "China": "https://countryflagsapi.com/png/cn",
    "Germany": "https://countryflagsapi.com/png/de",
    "Russia": "https://countryflagsapi.com/png/ru",
    "UK": "https://countryflagsapi.com/png/gb",
    "France": "https://countryflagsapi.com/png/fr",
    "Japan": "https://countryflagsapi.com/png/jp",
    "India": "https://countryflagsapi.com/png/in",
    "Canada": "https://countryflagsapi.com/png/ca",
    "Brazil": "https://countryflagsapi.com/png/br",
    "Italy": "https://countryflagsapi.com/png/it",
    "South Korea": "https://countryflagsapi.com/png/kr",
    "Australia": "https://countryflagsapi.com/png/au",
    "Spain": "https://countryflagsapi.com/png/es",
    "Mexico": "https://countryflagsapi.com/png/mx",
    "Netherlands": "https://countryflagsapi.com/png/nl",
    "Turkey": "https://countryflagsapi.com/png/tr",
    "Saudi Arabia": "https://countryflagsapi.com/png/sa",
    "Sweden": "https://countryflagsapi.com/png/se",
    "Switzerland": "https://countryflagsapi.com/png/ch"
}

# -------------------------------
# 3. Sidebar: user-adjustable country impacts
# -------------------------------
st.sidebar.header("Adjust Country Impact")
user_inputs = {}
for country, weight in zip(countries, weights):
    user_inputs[country] = st.sidebar.slider(
        f"{country} impact", 0.0, 1.0, float(weight), 0.01
    )

# Normalize the updated weights
weights_updated = np.array(list(user_inputs.values()))
weights_normalized = weights_updated / np.sum(weights_updated)

# -------------------------------
# 4. Display flags + normalized weights
# -------------------------------
st.subheader("Country Weights with Flags")
for country, flag_url, weight in zip(countries, [flags[c] for c in countries], weights_normalized):
    cols = st.columns([1, 3])
    with cols[0]:
        st.image(flag_url, width=40)
    with cols[1]:
        st.write(f"**{country}** — Normalized Weight: {weight:.3f}")

# -------------------------------
# 5. Bar chart visualization
# -------------------------------
df = pd.DataFrame({
    "Country": countries,
    "Normalized Weight": weights_normalized
})

fig = px.bar(
    df,
    x="Country",
    y="Normalized Weight",
    text="Normalized Weight",
    color="Normalized Weight",
    color_continuous_scale="Viridis",
    title="Country Weight Distribution"
)
fig.update_layout(xaxis_title="Country", yaxis_title="Normalized Weight", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 6. Simple neural network simulation
# -------------------------------
st.subheader("Neural Network Output Simulation")
inputs = np.random.rand(len(countries))  # Random input vector
output = np.dot(inputs, weights_normalized)
st.write(f"Simulated output (weighted sum): {output:.3f}")
