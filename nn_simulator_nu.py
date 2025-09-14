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
# 2. Country flags dictionary
# -------------------------------
flags = {
    "USA": "https://flagcdn.com/us.png",
    "China": "https://flagcdn.com/cn.png",
    "Germany": "https://flagcdn.com/de.png",
    "Russia": "https://flagcdn.com/ru.png",
    "UK": "https://flagcdn.com/gb.png",
    "France": "https://flagcdn.com/fr.png",
    "Japan": "https://flagcdn.com/jp.png",
    "India": "https://flagcdn.com/in.png",
    "Canada": "https://flagcdn.com/ca.png",
    "Brazil": "https://flagcdn.com/br.png",
    "Italy": "https://flagcdn.com/it.png",
    "South Korea": "https://flagcdn.com/kr.png",
    "Australia": "https://flagcdn.com/au.png",
    "Spain": "https://flagcdn.com/es.png",
    "Mexico": "https://flagcdn.com/mx.png",
    "Netherlands": "https://flagcdn.com/nl.png",
    "Turkey": "https://flagcdn.com/tr.png",
    "Saudi Arabia": "https://flagcdn.com/sa.png",
    "Sweden": "https://flagcdn.com/se.png",
    "Switzerland": "https://flagcdn.com/ch.png"
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
for country, flag
