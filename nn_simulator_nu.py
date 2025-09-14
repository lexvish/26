import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="International Security NN Simulator", layout="wide")
st.title("🌍 Neural Network Simulator for 20 Countries")

# -------------------------------
# 1. Define countries and their GDPs (in trillion USD)
# -------------------------------
countries = [
    "USA", "China", "Germany", "Russia", "UK", "France",
    "Japan", "India", "Canada", "Brazil", "Italy", "South Korea",
    "Australia", "Spain", "Mexico", "Netherlands", "Turkey", "Saudi Arabia",
    "Sweden", "Switzerland"
]

gdp_values = [
    27.721, 17.734, 4.659, 2.021, 3.070, 3.052, 4.026, 3.568, 2.229, 2.064,
    2.003, 1.803, 1.776, 1.531, 1.385, 1.269, 1.016, 0.794, 0.758, 0.673
]

# -------------------------------
# 2. Sidebar: user-adjustable country impacts
# -------------------------------
st.sidebar.header("Adjust Country Impact")
user_inputs = {}
for country, gdp in zip(countries, gdp_values):
    # scale GDP to roughly 0-1 for slider
    user_inputs[country] = st.sidebar.slider(
        f"{country} impact", 0.0, 1.0, float(gdp) / 30.0, 0.01
    )

# Normalize the updated weights
weights_updated = np.array(list(user_inputs.values()))
weights_normalized = weights_updated / np.sum(weights_updated)

# -------------------------------
# 3. Display normalized weights
# -------------------------------
st.subheader("Country Normalized Weights")
for country, weight in zip(countries, weights_normalized):
    st.write(f"**{country}** — Normalized Weight: {weight:.3f}")

# -------------------------------
# 4. Bar chart visualization (2 decimal digits)
# -------------------------------
df = pd.DataFrame({
    "Country": countries,
    "Normalized Weight": weights_normalized
})

# Format weights to 2 decimal digits for display
df["Normalized Weight Text"] = df["Normalized Weight"].apply(lambda x: f"{x:.2f}")

fig = px.bar(
    df,
    x="Country",
    y="Normalized Weight",
    text="Normalized Weight Text",
    color="Normalized Weight",
    color_continuous_scale="Viridis",
    title="Country Weight Distribution"
)
fig.update_layout(
    xaxis_title="Country",
    yaxis_title="Normalized Weight",
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 5. Simple neural network simulation
# -------------------------------
st.subheader("Neural Network Output Simulation")
inputs = np.random.rand(len(countries))  # Random input vector
output = np.dot(inputs, weights_normalized)
st.write(f"Simulated output (weighted sum): {output:.3f}")
