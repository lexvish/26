import streamlit as st
import numpy as np

st.title("International Security Neural Net Simulator")

st.write("""
A simple simulator where 20 countries are inputs.  
You can set each country's weight (GDP-based coefficient),  
choose a transfer function, and compute the network output.
""")

# --- Inputs (20 countries)
st.header("Inputs (Countries)")
inputs = []
weights = []
for i in range(20):
    col1, col2 = st.columns(2)
    with col1:
        x = st.number_input(f"Country {i+1} GDP input", value=1.0, step=0.1, key=f"x{i}")
    with col2:
        w = st.number_input(f"Weight for Country {i+1}", value=0.1, step=0.1, key=f"w{i}")
    inputs.append(x)
    weights.append(w)

inputs = np.array(inputs)
weights = np.array(weights)

# --- Transfer function selection
st.header("Transfer Function")
activation = st.selectbox("Choose activation function:",
                          ["linear", "sigmoid", "tanh", "ReLU"])

def transfer_fn(z, fn):
    if fn == "linear":
        return z
    elif fn == "sigmoid":
        return 1 / (1 + np.exp(-z))
    elif fn == "tanh":
        return np.tanh(z)
    elif fn == "ReLU":
        return np.maximum(0, z)

# --- Compute output
z = np.dot(inputs, weights)
output = transfer_fn(z, activation)

st.subheader("Result")
st.write(f"Weighted sum (z): {z:.4f}")
st.write(f"Output after {activation} activation: {output:.4f}")
