import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = tf.keras.models.load_model('saved_model/fraud_model.h5')
scaler = joblib.load('saved_model/scaler.pkl')

st.title("Credit Card Fraud Detection System")
st.markdown("Enter transaction details to check potential fraud")

# Input widgets
col1, col2 = st.columns(2)

with col1:
    distance_home = st.number_input("Distance from Home (miles)", min_value=0.0, format="%.4f")
    distance_last_trans = st.number_input("Distance from Last Transaction (miles)", min_value=0.0, format="%.4f")
    ratio_median = st.number_input("Ratio to Median Purchase Price", min_value=0.0, format="%.4f")
    transaction_amount = st.number_input("Transaction Amount ($)", min_value=0.0, format="%.2f")

with col2:
    repeat_retailer = st.selectbox("Repeat Retailer", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
    used_chip = st.selectbox("Used Chip", [1, 0], format_func=lambda x: 'Chip' if x == 1 else 'No Chip')
    used_pin = st.selectbox("Used PIN", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
    online_order = st.selectbox("Online Order", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Prepare input
input_data = np.array([[distance_home, distance_last_trans, ratio_median,
                       repeat_retailer, used_chip, used_pin, online_order]])

# Scale features
scaled_data = scaler.transform(input_data)

# Prediction
if st.button("Check Fraud Probability"):
    prediction_prob = model.predict(scaled_data, verbose=0)  # Get probability
    prediction = (prediction_prob >= 0.5).astype(int)[0][0]  # Convert to binary prediction
    probability = prediction_prob[0][0]  # Get the probability value
    
    # Show prediction
    st.subheader('Analysis Result')
    if prediction == 0:
        st.success(f'Transaction appears legitimate (Risk Score: {probability:.1%})')
    else:
        st.error(f'Potential fraudulent transaction detected! (Risk Score: {probability:.1%})')
    
    # Show risk factors
    st.subheader('Risk Factor Analysis')
    risk_factors = pd.DataFrame({
        'Factor': [
            'Transaction Amount',
            'Distance from Home',
            'Distance from Last Transaction',
            'Purchase Amount Pattern',
            'Payment Method',
            'PIN Usage',
            'Online Transaction'
        ],
        'Status': [
            '🟢 Normal' if transaction_amount < 10000 else '🔴 High',
            '🟢 Normal' if distance_home < 250 else '🔴 Unusual',
            '🟢 Normal' if distance_last_trans < 1000 else '🔴 Suspicious',
            '🟢 Typical' if 0.5 <= ratio_median <= 2.5 else '🔴 Unusual',
            '🟢 Secure' if used_chip == 'Chip' else '🔸 Less Secure',
            '🟢 Used' if used_pin else '🔸 Not Used',
            '🔸 Yes' if online_order else '🟢 No'
        ]
    })
    st.table(risk_factors)

# Add model info
st.sidebar.markdown("### Model Information")
st.sidebar.write("Deep Neural Network with:")
st.sidebar.write("- 3 Hidden Layers (128, 64, 32 neurons)")
st.sidebar.write("- Dropout Layers for Regularization")
st.sidebar.write("- Trained on 100,000 transactions")
st.sidebar.write("- Real-time fraud probability assessment")