# app.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
from src.preprocessor import CreditCardPreprocessor
import os

# Load the trained model and scaler
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model/best_model.h5')
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def main():
    st.title('Credit Card Fraud Detection System')
    st.write('Upload a credit card image or enter transaction details to check for potential fraud')
    
    # Load model
    model, scaler = load_model()
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Upload Credit Card Image')
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Credit Card', use_column_width=True)
            
            preprocessor = CreditCardPreprocessor()
            card_number = preprocessor.extract_card_number(image)
            
            if card_number:
                st.success(f"Detected Card Number: {card_number}")
                if preprocessor.validate_card_number(card_number):
                    st.info("Card number is valid")
                else:
                    st.warning("Card number is invalid")
            else:
                st.error("Could not detect card number from image")
    
    with col2:
        st.subheader('Enter Transaction Details')
        
        # Create user-friendly input fields
        transaction_amount = st.number_input('Transaction Amount ($)', min_value=0.0, value=100.0)
        
        distance_from_home = st.number_input(
            'Distance from Home Location (miles)',
            min_value=0.0,
            help="Approximate distance between transaction location and cardholder's home"
        )
        
        distance_from_last = st.number_input(
            'Distance from Last Transaction (miles)',
            min_value=0.0,
            help="Distance from the location of the last transaction"
        )
        
        ratio_to_median_purchase = st.slider(
            'Ratio to Median Purchase Price',
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            help="How does this purchase compare to your typical spending? (1.0 = typical)"
        )
        
        used_chip = st.selectbox(
            'Payment Method',
            options=['Chip', 'Swipe', 'Online'],
            help="How was the card used for this transaction?"
        )
        
        used_pin_number = st.checkbox(
            'PIN Number Used',
            help="Was a PIN number entered for this transaction?"
        )
        
        online_order = st.checkbox(
            'Online Order',
            help="Was this transaction made online?"
        )
        
    # Create a predict button
    if st.button('Check for Fraud'):
        # Prepare input data
        used_chip_encoded = 1 if used_chip == 'Chip' else 0
        used_pin_encoded = 1 if used_pin_number else 0
        online_order_encoded = 1 if online_order else 0
        
        input_data = np.array([[
            distance_from_home,
            distance_from_last,
            ratio_to_median_purchase,
            used_chip_encoded,
            used_pin_encoded,
            online_order_encoded,
            transaction_amount
        ]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction using Keras model
        prediction_prob = model.predict(input_scaled, verbose=0)  # Get probability
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
                '🟢 Normal' if transaction_amount < 1000 else '🔴 High',
                '🟢 Normal' if distance_from_home < 50 else '🔴 Unusual',
                '🟢 Normal' if distance_from_last < 100 else '🔴 Suspicious',
                '🟢 Typical' if 0.5 <= ratio_to_median_purchase <= 1.5 else '🔴 Unusual',
                '🟢 Secure' if used_chip == 'Chip' else '🔸 Less Secure',
                '🟢 Used' if used_pin_number else '🔸 Not Used',
                '🔸 Yes' if online_order else '🟢 No'
            ]
        })
        st.table(risk_factors)

if __name__ == '__main__':
    main()