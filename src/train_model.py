# src/train_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import os

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
    
    def create_model(self, input_dim):
        """Create a deep neural network model with better probability distribution"""
        model = tf.keras.Sequential([
            # Input layer with L2 regularization
            tf.keras.layers.Dense(
                32, 
                activation='relu',
                input_dim=input_dim,
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            # Hidden layer 1
            tf.keras.layers.Dense(
                16, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            # Hidden layer 2 with smaller size
            tf.keras.layers.Dense(
                8, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            
            # Output layer with reduced complexity
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Use a lower learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model

    def train(self, data_path):
        # Load and prepare data
        df = pd.read_csv(data_path)
        
        # Separate features and target
        X = df.drop('fraud', axis=1)
        y = df['fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE with lower sampling_strategy
        smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Create minority class at 50% of majority
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        # Create and train model
        self.model = self.create_model(X_train_balanced.shape[1])
        
        # Add early stopping with higher patience
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train with class weights to handle imbalance
        class_weight = {0: 1., 1: 2.}  # Give more weight to fraud class
        
        self.history = self.model.fit(
            X_train_balanced,
            y_train_balanced,
            epochs=5,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping],
            class_weight=class_weight,
            verbose=1
        )
        
        # Save model and scaler
        self.model.save('model/best_model.h5')
        with open('model/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Test predictions
        test_pred_probs = self.model.predict(X_test_scaled)
        print("\nSample of prediction probabilities:")
        print(test_pred_probs[:10])

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    model = FraudDetectionModel()
    model.train('data/card_transdata.csv')