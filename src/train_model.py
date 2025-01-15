# src/train_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
    
    def create_model(self, input_dim):
        """Create a deep neural network model"""
        model = Sequential([
            # Input layer
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            
            # Hidden layers
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model

    def load_and_prepare_data(self, data_path):
        """Load and prepare data for training"""
        # Load data
        df = pd.read_csv(data_path)
        
        # Create visualization directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        
        # Visualize class distribution
        plt.figure(figsize=(8, 6))
        df['fraud'].value_counts(normalize=True).plot(kind='bar')
        plt.title('Distribution of Fraud vs Normal Transactions')
        plt.savefig('visualizations/class_distribution.png')
        plt.close()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('visualizations/correlation_heatmap.png')
        plt.close()
        
        # Separate features and target
        X = df.drop('fraud', axis=1)
        y = df['fraud']
        
        # Split data into train, validation, and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE to training data only
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        return (X_train_balanced, y_train_balanced, 
                X_val_scaled, y_val,
                X_test_scaled, y_test)

    def train(self, data_path):
        """Train the model"""
        # Prepare data
        (X_train_balanced, y_train_balanced,
         X_val_scaled, y_val,
         X_test_scaled, y_test) = self.load_and_prepare_data(data_path)
        
        # Create model
        self.model = self.create_model(X_train_balanced.shape[1])
        
        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            'model/best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
        
        # Train model
        self.history = self.model.fit(
            X_train_balanced, y_train_balanced,
            validation_data=(X_val_scaled, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping, model_checkpoint]
        )
        
        # Plot training history
        self.plot_training_history()
        
        # Evaluate on test set
        test_results = self.model.evaluate(X_test_scaled, y_test)
        print("\nTest Set Results:")
        for metric, value in zip(self.model.metrics_names, test_results):
            print(f"{metric}: {value:.4f}")
        
        # Save the scaler
        with open('model/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def plot_training_history(self):
        """Plot training metrics"""
        metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
        plt.figure(figsize=(15, 10))
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 3, i)
            plt.plot(self.history.history[metric], label='Train')
            plt.plot(self.history.history[f'val_{metric}'], label='Validation')
            plt.title(f'Model {metric.capitalize()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/training_history.png')
        plt.close()

def main():
    # Create directories if they don't exist
    os.makedirs('model', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Initialize and train model
    fraud_detector = FraudDetectionModel()
    fraud_detector.train('data/card_transdata.csv')

if __name__ == "__main__":
    main()