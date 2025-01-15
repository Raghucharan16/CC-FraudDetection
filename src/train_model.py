import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle

def load_and_prepare_data():
    # Load the credit card fraud dataset
    # This dataset contains more interpretable features
    df = pd.read_csv('data/card_transdata.csv')
    
    print("Dataset Shape:", df.shape)
    print("\nFraud Distribution:")
    print(df['fraud'].value_counts(normalize=True))
    
    # Separate features and target
    X = df.drop('fraud', axis=1)
    y = df['fraud']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, scaler

def train_model():
    # Load and prepare data
    X_train_balanced, X_test_scaled, y_train_balanced, y_test, scaler = load_and_prepare_data()
    
    # Train the model with balanced dataset
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model and scaler
    with open('model/fraud_model.pkl', 'wb') as f:
        pickle.dump((model, scaler), f)

if __name__ == "__main__":
    train_model()