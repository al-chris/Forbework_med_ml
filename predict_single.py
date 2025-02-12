import sys
import os
import joblib
import pandas as pd
import io
import numpy as np

from feature_engineering import FeatureEngineer
from data_preprocessing import preprocess_data

def get_user_input():
    fields = {
        "Age": "Enter Age (number): ",
        "Blood Pressure": "Enter Blood Pressure (Low, Normal, High): ",
        "Cholesterol Level": "Enter Cholesterol Level (Low, Normal, High): ",
        "Fever": "Has Fever? (Yes/No): ",
        "Cough": "Has Cough? (Yes/No): ",
        "Fatigue": "Has Fatigue? (Yes/No): ",
        "Difficulty Breathing": "Has Difficulty Breathing? (Yes/No): ",
        "Disease": "Enter Disease (e.g., Flu): ",
        "Gender": "Enter Gender (Male/Female): "
    }
    data = {}
    for key, prompt_text in fields.items():
        data[key] = input(prompt_text)
    
    string_data = f"""Disease,Fever,Cough,Fatigue,Difficulty Breathing,Age,Gender,Blood Pressure,Cholesterol Level,Outcome Variable
{data["Disease"]},{data["Fever"]},{data["Cough"]},{data["Fatigue"]},{data["Difficulty Breathing"]},{data["Age"]},{data["Gender"]},{data["Blood Pressure"]},{data["Cholesterol Level"]},Negative"""

    df = pd.read_csv(io.StringIO(string_data))
    return df

def prepare_single_input(df_input, feature_selector, engineer_state=None):
    # Load the saved engineering state if provided
    if engineer_state is None:
        engineer_state = joblib.load('results/engineer_state.pkl')
    
    # Create a new engineer instance with saved state
    engineer = FeatureEngineer()
    engineer.scaler = engineer_state['scaler']
    engineer.label_encoders = engineer_state['label_encoders']
    
    # Create medical features
    df_engineered = engineer.create_medical_features(df_input)
    print("After feature engineering:")
    print(df_engineered)
    
    # Drop the Outcome Variable before feature preparation
    if 'Outcome Variable' in df_engineered.columns:
        df_engineered = df_engineered.drop('Outcome Variable', axis=1)
    
    # Prepare features using saved label encoders
    X_prepared = engineer.prepare_features(df_engineered)
    print("\nAfter preparing features:")
    print(X_prepared)
    
    # Scale features using saved scaler
    X_scaled = pd.DataFrame(
        engineer.scaler.transform(X_prepared),
        columns=X_prepared.columns,
        index=X_prepared.index
    )
    print("\nAfter scaling:")
    print(X_scaled)
    
    # Get required features
    required_features = feature_selector.get_feature_names_out()
    X_selected = X_scaled[required_features]
    print("\nFinal features:")
    print(X_selected)
    
    return X_selected

def main():
    try:
        # Load the saved states
        feature_selector = joblib.load('results/feature_selector.pkl')
        engineer_state = joblib.load('results/engineer_state.pkl')
        
        # Get user input
        print("Please enter the values for a single patient's data:")
        df_input = get_user_input()
        print("\nInput data:")
        print(df_input)
        
        # Prepare input with exact same features as training
        X_final = prepare_single_input(df_input, feature_selector, engineer_state)
        
        # Load the model
        model_path = 'results/best_model.pkl'
        best_model = joblib.load(model_path)
        
        # Make prediction
        prediction = best_model.predict(X_final)[0]
        pred_label = 'Positive' if prediction == 1 else 'Negative'
        
        print("\nPrediction for the entered data:")
        print(pred_label)
        
        # If the model supports probability estimates
        if hasattr(best_model, 'predict_proba'):
            pred_proba = best_model.predict_proba(X_final)[0]
            print(f"Confidence scores: Negative: {pred_proba[0]:.3f}, Positive: {pred_proba[1]:.3f}")
        
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        print("\nDebug information:")
        print(f"Required features: {feature_selector.get_feature_names_out()}")
        if 'X_final' in locals():
            print(f"Available features after preprocessing: {X_final.columns}")
            print("\nFeature values:")
            print(X_final)

if __name__ == "__main__":
    main()