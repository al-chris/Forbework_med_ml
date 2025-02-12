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

def prepare_single_input(df_input, feature_selector):
    engineer = FeatureEngineer()
    
    # Create medical features first
    df_engineered = engineer.create_medical_features(df_input)
    
    # Get the exact features that were selected during training
    required_features = feature_selector.get_feature_names_out()
    
    # Prepare features (encode categorical variables)
    X_prepared = engineer.prepare_features(df_engineered)
    
    # Scale the features
    X_scaled = engineer.scale_features(X_prepared)
    
    # Drop outcome variable if present
    if 'Outcome Variable' in X_scaled.columns:
        X_scaled = X_scaled.drop('Outcome Variable', axis=1)
    
    # Select only the features that were used in training
    X_selected = X_scaled[required_features]
    
    return X_selected

def main():
    # Load feature selector
    fs_path = 'results/feature_selector.pkl'
    if not os.path.exists(fs_path):
        print("Feature selector not found. Run training to generate selector.")
        sys.exit(1)
    feature_selector = joblib.load(fs_path)
    
    # Get user input
    print("Please enter the values for a single patient's data:")
    df_input = get_user_input()
    print("\nInput data:")
    print(df_input)
    
    try:
        # Prepare input with exact same features as training
        X_final = prepare_single_input(df_input, feature_selector)
        
        # Load the model
        model_path = 'results/best_model.pkl'
        if not os.path.exists(model_path):
            print("Trained model file not found. Run training to generate model.")
            sys.exit(1)
        best_model = joblib.load(model_path)
        
        # Make prediction
        prediction = best_model.predict(X_final)[0]
        pred_label = 'Positive' if prediction == 1 else 'Negative'
        
        print("\nPrediction for the entered data:")
        print(pred_label)
        
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        print("\nDebug information:")
        print(f"Required features: {feature_selector.get_feature_names_out()}")
        print(f"Available features after preprocessing: {X_final.columns if 'X_final' in locals() else 'Not available'}")

if __name__ == "__main__":
    main()