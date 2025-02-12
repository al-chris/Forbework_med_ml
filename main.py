from data_preprocessing import preprocess_data
from feature_engineering import FeatureEngineer
from model_selection import ModelSelector
import pandas as pd
import os
from datetime import datetime

def main():
    # Create output directory
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading data...")
    data = pd.read_csv('dataset\\Disease_symptom_and_patient_profile_dataset.csv')
    
    # Feature engineering
    print("\nPerforming feature engineering...")
    engineer = FeatureEngineer()
    
    # Create medical features
    data_engineered = engineer.create_medical_features(data)
    
    # Preprocess the engineered data
    X, y, le_disease, le_gender = preprocess_data(data_engineered, is_training=True)
    
    # Prepare features (encode categorical variables)
    print("Preparing features...")
    X_prepared = engineer.prepare_features(X)
    
    # Scale features
    print("Scaling features...")
    X_scaled = engineer.scale_features(X_prepared)
    
    # Select important features
    print("Selecting features...")
    X_selected = engineer.select_features(X_scaled, y, method='mutual_info')
    
    # Use selected features for model selection
    print("\nPerforming model selection...")
    selector = ModelSelector()
    
    # Find best model using selected features
    best_model, model_performances = selector.select_best_model(X_selected, y)
    
    # Save the feature selector for later use in prediction
    import joblib
    fs_path = os.path.join(output_dir, 'feature_selector.pkl')
    joblib.dump(engineer.feature_selector, fs_path)
    
    # Save results
    print("\nSaving results...")
    selector.save_model_results(output_dir)

    # Save the best model
    model_path = os.path.join(output_dir, 'best_model.pkl')
    selector.save_model(model_path)
    
    print(f"\nBest model: {type(best_model).__name__}")
    print(f"Best score: {selector.best_score:.4f}")

if __name__ == "__main__":
    main()