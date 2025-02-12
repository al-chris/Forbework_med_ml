import sys
import os
import pickle
import pandas as pd
import joblib

# Import necessary modules from the project
from feature_engineering import FeatureEngineer
from data_preprocessing import preprocess_data

def main():
    # Load the trained model
    model_path = 'results/best_model.pkl'
    if not os.path.exists(model_path):
        print("Trained model file not found. Train and save the model first.")
        sys.exit(1)
    # with open(model_path, 'rb') as f:
        # best_model = pickle.load(f)
    best_model = joblib.load(model_path)
    
    # Load prediction data (provide CSV file path as first argument or use default)
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'dataset\\Disease_symptom_and_patient_profile_dataset.csv'
    data = pd.read_csv(input_file)
    print(data.head(5))
    
    # Feature engineering and preprocessing pipeline
    engineer = FeatureEngineer()
    data_engineered = engineer.create_medical_features(data)
    # For prediction, is_training=False (target not available)
    X, _, _, _ = preprocess_data(data_engineered, is_training=False)
    X_prepared = engineer.prepare_features(X)
    X_scaled = engineer.scale_features(X_prepared)
    # After preprocessing but before feature selection
    print(X_scaled.columns)
    if 'Outcome Variable' in X_scaled.columns:
        X_scaled = X_scaled.drop('Outcome Variable', axis=1)
    # Load the saved feature selector and transform the scaled features
    fs_path = 'results/feature_selector.pkl'
    if not os.path.exists(fs_path):
        print("Feature selector not found. Run training to generate selector.")
        sys.exit(1)
    feature_selector = joblib.load(fs_path)
    X_selected = feature_selector.transform(X_scaled)
    
    # Make predictions using the loaded model
    predictions = best_model.predict(X_selected)
    # Map numeric predictions to labels
    pred_labels = ['Positive' if p == 1 else 'Negative' for p in predictions]
    
    print("Predictions:")
    print(pred_labels)

if __name__ == "__main__":
    main()
