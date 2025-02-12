import sys
import os
import pandas as pd
import joblib

# Import necessary modules from the project
from feature_engineering import FeatureEngineer
from data_preprocessing import preprocess_data

def main():
    # Load the trained model and other saved states
    model_path = 'results/best_model.pkl'
    fs_path = 'results/feature_selector.pkl'
    engineer_state_path = 'results/engineer_state.pkl'

    # Check if all required files exist
    for path in [model_path, fs_path, engineer_state_path]:
        if not os.path.exists(path):
            print(f"Required file {path} not found. Run training first.")
            sys.exit(1)

    # Load model and states
    best_model = joblib.load(model_path)
    feature_selector = joblib.load(fs_path)
    engineer_state = joblib.load(engineer_state_path)
    
    # Load prediction data
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'dataset\\Disease_symptom_and_patient_profile_dataset.csv'
    data = pd.read_csv(input_file)
    print(data.head(5))
    
    # Initialize feature engineer with saved state
    engineer = FeatureEngineer()
    engineer.scaler = engineer_state['scaler']
    engineer.label_encoders = engineer_state['label_encoders']
    
    # Feature engineering
    data_engineered = engineer.create_medical_features(data)
    print("After feature engineering:")
    print(data_engineered.head())

    # Remove outcome variable if present
    if 'Outcome Variable' in data_engineered.columns:
        data_engineered = data_engineered.drop('Outcome Variable', axis=1)

    # Prepare features using saved label encoders
    X_prepared = engineer.prepare_features(data_engineered)
    print("\nAfter preparing features:")
    print(X_prepared.head())

    # Scale features using saved scaler
    X_scaled = pd.DataFrame(
        engineer.scaler.transform(X_prepared),
        columns=X_prepared.columns,
        index=X_prepared.index
    )
    print("\nAfter scaling:")
    print(X_scaled.head())

    # Get required features and select them
    required_features = feature_selector.get_feature_names_out()
    X_selected = X_scaled[required_features]

    print("\nFinal features:")
    print(X_selected)
    
    # Make predictions using the loaded model
    predictions = best_model.predict(X_selected)
    pred_labels = ['Positive' if p == 1 else 'Negative' for p in predictions]
    
    print("\nPredictions:")
    print(pred_labels)

    # If the model supports probability estimates, show confidence scores
    if hasattr(best_model, 'predict_proba'):
        probabilities = best_model.predict_proba(X_selected)
        print("\nConfidence scores:")
        for i, (prob_neg, prob_pos) in enumerate(probabilities):
            print(f"Sample {i + 1}: Negative: {prob_neg:.3f}, Positive: {prob_pos:.3f}")

if __name__ == "__main__":
    main()