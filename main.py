import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocess_data
from model import PatientRiskPredictor
from evaluate import evaluate_model
import os
from datetime import datetime, timezone

def main():
    # Create output directory for results
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    data = pd.read_csv('dataset\\Disease_symptom_and_patient_profile_dataset.csv')
    
    # Preprocess the data
    X, y, le_disease, le_gender = preprocess_data(data)
    
    # Initialize the model
    predictor = PatientRiskPredictor()
    
    # Perform cross-validation
    cv_results = predictor.cross_validate(X, y)
    
    # Save cross-validation results
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    
    # Plot cross-validation results
    predictor.cv.plot_cv_results(
        cv_results,
        save_path=os.path.join(output_dir, f'cv_results_{timestamp}.png')
    )
    
    # Generate and save CV report
    cv_report = predictor.cv.generate_cv_report(cv_results)
    with open(os.path.join(output_dir, f'cv_report_{timestamp}.txt'), 'w') as f:
        f.write(cv_report)
    
    # Train final model on full dataset
    predictor.train(X, y)
    
    # Save the model
    predictor.save_model(os.path.join(output_dir, 'patient_risk_model.joblib'))
    
    # Print feature importance
    feature_importance = predictor.get_feature_importance(X.columns)
    print("\nFeature Importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")

if __name__ == "__main__":
    main()