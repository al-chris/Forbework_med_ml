import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data, is_training=True):
    """
    Preprocess the patient data for model training or prediction.
    
    Args:
        data (pd.DataFrame): Patient data
        is_training (bool): Whether this is training data (with outcome) or prediction data
    Returns:
        Preprocessed features and optionally target variables
    """
    # Create a copy of the data
    df = data.copy()
    
    # Initialize label encoders for categorical variables
    le_disease = LabelEncoder()
    le_gender = LabelEncoder()
    
    # Convert binary variables to numeric
    binary_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
    for col in binary_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Encode categorical variables
    df['Disease'] = le_disease.fit_transform(df['Disease'])
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    
    # Convert Blood Pressure to numeric
    bp_map = {'Low': 0, 'Normal': 1, 'High': 2}
    df['Blood Pressure'] = df['Blood Pressure'].map(bp_map)
    
    # Convert Cholesterol Level to numeric
    chol_map = {'Low': 0, 'Normal': 1, 'High': 2}
    df['Cholesterol Level'] = df['Cholesterol Level'].map(chol_map)
    
    if is_training:
        # For training data, separate features and target
        X = df.drop('Outcome Variable', axis=1)
        y = df['Outcome Variable'].map({'Positive': 1, 'Negative': 0})
        return X, y, le_disease, le_gender
    else:
        # For prediction data, return only features
        return df, None, le_disease, le_gender