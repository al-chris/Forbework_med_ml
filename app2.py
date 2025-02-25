import streamlit as st
import pandas as pd
import joblib
import io
import numpy as np
from feature_engineering import FeatureEngineer
from data_preprocessing import preprocess_data

def load_models():
    try:
        feature_selector = joblib.load('results/feature_selector.pkl')
        engineer_state = joblib.load('results/engineer_state.pkl')
        model = joblib.load('results/best_model.pkl')
        return feature_selector, engineer_state, model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def create_single_row_dataframe(inputs):
    string_data = f"""Disease,Fever,Cough,Fatigue,Difficulty Breathing,Age,Gender,Blood Pressure,Cholesterol Level,Outcome Variable
{inputs['Disease']},{inputs['Fever']},{inputs['Cough']},{inputs['Fatigue']},{inputs['Difficulty Breathing']},{inputs['Age']},{inputs['Gender']},{inputs['Blood Pressure']},{inputs['Cholesterol Level']},Negative"""
    return pd.read_csv(io.StringIO(string_data))

def prepare_single_input(df_input, feature_selector, engineer_state):
    try:
        # Create a new engineer instance with saved state
        engineer = FeatureEngineer()
        engineer.scaler = engineer_state['scaler']
        engineer.label_encoders = engineer_state['label_encoders']
        
        # Create medical features
        df_engineered = engineer.create_medical_features(df_input)
        
        # Drop the Outcome Variable before feature preparation
        if 'Outcome Variable' in df_engineered.columns:
            df_engineered = df_engineered.drop('Outcome Variable', axis=1)
        
        # Prepare features using saved label encoders
        X_prepared = engineer.prepare_features(df_engineered)
        
        # Scale features using saved scaler
        X_scaled = pd.DataFrame(
            engineer.scaler.transform(X_prepared),
            columns=X_prepared.columns,
            index=X_prepared.index
        )
        
        # Get required features
        required_features = feature_selector.get_feature_names_out()
        X_selected = X_scaled[required_features]
        
        return X_selected
    except Exception as e:
        st.error(f"Error in data preparation: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Patient Risk Prediction System",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Patient Risk Prediction System")
    st.write("Enter patient information to get a prediction.")
    
    # Load models
    feature_selector, engineer_state, model = load_models()
    if None in (feature_selector, engineer_state, model):
        st.error("Failed to load required models. Please check if all model files exist.")
        return

    # Create form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            disease = st.text_input("Disease", placeholder="e.g., Flu")
            age = st.number_input("Age", min_value=0, max_value=120, value=25)
            gender = st.selectbox("Gender", ["Male", "Female"])
            blood_pressure = st.selectbox("Blood Pressure", ["Low", "Normal", "High"])
            cholesterol = st.selectbox("Cholesterol Level", ["Low", "Normal", "High"])

        with col2:
            fever = st.selectbox("Fever", ["Yes", "No"])
            cough = st.selectbox("Cough", ["Yes", "No"])
            fatigue = st.selectbox("Fatigue", ["Yes", "No"])
            difficulty_breathing = st.selectbox("Difficulty Breathing", ["Yes", "No"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        if not disease:
            st.error("Please enter a disease.")
            return

        # Collect inputs
        inputs = {
            "Disease": disease,
            "Age": age,
            "Gender": gender,
            "Blood Pressure": blood_pressure,
            "Cholesterol Level": cholesterol,
            "Fever": fever,
            "Cough": cough,
            "Fatigue": fatigue,
            "Difficulty Breathing": difficulty_breathing
        }

        # Create dataframe
        df_input = create_single_row_dataframe(inputs)

        # Prepare input
        X_final = prepare_single_input(df_input, feature_selector, engineer_state)
        
        if X_final is not None:
            try:
                # Make prediction
                prediction = model.predict(X_final)[0]
                pred_label = 'Positive' if prediction == 1 else 'Negative'
                
                # Calculate probability if available
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X_final)[0]
                    
                    # Display results in a nice format
                    st.markdown("### Prediction Results")
                    
                    # Create three columns for the results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Prediction", pred_label)
                    
                    with col2:
                        st.metric("Negative Probability", f"{pred_proba[0]:.1%}")
                    
                    with col3:
                        st.metric("Positive Probability", f"{pred_proba[1]:.1%}")
                    
                    # Add confidence indicator
                    confidence = max(pred_proba)
                    st.progress(confidence)
                    st.caption(f"Prediction Confidence: {confidence:.1%}")
                
                else:
                    st.success(f"Prediction: {pred_label}")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()