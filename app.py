import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import plotly.express as px
import plotly.graph_objects as go
from model import PatientRiskPredictor
from data_preprocessing import preprocess_data
from feature_engineering import FeatureEngineer
from model_selection import ModelSelector

# Constants
CURRENT_USER = "al-chris"
CURRENT_UTC = "2025-02-11 14:23:36"

st.set_page_config(
    page_title="Patient Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

class PatientRiskApp:
    def __init__(self):
        self.initialize_session_state()
        
    @staticmethod
    def initialize_session_state():
        """Initialize session state variables"""
        if 'predictor' not in st.session_state:
            st.session_state.predictor = PatientRiskPredictor()
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        if 'feature_engineer' not in st.session_state:
            st.session_state.feature_engineer = FeatureEngineer()
        if 'model_selector' not in st.session_state:
            st.session_state.model_selector = ModelSelector()
        if 'last_features' not in st.session_state:
            st.session_state.last_features = None
            
    def render_header(self):
        """Render the header section of the app"""
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("🏥 Patient Risk Prediction System")
            st.markdown("Advanced ML-powered healthcare risk prediction")
        with col2:
            st.text(f"Current user: {CURRENT_USER}")
            st.text(f"UTC: {CURRENT_UTC}")
            if st.session_state.model_trained:
                st.success("Model Status: Trained")
            else:
                st.warning("Model Status: Not Trained")
            
    def render_sidebar(self):
        """Render the sidebar"""
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/hospital-3.png")
            st.title("Navigation")
            
            # Add model selection in sidebar
            if st.session_state.model_trained:
                st.subheader("Model Settings")
                selected_model = st.selectbox(
                    "Select Model",
                    list(st.session_state.model_selector.model_performances.keys()),
                    index=0
                )
                if selected_model:
                    model_info = st.session_state.model_selector.model_performances[selected_model]
                    st.metric("Model Score", f"{model_info['score']:.4f}")
            
            return st.radio(
                "Choose a page",
                ["Home", "Train Model", "Make Predictions", "Model Analysis"]
            )

    def render_train_model(self):
        """Render the model training page"""
        st.header("Train Model")
        
        # Add feature engineering options
        with st.expander("Feature Engineering Settings"):
            scaling_method = st.selectbox(
                "Scaling Method",
                ["standard", "minmax"],
                help="Choose how to scale the features"
            )
            
            feature_selection = st.checkbox(
                "Enable Feature Selection",
                value=True,
                help="Select most important features"
            )
            
            if feature_selection:
                selection_method = st.selectbox(
                    "Feature Selection Method",
                    ["mutual_info", "f_score"]
                )
                
            poly_features = st.checkbox(
                "Enable Polynomial Features",
                value=False,
                help="Create polynomial feature combinations"
            )
            
            if poly_features:
                poly_degree = st.slider(
                    "Polynomial Degree",
                    min_value=2,
                    max_value=3,
                    value=2
                )
        
        # Add model selection options
        with st.expander("Model Selection Settings"):
            search_method = st.selectbox(
                "Search Method",
                ["grid", "random"],
                help="Choose how to search for best parameters"
            )
            
            cv_folds = st.slider(
                "Cross-validation Folds",
                min_value=3,
                max_value=10,
                value=5
            )
            
            if search_method == "random":
                n_iter = st.slider(
                    "Number of Iterations",
                    min_value=10,
                    max_value=100,
                    value=20
                )
        
        uploaded_file = st.file_uploader("Upload training data (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_columns = [
                    'Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
                    'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Outcome Variable'
                ]
                
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    st.info("Please ensure your CSV file contains all required columns.")
                    return
                    
                st.success("Data uploaded successfully!")
                
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                st.subheader("Data Statistics")
                st.dataframe(data.describe())
                
                # Add data validation
                st.subheader("Data Validation")
                validation_passed = True
                
                # Check for missing values
                missing_values = data.isnull().sum()
                if missing_values.sum() > 0:
                    st.warning("Found missing values in the following columns:")
                    st.write(missing_values[missing_values > 0])
                    validation_passed = False
                
                # Check binary columns
                binary_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
                for col in binary_columns:
                    invalid_values = data[~data[col].isin(['Yes', 'No'])][col].unique()
                    if len(invalid_values) > 0:
                        st.warning(f"Invalid values in {col}: {invalid_values}")
                        st.info(f"{col} should only contain 'Yes' or 'No'")
                        validation_passed = False
                
                # Check age range
                if (data['Age'] < 0).any() or (data['Age'] > 120).any():
                    st.warning("Age values should be between 0 and 120")
                    validation_passed = False
                
                if not validation_passed:
                    st.error("Please fix the data issues before training the model")
                    return
                
                if st.button("Train Model"):
                    with st.spinner("Training model..."):
                        try:
                            # Create progress container
                            progress_container = st.empty()
                            
                            # Feature engineering
                            progress_container.text("Performing feature engineering...")
                            data_engineered = st.session_state.feature_engineer.create_medical_features(data)
                            
                            # Preprocess data
                            progress_container.text("Preprocessing data...")
                            X, y, le_disease, le_gender = preprocess_data(data_engineered, is_training=True)
                            
                            # Prepare features
                            progress_container.text("Preparing features...")
                            X_prepared = st.session_state.feature_engineer.prepare_features(X)
                            
                            # Scale features
                            progress_container.text("Scaling features...")
                            X_scaled = st.session_state.feature_engineer.scale_features(X_prepared, method=scaling_method)
                            
                            # Feature selection if enabled
                            if feature_selection:
                                progress_container.text("Selecting features...")
                                X_selected = st.session_state.feature_engineer.select_features(
                                    X_scaled, y, method=selection_method
                                )
                            else:
                                X_selected = X_scaled
                            
                            # Polynomial features if enabled
                            if poly_features:
                                progress_container.text("Creating polynomial features...")
                                X_final = st.session_state.feature_engineer.create_polynomial_features(
                                    X_selected, degree=poly_degree
                                )
                            else:
                                X_final = X_selected
                            
                            # Store feature names
                            st.session_state.last_features = X_final.columns
                            
                            # Model selection
                            progress_container.text("Selecting best model...")
                            best_model, model_performances = st.session_state.model_selector.select_best_model(
                                X_final, y, cv=cv_folds
                            )
                            
                            # Store the best model
                            st.session_state.predictor.model = best_model
                            st.session_state.model_trained = True
                            
                            # Clear progress container
                            progress_container.empty()
                            
                            st.success("Model trained successfully!")
                            
                            # Display model performance summary
                            st.subheader("Model Performance Summary")
                            for model_name, info in model_performances.items():
                                st.metric(
                                    label=model_name,
                                    value=f"{info['score']:.4f}",
                                    delta=f"{info['score'] - st.session_state.model_selector.best_score:.4f}"
                                )
                            
                        except Exception as e:
                            st.error(f"Error during model training: {str(e)}")
                            st.info("Please ensure your data is formatted correctly and try again.")
                            
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                st.info("Please ensure your file is a valid CSV format.")

    def render_predictions(self):
        """Render the predictions page"""
        st.header("Make Predictions")
        
        if not st.session_state.model_trained:
            st.warning("Please train the model first!")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            disease = st.selectbox("Disease", ["Influenza", "Common Cold", "Eczema"])
            fever = st.selectbox("Fever", ["Yes", "No"])
            cough = st.selectbox("Cough", ["Yes", "No"])
            fatigue = st.selectbox("Fatigue", ["Yes", "No"])
            difficulty_breathing = st.selectbox("Difficulty Breathing", ["Yes", "No"])
            
        with col2:
            age = st.number_input("Age", min_value=0, max_value=120)
            gender = st.selectbox("Gender", ["Male", "Female"])
            blood_pressure = st.selectbox("Blood Pressure", ["Low", "Normal", "High"])
            cholesterol = st.selectbox("Cholesterol Level", ["Low", "Normal", "High"])
            
        if st.button("Predict"):
            # Create input data
            input_data = pd.DataFrame({
                'Disease': [disease],
                'Fever': [fever],
                'Cough': [cough],
                'Fatigue': [fatigue],
                'Difficulty Breathing': [difficulty_breathing],
                'Age': [age],
                'Gender': [gender],
                'Blood Pressure': [blood_pressure],
                'Cholesterol Level': [cholesterol]
            })
            
            try:
                # Feature engineering for prediction
                input_engineered = st.session_state.feature_engineer.create_medical_features(input_data)
                
                # Preprocess input
                X, _, _, _ = preprocess_data(input_engineered, is_training=False)
                
                # Prepare features
                X_prepared = st.session_state.feature_engineer.prepare_features(X)
                
                # Scale features
                X_scaled = st.session_state.feature_engineer.scale_features(X_prepared)
                
                # Apply same feature selection if it was used
                if hasattr(st.session_state.feature_engineer, 'feature_selector') and \
                   st.session_state.feature_engineer.feature_selector is not None:
                    X_selected = pd.DataFrame(
                        st.session_state.feature_engineer.feature_selector.transform(X_scaled),
                        columns=st.session_state.feature_engineer.selected_features
                    )
                else:
                    X_selected = X_scaled
                
                # Apply polynomial features if they were used
                if hasattr(st.session_state.feature_engineer, 'poly_features') and \
                   st.session_state.feature_engineer.poly_features is not None:
                    X_final = pd.DataFrame(
                        st.session_state.feature_engineer.poly_features.transform(X_selected),
                        columns=st.session_state.last_features
                    )
                else:
                    X_final = X_selected
                
                # Make prediction
                prediction = st.session_state.predictor.predict(X_final)[0]
                
                # Display result
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.error("⚠️ Positive Risk")
                    st.markdown("""
                        ### Recommendations:
                        - Schedule immediate follow-up with healthcare provider
                        - Monitor symptoms closely
                        - Consider additional testing
                    """)
                else:
                    st.success("✅ Negative Risk")
                    st.markdown("""
                        ### Recommendations:
                        - Continue regular health monitoring
                        - Maintain healthy lifestyle
                        - Schedule routine check-ups
                    """)
                    
                # Display confidence metrics
                st.subheader("Prediction Details")
                prediction_proba = st.session_state.predictor.model.predict_proba(X_final)[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Positive Risk Probability",
                        f"{prediction_proba[1]:.2%}"
                    )
                with col2:
                    st.metric(
                        "Negative Risk Probability",
                        f"{prediction_proba[0]:.2%}"
                    )
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please ensure all input fields are filled correctly.")

    def render_analysis(self):
        """Render the model analysis page"""
        st.header("Model Analysis")
        
        if not st.session_state.model_trained:
            st.warning("Please train the model first!")
            return
            
        try:
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs([
                "Model Comparison",
                "Feature Importance",
                "Performance Metrics",
                "Model Details"
            ])
            
            with tab1:
                st.subheader("Model Comparison")
                if hasattr(st.session_state.model_selector, 'model_performances'):
                    # Create comparison DataFrame
                    comparison_data = {
                        'Model': [],
                        'Score': [],
                        'Timestamp': []
                    }
                    
                    for model_name, info in st.session_state.model_selector.model_performances.items():
                        comparison_data['Model'].append(model_name)
                        comparison_data['Score'].append(info['score'])
                        comparison_data['Timestamp'].append(info['timestamp'])
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Plot comparison
                    fig = px.bar(
                        comparison_df,
                        x='Model',
                        y='Score',
                        title='Model Performance Comparison'
                    )
                    st.plotly_chart(fig)
                    
                    # Display comparison table
                    st.dataframe(comparison_df.style.format({
                        'Score': '{:.4f}'
                    }))
                else:
                    st.info("No model comparison data available.")
            
            with tab2:
                st.subheader("Feature Importance")
                if hasattr(st.session_state.predictor, 'model'):
                    try:
                        feature_importance = st.session_state.predictor.get_feature_importance(
                            st.session_state.last_features if st.session_state.last_features is not None
                            else ['Feature ' + str(i) for i in range(X.shape[1])]
                        )
                        
                        # Create feature importance DataFrame
                        feat_imp_df = pd.DataFrame({
                            'Feature': list(feature_importance.keys()),
                            'Importance': list(feature_importance.values())
                        }).sort_values('Importance', ascending=False)
                        
                        # Plot feature importance
                        fig = px.bar(
                            feat_imp_df,
                            x='Feature',
                            y='Importance',
                            title='Feature Importance Analysis'
                        )
                        st.plotly_chart(fig)
                        
                        # Display feature importance table
                        st.dataframe(feat_imp_df.style.format({
                            'Importance': '{:.3f}'
                        }))
                    except Exception as e:
                        st.warning(f"Cannot calculate feature importance for this model type: {type(st.session_state.predictor.model).__name__}")
                        st.info("Feature importance is only available for tree-based models, linear SVM, and logistic regression.")
                else:
                    st.info("No feature importance data available.")
            
            with tab3:
                st.subheader("Performance Metrics")
                
                # Get cross-validation results
                if hasattr(st.session_state.predictor, 'last_cv_results'):
                    cv_results = st.session_state.predictor.last_cv_results
                    
                    # Create metrics DataFrame
                    metrics_data = {
                        'Metric': [],
                        'Mean': [],
                        'Std': [],
                        'Min': [],
                        'Max': []
                    }
                    
                    for metric, values in cv_results.items():
                        metrics_data['Metric'].append(metric.capitalize())
                        metrics_data['Mean'].append(values['mean'])
                        metrics_data['Std'].append(values['std'])
                        metrics_data['Min'].append(min(values['scores']))
                        metrics_data['Max'].append(max(values['scores']))
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    # Display metrics table
                    st.dataframe(metrics_df.style.format({
                        'Mean': '{:.3f}',
                        'Std': '{:.3f}',
                        'Min': '{:.3f}',
                        'Max': '{:.3f}'
                    }))
                    
                    # Create radar chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=metrics_df['Mean'],
                        theta=metrics_df['Metric'],
                        fill='toself',
                        name='Mean Performance'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Add distribution plots for each metric
                    st.subheader("Metric Distributions")
                    for metric, values in cv_results.items():
                        fig = px.box(
                            y=values['scores'],
                            title=f"{metric.capitalize()} Distribution"
                        )
                        st.plotly_chart(fig)
                        
                else:
                    st.info("No performance metrics available.")
            
            with tab4:
                st.subheader("Model Details")
                if hasattr(st.session_state.predictor, 'model'):
                    # Display model parameters
                    st.json({
                        'Model Type': str(type(st.session_state.predictor.model).__name__),
                        'Number of Features': len(st.session_state.last_features) if st.session_state.last_features is not None 
                            else len(st.session_state.predictor.model.feature_importances_),
                        'Model Parameters': st.session_state.predictor.model.get_params()
                    })
                    
                    # Feature engineering details
                    st.subheader("Feature Engineering Details")
                    fe_details = {
                        "Original Features": len(st.session_state.feature_engineer.original_features) 
                            if st.session_state.feature_engineer.original_features is not None else "N/A",
                        "Selected Features": len(st.session_state.feature_engineer.selected_features)
                            if st.session_state.feature_engineer.selected_features is not None else "N/A",
                        "Polynomial Features": "Enabled" if st.session_state.feature_engineer.poly_features is not None else "Disabled",
                        "Scaling Method": st.session_state.feature_engineer.scaler.__class__.__name__
                            if st.session_state.feature_engineer.scaler is not None else "None"
                    }
                    st.json(fe_details)
                    
                    # Add download buttons for reports
                    st.subheader("Download Reports")
                    if st.button("Generate Report"):
                        report = self.generate_model_report()
                        st.download_button(
                            label="Download Report",
                            data=report,
                            file_name=f"model_report_{CURRENT_UTC.replace(' ', '_').replace(':', '-')}.txt",
                            mime="text/plain"
                        )
                else:
                    st.info("No model details available.")
                    
        except Exception as e:
            st.error(f"Error in model analysis: {str(e)}")
            st.info("Please ensure the model has been properly trained before accessing analysis.")

    def generate_model_report(self):
        """Generate a comprehensive model report"""
        report = []
        report.append("=" * 50)
        report.append("MODEL ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Generated on: {CURRENT_UTC}")
        report.append(f"Generated by: {CURRENT_USER}")
        report.append("\n")
        
        # Add model comparison results
        if hasattr(st.session_state.model_selector, 'model_performances'):
            report.append("MODEL COMPARISON")
            report.append("-" * 30)
            for model_name, info in st.session_state.model_selector.model_performances.items():
                report.append(f"\n{model_name}")
                report.append(f"Score: {info['score']:.4f}")
                report.append(f"Timestamp: {info['timestamp']}")
        
        # Add cross-validation results
        if hasattr(st.session_state.predictor, 'last_cv_results'):
            report.append("\nCROSS-VALIDATION RESULTS")
            report.append("-" * 30)
            for metric, values in st.session_state.predictor.last_cv_results.items():
                report.append(f"\n{metric.upper()}")
                report.append(f"Mean: {values['mean']:.3f}")
                report.append(f"Std: {values['std']:.3f}")
                report.append(f"Min: {min(values['scores']):.3f}")
                report.append(f"Max: {max(values['scores']):.3f}")
        
        # Add feature engineering details
        report.append("\nFEATURE ENGINEERING DETAILS")
        report.append("-" * 30)
        report.append(f"Original Features: {len(st.session_state.feature_engineer.original_features) if st.session_state.feature_engineer.original_features is not None else 'N/A'}")
        report.append(f"Selected Features: {len(st.session_state.feature_engineer.selected_features) if st.session_state.feature_engineer.selected_features is not None else 'N/A'}")
        report.append(f"Polynomial Features: {'Enabled' if st.session_state.feature_engineer.poly_features is not None else 'Disabled'}")
        report.append(f"Scaling Method: {st.session_state.feature_engineer.scaler.__class__.__name__ if st.session_state.feature_engineer.scaler is not None else 'None'}")
        
        # Add feature importance
        if hasattr(st.session_state.predictor, 'model'):
            report.append("\nFEATURE IMPORTANCE")
            report.append("-" * 30)
            feature_importance = st.session_state.predictor.get_feature_importance(
                st.session_state.last_features if st.session_state.last_features is not None
                else ['Feature ' + str(i) for i in range(len(st.session_state.predictor.model.feature_importances_))]
            )
            for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                report.append(f"{feature}: {importance:.3f}")
        
        return "\n".join(report)
    
    def render_home(self):
        """Render the home page"""
        st.markdown("""
        ## Welcome to the Patient Risk Prediction System
        
        This application helps healthcare professionals predict potential patient health risks 
        using advanced machine learning techniques. The system provides:
        
        - 📊 Advanced feature engineering and model selection
        - 🔍 Individual patient risk prediction
        - 📈 Detailed model analysis and metrics
        - 📋 Comprehensive performance reports
        
        ### Getting Started
        1. Go to the "Train Model" page to upload your training data
        2. Configure feature engineering and model selection settings
        3. Use the "Make Predictions" page to predict individual patient risks
        4. Check the "Model Analysis" page for detailed performance metrics
        
        ### Data Format Requirements
        Upload CSV files with the following columns:
        - Disease
        - Fever (Yes/No)
        - Cough (Yes/No)
        - Fatigue (Yes/No)
        - Difficulty Breathing (Yes/No)
        - Age
        - Gender
        - Blood Pressure (Low/Normal/High)
        - Cholesterol Level (Low/Normal/High)
        - Outcome Variable (Positive/Negative)
        """)
        
        # Display sample data format
        st.subheader("Sample Data Format")
        sample_data = pd.DataFrame({
            'Disease': ['Influenza', 'Common Cold'],
            'Fever': ['Yes', 'No'],
            'Cough': ['Yes', 'Yes'],
            'Fatigue': ['Yes', 'Yes'],
            'Difficulty Breathing': ['Yes', 'No'],
            'Age': [45, 32],
            'Gender': ['Male', 'Female'],
            'Blood Pressure': ['Normal', 'Low'],
            'Cholesterol Level': ['High', 'Normal'],
            'Outcome Variable': ['Positive', 'Negative']
        })
        st.dataframe(sample_data)
        
        # Display current system status
        st.subheader("System Status")
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.metric(
                "Model Status",
                "Trained" if st.session_state.model_trained else "Not Trained"
            )
        with status_col2:
            st.metric(
                "Last Updated",
                CURRENT_UTC if st.session_state.model_trained else "N/A"
            )
    
    def run(self):
        """Run the Streamlit app"""
        self.render_header()
        page = self.render_sidebar()
        
        if page == "Home":
            self.render_home()
        elif page == "Train Model":
            self.render_train_model()
        elif page == "Make Predictions":
            self.render_predictions()
        elif page == "Model Analysis":
            self.render_analysis()

if __name__ == "__main__":
    app = PatientRiskApp()
    app.run()