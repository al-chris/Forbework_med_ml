# AI Model for Predictive Patient Care

## Project Overview
This project implements a machine learning model to predict potential patient health risks based on historical patient data. The model uses various health indicators to predict the likelihood of positive or negative outcomes.

## Features
- Data preprocessing pipeline for handling medical data
- Random Forest Classifier for risk prediction
- Model evaluation metrics and visualizations
- Feature importance analysis
- Model persistence functionality
- Interactive Streamlit web interface
- Comprehensive cross-validation system
- Detailed performance analytics

## Requirements
- Python 3.8+
- pandas >= 1.3.0
- scikit-learn >= 0.24.2
- streamlit >= 1.10.0
- plotly >= 5.3.1
- matplotlib >= 3.4.3
- seaborn >= 0.11.2
- joblib >= 1.0.1

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Using the Streamlit Interface
1. Start the web interface:
```bash
streamlit run app.py
```

2. Navigate to the provided URL (typically http://localhost:8501)

3. Use the sidebar to navigate between:
   - Home: Overview and instructions
   - Train Model: Upload and train your model
   - Make Predictions: Input patient data for predictions
   - Model Analysis: View detailed performance metrics

### Data Format
Prepare your data in CSV format with the following columns:
- Disease (categorical)
- Fever (Yes/No)
- Cough (Yes/No)
- Fatigue (Yes/No)
- Difficulty Breathing (Yes/No)
- Age (numeric, 0-120)
- Gender (Male/Female)
- Blood Pressure (Low/Normal/High)
- Cholesterol Level (Low/Normal/High)
- Outcome Variable (Positive/Negative)

### Command Line Usage
For command-line operation:
```bash
python main.py
```

## Model Evaluation
The model is evaluated using multiple metrics:
- Accuracy: Overall prediction accuracy
- Precision: Positive prediction precision
- Recall: Sensitivity/True Positive Rate
- F1 Score: Harmonic mean of precision and recall
- Cross-validation scores with standard deviations

## Cross-Validation System
Comprehensive cross-validation including:
- K-Fold cross-validation (k=5 by default)
- Multiple evaluation metrics
- Performance visualization
- Detailed reports with timestamps

### Cross-Validation Results
The system generates:
1. Cross-validation scores for each metric
2. Standard deviation of scores to assess model stability
3. Interactive visualizations with error bars
4. Detailed reports with timestamps
5. Downloadable performance reports

## Implementation in Health App
To implement this model in a production health app:

1. Deploy the trained model:
   - Use Flask/FastAPI for REST API
   - Implement containerization (Docker)
   - Set up load balancing

2. Create API endpoints for:
   - Risk prediction
   - Model retraining
   - Metrics visualization
   - Batch predictions

3. Security measures:
   - Implement proper error handling
   - Add user authentication
   - Enable data encryption
   - Set up API key validation
   - Implement rate limiting

4. Monitoring:
   - Include regular model monitoring
   - Set up performance alerts
   - Implement logging system
   - Track prediction accuracy

## Project Structure
```
├── dataset/                 # Folder to store the  dataset
├── app.py                   # Streamlit web interface
├── data_preprocessing.py    # Data cleaning and feature engineering
├── model.py                 # Main predictor class
├── evaluate.py              # Evaluation metrics
├── main.py                  # Command-line interface
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Documentation
- `app.py`: Streamlit web interface implementation
- `data_preprocessing.py`: Handles data cleaning and feature engineering
- `model.py`: Contains the main predictor class
- `evaluate.py`: Implements evaluation metrics
- `main.py`: Command-line interface

## Dataset
- [Disease Symptoms and Patient Profile Dataset](https://www.kaggle.com/datasets/uom190346a/disease-symptoms-and-patient-profile-dataset/data)

## Future Improvements
1. Advanced Feature Engineering:
   - Implement feature selection algorithms
   - Add feature scaling options
   - Include polynomial features
   - Bigger dataset

2. Enhanced Model Selection:
   - Add support for multiple algorithms
   - Implement automated model selection

3. Extended Functionality:
   - Add support for more health indicators
   - Implement batch prediction
   - Add data export options
   - Enable model versioning

4. Monitoring and Maintenance:
   - Implement real-time monitoring
   - Add automated retraining
   - Include data drift detection
   - Add model performance tracking

## Demo
- [Demo](https://forbework-med-ml.onrender.com/)

## Contributors
- Current Maintainer: al-chris

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Disease Symptoms Dataset contributors
- Streamlit community
- scikit-learn developers
