from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from cross_validation import CrossValidator

class PatientRiskPredictor:
    """
    A class to handle the training and prediction of patient health risks.
    """
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.cv = CrossValidator(n_splits=5)
        self.last_cv_results = None
        
    def train(self, X_train, y_train):
        """
        Train the model with the provided data.
        
        Args:
            X_train: Training features
            y_train: Training target variables
        """
        self.model.fit(X_train, y_train)
        
    def cross_validate(self, X, y):
        """
        Perform cross-validation and store results.
        
        Args:
            X: Features
            y: Target variable
        Returns:
            Dictionary containing cross-validation results
        """
        self.last_cv_results = self.cv.perform_cross_validation(self.model, X, y)
        return self.last_cv_results
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names):
        """
        Get the importance of each feature in the model.
        
        Args:
            feature_names: List of feature names
        Returns:
            Dictionary of feature importance
        """
        importance = self.model.feature_importances_
        return dict(zip(feature_names, importance))
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        joblib.dump(self.model, filepath)
    
    @staticmethod
    def load_model(filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
        Returns:
            Loaded model
        """
        return joblib.load(filepath)