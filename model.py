from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import numpy as np

class PatientRiskPredictor:
    def __init__(self):
        self.model = None
        self.last_cv_results = None

    def train(self, X, y):
        """Train the model"""
        if self.model is None:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def get_feature_importance(self, feature_names):
        """
        Get feature importance based on model type.
        Different models have different ways of determining feature importance.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        importances = {}
        
        if isinstance(self.model, (RandomForestClassifier, GradientBoostingClassifier)):
            # Tree-based models have feature_importances_
            importance_values = self.model.feature_importances_
        elif isinstance(self.model, LogisticRegression):
            # Logistic regression uses coefficients
            importance_values = np.abs(self.model.coef_[0])
        elif isinstance(self.model, SVC):
            if self.model.kernel == 'linear':
                # Linear SVM uses coefficients
                importance_values = np.abs(self.model.coef_[0])
            else:
                # For non-linear SVM, we'll use a simple feature ranking based on correlation
                if not hasattr(self, '_last_X') or self._last_X is None:
                    raise ValueError("Feature importance for non-linear SVM requires training data")
                importance_values = np.abs(np.corrcoef(self._last_X.T, self.model.decision_function(self._last_X))[-1, :-1])
        else:
            raise ValueError(f"Feature importance not implemented for model type: {type(self.model).__name__}")

        # Normalize importance values
        importance_values = importance_values / np.sum(importance_values)
        
        # Create dictionary of feature importances
        for name, importance in zip(feature_names, importance_values):
            importances[name] = float(importance)  # Convert to float for JSON serialization
            
        return importances

    def cross_validate(self, X, y):
        """Perform cross-validation"""
        # Store X for potential use in feature importance calculation
        self._last_X = X
        
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1'
        }
        
        cv_results = cross_validate(
            self.model,
            X,
            y,
            cv=5,
            scoring=scoring,
            return_train_score=False
        )
        
        # Format results
        self.last_cv_results = {}
        for metric in scoring.keys():
            scores = cv_results[f'test_{metric}']
            self.last_cv_results[metric] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std()
            }
            
        return self.last_cv_results