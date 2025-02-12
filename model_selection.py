from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timezone
import pickle
import joblib

class ModelSelector:
    """
    Enhanced model selection class that handles:
    - Multiple algorithms
    - Automated model selection
    - Hyperparameter tuning
    - Model comparison
    - Performance tracking
    """
    def __init__(self):
        self.models = {
            'random_forest': {
                'model': RandomForestClassifier(),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 4, 5],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'svm': {
                'model': SVC(probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.1, 0.01]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
        }
        
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.model_performances = {}
        
    def get_scoring_metrics(self):
        """Define scoring metrics for model evaluation"""
        return {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score)
        }
    
    def tune_model(self, model_name, X, y, search_method='grid', cv=5, n_iter=20):
        """
        Tune hyperparameters for a specific model
        
        Args:
            model_name: Name of the model to tune
            X: Feature matrix
            y: Target variable
            search_method: 'grid' or 'random'
            cv: Number of cross-validation folds
            n_iter: Number of iterations for random search
        Returns:
            Tuned model and best parameters
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not supported")
            
        model_info = self.models[model_name]
        scoring = self.get_scoring_metrics()
        
        if search_method == 'grid':
            search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=cv,
                scoring=scoring,
                refit='f1',
                n_jobs=-1
            )
        else:
            search = RandomizedSearchCV(
                model_info['model'],
                model_info['params'],
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                refit='f1',
                n_jobs=-1
            )
            
        search.fit(X, y)
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def select_best_model(self, X, y, cv=5):
        """
        Try all models and select the best one
        
        Args:
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds
        Returns:
            Best model and its performance metrics
        """
        for model_name in self.models:
            try:
                print(f"Tuning {model_name}...")
                model, params, score = self.tune_model(
                    model_name, X, y, search_method='random', cv=cv
                )
                
                self.model_performances[model_name] = {
                    'model': model,
                    'params': params,
                    'score': score,
                    'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                }
                
                if self.best_score is None or score > self.best_score:
                    self.best_model = model
                    self.best_params = params
                    self.best_score = score
                    
            except Exception as e:
                print(f"Error tuning {model_name}: {str(e)}")
                
        return self.best_model, self.model_performances
    
    def save_model_results(self, save_path='model_results'):
        """Save model performance results"""
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        
        # Save performance metrics
        performance_data = {
            name: {
                'params': info['params'],
                'score': float(info['score']),
                'timestamp': info['timestamp']
            }
            for name, info in self.model_performances.items()
        }
        
        with open(f'{save_path}/model_performance_{timestamp}.json', 'w') as f:
            json.dump(performance_data, f, indent=4)
            
        # Create performance report
        report = [
            "Model Performance Report",
            f"Generated on: {timestamp}",
            f"Generated by: {os.getenv('USER', 'al-chris')}",
            "\nModel Comparison:",
        ]
        
        for name, info in self.model_performances.items():
            report.extend([
                f"\n{name.upper()}:",
                f"Score: {info['score']:.4f}",
                f"Best Parameters: {info['params']}"
            ])
            
        with open(f'{save_path}/performance_report_{timestamp}.txt', 'w') as f:
            f.write('\n'.join(report))

    def save_model(self, save_path='best_model.pkl'):
        """Save the best model to a file"""
        if self.best_model is not None:
            # with open(save_path, 'wb') as f:
            #     pickle.dump(self.best_model, f)
            joblib.dump(self.best_model, save_path)
        else:
            raise ValueError("No model to save. Train the model first.")