from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class CrossValidator:
    """
    Handles cross-validation for the patient risk prediction model.
    """
    def __init__(self, n_splits=5, random_state=42):
        """
        Initialize cross-validator.
        
        Args:
            n_splits (int): Number of folds for cross-validation
            random_state (int): Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
    def get_scoring_metrics(self):
        """
        Define scoring metrics for cross-validation.
        
        Returns:
            Dictionary of scoring metrics
        """
        return {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score)
        }
    
    def perform_cross_validation(self, model, X, y):
        """
        Perform cross-validation with multiple metrics.
        
        Args:
            model: The machine learning model to validate
            X: Features
            y: Target variable
            
        Returns:
            Dictionary containing cross-validation results
        """
        scoring = self.get_scoring_metrics()
        results = {}
        
        for metric_name, scorer in scoring.items():
            scores = cross_val_score(model, X, y, cv=self.kf, scoring=scorer)
            results[metric_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            
        return results
    
    def plot_cv_results(self, results, save_path=None):
        """
        Plot cross-validation results.
        
        Args:
            results: Dictionary containing cross-validation results
            save_path: Path to save the plot
        """
        metrics = list(results.keys())
        means = [results[m]['mean'] for m in metrics]
        stds = [results[m]['std'] for m in metrics]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, means, yerr=stds, capsize=5)
        plt.title('Cross-Validation Results')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def generate_cv_report(self, results):
        """
        Generate a detailed cross-validation report.
        
        Args:
            results: Dictionary containing cross-validation results
        
        Returns:
            Formatted report string
        """
        report = ["Cross-Validation Report"]
        report.append(f"Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        report.append(f"Number of folds: {self.n_splits}")
        report.append("\nMetrics Summary:")
        
        for metric, values in results.items():
            report.append(f"\n{metric.capitalize()}:")
            report.append(f"  Mean: {values['mean']:.4f}")
            report.append(f"  Std: {values['std']:.4f}")
            report.append(f"  Scores: {', '.join([f'{s:.4f}' for s in values['scores']])}")
            
        return "\n".join(report)