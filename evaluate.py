from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_model(y_true, y_pred, X_test, feature_names):
    """
    Evaluate the model performance and generate visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        X_test: Test features
        feature_names: Names of the features
    """
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }
    
    # Create evaluation report
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    feature_imp = pd.Series(feature_names).value_counts()
    sns.barplot(x=feature_imp.values, y=feature_imp.index)
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png')