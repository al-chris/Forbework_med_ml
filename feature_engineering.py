import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Advanced feature engineering class that handles:
    - Feature selection
    - Feature scaling
    - Polynomial features
    - Feature interactions
    - Dimensionality reduction
    """
    def __init__(self):
        self.scaler = None
        self.poly_features = None
        self.feature_selector = None
        self.pca = None
        self.original_features = None
        self.selected_features = None
        self.label_encoders = {}
        
    def create_medical_features(self, df):
        """
        Create domain-specific medical features
        
        Args:
            df: DataFrame with medical data
        Returns:
            DataFrame with additional medical features
        """
        df = df.copy()
        
        # Convert Age column to numeric (handle string inputs)
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        
        # Create risk score based on symptoms
        symptom_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
        df['symptom_count'] = df[symptom_cols].apply(
            lambda x: sum(1 for val in x if val == 'Yes'), axis=1
        )
        
        # Create age groups (but keep original Age column)
        df['age_group'] = pd.cut(
            df['Age'],
            bins=[0, 12, 18, 35, 50, 65, 120],
            labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Middle Aged', 'Senior']
        )
        
        # Create health index
        df['health_index'] = self._calculate_health_index(df)
        
        return df
    
    def _calculate_health_index(self, df):
        """Calculate a general health index based on various factors"""
        health_index = np.zeros(len(df))
        
        # Add risk factors
        health_index += (df['Age'] > 65).astype(int) * 2
        health_index += (df['Blood Pressure'] == 'High').astype(int) * 1.5
        health_index += (df['Cholesterol Level'] == 'High').astype(int) * 1.5
        health_index += df['symptom_count'] * 0.5
        
        return health_index
    
    def prepare_features(self, df):
        """
        Prepare features by encoding categorical variables
        
        Args:
            df: DataFrame with features
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])
        
        return df
    
    def select_features(self, X, y, method='mutual_info', n_features=None):
        """
        Select the most important features
        
        Args:
            X: Feature matrix
            y: Target variable
            method: 'mutual_info' or 'f_score'
            n_features: Number of features to select
        Returns:
            Selected features matrix
        """
        if n_features is None:
            n_features = X.shape[1] // 2
            
        if method == 'mutual_info':
            self.feature_selector = SelectKBest(
                mutual_info_classif, k=n_features
            )
        else:
            self.feature_selector = SelectKBest(
                f_classif, k=n_features
            )
            
        self.original_features = X.columns
        X_selected = self.feature_selector.fit_transform(X, y)
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X.columns[selected_mask]
        # Save the selected feature order in the selector for later use
        self.feature_selector.selected_features = list(self.selected_features)
        
        return pd.DataFrame(X_selected, columns=self.selected_features)
    
    def create_polynomial_features(self, X, degree=2, include_bias=False):
        """
        Create polynomial features
        
        Args:
            X: Feature matrix
            degree: Polynomial degree
            include_bias: Whether to include bias term
        Returns:
            Matrix with polynomial features
        """
        self.poly_features = PolynomialFeatures(
            degree=degree,
            include_bias=include_bias
        )
        X_poly = self.poly_features.fit_transform(X)
        
        # Generate feature names
        feature_names = self.poly_features.get_feature_names_out(X.columns)
        
        return pd.DataFrame(X_poly, columns=feature_names)
    
    def scale_features(self, X, method='standard'):
        """
        Scale features using specified method
        
        Args:
            X: Feature matrix
            method: 'standard' or 'minmax'
        Returns:
            Scaled feature matrix
        """
        # Ensure all features are numeric
        X = X.astype(float)
        
        if method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
            
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)
    
    def reduce_dimensions(self, X, n_components=None):
        """
        Reduce dimensionality using PCA
        
        Args:
            X: Feature matrix
            n_components: Number of components to keep
        Returns:
            Reduced feature matrix
        """
        if n_components is None:
            n_components = min(X.shape[1], 10)
            
        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X)
        
        # Generate feature names
        feature_names = [f'PC{i+1}' for i in range(n_components)]
        
        return pd.DataFrame(X_reduced, columns=feature_names)