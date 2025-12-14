"""
KubeShield Data Preprocessing Module
Handles data cleaning, normalization, and feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import logging
from typing import Tuple, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses raw Kubernetes audit logs and metrics for ML models
    
    Operations:
    1. Loading and parsing raw data
    2. Handling missing values
    3. Feature scaling and normalization
    4. Outlier detection and removal
    5. Time-series alignment
    6. Feature engineering
    """
    
    def __init__(self, scaler_type='standard'):
        """
        Initialize preprocessor
        
        Args:
            scaler_type: 'standard', 'minmax', or 'robust'
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = RobustScaler()
        
        self.pca = None
        logger.info(f"Initialized DataPreprocessor with {scaler_type} scaling")
    
    def load_data(self, filepath: str) -> np.ndarray:
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
        
        Returns:
            Loaded data array
        """
        logger.info(f"Loading data from {filepath}...")
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        return df.values, df.columns.tolist()
    
    def handle_missing_values(self, X: np.ndarray, strategy='mean') -> np.ndarray:
        """
        Handle missing values in data
        
        Args:
            X: Input data
            strategy: 'mean', 'median', 'forward_fill', 'drop'
        
        Returns:
            Data with missing values handled
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        df = pd.DataFrame(X)
        missing_count = df.isnull().sum().sum()
        
        if missing_count == 0:
            logger.info("No missing values found")
            return X
        
        logger.info(f"Found {missing_count} missing values")
        
        if strategy == 'mean':
            df.fillna(df.mean(), inplace=True)
        elif strategy == 'median':
            df.fillna(df.median(), inplace=True)
        elif strategy == 'forward_fill':
            df.fillna(method='ffill', inplace=True)
        elif strategy == 'drop':
            df.dropna(inplace=True)
        
        logger.info(f"After handling: {df.shape}")
        return df.values
    
    def detect_outliers(self, X: np.ndarray, method='iqr', threshold=3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers in data
        
        Args:
            X: Input data
            method: 'iqr' or 'zscore'
            threshold: Threshold for zscore method
        
        Returns:
            Tuple of (X_cleaned, outlier_mask)
        """
        logger.info(f"Detecting outliers using {method} method...")
        
        outlier_mask = np.zeros(len(X), dtype=bool)
        
        if method == 'iqr':
            for col in range(X.shape[1]):
                Q1 = np.percentile(X[:, col], 25)
                Q3 = np.percentile(X[:, col], 75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask |= (X[:, col] < lower_bound) | (X[:, col] > upper_bound)
        
        elif method == 'zscore':
            zscore = np.abs((X - X.mean()) / X.std())
            outlier_mask = (zscore > threshold).any(axis=1)
        
        num_outliers = np.sum(outlier_mask)
        logger.info(f"Detected {num_outliers} outliers ({100*num_outliers/len(X):.2f}%)")
        
        X_cleaned = X[~outlier_mask]
        return X_cleaned, outlier_mask
    
    def normalize(self, X: np.ndarray, fit=True) -> np.ndarray:
        """
        Normalize features using fitted scaler
        
        Args:
            X: Input data
            fit: Whether to fit the scaler
        
        Returns:
            Normalized data
        """
        logger.info(f"Normalizing features (fit={fit})...")
        
        if fit:
            X_normalized = self.scaler.fit_transform(X)
            logger.info("Scaler fitted")
        else:
            X_normalized = self.scaler.transform(X)
        
        logger.info(f"Normalized data - Min: {X_normalized.min():.4f}, "
                   f"Max: {X_normalized.max():.4f}, Mean: {X_normalized.mean():.4f}")
        
        return X_normalized
    
    def apply_pca(self, X: np.ndarray, n_components=0.95, fit=True) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction
        
        Args:
            X: Input data
            n_components: Number of components or variance ratio
            fit: Whether to fit PCA
        
        Returns:
            PCA-transformed data
        """
        logger.info(f"Applying PCA (n_components={n_components})...")
        
        if fit:
            self.pca = PCA(n_components=n_components)
            X_pca = self.pca.fit_transform(X)
            logger.info(f"PCA fitted - Output shape: {X_pca.shape}")
            logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        else:
            X_pca = self.pca.transform(X)
        
        return X_pca
    
    def create_sequences(self, X: np.ndarray, seq_length: int = 60) -> np.ndarray:
        """
        Create sequences for LSTM models
        
        Args:
            X: Input data (num_samples, features)
            seq_length: Sequence length
        
        Returns:
            Sequences (num_sequences, seq_length, features)
        """
        logger.info(f"Creating sequences of length {seq_length}...")
        
        num_samples = X.shape[0]
        num_features = X.shape[1]
        num_sequences = max(0, num_samples - seq_length + 1)
        
        sequences = np.zeros((num_sequences, seq_length, num_features), dtype=np.float32)
        
        for i in range(num_sequences):
            sequences[i] = X[i:i+seq_length]
        
        logger.info(f"Created sequences shape: {sequences.shape}")
        return sequences
    
    def preprocess_pipeline(self, X: np.ndarray, 
                           remove_outliers: bool = True,
                           normalize: bool = True,
                           use_pca: bool = False) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            X: Input data
            remove_outliers: Whether to remove outliers
            normalize: Whether to normalize
            use_pca: Whether to apply PCA
        
        Returns:
            Preprocessed data
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Remove outliers
        if remove_outliers:
            X, _ = self.detect_outliers(X)
        
        # Normalize
        if normalize:
            X = self.normalize(X, fit=True)
        
        # PCA
        if use_pca:
            X = self.apply_pca(X, fit=True)
        
        logger.info(f"Pipeline complete - Output shape: {X.shape}")
        return X


def main():
    """Example usage"""
    logger.info("Testing DataPreprocessor...")
    
    # Generate synthetic data
    X = np.random.randn(1000, 73) * 20 + 50
    
    # Initialize preprocessor
    prep = DataPreprocessor(scaler_type='standard')
    
    # Preprocessing pipeline
    X_processed = prep.preprocess_pipeline(X, remove_outliers=True, normalize=True)
    
    logger.info(f"Original shape: {X.shape}")
    logger.info(f"Processed shape: {X_processed.shape}")
    
    # Create sequences
    sequences = prep.create_sequences(X_processed, seq_length=60)
    logger.info(f"Sequences shape: {sequences.shape}")
    
    print("✓ Preprocessing test completed successfully")


if __name__ == '__main__':
    main()
