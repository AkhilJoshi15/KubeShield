"""
KubeShield Isolation Forest Model
Detects outlier samples using isolation-based anomaly detection
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IsolationForestModel:
    """
    Isolation Forest for detecting point anomalies
    
    Isolation Forest works by randomly selecting a feature and then randomly
    selecting a split value between the max and min values of that feature.
    Anomalies are isolated faster (fewer splits required), leading to lower
    scores in the decision tree.
    
    Hyperparameters (tuned for KubeShield):
    - n_estimators: 150 (number of isolation trees)
    - contamination: 0.1 (expected proportion of outliers)
    - random_state: 42 (reproducibility)
    """
    
    def __init__(self, n_estimators=150, contamination=0.1, random_state=42):
        """
        Initialize Isolation Forest
        
        Args:
            n_estimators: Number of isolation trees in the ensemble
            contamination: Expected proportion of anomalies in dataset
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        logger.info(f"Initialized IsolationForest: n_estimators={n_estimators}, "
                   f"contamination={contamination}")
    
    def fit(self, X_train):
        """
        Fit the Isolation Forest model
        
        Args:
            X_train: Training data (num_samples, feature_dim)
        """
        logger.info(f"Fitting Isolation Forest on {X_train.shape[0]} samples...")
        
        # Flatten if 3D (for compatibility with sequence models)
        if len(X_train.shape) == 3:
            logger.info(f"Flattening input from {X_train.shape} to 2D")
            num_samples = X_train.shape[0]
            X_train = X_train.reshape(num_samples, -1)
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Fit model
        self.model.fit(X_train_scaled)
        self.is_fitted = True
        
        logger.info("Isolation Forest fitting completed")
    
    def predict_anomalies(self, X):
        """
        Predict anomalies (-1 for anomaly, 1 for normal)
        
        Args:
            X: Input data (num_samples, feature_dim or num_samples, seq_len, feature_dim)
        
        Returns:
            Predictions array (-1 or 1 for each sample)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_transformed = self._preprocess(X)
        predictions = self.model.predict(X_transformed)
        return predictions
    
    def compute_anomaly_score(self, X):
        """
        Compute anomaly scores (probability estimates)
        
        Args:
            X: Input data (num_samples, feature_dim or 3D array)
        
        Returns:
            Anomaly scores (num_samples,) - higher values indicate more anomalous
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        X_transformed = self._preprocess(X)
        
        # Get decision scores (negative is anomaly, positive is normal)
        decision_scores = self.model.decision_function(X_transformed)
        
        # Convert to anomaly probability scores [0, 1]
        # Lower decision scores -> higher anomaly scores
        min_score = np.min(decision_scores)
        max_score = np.max(decision_scores)
        
        if max_score > min_score:
            anomaly_scores = (max_score - decision_scores) / (max_score - min_score)
        else:
            anomaly_scores = np.zeros_like(decision_scores)
        
        # Clip to [0, 1] range
        anomaly_scores = np.clip(anomaly_scores, 0, 1)
        
        return anomaly_scores
    
    def _preprocess(self, X):
        """
        Preprocess input data
        
        Args:
            X: Input data (2D or 3D array)
        
        Returns:
            Preprocessed and scaled data
        """
        # Flatten if 3D
        if len(X.shape) == 3:
            num_samples = X.shape[0]
            X = X.reshape(num_samples, -1)
        
        # Scale using fitted scaler
        return self.scaler.transform(X)
    
    def save_model(self, filepath):
        """
        Save model and scaler to disk
        
        Args:
            filepath: Path to save model (without extension)
        """
        joblib.dump(self.model, f"{filepath}_if.joblib")
        joblib.dump(self.scaler, f"{filepath}_scaler.joblib")
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model and scaler from disk
        
        Args:
            filepath: Path to load model (without extension)
        """
        self.model = joblib.load(f"{filepath}_if.joblib")
        self.scaler = joblib.load(f"{filepath}_scaler.joblib")
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    logger.info("Testing Isolation Forest Model...")
    
    # Generate synthetic data
    num_samples = 1000
    feature_dim = 73
    
    # Normal data
    X_train = np.random.randn(num_samples, feature_dim).astype(np.float32) * 10 + 50
    
    # Test data with some anomalies
    X_test = np.random.randn(100, feature_dim).astype(np.float32) * 10 + 50
    X_test[0:5] = np.random.randn(5, feature_dim) * 50  # Add anomalies
    
    # Create and fit model
    model = IsolationForestModel(n_estimators=150, contamination=0.05)
    model.fit(X_train)
    
    # Compute anomaly scores
    scores = model.compute_anomaly_score(X_test)
    predictions = model.predict_anomalies(X_test)
    
    logger.info(f"Anomaly score statistics - Min: {scores.min():.4f}, "
               f"Max: {scores.max():.4f}, Mean: {scores.mean():.4f}")
    logger.info(f"Detected {np.sum(predictions == -1)} anomalies out of {len(predictions)} samples")
    
    print("✓ Isolation Forest model test completed successfully")
