"""
KubeShield LSTM Autoencoder Model
Detects temporal anomalies in feature sequences using LSTM reconstruction error
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMAutoencoder(keras.Model):
    """
    LSTM-based Autoencoder for detecting temporal anomalies
    
    Architecture:
    - Encoder: 2 LSTM layers (128 -> 32 units) with dropout
    - Decoder: 2 LSTM layers (32 -> 128 units) with dropout
    - Output: Reconstructed sequence with temporal continuity
    
    The model is trained on normal behavior and uses reconstruction error
    as an anomaly score (higher error = more anomalous).
    """
    
    def __init__(self, sequence_length=60, feature_dim=73, encoding_dim=16):
        """
        Initialize LSTM Autoencoder
        
        Args:
            sequence_length: Number of timesteps in input sequence
            feature_dim: Number of features per timestep (73 for KubeShield)
            encoding_dim: Dimension of latent encoding
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.encoding_dim = encoding_dim
        
        # Encoder
        self.encoder = keras.Sequential([
            layers.LSTM(128, activation='relu', input_shape=(sequence_length, feature_dim),
                       return_sequences=True, name='encoder_lstm_1'),
            layers.Dropout(0.2),
            layers.LSTM(32, activation='relu', return_sequences=False, name='encoder_lstm_2'),
            layers.Dropout(0.2),
            layers.Dense(encoding_dim, activation='relu', name='encoding')
        ], name='encoder')
        
        # Decoder
        self.decoder = keras.Sequential([
            layers.RepeatVector(sequence_length, input_shape=(encoding_dim,)),
            layers.LSTM(32, activation='relu', return_sequences=True, name='decoder_lstm_1'),
            layers.Dropout(0.2),
            layers.LSTM(128, activation='relu', return_sequences=True, name='decoder_lstm_2'),
            layers.Dropout(0.2),
            layers.TimeDistributed(layers.Dense(feature_dim), name='reconstruction')
        ], name='decoder')
        
        logger.info(f"Initialized LSTMAutoencoder: seq_len={sequence_length}, "
                   f"features={feature_dim}, encoding_dim={encoding_dim}")
    
    def call(self, inputs, training=False):
        """Forward pass through encoder and decoder"""
        encoded = self.encoder(inputs, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded
    
    def train_model(self, X_train, X_val=None, epochs=50, batch_size=32):
        """
        Train the autoencoder
        
        Args:
            X_train: Training data (num_samples, sequence_length, feature_dim)
            X_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        logger.info(f"Training LSTM Autoencoder for {epochs} epochs...")
        logger.info(f"Training data shape: {X_train.shape}")
        
        self.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',
            metrics=['mae']
        )
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        history = self.fit(
            X_train, X_train,
            validation_data=(X_val, X_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        logger.info("LSTM Autoencoder training completed")
        return history
    
    def compute_anomaly_score(self, X):
        """
        Compute reconstruction error as anomaly score
        
        Args:
            X: Input data (num_samples, sequence_length, feature_dim)
        
        Returns:
            Anomaly scores (num_samples,) - values between 0 and 1
        """
        predictions = self.predict(X, verbose=0)
        mse = np.mean(np.power(X - predictions, 2), axis=(1, 2))
        
        # Normalize to [0, 1]
        min_mse = np.min(mse)
        max_mse = np.max(mse)
        if max_mse > min_mse:
            anomaly_scores = (mse - min_mse) / (max_mse - min_mse)
        else:
            anomaly_scores = np.zeros_like(mse)
        
        return anomaly_scores
    
    def save_model(self, filepath):
        """Save model weights"""
        self.save_weights(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model weights"""
        self.load_weights(filepath)
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    logger.info("Testing LSTM Autoencoder...")
    
    # Generate synthetic data
    sequence_length = 60
    feature_dim = 73
    num_samples = 1000
    
    X_train = np.random.randn(num_samples, sequence_length, feature_dim).astype(np.float32)
    X_test = np.random.randn(100, sequence_length, feature_dim).astype(np.float32)
    
    # Create and train model
    model = LSTMAutoencoder(sequence_length=sequence_length, feature_dim=feature_dim)
    history = model.train_model(X_train, epochs=5, batch_size=32)
    
    # Compute anomaly scores
    scores = model.compute_anomaly_score(X_test)
    logger.info(f"Anomaly score statistics - Min: {scores.min():.4f}, "
               f"Max: {scores.max():.4f}, Mean: {scores.mean():.4f}")
    
    print("✓ LSTM Autoencoder test completed successfully")
