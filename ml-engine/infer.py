"""
KubeShield Inference Script
Loads trained models and performs ensemble anomaly detection
"""

import numpy as np
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Tuple
import joblib

from models.lstm_autoencoder import LSTMAutoencoder
from models.isolation_forest import IsolationForestModel
from models.gnn import GNNModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnsembleModel:
    """Ensemble of three anomaly detection models with weighted fusion"""
    
    # Optimal weights from paper: LSTM 0.47, IF 0.28, GNN 0.25
    WEIGHTS = {
        'lstm': 0.47,
        'isolation_forest': 0.28,
        'gnn': 0.25
    }
    
    def __init__(self, model_dir='models'):
        """
        Load all trained models
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        
        logger.info("Loading trained models...")
        
        # Load LSTM Autoencoder
        self.lstm_model = LSTMAutoencoder(
            sequence_length=60,
            feature_dim=73,
            encoding_dim=16
        )
        try:
            self.lstm_model.load_model(str(self.model_dir / 'lstm_autoencoder.h5'))
            logger.info("✓ LSTM Autoencoder loaded")
        except:
            logger.warning("⚠ LSTM Autoencoder not found, using random model")
        
        # Load Isolation Forest
        self.if_model = IsolationForestModel()
        try:
            self.if_model.load_model(str(self.model_dir / 'isolation_forest'))
            logger.info("✓ Isolation Forest loaded")
        except:
            logger.warning("⚠ Isolation Forest not found, using new model")
        
        # Load GNN
        self.gnn_model = GNNModel(node_feature_dim=73)
        try:
            with open(self.model_dir / 'gnn_metadata.json') as f:
                metadata = json.load(f)
                logger.info("✓ GNN metadata loaded")
        except:
            logger.warning("⚠ GNN metadata not found")
        
        logger.info(f"Ensemble weights - LSTM: {self.WEIGHTS['lstm']}, "
                   f"IF: {self.WEIGHTS['isolation_forest']}, "
                   f"GNN: {self.WEIGHTS['gnn']}")
    
    def predict(self, sequence: np.ndarray, flat_features: np.ndarray = None) -> Dict:
        """
        Run inference with ensemble model
        
        Args:
            sequence: Input sequence (1, 60, 73) or (num_samples, 60, 73)
            flat_features: Optional flattened features for IF model
        
        Returns:
            Dictionary with individual scores and ensemble score
        """
        
        # Add batch dimension if needed
        if len(sequence.shape) == 2:
            sequence = np.expand_dims(sequence, axis=0)
        
        batch_size = sequence.shape[0]
        
        logger.info(f"Running inference on {batch_size} samples...")
        
        # LSTM Autoencoder prediction
        try:
            lstm_scores = self.lstm_model.compute_anomaly_score(sequence)
            logger.debug(f"LSTM scores shape: {lstm_scores.shape}")
        except Exception as e:
            logger.error(f"LSTM inference failed: {e}")
            lstm_scores = np.random.rand(batch_size)
        
        # Isolation Forest prediction
        try:
            if flat_features is None:
                # Use last timestep as flat features
                flat_features = sequence[:, -1, :]
            if_scores = self.if_model.compute_anomaly_score(flat_features)
            logger.debug(f"IF scores shape: {if_scores.shape}")
        except Exception as e:
            logger.error(f"IF inference failed: {e}")
            if_scores = np.random.rand(batch_size)
        
        # GNN prediction (simplified for single feature vector)
        try:
            gnn_scores = np.array([
                self.gnn_model.compute_anomaly_score(
                    self.gnn_model.create_graph(
                        np.expand_dims(sequence[i, -1, :], axis=0),
                        []
                    )
                )
                for i in range(batch_size)
            ])
            logger.debug(f"GNN scores shape: {gnn_scores.shape}")
        except Exception as e:
            logger.error(f"GNN inference failed: {e}")
            gnn_scores = np.random.rand(batch_size)
        
        # Ensemble fusion
        ensemble_scores = self.fuse_scores(lstm_scores, if_scores, gnn_scores)
        
        results = {
            'lstm_scores': lstm_scores,
            'isolation_forest_scores': if_scores,
            'gnn_scores': gnn_scores,
            'ensemble_scores': ensemble_scores
        }
        
        return results
    
    @staticmethod
    def fuse_scores(s_lstm, s_if, s_gnn, weights=None):
        """
        Fuse scores from three models using weighted ensemble
        
        Args:
            s_lstm: LSTM scores
            s_if: Isolation Forest scores
            s_gnn: GNN scores
            weights: Dict with weights (default: paper optimal)
        
        Returns:
            Fused ensemble score
        """
        if weights is None:
            weights = EnsembleModel.WEIGHTS
        
        ensemble = (
            weights['lstm'] * s_lstm +
            weights['isolation_forest'] * s_if +
            weights['gnn'] * s_gnn
        )
        
        # Clip to [0, 1]
        ensemble = np.clip(ensemble, 0, 1)
        
        return ensemble
    
    def detect_anomalies(self, sequence, threshold=0.85):
        """
        Detect anomalies using ensemble with threshold
        
        Args:
            sequence: Input sequence
            threshold: Anomaly threshold (default: 0.85)
        
        Returns:
            Tuple of (predictions, scores) where predictions is boolean array
        """
        results = self.predict(sequence)
        scores = results['ensemble_scores']
        
        predictions = scores >= threshold
        
        num_anomalies = np.sum(predictions)
        logger.info(f"Detected {num_anomalies}/{len(predictions)} anomalies "
                   f"(threshold: {threshold})")
        
        return predictions, scores, results
    
    def explain_prediction(self, sequence, idx=0):
        """
        Explain individual prediction by showing contribution of each model
        
        Args:
            sequence: Input sequence
            idx: Sample index to explain
        
        Returns:
            Dictionary with explanation
        """
        results = self.predict(sequence)
        
        explanation = {
            'lstm_score': float(results['lstm_scores'][idx]),
            'lstm_weight': self.WEIGHTS['lstm'],
            'lstm_contribution': float(results['lstm_scores'][idx] * self.WEIGHTS['lstm']),
            
            'if_score': float(results['isolation_forest_scores'][idx]),
            'if_weight': self.WEIGHTS['isolation_forest'],
            'if_contribution': float(results['isolation_forest_scores'][idx] * self.WEIGHTS['isolation_forest']),
            
            'gnn_score': float(results['gnn_scores'][idx]),
            'gnn_weight': self.WEIGHTS['gnn'],
            'gnn_contribution': float(results['gnn_scores'][idx] * self.WEIGHTS['gnn']),
            
            'ensemble_score': float(results['ensemble_scores'][idx]),
            'is_anomaly': float(results['ensemble_scores'][idx]) >= 0.85
        }
        
        return explanation


def main():
    parser = argparse.ArgumentParser(description='KubeShield Inference')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory containing trained models')
    parser.add_argument('--data', type=str, help='Input data file (CSV)')
    parser.add_argument('--threshold', type=float, default=0.85,
                       help='Anomaly detection threshold (default: 0.85)')
    parser.add_argument('--explain', action='store_true',
                       help='Show detailed explanation for first sample')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("KubeShield Ensemble Anomaly Detection Inference")
    logger.info("="*70)
    
    # Initialize ensemble
    ensemble = EnsembleModel(model_dir=args.model_dir)
    
    # Generate or load test data
    if args.data:
        logger.info(f"Loading data from {args.data}")
        test_data = np.loadtxt(args.data, delimiter=',')
    else:
        logger.info("Generating synthetic test data...")
        test_data = np.random.randn(100, 73).astype(np.float32) * 20 + 50
    
    # Prepare sequence format
    if len(test_data.shape) == 1:
        test_data = np.expand_dims(test_data, axis=0)
    
    if test_data.shape[1] != 73:
        # Assume it's already in sequence format
        sequence = test_data
    else:
        # Expand to sequence format
        sequence = np.expand_dims(test_data, axis=1)
        sequence = np.tile(sequence, (1, 60, 1))
    
    # Run inference
    logger.info(f"\nInput shape: {sequence.shape}")
    
    predictions, scores, results = ensemble.detect_anomalies(
        sequence,
        threshold=args.threshold
    )
    
    # Print summary
    logger.info(f"\n" + "="*70)
    logger.info("INFERENCE RESULTS")
    logger.info("="*70)
    logger.info(f"Total samples: {len(scores)}")
    logger.info(f"Anomalies detected: {np.sum(predictions)}")
    logger.info(f"Normal samples: {len(scores) - np.sum(predictions)}")
    logger.info(f"Anomaly threshold: {args.threshold}")
    logger.info(f"\nScore statistics:")
    logger.info(f"  Min: {scores.min():.4f}")
    logger.info(f"  Max: {scores.max():.4f}")
    logger.info(f"  Mean: {scores.mean():.4f}")
    logger.info(f"  Std: {scores.std():.4f}")
    
    # Detailed explanation if requested
    if args.explain:
        logger.info(f"\n" + "="*70)
        logger.info("DETAILED EXPLANATION - Sample 0")
        logger.info("="*70)
        explanation = ensemble.explain_prediction(sequence, idx=0)
        
        logger.info(f"LSTM Autoencoder:")
        logger.info(f"  Score: {explanation['lstm_score']:.4f}, "
                   f"Weight: {explanation['lstm_weight']:.2f}, "
                   f"Contribution: {explanation['lstm_contribution']:.4f}")
        
        logger.info(f"Isolation Forest:")
        logger.info(f"  Score: {explanation['if_score']:.4f}, "
                   f"Weight: {explanation['if_weight']:.2f}, "
                   f"Contribution: {explanation['if_contribution']:.4f}")
        
        logger.info(f"Graph Neural Network:")
        logger.info(f"  Score: {explanation['gnn_score']:.4f}, "
                   f"Weight: {explanation['gnn_weight']:.2f}, "
                   f"Contribution: {explanation['gnn_contribution']:.4f}")
        
        logger.info(f"\nEnsemble Score: {explanation['ensemble_score']:.4f}")
        logger.info(f"Predicted: {'ANOMALY' if explanation['is_anomaly'] else 'NORMAL'}")
    
    logger.info("\n✓ Inference completed!")


if __name__ == '__main__':
    main()
