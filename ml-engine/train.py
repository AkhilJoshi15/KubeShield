"""
KubeShield Model Training Pipeline
Trains all three models: LSTM Autoencoder, Isolation Forest, and GNN
"""

import numpy as np
import argparse
import logging
import os
from pathlib import Path
import pickle
import json

from models.lstm_autoencoder import LSTMAutoencoder
from models.isolation_forest import IsolationForestModel
from models.gnn import GNNModel, GraphData

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and preprocessing"""
    
    @staticmethod
    def generate_synthetic_data(num_samples=10000, sequence_length=60, feature_dim=73):
        """
        Generate synthetic training data
        
        Args:
            num_samples: Number of samples
            sequence_length: Length of sequences for LSTM
            feature_dim: Number of features
        
        Returns:
            Tuple of (sequences, flat_features)
        """
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        # Generate sequences for LSTM (normal behavior pattern)
        sequences = np.zeros((num_samples, sequence_length, feature_dim), dtype=np.float32)
        
        for i in range(num_samples):
            # Create temporal patterns
            for t in range(sequence_length):
                # Sinusoidal pattern + noise
                sequences[i, t, :] = (
                    50 * np.sin(2 * np.pi * t / sequence_length) +
                    np.random.randn(feature_dim) * 10
                )
        
        # Flat features for isolation forest and reference
        flat_features = sequences[:, -1, :]  # Use last timestep
        
        logger.info(f"Sequences shape: {sequences.shape}")
        logger.info(f"Flat features shape: {flat_features.shape}")
        
        return sequences, flat_features
    
    @staticmethod
    def load_data(filepath):
        """Load data from CSV file"""
        logger.info(f"Loading data from {filepath}")
        data = np.loadtxt(filepath, delimiter=',')
        logger.info(f"Data shape: {data.shape}")
        return data


class TrainingPipeline:
    """Orchestrates training of all three models"""
    
    def __init__(self, output_dir='models'):
        """
        Initialize training pipeline
        
        Args:
            output_dir: Directory to save trained models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
    
    def train_lstm(self, sequences, val_split=0.2, epochs=50, batch_size=32):
        """Train LSTM Autoencoder"""
        logger.info("\n" + "="*70)
        logger.info("TRAINING LSTM AUTOENCODER")
        logger.info("="*70)
        
        # Split data
        split_idx = int(len(sequences) * (1 - val_split))
        X_train = sequences[:split_idx]
        X_val = sequences[split_idx:]
        
        logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Create and train model
        model = LSTMAutoencoder(
            sequence_length=sequences.shape[1],
            feature_dim=sequences.shape[2],
            encoding_dim=16
        )
        
        history = model.train_model(
            X_train, X_val=X_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Compute anomaly scores on training data
        scores = model.compute_anomaly_score(X_train[:100])
        logger.info(f"Anomaly score stats - Min: {scores.min():.4f}, "
                   f"Max: {scores.max():.4f}, Mean: {scores.mean():.4f}")
        
        # Save model
        model.save_model(str(self.output_dir / 'lstm_autoencoder.h5'))
        
        return model
    
    def train_isolation_forest(self, flat_features):
        """Train Isolation Forest"""
        logger.info("\n" + "="*70)
        logger.info("TRAINING ISOLATION FOREST")
        logger.info("="*70)
        
        model = IsolationForestModel(
            n_estimators=150,
            contamination=0.05,
            random_state=42
        )
        
        model.fit(flat_features)
        
        # Compute anomaly scores
        scores = model.compute_anomaly_score(flat_features[:100])
        logger.info(f"Anomaly score stats - Min: {scores.min():.4f}, "
                   f"Max: {scores.max():.4f}, Mean: {scores.mean():.4f}")
        
        # Save model
        model.save_model(str(self.output_dir / 'isolation_forest'))
        
        return model
    
    def train_gnn(self, flat_features, num_graphs=100):
        """Train Graph Neural Network"""
        logger.info("\n" + "="*70)
        logger.info("TRAINING GRAPH NEURAL NETWORK")
        logger.info("="*70)
        
        model = GNNModel(
            node_feature_dim=flat_features.shape[1],
            edge_feature_dim=16,
            hidden_dim=128,
            num_layers=3
        )
        
        # Create synthetic graphs
        logger.info(f"Creating {num_graphs} synthetic graphs...")
        embeddings = []
        
        for i in range(num_graphs):
            # Select random nodes from features
            num_nodes = np.random.randint(5, 25)
            node_indices = np.random.choice(len(flat_features), num_nodes, replace=False)
            node_features = flat_features[node_indices]
            
            # Create random connections
            num_edges = np.random.randint(5, num_nodes * 2)
            connections = [
                (np.random.randint(0, num_nodes),
                 np.random.randint(0, num_nodes),
                 np.random.rand())
                for _ in range(num_edges)
            ]
            
            # Create graph and compute embedding
            graph = model.create_graph(node_features, connections)
            embedding = model.compute_graph_embedding(graph)
            embeddings.append(embedding)
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Processed {i + 1}/{num_graphs} graphs")
        
        embeddings = np.array(embeddings)
        logger.info(f"Embedding statistics - Shape: {embeddings.shape}")
        logger.info(f"Embedding mean: {embeddings.mean():.4f}, "
                   f"std: {embeddings.std():.4f}")
        
        # Save model metadata
        model_metadata = {
            'node_feature_dim': model.node_feature_dim,
            'hidden_dim': model.hidden_dim,
            'num_layers': model.num_layers,
            'num_graphs_trained': num_graphs
        }
        
        with open(self.output_dir / 'gnn_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        logger.info(f"GNN model metadata saved")
        
        return model, embeddings
    
    def train_all(self, sequences=None, flat_features=None, epochs=50):
        """Train all models"""
        
        # Generate or load data
        if sequences is None or flat_features is None:
            sequences, flat_features = DataLoader.generate_synthetic_data(
                num_samples=10000,
                sequence_length=60,
                feature_dim=73
            )
        
        # Train models
        lstm_model = self.train_lstm(sequences, epochs=epochs)
        if_model = self.train_isolation_forest(flat_features)
        gnn_model, gnn_embeddings = self.train_gnn(flat_features)
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETED")
        logger.info("="*70)
        logger.info(f"Models saved to: {self.output_dir}")
        
        return {
            'lstm': lstm_model,
            'isolation_forest': if_model,
            'gnn': gnn_model,
            'gnn_embeddings': gnn_embeddings
        }


def main():
    parser = argparse.ArgumentParser(description='Train KubeShield ML models')
    parser.add_argument('--data', type=str, help='Path to training data (CSV)')
    parser.add_argument('--output', type=str, default='models',
                       help='Output directory for models (default: models)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of synthetic samples if no data provided (default: 10000)')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("KubeShield ML Model Training Pipeline")
    logger.info("="*70)
    
    # Initialize pipeline
    pipeline = TrainingPipeline(output_dir=args.output)
    
    # Load or generate data
    if args.data:
        data = DataLoader.load_data(args.data)
        # Assume data has shape (num_samples, 73)
        sequences = np.expand_dims(data, axis=1)
        sequences = np.tile(sequences, (1, 60, 1))
        flat_features = data
    else:
        sequences, flat_features = DataLoader.generate_synthetic_data(
            num_samples=args.samples,
            sequence_length=60,
            feature_dim=73
        )
    
    # Train all models
    models = pipeline.train_all(sequences, flat_features, epochs=args.epochs)
    
    logger.info("\n✓ All models trained and saved successfully!")


if __name__ == '__main__':
    main()
