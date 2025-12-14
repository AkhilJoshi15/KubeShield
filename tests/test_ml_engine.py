"""
KubeShield ML Engine Test Suite
Comprehensive unit tests for all ML components
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'ml-engine'))

from models.lstm_autoencoder import LSTMAutoencoder
from models.isolation_forest import IsolationForestModel
from models.gnn import GNNModel
from infer import EnsembleModel


class TestLSTMAutoencoder(unittest.TestCase):
    """Test LSTM Autoencoder model"""
    
    def setUp(self):
        """Setup test data"""
        self.sequence_length = 60
        self.feature_dim = 73
        self.num_samples = 100
        
        self.X_test = np.random.randn(
            self.num_samples, 
            self.sequence_length, 
            self.feature_dim
        ).astype(np.float32)
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        model = LSTMAutoencoder(
            sequence_length=self.sequence_length,
            feature_dim=self.feature_dim
        )
        self.assertIsNotNone(model)
        self.assertEqual(model.sequence_length, self.sequence_length)
        self.assertEqual(model.feature_dim, self.feature_dim)
    
    def test_forward_pass(self):
        """Test forward pass through model"""
        model = LSTMAutoencoder(
            sequence_length=self.sequence_length,
            feature_dim=self.feature_dim
        )
        
        # Forward pass
        output = model(self.X_test[:5])
        
        # Check output shape
        self.assertEqual(output.shape, (5, self.sequence_length, self.feature_dim))
    
    def test_anomaly_score_range(self):
        """Test anomaly scores are in [0, 1]"""
        model = LSTMAutoencoder(
            sequence_length=self.sequence_length,
            feature_dim=self.feature_dim
        )
        
        # Compute scores
        scores = model.compute_anomaly_score(self.X_test)
        
        # Check range
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))
        self.assertEqual(len(scores), self.num_samples)


class TestIsolationForest(unittest.TestCase):
    """Test Isolation Forest model"""
    
    def setUp(self):
        """Setup test data"""
        self.num_samples = 1000
        self.feature_dim = 73
        
        self.X_train = np.random.randn(self.num_samples, self.feature_dim).astype(np.float32)
        self.X_test = np.random.randn(100, self.feature_dim).astype(np.float32)
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        model = IsolationForestModel(n_estimators=150)
        self.assertIsNotNone(model)
        self.assertEqual(model.n_estimators, 150)
    
    def test_fit(self):
        """Test model fitting"""
        model = IsolationForestModel(n_estimators=50)
        model.fit(self.X_train)
        
        self.assertTrue(model.is_fitted)
    
    def test_anomaly_score_range(self):
        """Test anomaly scores are in [0, 1]"""
        model = IsolationForestModel(n_estimators=50)
        model.fit(self.X_train)
        
        scores = model.compute_anomaly_score(self.X_test)
        
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))
        self.assertEqual(len(scores), 100)
    
    def test_predictions(self):
        """Test predictions return -1 or 1"""
        model = IsolationForestModel(n_estimators=50)
        model.fit(self.X_train)
        
        predictions = model.predict_anomalies(self.X_test)
        
        unique_values = np.unique(predictions)
        self.assertTrue(np.all(np.isin(unique_values, [-1, 1])))


class TestGNN(unittest.TestCase):
    """Test Graph Neural Network model"""
    
    def setUp(self):
        """Setup test data"""
        self.num_nodes = 10
        self.feature_dim = 73
        
        self.node_features = np.random.randn(self.num_nodes, self.feature_dim).astype(np.float32)
        
        self.connections = [
            (0, 1, 0.5),
            (1, 2, 0.8),
            (2, 3, 0.3),
            (3, 4, 0.9),
            (4, 0, 0.6)
        ]
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        model = GNNModel(node_feature_dim=self.feature_dim)
        self.assertIsNotNone(model)
        self.assertEqual(model.node_feature_dim, self.feature_dim)
    
    def test_graph_creation(self):
        """Test graph creation"""
        model = GNNModel(node_feature_dim=self.feature_dim)
        graph = model.create_graph(self.node_features, self.connections)
        
        self.assertEqual(graph.node_features.shape, (self.num_nodes, self.feature_dim))
        self.assertEqual(graph.edge_index.shape[1], len(self.connections))
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = GNNModel(node_feature_dim=self.feature_dim)
        graph = model.create_graph(self.node_features, self.connections)
        
        embeddings = model.forward(graph)
        
        self.assertEqual(embeddings.shape[0], self.num_nodes)
        self.assertEqual(embeddings.shape[1], model.hidden_dim)
    
    def test_anomaly_score_range(self):
        """Test anomaly scores are in [0, 1]"""
        model = GNNModel(node_feature_dim=self.feature_dim)
        graph = model.create_graph(self.node_features, self.connections)
        
        score = model.compute_anomaly_score(graph)
        
        self.assertTrue(0 <= score <= 1)


class TestEnsembleModel(unittest.TestCase):
    """Test ensemble fusion"""
    
    def test_fuse_scores(self):
        """Test score fusion"""
        s_lstm = np.array([0.3, 0.7, 0.9])
        s_if = np.array([0.2, 0.6, 0.8])
        s_gnn = np.array([0.4, 0.5, 0.85])
        
        ensemble_scores = EnsembleModel.fuse_scores(s_lstm, s_if, s_gnn)
        
        # Check output range
        self.assertTrue(np.all(ensemble_scores >= 0))
        self.assertTrue(np.all(ensemble_scores <= 1))
        self.assertEqual(len(ensemble_scores), 3)
        
        # Verify weighted sum (0.47*lstm + 0.28*if + 0.25*gnn)
        expected_0 = 0.47 * 0.3 + 0.28 * 0.2 + 0.25 * 0.4
        self.assertAlmostEqual(ensemble_scores[0], expected_0, places=5)
    
    def test_fuse_scores_bounds(self):
        """Test fusion with extreme values"""
        s_lstm = np.array([1.0, 0.0])
        s_if = np.array([1.0, 0.0])
        s_gnn = np.array([1.0, 0.0])
        
        ensemble_scores = EnsembleModel.fuse_scores(s_lstm, s_if, s_gnn)
        
        self.assertAlmostEqual(ensemble_scores[0], 1.0, places=5)
        self.assertAlmostEqual(ensemble_scores[1], 0.0, places=5)


class TestConfigValidation(unittest.TestCase):
    """Test configuration files and constants"""
    
    def test_ensemble_weights_sum(self):
        """Test ensemble weights sum to 1"""
        weights = EnsembleModel.WEIGHTS
        total = sum(weights.values())
        
        self.assertAlmostEqual(total, 1.0, places=10)
    
    def test_feature_dimensions(self):
        """Test feature dimensions are consistent"""
        expected_feature_dim = 73
        
        # Check LSTM
        lstm = LSTMAutoencoder(sequence_length=60, feature_dim=expected_feature_dim)
        self.assertEqual(lstm.feature_dim, expected_feature_dim)
        
        # Check GNN
        gnn = GNNModel(node_feature_dim=expected_feature_dim)
        self.assertEqual(gnn.node_feature_dim, expected_feature_dim)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLSTMAutoencoder))
    suite.addTests(loader.loadTestsFromTestCase(TestIsolationForest))
    suite.addTests(loader.loadTestsFromTestCase(TestGNN))
    suite.addTests(loader.loadTestsFromTestCase(TestEnsembleModel))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    print("="*70)
    print("KubeShield ML Engine Test Suite")
    print("="*70)
    
    exit_code = run_tests()
    
    if exit_code == 0:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
    
    sys.exit(exit_code)
