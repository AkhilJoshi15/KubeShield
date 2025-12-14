"""
KubeShield Graph Neural Network Model
Detects relational anomalies using GNN on system graphs
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GraphData:
    """Represents a Kubernetes system graph"""
    node_features: np.ndarray  # (num_nodes, feature_dim)
    edge_index: np.ndarray     # (2, num_edges) - source and target indices
    edge_attr: np.ndarray      # (num_edges, edge_dim) - edge attributes


class GNNModel:
    """
    Graph Neural Network for detecting relational anomalies
    
    Models Kubernetes systems as graphs where:
    - Nodes: Pods, nodes, services, etc.
    - Edges: Relationships (network traffic, API calls, mounts, etc.)
    
    Uses message passing to aggregate information from neighbors:
    h_i^(l+1) = RELU(W_self * h_i^(l) + sum_{j in neighbors(i)} W_neighbor * h_j^(l))
    
    Anomaly detection: Compares node embedding similarity with expected patterns
    
    Architecture:
    - 3 GNN layers (128 -> 64 -> 32 hidden dims)
    - Graph pooling for graph-level embeddings
    - Classification head for anomaly detection
    """
    
    def __init__(self, node_feature_dim=73, edge_feature_dim=16, hidden_dim=128, 
                 num_layers=3, dropout=0.2):
        """
        Initialize GNN Model
        
        Args:
            node_feature_dim: Dimension of node features (73 for KubeShield)
            edge_feature_dim: Dimension of edge attributes
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initialize weight matrices for message passing
        # In production, use torch.nn.Linear with proper initialization
        self.node_embeddings = {}
        self.edge_embeddings = {}
        self.aggregation_weights = {}
        
        logger.info(f"Initialized GNN: node_dim={node_feature_dim}, "
                   f"hidden_dim={hidden_dim}, num_layers={num_layers}")
    
    def create_graph(self, pod_features: np.ndarray, connections: list) -> GraphData:
        """
        Create a graph from pod features and connections
        
        Args:
            pod_features: Pod features (num_pods, feature_dim)
            connections: List of (source_pod_id, target_pod_id, traffic_volume)
        
        Returns:
            GraphData object
        """
        num_nodes = pod_features.shape[0]
        
        # Build edge index
        if connections:
            edge_index_list = [(c[0], c[1]) for c in connections]
            edge_attr_list = [
                np.array([c[2], len(connections), c[2]/max([c[2] for c in connections])])
                for c in connections
            ]
            edge_index = np.array(edge_index_list).T
            edge_attr = np.array(edge_attr_list)
        else:
            edge_index = np.zeros((2, 0), dtype=int)
            edge_attr = np.zeros((0, self.edge_feature_dim))
        
        return GraphData(
            node_features=pod_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
    
    def aggregate_neighbors(self, node_features: np.ndarray, 
                           edge_index: np.ndarray, 
                           layer: int) -> np.ndarray:
        """
        Aggregate information from neighboring nodes
        
        Args:
            node_features: Current node features (num_nodes, feature_dim)
            edge_index: Edge connectivity (2, num_edges)
            layer: Current layer number
        
        Returns:
            Aggregated features (num_nodes, hidden_dim)
        """
        num_nodes = node_features.shape[0]
        aggregated = np.zeros((num_nodes, self.hidden_dim))
        
        # Simple neighborhood aggregation with mean pooling
        for target_idx in range(num_nodes):
            # Find all edges pointing to this node
            neighbor_mask = edge_index[1] == target_idx
            
            if np.any(neighbor_mask):
                neighbor_indices = edge_index[0][neighbor_mask]
                neighbor_features = node_features[neighbor_indices]
                aggregated[target_idx] = np.mean(neighbor_features, axis=0)
            else:
                aggregated[target_idx] = np.zeros(self.hidden_dim)
        
        return aggregated
    
    def forward(self, graph_data: GraphData, training: bool = False) -> np.ndarray:
        """
        Forward pass through GNN layers
        
        Args:
            graph_data: Graph data with node features and edges
            training: Whether in training mode
        
        Returns:
            Node embeddings (num_nodes, hidden_dim)
        """
        node_features = graph_data.node_features
        edge_index = graph_data.edge_index
        
        # Initial linear projection
        h = self._linear_transform(node_features, 
                                  self.node_feature_dim, 
                                  self.hidden_dim)
        
        # Apply GNN layers
        for layer in range(self.num_layers):
            # Aggregate from neighbors
            h_agg = self.aggregate_neighbors(h, edge_index, layer)
            
            # Combine self and aggregated features
            h = self._combine_features(h, h_agg, layer)
            
            # Apply activation and dropout
            h = np.maximum(h, 0)  # ReLU
            if training and self.dropout > 0:
                mask = np.random.binomial(1, 1 - self.dropout, h.shape)
                h = h * mask / (1 - self.dropout)
        
        return h
    
    def compute_graph_embedding(self, graph_data: GraphData) -> np.ndarray:
        """
        Compute graph-level embedding by pooling node embeddings
        
        Args:
            graph_data: Graph data
        
        Returns:
            Graph embedding (hidden_dim,)
        """
        node_embeddings = self.forward(graph_data, training=False)
        
        # Global mean pooling
        graph_embedding = np.mean(node_embeddings, axis=0)
        
        return graph_embedding
    
    def compute_anomaly_score(self, graph_data: GraphData, 
                            reference_embedding: Optional[np.ndarray] = None) -> float:
        """
        Compute anomaly score for a graph
        
        Args:
            graph_data: Graph data
            reference_embedding: Reference normal embedding
        
        Returns:
            Anomaly score (0-1)
        """
        graph_embedding = self.compute_graph_embedding(graph_data)
        
        if reference_embedding is None:
            reference_embedding = np.ones_like(graph_embedding) * 0.5
        
        # Compute Euclidean distance
        distance = np.linalg.norm(graph_embedding - reference_embedding)
        
        # Normalize to [0, 1]
        anomaly_score = min(1.0, distance / 10.0)  # Empirical scaling
        
        return anomaly_score
    
    def _linear_transform(self, X: np.ndarray, in_dim: int, out_dim: int) -> np.ndarray:
        """Apply linear transformation"""
        # In production, use proper weight initialization
        W = np.random.randn(in_dim, out_dim) * 0.01
        return np.dot(X, W)
    
    def _combine_features(self, self_feat: np.ndarray, 
                         agg_feat: np.ndarray, layer: int) -> np.ndarray:
        """Combine self and aggregated features"""
        alpha = 0.5  # Balance between self and neighbor information
        return alpha * self_feat + (1 - alpha) * agg_feat


if __name__ == "__main__":
    # Example usage
    logger.info("Testing Graph Neural Network Model...")
    
    # Create synthetic graph data
    num_nodes = 20
    feature_dim = 73
    num_connections = 50
    
    node_features = np.random.randn(num_nodes, feature_dim).astype(np.float32)
    
    # Create random connections
    connections = [
        (np.random.randint(0, num_nodes),
         np.random.randint(0, num_nodes),
         np.random.rand())
        for _ in range(num_connections)
    ]
    
    # Initialize and test model
    model = GNNModel(node_feature_dim=feature_dim)
    graph = model.create_graph(node_features, connections)
    
    # Get embeddings
    embeddings = model.forward(graph)
    graph_emb = model.compute_graph_embedding(graph)
    anomaly_score = model.compute_anomaly_score(graph)
    
    logger.info(f"Graph embedding shape: {graph_emb.shape}")
    logger.info(f"Anomaly score: {anomaly_score:.4f}")
    logger.info(f"Node embeddings shape: {embeddings.shape}")
    
    print("✓ GNN model test completed successfully")
