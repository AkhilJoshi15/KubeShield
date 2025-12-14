# KubeShield

**KubeShield: Ensemble Machine Learning for Anomaly Detection in Multi-Cloud Kubernetes Environments**

This is a comprehensive reference implementation accompanying the IEEE paper on KubeShield. It provides production-ready source code, scripts, and configuration for core platform components (Data Collection in Go, Stream Processing in Java, ML Engine in Python), as well as complete guidance for deployment on Kubernetes clusters.

**Note:** This is a research artifact with reference implementations for academic purposes. For production use, additional hardening, monitoring, and testing is recommended.

## 📋 Overview

KubeShield is an ensemble machine learning system designed to detect anomalies in multi-cloud Kubernetes environments. It combines:

- **LSTM Autoencoders** for temporal pattern detection
- **Isolation Forest** for isolation-based anomaly detection
- **Graph Neural Networks (GNN)** for relationship-based anomaly detection
- **Weighted Ensemble Fusion** for robust final predictions

The system ingests Kubernetes audit logs and system metrics, extracts features using Apache Flink, and applies ML models for real-time anomaly detection.

## 📁 Directory Structure

```
KubeShield/
├── README.md                      # This file
├── LICENSE                        # MIT License
│
├── helm/                          # Kubernetes deployment via Helm
│   ├── Chart.yaml                 # Helm chart metadata
│   ├── values.yaml                # Default Helm values
│   └── templates/
│       └── deployment.yaml        # K8s deployment template
│
├── data-collection/               # Go agent for telemetry collection
│   └── main.go                    # Data collection service
│
├── stream-processing/             # Apache Flink feature engineering
│   └── FlinkJob.java              # Streaming feature extraction
│
├── ml-engine/                     # Python ML models and inference
│   ├── train.py                   # Model training pipeline
│   ├── infer.py                   # Inference and ensemble fusion
│   └── models/
│       ├── lstm_autoencoder.py    # LSTM autoencoder model
│       ├── isolation_forest.py    # Isolation forest model
│       └── gnn.py                 # Graph neural network model
│
├── configs/                       # Configuration files
│   ├── fluent-bit.conf            # Log forwarding configuration
│   ├── kafka-config.yaml          # Kafka broker configuration
│   ├── opa-policy.yaml            # OPA policy definitions
│   └── cilium-values.yaml         # Cilium network policy values
│
├── scripts/                       # Utility and deployment scripts
│   ├── preprocess.py              # Data preprocessing utilities
│   ├── generate_synthetic_audit.py# Synthetic audit log generation
│   └── evaluate.py                # Model evaluation metrics
│
└── tests/                         # Unit and integration tests
    └── test_ml_engine.py          # ML engine test suite
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Go 1.16+
- Java 11+
- Docker & Kubernetes 1.20+
- Helm 3.0+
- Apache Flink 1.14+
- Apache Kafka 2.8+

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AkhilJoshi15/KubeShield.git
   cd KubeShield
   ```

2. **Build components:**

   **Data Collection Agent (Go):**
   ```bash
   cd data-collection
   go build -o kubeshield-collector main.go
   ```

   **Stream Processing Job (Java):**
   ```bash
   cd stream-processing
   mvn clean package
   ```

   **ML Engine (Python):**
   ```bash
   cd ml-engine
   pip install -r requirements.txt
   ```

3. **Configure your environment:**
   ```bash
   # Update configs with your cluster details
   vim configs/kafka-config.yaml
   vim configs/fluent-bit.conf
   ```

## 📊 Components

### Data Collection Agent
- **Language:** Go
- **Purpose:** Collects Kubernetes audit logs and node system metrics
- **Output:** Sends telemetry to Kafka topic `kubeshield-telemetry`
- **Location:** `data-collection/main.go`

### Stream Processing
- **Language:** Java (Apache Flink)
- **Purpose:** Real-time feature extraction and engineering from raw telemetry
- **Features:** Sliding window aggregation, statistical features, relationship graphs
- **Location:** `stream-processing/FlinkJob.java`

### ML Engine
- **Language:** Python (TensorFlow, scikit-learn, DGL)
- **Models:**
  - **LSTM Autoencoder:** Detects temporal anomalies (47% ensemble weight)
  - **Isolation Forest:** Detects outliers (28% ensemble weight)
  - **GNN:** Detects relationship anomalies (25% ensemble weight)
- **Fusion:** Weighted ensemble for final anomaly score (0.0-1.0)
- **Location:** `ml-engine/`

## 🔧 Configuration

### Kafka Configuration (`configs/kafka-config.yaml`)
Configure broker addresses, topic names, and partitions:
```yaml
broker: localhost:9092
topics:
  raw: kubeshield-telemetry
  features: kubeshield-features
```

### OPA Policies (`configs/opa-policy.yaml`)
Define security policies for anomaly response:
```yaml
policy:
  isolation: true
  alert_threshold: 0.85
```

### Cilium Network Policies (`configs/cilium-values.yaml`)
Network segmentation and traffic control for Kubernetes pods.

## 🤖 Model Training

```bash
cd ml-engine
python train.py --data features.csv --output models/
```

This trains all three models and saves them to the `models/` directory.

## 🎯 Inference

```bash
python infer.py --model-dir models/ --data test_features.csv
```

The inference pipeline loads all models, generates predictions, and fuses them using weighted ensemble:
```
final_score = 0.28 * isolation_forest_score + 0.47 * lstm_score + 0.25 * gnn_score
```

## 📈 Evaluation

```bash
python scripts/evaluate.py --predictions predictions.csv --ground_truth ground_truth.csv
```

Generates evaluation metrics (Precision, Recall, F1, AUC-ROC, etc.)

## 🧪 Testing

```bash
python -m pytest tests/
```

Runs the test suite, including:
- Model output validation
- Ensemble fusion correctness
- Score normalization checks
- Configuration validation

## 🐳 Docker Deployment

**Build images:**
```bash
# Data collector
docker build -t akhiljoshi/kubeshield-collector:latest data-collection/

# ML engine
docker build -t akhiljoshi/kubeshield-ml:latest ml-engine/
```

**Push to registry:**
```bash
docker push akhiljoshi/kubeshield-collector:latest
docker push akhiljoshi/kubeshield-ml:latest
```

## ☸️ Kubernetes Deployment

**Deploy via Helm:**
```bash
helm install kubeshield ./helm/ \
  --set collector.image=akhiljoshi/kubeshield-collector:latest \
  --set mlEngine.image=akhiljoshi/kubeshield-ml:latest
```

**Verify deployment:**
```bash
kubectl get pods -l app=kubeshield
kubectl logs -f deployment/kubeshield-mlengine
```

## 📝 Data Pipeline

```
K8s Audit Logs & Metrics
       ↓
Data Collection Agent (Go)
       ↓
Kafka Topic: kubeshield-telemetry
       ↓
Apache Flink (Feature Engineering)
       ↓
Kafka Topic: kubeshield-features
       ↓
ML Engine (Python)
  ├─ LSTM Autoencoder
  ├─ Isolation Forest
  └─ GNN
       ↓
Ensemble Fusion
       ↓
Anomaly Score (0.0-1.0)
```

## 📊 Evaluation Results

Based on the accompanying paper:

| Model | Precision | Recall | F1-Score | AUC-ROC |
|-------|-----------|--------|----------|---------|
| LSTM  | 0.92      | 0.88   | 0.90     | 0.95    |
| IF    | 0.85      | 0.81   | 0.83     | 0.91    |
| GNN   | 0.89      | 0.86   | 0.87     | 0.93    |
| **Ensemble** | **0.94** | **0.91** | **0.92** | **0.97** |

## 📚 Paper Citation

If you use KubeShield in your research, please cite the accompanying IEEE paper:

```bibtex
@article{kubeshield2024,
  title={KubeShield: Ensemble Machine Learning for Anomaly Detection in Multi-Cloud Kubernetes Environments},
  author={Joshi, Akhil and ...},
  journal={IEEE Transactions on Cloud Computing},
  year={2024}
}
```

## 🔐 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or discussions about KubeShield, please open an issue on GitHub or contact the authors.

## 🙏 Acknowledgments

This work builds upon the Kubernetes audit logging capabilities and the broader open-source ML and cloud-native communities.

---

**Last Updated:** December 2024  
**Version:** 0.1.0 (Research Artifact)
