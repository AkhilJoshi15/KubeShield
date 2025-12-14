# KubeShield

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://python.org)
[![Paper](https://img.shields.io/badge/Paper-IEEE-orange.svg)](paper/)

**Ensemble Machine Learning for Anomaly Detection in Multi-Cloud Kubernetes Environments**

## 📖 Overview

KubeShield is an AI-driven security framework for detecting anomalies in Kubernetes clusters.  It combines three complementary ML models: 

| Model | Purpose | Architecture |
|-------|---------|--------------|
| **Isolation Forest** | Point anomaly detection | 150 trees, contamination=0.015 |
| **LSTM Autoencoder** | Temporal pattern analysis | 2-layer BiLSTM, 128 hidden |
| **Graph Neural Network** | Service relationship anomalies | 3-layer GAT, 4 attention heads |

## 🎯 Key Results

- **94.2% F1-Score** on combined LID-DS + CICIDS2018 dataset
- **18.7% improvement** over best baseline (LSTM-only)
- **3.8% False Positive Rate** with confidence-based routing
- **89ms end-to-end latency** with GPU acceleration

## 📊 Datasets

We evaluate on two public datasets: 

| Dataset | Type | Size | Source |
|---------|------|------|--------|
| **LID-DS** | System Calls | 1.5M sequences | [Leipzig University](https://2id-ds.2gi.2ww3.2cs.2uni-leipzig.de/) |
| **CICIDS2018** | Network Flows | 16M flows | [UNB CIC](https://www.unb.ca/cic/datasets/ids-2018.html) |

### Download Datasets

```bash
# Download and preprocess datasets
./data/download_datasets.sh

# Generate synthetic Kubernetes audit logs
python data/preprocess/generate_audit_logs.py
