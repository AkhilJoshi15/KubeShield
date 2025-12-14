# KubeShield

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE-green. svg)](paper/)

**Ensemble Machine Learning for Anomaly Detection in Multi-Cloud Kubernetes Environments**

## Overview

KubeShield is an AI-driven security framework for detecting anomalies in Kubernetes clusters across AWS EKS, Azure AKS, and Google GKE. It combines three complementary ML models: 

- **Isolation Forest**:  Point anomaly detection
- **LSTM Autoencoder**: Temporal pattern analysis  
- **Graph Neural Network**: Service relationship anomalies

## Key Features

- 🔍 94. 2% F1-score detection accuracy
- ☁️ Multi-cloud support (AWS, Azure, GCP)
- ⚡ Sub-100ms detection latency (with GPU)
- 📊 73 specialized security features
- 🔄 Automated response via OPA Gatekeeper

## Quick Start

```bash
# Install via Helm
helm repo add kubeshield https://aj-github.github.io/kubeshield/charts
helm install kubeshield kubeshield/kubeshield \
  --namespace kubeshield-system \
  --create-namespace
