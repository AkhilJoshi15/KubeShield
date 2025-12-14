# KubeShield: Ensemble ML for Kubernetes Security

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**94.2% F1-score** ensemble anomaly detection for multi-cloud Kubernetes environments.

## Overview

KubeShield combines Isolation Forest, LSTM Autoencoder, and Graph Neural Networks to detect security anomalies across audit logs, network flows, and system calls.

## Key Results

- **F1-Score**: 94.2% (18.7% improvement over baselines)
- **False Positive Rate**: 3.8%
- **Multi-Cloud Support**: AWS EKS, Azure AKS, GCP GKE
- **Latency**: 34ms end-to-end

## Installation

```bash
git clone https://github.com/aj-github/kubeshield.git
cd kubeshield
pip install -r requirements.txt
