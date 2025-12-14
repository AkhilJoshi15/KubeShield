"""
CICIDS2018 Dataset Preprocessing for KubeShield

Extracts network flow features from CICIDS2018 dataset.
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CICIDSPreprocessor:
    """Preprocess CICIDS2018 network flows into features."""

    # Attack label mapping
    ATTACK_LABELS = {
        "Benign": 0,
        "Bot": 1,
        "DDoS attack-HOIC": 1,
        "DDoS attack-LOIC-UDP": 1,
        "DDoS attacks-LOIC-HTTP": 1,
        "DoS attacks-GoldenEye": 1,
        "DoS attacks-Hulk": 1,
        "DoS attacks-SlowHTTPTest": 1,
        "DoS attacks-Slowloris": 1,
        "FTP-BruteForce": 1,
        "Infiltration": 1,
        "SQL Injection": 1,
        "SSH-BruteForce": 1,
        "Brute Force -Web": 1,
        "Brute Force -XSS": 1,
        "DDOS attack-HOIC": 1,
        "DDOS attack-LOIC-UDP": 1,
    }

    # Selected features for our 23 network features
    SELECTED_FEATURES = [
        # Volume metrics (6)
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Flow Bytes/s",
        # Connection patterns (10)
        "Flow Packets/s",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Mean",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Fwd IAT Mean",
        "Bwd IAT Mean",
        "Fwd PSH Flags",
        "Bwd PSH Flags",
        "Fwd URG Flags",
        # Protocol/behavioral (7)
        "Fwd Header Length",
        "Bwd Header Length",
        "Packet Length Mean",
        "Packet Length Std",
        "FIN Flag Count",
        "SYN Flag Count",
        "RST Flag Count",
    ]

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_and_process_file(self, filepath: Path) -> pd.DataFrame:
        """Load and process a single CICIDS CSV file."""
        logger.info(f"Processing {filepath.name}")

        try:
            # Read CSV
            df = pd.read_csv(filepath, low_memory=False)

            # Clean column names
            df.columns = df.columns.str.strip()

            # Map labels
            if "Label" in df.columns:
                df["label"] = df["Label"].map(lambda x: self.ATTACK_LABELS.get(str(x).strip(), 1))
            else:
                df["label"] = 0

            # Select features (handle missing columns gracefully)
            available_features = [f for f in self.SELECTED_FEATURES if f in df.columns]

            if len(available_features) < len(self.SELECTED_FEATURES):
                missing = set(self.SELECTED_FEATURES) - set(available_features)
                logger.warning(f"Missing features: {missing}")

            # Extract selected features
            feature_df = df[available_features + ["label"]].copy()

            # Handle infinities and NaNs
            feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
            feature_df = feature_df.fillna(0)

            # Normalize features
            for col in available_features:
                if feature_df[col].std() > 0:
                    feature_df[col] = (feature_df[col] - feature_df[col].mean()) / feature_df[col].std()

            return feature_df

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return pd.DataFrame()

    def process_dataset(self) -> pd.DataFrame:
        """Process entire CICIDS2018 dataset."""
        logger.info(f"Processing CICIDS2018 dataset from {self.data_dir}")

        if not self.data_dir.exists():
            logger.error(f"Data directory {self.data_dir} does not exist")
            return pd.DataFrame()

        all_dfs: List[pd.DataFrame] = []

        # Process each CSV file
        for csv_file in sorted(self.data_dir.glob("*.csv")):
            df = self.load_and_process_file(csv_file)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            logger.error("No data processed!")
            return pd.DataFrame()

        # Combine all files
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Rename columns to our feature naming convention
        feature_mapping = {
            "Flow Duration": "net_flow_duration",
            "Total Fwd Packets": "net_fwd_packets",
            "Total Backward Packets": "net_bwd_packets",
            "Total Length of Fwd Packets": "net_fwd_bytes",
            "Total Length of Bwd Packets": "net_bwd_bytes",
            "Flow Bytes/s": "net_bytes_per_sec",
            "Flow Packets/s": "net_packets_per_sec",
            "Fwd Packet Length Mean": "net_fwd_pkt_mean",
            "Bwd Packet Length Mean": "net_bwd_pkt_mean",
            "Flow IAT Mean": "net_iat_mean",
            "Flow IAT Std": "net_iat_std",
            "Fwd IAT Mean": "net_fwd_iat_mean",
            "Bwd IAT Mean": "net_bwd_iat_mean",
            "Fwd PSH Flags": "net_fwd_psh",
            "Bwd PSH Flags": "net_bwd_psh",
            "Fwd URG Flags": "net_fwd_urg",
            "Fwd Header Length": "net_fwd_header_len",
            "Bwd Header Length": "net_bwd_header_len",
            "Packet Length Mean": "net_pkt_len_mean",
            "Packet Length Std": "net_pkt_len_std",
            "FIN Flag Count": "net_fin_count",
            "SYN Flag Count": "net_syn_count",
            "RST Flag Count": "net_rst_count",
        }
        combined_df = combined_df.rename(columns=feature_mapping)

        # Summary statistics
        if "label" in combined_df.columns:
            n_attack = int(combined_df["label"].sum())
            n_normal = len(combined_df) - n_attack
        else:
            n_attack = 0
            n_normal = len(combined_df)
        logger.info(f"Processed {len(combined_df)} flows: {n_attack} attacks, {n_normal} normal")

        return combined_df

    def save_features(self, df: pd.DataFrame):
        """Save processed features."""
        output_path = self.output_dir / "cicids_features.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved features to {output_path}")

        # Also save sample CSV
        csv_path = self.output_dir / "cicids_features_sample.csv"
        df.head(1000).to_csv(csv_path, index=False)
        logger.info(f"Saved sample CSV to {csv_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess CICIDS2018 dataset")
    parser.add_argument("--data-dir", type=str, default="raw/cicids2018", help="Path to raw CICIDS2018 data")
    parser.add_argument("--output-dir", type=str, default="data/processed/cicids", help="Output directory for processed features")
    args = parser.parse_args()

    preprocessor = CICIDSPreprocessor(args.data_dir, args.output_dir)
    df = preprocessor.process_dataset()
    if not df.empty:
        preprocessor.save_features(df)


if __name__ == "__main__":
    main()