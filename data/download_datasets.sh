#!/bin/bash
# Download LID-DS and CICIDS2018 datasets for KubeShield
# Usage: ./data/download_datasets.sh

set -euo pipefail

echo "================================================"
echo "KubeShield Dataset Downloader"
echo "================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/raw"
mkdir -p "${DATA_DIR}"

# ------------------------------------------------------------------------------
# LID-DS Dataset
# ------------------------------------------------------------------------------
echo ""
echo "[1/2] LID-DS Dataset (manual download required)"
echo "Source: https://2id-ds.2gi.2ww3.2cs.2uni-leipzig.de/"

LIDDS_DIR="${DATA_DIR}/lidds"
mkdir -p "${LIDDS_DIR}"

# NOTE: LID-DS may require manual download due to terms / captcha / redirects.
echo ""
echo "Please download LID-DS manually from:"
echo "  https://2id-ds.2gi.2ww3.2cs.2uni-leipzig.de/"
echo "and place files in: ${LIDDS_DIR}"
echo ""
echo "Required files (example):"
echo "  - LID-DS-2021.tar.gz"
echo ""
echo "If you have direct URLs, uncomment and edit the wget/tar commands below."

# Example:
# wget -O "${LIDDS_DIR}/LID-DS-2021.tar.gz" "https://example.com/LID-DS-2021.tar.gz"
# tar -xzf "${LIDDS_DIR}/LID-DS-2021.tar.gz" -C "${LIDDS_DIR}"

# ------------------------------------------------------------------------------
# CICIDS2018 Dataset
# ------------------------------------------------------------------------------
echo ""
echo "[2/2] CICIDS2018 Dataset (AWS S3 sync or manual download)"
echo "Source: https://www.unb.ca/cic/datasets/ids-2018.html"

CICIDS_DIR="${DATA_DIR}/cicids2018"
mkdir -p "${CICIDS_DIR}"

# CICIDS2018 is available via AWS S3 (large, ~7GB)
echo "Attempting to download from known S3 bucket (no-sign-request)..."

if command -v aws &> /dev/null; then
    aws s3 sync --no-sign-request \
        "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" \
        "${CICIDS_DIR}/" || {
            echo "AWS S3 sync failed or bucket layout changed. Please download manually."
        }
else
    echo "AWS CLI not found. Please install it or download manually from:"
    echo "  https://www.unb.ca/cic/datasets/ids-2018.html"
    echo ""
    echo "Alternative: Download CSV files and place them in: ${CICIDS_DIR}"
fi

# ------------------------------------------------------------------------------
# Verify downloads
# ------------------------------------------------------------------------------
echo ""
echo "================================================"
echo "Download Summary"
echo "================================================"
echo "LID-DS:     ${LIDDS_DIR}"
echo "CICIDS2018: ${CICIDS_DIR}"
echo ""
echo "Next steps:"
echo "  1. Verify files are present"
echo "  2. Run:  python data/preprocess/preprocess_lidds.py --data-dir ${LIDDS_DIR} --output-dir data/processed/lidds"
echo "  3. Run:  python data/preprocess/preprocess_cicids.py --data-dir ${CICIDS_DIR} --output-dir data/processed/cicids"
echo "================================================"
