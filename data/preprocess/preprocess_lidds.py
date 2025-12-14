"""
LID-DS Dataset Preprocessing for KubeShield

Extracts system call features from LID-DS dataset.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LIDDSPreprocessor:
    """Preprocess LID-DS system call traces into features."""

    # Security-relevant system calls
    SECURITY_SYSCALLS = {
        "execve",
        "execveat",  # Process execution
        "open",
        "openat",
        "creat",  # File access
        "read",
        "write",
        "pread64",
        "pwrite64",  # File I/O
        "connect",
        "accept",
        "bind",
        "listen",  # Network
        "socket",
        "socketpair",  # Socket creation
        "clone",
        "fork",
        "vfork",  # Process creation
        "ptrace",  # Debugging (often malicious)
        "mmap",
        "mprotect",  # Memory operations
        "unlink",
        "unlinkat",
        "rmdir",  # File deletion
        "chmod",
        "fchmod",
        "chown",
        "fchown",  # Permission changes
        "setuid",
        "setgid",
        "setreuid",
        "setregid",  # Privilege changes
        "mount",
        "umount",  # Filesystem mount
        "prctl",  # Process control
    }

    # Sensitive paths
    SENSITIVE_PATHS = [
        "/etc/passwd",
        "/etc/shadow",
        "/etc/sudoers",
        "/var/run/secrets",
        "/var/run/docker.sock",
        "/proc/self",
        "/proc/1",
        "/root",
        "/home",
    ]

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_features(self, syscall_sequence: List[Dict]) -> Dict[str, float]:
        """Extract 16 system call features from a sequence."""
        features: Dict[str, float] = {}

        # Count syscalls
        syscall_counts = Counter(s.get("syscall", "") for s in syscall_sequence)
        total_calls = len(syscall_sequence)

        # Feature names to return when empty
        feature_names = [
            "process_creation_rate",
            "exec_rate",
            "unique_binary_count",
            "shell_invocation_rate",
            "interpreter_rate",
            "compiler_rate",
            "sensitive_path_access",
            "temp_file_rate",
            "config_modification_rate",
            "permission_change_rate",
            "file_deletion_rate",
            "socket_creation_rate",
            "connection_rate",
            "raw_socket_rate",
            "ptrace_rate",
            "privilege_change_rate",
        ]

        if total_calls == 0:
            return {fn: 0.0 for fn in feature_names}

        # Feature 1-6: Execution behavior
        features["process_creation_rate"] = (
            syscall_counts.get("clone", 0)
            + syscall_counts.get("fork", 0)
            + syscall_counts.get("vfork", 0)
        ) / total_calls

        features["exec_rate"] = (
            syscall_counts.get("execve", 0) + syscall_counts.get("execveat", 0)
        ) / total_calls

        # Unique binaries (from execve arguments)
        exec_calls = [s for s in syscall_sequence if s.get("syscall") in ("execve", "execveat")]
        unique_binaries = len(
            {
                str(s.get("args", [""])[0])
                for s in exec_calls
                if s.get("args") and len(s.get("args", [])) > 0
            }
        )
        features["unique_binary_count"] = float(unique_binaries)

        # Shell invocations
        shell_patterns = ["/bin/sh", "/bin/bash", "/bin/zsh", "/bin/dash"]
        shell_count = sum(
            1
            for s in exec_calls
            if any(p in str(s.get("args", "")) for p in shell_patterns)
        )
        features["shell_invocation_rate"] = shell_count / max(len(exec_calls), 1)

        # Interpreted languages
        interp_patterns = ["python", "ruby", "node", "perl", "php"]
        interp_count = sum(
            1
            for s in exec_calls
            if any(p in str(s.get("args", "")) for p in interp_patterns)
        )
        features["interpreter_rate"] = interp_count / max(len(exec_calls), 1)

        # Compiler usage
        compiler_patterns = ["gcc", "g++", "clang", "make", "cmake"]
        compiler_count = sum(
            1
            for s in exec_calls
            if any(p in str(s.get("args", "")) for p in compiler_patterns)
        )
        features["compiler_rate"] = compiler_count / max(len(exec_calls), 1)

        # Feature 7-11: File system access
        file_calls = [
            s
            for s in syscall_sequence
            if s.get("syscall") in ("open", "openat", "creat", "read", "write")
        ]

        # Sensitive path access
        sensitive_access = sum(
            1
            for s in file_calls
            if any(p in str(s.get("args", "")) for p in self.SENSITIVE_PATHS)
        )
        features["sensitive_path_access"] = sensitive_access / max(len(file_calls), 1)

        # Temp file creation
        temp_patterns = ["/tmp/", "/var/tmp/", "/dev/shm/"]
        temp_access = sum(
            1 for s in file_calls if any(p in str(s.get("args", "")) for p in temp_patterns)
        )
        features["temp_file_rate"] = temp_access / max(len(file_calls), 1)

        # Config file modifications
        config_patterns = ["/etc/", ".conf", ".cfg", ".ini"]
        config_access = sum(
            1 for s in file_calls if any(p in str(s.get("args", "")) for p in config_patterns)
        )
        features["config_modification_rate"] = config_access / max(len(file_calls), 1)

        # Permission changes
        perm_calls = syscall_counts.get("chmod", 0) + syscall_counts.get("fchmod", 0)
        features["permission_change_rate"] = perm_calls / total_calls

        # File deletion
        delete_calls = (
            syscall_counts.get("unlink", 0)
            + syscall_counts.get("unlinkat", 0)
            + syscall_counts.get("rmdir", 0)
        )
        features["file_deletion_rate"] = delete_calls / total_calls

        # Feature 12-16: Network and IPC
        socket_calls = syscall_counts.get("socket", 0) + syscall_counts.get("socketpair", 0)
        features["socket_creation_rate"] = socket_calls / total_calls

        connect_calls = syscall_counts.get("connect", 0)
        features["connection_rate"] = connect_calls / total_calls

        # Raw socket detection (SOCK_RAW = 3)
        socket_events = [s for s in syscall_sequence if s.get("syscall") == "socket"]
        raw_sockets = sum(1 for s in socket_events if "3" in str(s.get("args", "")))
        features["raw_socket_rate"] = raw_sockets / max(len(socket_events), 1)

        # ptrace usage (often malicious)
        features["ptrace_rate"] = syscall_counts.get("ptrace", 0) / total_calls

        # Privilege escalation attempts
        priv_calls = (
            syscall_counts.get("setuid", 0)
            + syscall_counts.get("setgid", 0)
            + syscall_counts.get("setreuid", 0)
            + syscall_counts.get("setregid", 0)
        )
        features["privilege_change_rate"] = priv_calls / total_calls

        # Ensure all expected feature keys exist
        for fn in feature_names:
            features.setdefault(fn, 0.0)

        return features

    def process_dataset(self) -> pd.DataFrame:
        """Process entire LID-DS dataset."""
        logger.info(f"Processing LID-DS dataset from {self.data_dir}")

        all_features = []
        all_labels = []

        # Process each scenario directory
        if not self.data_dir.exists():
            logger.error(f"Data directory {self.data_dir} does not exist")
            return pd.DataFrame()

        for scenario_dir in sorted(self.data_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue

            logger.info(f"Processing scenario: {scenario_dir.name}")

            # Determine label from directory name
            is_attack = "attack" in scenario_dir.name.lower() or "exploit" in scenario_dir.name.lower()

            # Process each trace file
            for trace_file in scenario_dir.glob("*.json"):
                try:
                    with open(trace_file, "r") as f:
                        syscall_sequence = json.load(f)

                    features = self.extract_features(syscall_sequence)
                    features["scenario"] = scenario_dir.name
                    features["file"] = trace_file.name

                    all_features.append(features)
                    all_labels.append(1 if is_attack else 0)

                except Exception as e:
                    logger.warning(f"Error processing {trace_file}: {e}")

        # Create DataFrame
        if not all_features:
            logger.error("No features extracted.")
            return pd.DataFrame()

        df = pd.DataFrame(all_features)
        df["label"] = all_labels

        n_attack = int(df["label"].sum())
        n_normal = len(df) - n_attack
        logger.info(f"Processed {len(df)} samples: {n_attack} attacks, {n_normal} normal")

        return df

    def save_features(self, df: pd.DataFrame):
        """Save processed features."""
        output_path = self.output_dir / "lidds_features.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved features to {output_path}")

        # Also save as CSV for inspection
        csv_path = self.output_dir / "lidds_features_sample.csv"
        df.head(1000).to_csv(csv_path, index=False)
        logger.info(f"Saved sample CSV to {csv_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess LID-DS dataset")
    parser.add_argument("--data-dir", type=str, default="raw/lidds", help="Path to raw LID-DS data")
    parser.add_argument("--output-dir", type=str, default="data/processed/lidds", help="Output directory for processed features")
    args = parser.parse_args()

    preprocessor = LIDDSPreprocessor(args.data_dir, args.output_dir)
    df = preprocessor.process_dataset()
    if not df.empty:
        preprocessor.save_features(df)


if __name__ == "__main__":
    main()
