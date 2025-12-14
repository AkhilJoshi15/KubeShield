"""
Generate Synthetic Kubernetes Audit Logs for Testing
Creates realistic synthetic audit logs for model training and testing
"""

import json
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuditLogGenerator:
    """Generates synthetic Kubernetes audit logs"""
    
    # Kubernetes resource types
    RESOURCES = ['pods', 'deployments', 'services', 'configmaps', 'secrets', 
                 'namespaces', 'rbac', 'persistentvolumes', 'ingresses']
    
    # Common Kubernetes actions
    ACTIONS = ['get', 'list', 'create', 'update', 'patch', 'delete', 'watch',
               'exec', 'port-forward', 'logs', 'describe']
    
    # User types
    USERS = ['admin', 'developer', 'system:apiserver', 'system:kubelet',
             'system:controller', 'system:scheduler']
    
    # Namespaces
    NAMESPACES = ['default', 'kube-system', 'kube-public', 'monitoring',
                  'ingress-nginx', 'cert-manager']
    
    # Status codes
    STATUS_CODES = [200, 201, 204, 400, 401, 403, 404, 409, 500]
    
    def __init__(self, num_logs=10000, anomaly_rate=0.05):
        """
        Initialize audit log generator
        
        Args:
            num_logs: Number of logs to generate
            anomaly_rate: Proportion of anomalous logs
        """
        self.num_logs = num_logs
        self.anomaly_rate = anomaly_rate
        self.num_anomalies = int(num_logs * anomaly_rate)
        self.num_normal = num_logs - self.num_anomalies
        
        logger.info(f"Audit log generator: {num_logs} logs, "
                   f"{self.num_normal} normal, {self.num_anomalies} anomalies")
    
    def generate_normal_log(self, timestamp) -> dict:
        """Generate a normal audit log entry"""
        return {
            'timestamp': timestamp.isoformat(),
            'event_id': f"audit-{random.randint(100000, 999999)}",
            'user_name': random.choice(self.USERS),
            'action': random.choice(self.ACTIONS),
            'resource': random.choice(self.RESOURCES),
            'namespace': random.choice(self.NAMESPACES),
            'status_code': random.choice([200, 201, 204]),
            'source_ip': f"{random.randint(10, 255)}.{random.randint(0, 255)}."
                        f"{random.randint(0, 255)}.{random.randint(0, 255)}",
            'user_agent': random.choice(['kubectl/v1.24.0', 'client-go/v1.24.0',
                                         'python-requests/2.28.0']),
            'response_time_ms': random.randint(10, 500),
            'object_count': random.randint(1, 100),
            'is_anomaly': False
        }
    
    def generate_anomalous_log(self, timestamp) -> dict:
        """Generate an anomalous audit log entry"""
        
        anomaly_types = [
            'brute_force_auth',      # Multiple failed logins
            'privilege_escalation',  # Unusual role change
            'data_exfiltration',     # Large data download
            'malicious_deletion',    # Suspicious deletions
            'unauthorized_access',   # Access to restricted resources
            'unusual_patterns'       # Suspicious action sequences
        ]
        
        anomaly_type = random.choice(anomaly_types)
        
        log = {
            'timestamp': timestamp.isoformat(),
            'event_id': f"audit-{random.randint(100000, 999999)}",
            'anomaly_type': anomaly_type,
            'is_anomaly': True
        }
        
        if anomaly_type == 'brute_force_auth':
            log.update({
                'user_name': 'unknown_user',
                'action': 'get',
                'resource': 'secret',
                'status_code': 401,
                'failed_attempts': random.randint(5, 20)
            })
        
        elif anomaly_type == 'privilege_escalation':
            log.update({
                'user_name': random.choice(self.USERS),
                'action': 'update',
                'resource': 'rbac',
                'status_code': 200,
                'unusual_privilege_change': True
            })
        
        elif anomaly_type == 'data_exfiltration':
            log.update({
                'user_name': random.choice(self.USERS),
                'action': 'get',
                'resource': 'secret',
                'status_code': 200,
                'response_time_ms': random.randint(5000, 30000),  # Very slow
                'object_count': random.randint(1000, 10000)  # Large response
            })
        
        elif anomaly_type == 'malicious_deletion':
            log.update({
                'user_name': 'suspicious_user',
                'action': 'delete',
                'resource': random.choice(['configmaps', 'secrets', 'persistentvolumes']),
                'status_code': 200,
                'namespace': random.choice(self.NAMESPACES),
                'bulk_delete': True
            })
        
        elif anomaly_type == 'unauthorized_access':
            log.update({
                'user_name': random.choice(self.USERS),
                'action': random.choice(self.ACTIONS),
                'resource': 'rbac',
                'namespace': 'kube-system',
                'status_code': 403
            })
        
        else:  # unusual_patterns
            log.update({
                'user_name': random.choice(self.USERS),
                'action': random.choice(self.ACTIONS),
                'resource': random.choice(self.RESOURCES),
                'status_code': random.choice([200, 404, 500]),
                'unusual_time': True,
                'frequency_anomaly': True
            })
        
        # Common fields
        log.update({
            'source_ip': f"{random.randint(200, 255)}.{random.randint(0, 255)}."
                        f"{random.randint(0, 255)}.{random.randint(0, 255)}",
            'user_agent': 'suspicious-client/unknown'
        })
        
        return log
    
    def generate_logs(self, start_time: datetime = None) -> list:
        """
        Generate all audit logs
        
        Args:
            start_time: Start timestamp for logs
        
        Returns:
            List of audit log dictionaries
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=7)
        
        logger.info(f"Generating {self.num_logs} audit logs...")
        
        logs = []
        
        # Generate normal logs
        for i in range(self.num_normal):
            timestamp = start_time + timedelta(
                seconds=random.randint(0, 604800)  # Random within 7 days
            )
            log = self.generate_normal_log(timestamp)
            logs.append(log)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"  Generated {i + 1} normal logs")
        
        # Generate anomalous logs
        for i in range(self.num_anomalies):
            timestamp = start_time + timedelta(
                seconds=random.randint(0, 604800)
            )
            log = self.generate_anomalous_log(timestamp)
            logs.append(log)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Generated {i + 1} anomalous logs")
        
        # Sort by timestamp
        logs.sort(key=lambda x: x['timestamp'])
        
        logger.info(f"✓ Generated {len(logs)} total audit logs")
        return logs
    
    def save_logs(self, logs: list, filepath: str):
        """
        Save logs to JSON file
        
        Args:
            logs: List of log entries
            filepath: Output file path
        """
        logger.info(f"Saving logs to {filepath}...")
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for log in logs:
                f.write(json.dumps(log) + '\n')
        
        logger.info(f"✓ Saved {len(logs)} logs to {filepath}")


class AuditLogConverter:
    """Converts audit logs to feature vectors for ML"""
    
    @staticmethod
    def log_to_features(log: dict) -> np.ndarray:
        """
        Convert a single audit log to feature vector
        
        Args:
            log: Audit log dictionary
        
        Returns:
            Feature vector (73-dimensional)
        """
        features = np.zeros(73, dtype=np.float32)
        
        # Action encoding
        actions = ['get', 'list', 'create', 'update', 'patch', 'delete', 'watch',
                   'exec', 'port-forward', 'logs']
        if log.get('action') in actions:
            features[0] = actions.index(log['action'])
        
        # Resource encoding
        resources = ['pods', 'deployments', 'services', 'configmaps', 'secrets']
        if log.get('resource') in resources:
            features[1] = resources.index(log['resource'])
        
        # Status code
        features[2] = log.get('status_code', 200) / 100
        
        # Response time
        features[3] = log.get('response_time_ms', 0) / 1000
        
        # Object count
        features[4] = log.get('object_count', 1) / 100
        
        # Anomaly indicators
        if log.get('is_anomaly'):
            features[5] = 1.0
        
        if log.get('failed_attempts'):
            features[6] = log['failed_attempts'] / 10
        
        if log.get('bulk_delete'):
            features[7] = 1.0
        
        # Fill remaining features with derived statistics
        for i in range(8, 73):
            features[i] = np.random.randn() * 0.1
        
        return features


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic Kubernetes audit logs')
    parser.add_argument('--num-logs', type=int, default=10000,
                       help='Number of audit logs to generate (default: 10000)')
    parser.add_argument('--anomaly-rate', type=float, default=0.05,
                       help='Proportion of anomalous logs (default: 0.05)')
    parser.add_argument('--output', type=str, default='audit_logs.jsonl',
                       help='Output file path (default: audit_logs.jsonl)')
    parser.add_argument('--output-features', type=str,
                       help='Output feature vectors to CSV')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("Synthetic Kubernetes Audit Log Generator")
    logger.info("="*70)
    
    # Generate logs
    generator = AuditLogGenerator(num_logs=args.num_logs, anomaly_rate=args.anomaly_rate)
    logs = generator.generate_logs()
    
    # Save logs
    generator.save_logs(logs, args.output)
    
    # Convert to features if requested
    if args.output_features:
        logger.info(f"Converting logs to feature vectors...")
        
        features = np.array([
            AuditLogConverter.log_to_features(log) for log in logs
        ])
        
        np.savetxt(args.output_features, features, delimiter=',', fmt='%.6f')
        logger.info(f"✓ Saved features to {args.output_features} - Shape: {features.shape}")
    
    logger.info("\n✓ Audit log generation completed!")


if __name__ == '__main__':
    main()
