"""
Synthetic Kubernetes Audit Log Generator

This module generates synthetic Kubernetes audit logs with 34 audit features
for testing, training, and anomaly detection purposes.

Features (34 total):
1. timestamp - Audit event timestamp
2. event_type - Type of audit event (Create, Update, Delete, Get, List, Watch, etc.)
3. user_name - Name of the user performing the action
4. user_groups - Groups the user belongs to
5. impersonated_user - User being impersonated (if applicable)
6. source_ips - Source IP address(es)
7. user_agent - Client user agent string
8. verb - HTTP verb (get, list, create, update, patch, delete, watch)
9. api_group - Kubernetes API group
10. api_version - API version
11. resource_kind - Kind of resource (Pod, Deployment, Service, etc.)
12. resource_name - Name of the resource
13. resource_namespace - Kubernetes namespace
14. request_uri - Full request URI
15. request_object - Request object content
16. request_object_size - Size of request object in bytes
17. response_status_code - HTTP response status code
18. response_status_reason - Response status reason
19. response_object - Response object content
20. response_object_size - Size of response object in bytes
21. request_received_timestamp - Timestamp when request was received
22. stage - Audit event stage (RequestReceived, ResponseStarted, ResponseComplete, Panic)
23. audit_id - Unique audit event ID
24. audit_level - Audit level (Metadata, RequestResponse, None)
25. privileged_escalation - Whether escalation was attempted
26. allowed - Whether the action was allowed
27. annotations_authorization_decision - Authorization decision annotation
28. annotations_authorization_reason - Authorization reason annotation
29. request_object_hash - Hash of request object
30. response_object_hash - Hash of response object
31. request_headers - Request headers
32. response_headers - Response headers
33. client_certificate_expiration - Client certificate expiration time
34. client_certificate_serial_number - Client certificate serial number
"""

import json
import random
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class AuditLogEvent:
    """Data class representing a single Kubernetes audit log event"""
    timestamp: str
    event_type: str
    user_name: str
    user_groups: List[str]
    impersonated_user: Optional[str]
    source_ips: List[str]
    user_agent: str
    verb: str
    api_group: str
    api_version: str
    resource_kind: str
    resource_name: str
    resource_namespace: str
    request_uri: str
    request_object: Dict[str, Any]
    request_object_size: int
    response_status_code: int
    response_status_reason: str
    response_object: Dict[str, Any]
    response_object_size: int
    request_received_timestamp: str
    stage: str
    audit_id: str
    audit_level: str
    privileged_escalation: bool
    allowed: bool
    annotations_authorization_decision: str
    annotations_authorization_reason: str
    request_object_hash: str
    response_object_hash: str
    request_headers: Dict[str, str]
    response_headers: Dict[str, str]
    client_certificate_expiration: Optional[str]
    client_certificate_serial_number: Optional[str]


class KubernetesAuditLogGenerator:
    """Generate synthetic Kubernetes audit logs with realistic patterns"""

    # Constants for log generation
    EVENT_TYPES = ["Create", "Update", "Delete", "Get", "List", "Watch", "Patch"]
    VERBS = ["get", "list", "create", "update", "patch", "delete", "watch", "post", "put"]
    API_GROUPS = ["", "apps", "batch", "rbac.authorization.k8s.io", "policy", "networking.k8s.io", "storage.k8s.io"]
    API_VERSIONS = ["v1", "v1beta1", "v1alpha1"]
    RESOURCE_KINDS = ["Pod", "Deployment", "StatefulSet", "Service", "ConfigMap", "Secret", "PersistentVolume", "Role", "RoleBinding", "ClusterRole", "ClusterRoleBinding", "ServiceAccount"]
    NAMESPACES = ["default", "kube-system", "kube-public", "kube-node-lease", "monitoring", "ingress-nginx", "cert-manager"]
    AUDIT_LEVELS = ["Metadata", "RequestResponse", "None"]
    STAGES = ["RequestReceived", "ResponseStarted", "ResponseComplete", "Panic"]
    STATUS_CODES = [200, 201, 202, 204, 400, 401, 403, 404, 409, 500, 503]
    USER_AGENTS = [
        "kubectl/v1.24.0",
        "kubectl/v1.25.0",
        "kubernetes-client/python",
        "helm/v3.10.0",
        "kube-controller-manager",
        "kube-scheduler",
        "kubelet/v1.24.0"
    ]
    USERS = [
        "system:admin", "system:kube-controller-manager", "system:kube-scheduler",
        "system:kubelet", "alice", "bob", "charlie", "admin", "developer",
        "system:serviceaccount:default:default"
    ]
    USER_GROUPS = [
        "system:masters", "system:nodes", "system:authenticated",
        "developers", "admins", "viewers", "editors"
    ]
    SOURCE_IPS = [
        "192.168.1.100", "192.168.1.101", "192.168.1.102",
        "10.0.0.1", "10.0.0.2", "172.17.0.1", "172.17.0.2"
    ]

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the audit log generator

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        self.base_time = datetime.utcnow()

    def _generate_hash(self, data: Dict[str, Any]) -> str:
        """Generate SHA256 hash of an object"""
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _generate_request_object(self, resource_kind: str) -> Dict[str, Any]:
        """Generate a realistic request object based on resource kind"""
        base_object = {
            "apiVersion": random.choice(self.API_VERSIONS),
            "kind": resource_kind,
            "metadata": {
                "name": f"{resource_kind.lower()}-{random.randint(1, 1000)}",
                "namespace": random.choice(self.NAMESPACES)
            }
        }

        # Add spec based on resource kind
        if resource_kind == "Pod":
            base_object["spec"] = {
                "containers": [
                    {
                        "name": "main",
                        "image": f"nginx:{random.choice(['latest', '1.21', '1.22'])}",
                        "ports": [{"containerPort": random.choice([80, 443, 8080])}]
                    }
                ]
            }
        elif resource_kind == "Deployment":
            base_object["spec"] = {
                "replicas": random.randint(1, 5),
                "selector": {"matchLabels": {"app": "test"}},
                "template": {
                    "metadata": {"labels": {"app": "test"}},
                    "spec": {"containers": [{"name": "app", "image": "nginx:latest"}]}
                }
            }
        elif resource_kind == "Service":
            base_object["spec"] = {
                "type": random.choice(["ClusterIP", "NodePort", "LoadBalancer"]),
                "ports": [{"port": 80, "targetPort": 8080}],
                "selector": {"app": "test"}
            }
        elif resource_kind == "ConfigMap":
            base_object["data"] = {
                "config.yaml": "key: value",
                "app.config": "setting=value"
            }

        return base_object

    def _generate_response_object(self, request_object: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a response object from a request object"""
        response = request_object.copy()
        response["metadata"] = response.get("metadata", {}).copy()
        response["metadata"]["uid"] = str(uuid.uuid4())
        response["metadata"]["resourceVersion"] = str(random.randint(1000, 999999))
        response["metadata"]["creationTimestamp"] = datetime.utcnow().isoformat() + "Z"
        response["metadata"]["generation"] = random.randint(1, 10)

        if "status" not in response:
            response["status"] = {
                "phase": random.choice(["Pending", "Running", "Succeeded", "Failed"]),
                "conditions": []
            }

        return response

    def _generate_request_headers(self) -> Dict[str, str]:
        """Generate realistic request headers"""
        return {
            "Content-Type": "application/json",
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Authorization": f"Bearer {uuid.uuid4().hex[:50]}"
        }

    def _generate_response_headers(self, status_code: int) -> Dict[str, str]:
        """Generate realistic response headers"""
        return {
            "Content-Type": "application/json",
            "Content-Length": str(random.randint(100, 10000)),
            "Date": datetime.utcnow().isoformat(),
            "Server": "kube-apiserver",
            "Cache-Control": "no-cache",
            "X-Request-Id": uuid.uuid4().hex
        }

    def generate_event(self, event_time: Optional[datetime] = None) -> AuditLogEvent:
        """
        Generate a single audit log event

        Args:
            event_time: Optional timestamp for the event

        Returns:
            AuditLogEvent instance with all 34 features
        """
        if event_time is None:
            # Generate random offset from base time (within 24 hours)
            offset = random.randint(0, 86400)
            event_time = self.base_time + timedelta(seconds=offset)

        timestamp_str = event_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Generate basic audit fields
        event_type = random.choice(self.EVENT_TYPES)
        user_name = random.choice(self.USERS)
        verb = random.choice(self.VERBS)
        api_group = random.choice(self.API_GROUPS)
        api_version = random.choice(self.API_VERSIONS)
        resource_kind = random.choice(self.RESOURCE_KINDS)
        resource_name = f"{resource_kind.lower()}-{random.randint(1, 10000)}"
        resource_namespace = random.choice(self.NAMESPACES)
        status_code = random.choice(self.STATUS_CODES)

        # Generate objects
        request_object = self._generate_request_object(resource_kind)
        response_object = self._generate_response_object(request_object)

        # Generate sizes
        request_size = len(json.dumps(request_object))
        response_size = len(json.dumps(response_object))

        # Generate headers
        request_headers = self._generate_request_headers()
        response_headers = self._generate_response_headers(status_code)

        # Generate hashes
        request_hash = self._generate_hash(request_object)
        response_hash = self._generate_hash(response_object)

        # Build request URI
        group_part = f"/{api_group}" if api_group else ""
        request_uri = f"/api{group_part}/{api_version}/namespaces/{resource_namespace}/{resource_kind.lower()}s/{resource_name}"

        # Generate certificate info
        cert_expiration = (event_time + timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
        cert_serial = format(random.getrandbits(128), '064x')

        # Determine if request was allowed (90% success rate)
        allowed = random.random() < 0.9

        # Create audit event
        event = AuditLogEvent(
            timestamp=timestamp_str,
            event_type=event_type,
            user_name=user_name,
            user_groups=random.sample(self.USER_GROUPS, random.randint(1, 3)),
            impersonated_user=user_name if random.random() < 0.1 else None,
            source_ips=random.sample(self.SOURCE_IPS, random.randint(1, 2)),
            user_agent=random.choice(self.USER_AGENTS),
            verb=verb,
            api_group=api_group,
            api_version=api_version,
            resource_kind=resource_kind,
            resource_name=resource_name,
            resource_namespace=resource_namespace,
            request_uri=request_uri,
            request_object=request_object,
            request_object_size=request_size,
            response_status_code=status_code,
            response_status_reason="OK" if status_code == 200 else "Error",
            response_object=response_object,
            response_object_size=response_size,
            request_received_timestamp=timestamp_str,
            stage=random.choice(self.STAGES),
            audit_id=str(uuid.uuid4()),
            audit_level=random.choice(self.AUDIT_LEVELS),
            privileged_escalation=random.random() < 0.05,
            allowed=allowed,
            annotations_authorization_decision="allow" if allowed else "deny",
            annotations_authorization_reason="RBAC" if allowed else "insufficient permissions",
            request_object_hash=request_hash,
            response_object_hash=response_hash,
            request_headers=request_headers,
            response_headers=response_headers,
            client_certificate_expiration=cert_expiration,
            client_certificate_serial_number=cert_serial
        )

        return event

    def generate_events(self, count: int, start_time: Optional[datetime] = None) -> List[AuditLogEvent]:
        """
        Generate multiple audit log events

        Args:
            count: Number of events to generate
            start_time: Optional start time for events

        Returns:
            List of AuditLogEvent instances
        """
        events = []
        current_time = start_time or self.base_time

        for i in range(count):
            event = self.generate_event(current_time)
            events.append(event)
            # Increment time by random seconds (1-10 seconds between events)
            current_time += timedelta(seconds=random.randint(1, 10))

        return events

    def to_json(self, event: AuditLogEvent) -> str:
        """
        Convert audit event to JSON string

        Args:
            event: AuditLogEvent instance

        Returns:
            JSON string representation
        """
        event_dict = asdict(event)
        return json.dumps(event_dict, indent=2)

    def to_jsonl(self, events: List[AuditLogEvent]) -> str:
        """
        Convert audit events to JSONL format (one JSON object per line)

        Args:
            events: List of AuditLogEvent instances

        Returns:
            JSONL string representation
        """
        lines = []
        for event in events:
            event_dict = asdict(event)
            lines.append(json.dumps(event_dict))
        return "\n".join(lines)


def main():
    """Example usage of the audit log generator"""
    # Create generator with optional seed for reproducibility
    generator = KubernetesAuditLogGenerator(seed=42)

    # Generate 10 audit log events
    events = generator.generate_events(10)

    # Print first event in JSON format
    print("Sample Audit Log Event:")
    print(generator.to_json(events[0]))

    # Print all events in JSONL format
    print("\n\nAll Events (JSONL format):")
    print(generator.to_jsonl(events))

    # Print event statistics
    print("\n\nEvent Statistics:")
    print(f"Total events generated: {len(events)}")
    print(f"Success rate: {sum(1 for e in events if e.allowed) / len(events) * 100:.1f}%")
    print(f"Average request size: {sum(e.request_object_size for e in events) / len(events):.0f} bytes")
    print(f"Average response size: {sum(e.response_object_size for e in events) / len(events):.0f} bytes")


if __name__ == "__main__":
    main()
