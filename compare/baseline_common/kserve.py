from __future__ import annotations

import time
from typing import Dict

from kubernetes import client, config


def load_kubernetes() -> client.CoreV1Api:
    try:
        config.load_kube_config()
    except Exception:
        config.load_incluster_config()
    return client.CoreV1Api()


def logical_name(pod_name: str) -> str:
    return pod_name.split("-predictor-")[0] if "-predictor-" in pod_name else pod_name


def discover_instances(
    api: client.CoreV1Api,
    namespace: str = "like",
    label_selector: str = "component=predictor",
) -> Dict[str, str]:
    result: Dict[str, str] = {}
    pods = api.list_namespaced_pod(
        namespace=namespace,
        label_selector=label_selector,
        field_selector="status.phase=Running",
    )
    for pod in pods.items:
        name = logical_name(pod.metadata.name)
        if name.startswith("qwen-instance-") and pod.status.pod_ip:
            result[name] = pod.status.pod_ip
    return result


def wait_for_pool(
    api: client.CoreV1Api,
    pool_size: int,
    namespace: str = "like",
    label_selector: str = "component=predictor",
    poll_seconds: float = 5.0,
) -> Dict[str, str]:
    expected = {f"qwen-instance-{index:02d}" for index in range(1, pool_size + 1)}
    while True:
        instances = discover_instances(api, namespace, label_selector)
        missing = expected.difference(instances)
        if not missing:
            return {name: instances[name] for name in sorted(expected)}
        print(f"waiting for warm pool; missing={sorted(missing)}", flush=True)
        time.sleep(poll_seconds)

