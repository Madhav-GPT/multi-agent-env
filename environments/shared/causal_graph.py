"""Causal structure used by the deterministic judge."""

from __future__ import annotations

from collections import deque

from .state import SERVICE_NAMES, ServiceName

DEFAULT_SERVICE_GRAPH: dict[ServiceName, list[ServiceName]] = {
    "api-gateway": ["cache", "database", "auth_service"],
    "database": ["cache", "worker"],
    "cache": ["api-gateway"],
    "worker": ["database"],
    "auth_service": ["cache", "api-gateway"],
}


def impacted_services(root: ServiceName) -> list[ServiceName]:
    """Return services downstream from a root-cause service."""

    queue: deque[ServiceName] = deque([root])
    seen: set[ServiceName] = set()
    ordered: list[ServiceName] = []
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
        for downstream in DEFAULT_SERVICE_GRAPH.get(current, []):
            if downstream not in seen:
                queue.append(downstream)
    return [service for service in ordered if service in SERVICE_NAMES]

