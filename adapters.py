"""
adapters.py — plug-in HTTP adapters.

Any compatible software can be registered by supplying a base URL and a POST
path. The adapter layer makes no assumption about downstream internals; each
target is treated as an opaque HTTP endpoint that accepts a KAIROS hand-off
payload and returns JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import httpx


@dataclass
class Adapter:
    name: str
    base_url: str
    path: str
    timeout: float = 30.0
    headers: dict[str, str] | None = None

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = urljoin(self.base_url.rstrip("/") + "/", self.path.lstrip("/"))
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=self.headers or {})
            resp.raise_for_status()
            return resp.json()


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, Adapter] = {}

    def register(
        self,
        name: str,
        base_url: str,
        path: str,
        *,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        if not name:
            raise ValueError("name is required")
        if not base_url:
            raise ValueError("base_url is required")
        self._adapters[name] = Adapter(
            name=name, base_url=base_url, path=path,
            timeout=timeout, headers=headers,
        )

    def unregister(self, name: str) -> None:
        self._adapters.pop(name, None)

    def get(self, name: str) -> Adapter:
        if name not in self._adapters:
            raise KeyError(f"adapter {name!r} is not registered")
        return self._adapters[name]

    def names(self) -> list[str]:
        return sorted(self._adapters)

    def call(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.get(name).call(payload)


registry = AdapterRegistry()