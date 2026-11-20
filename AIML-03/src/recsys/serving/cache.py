"""Recommendation cache (data tier): Redis in production, in-process TTL-LRU otherwise."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Protocol

log = logging.getLogger(__name__)


class Cache(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl_s: int) -> None: ...
    def delete_prefix(self, prefix: str) -> int: ...


class MemoryCache:
    def __init__(self, max_items: int = 50_000) -> None:
        self._d: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.max_items = max_items
        self.hits = self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._d.get(key)
            if item is None or item[0] <= time.monotonic():
                self._d.pop(key, None)
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return item[1]

    def set(self, key: str, value: Any, ttl_s: int) -> None:
        with self._lock:
            self._d[key] = (time.monotonic() + ttl_s, value)
            self._d.move_to_end(key)
            while len(self._d) > self.max_items:
                self._d.popitem(last=False)

    def delete_prefix(self, prefix: str) -> int:
        with self._lock:
            keys = [k for k in self._d if k.startswith(prefix)]
            for k in keys:
                del self._d[k]
            return len(keys)


class RedisCache:  # pragma: no cover - requires redis
    def __init__(self, url: str, namespace: str = "recsys") -> None:
        import redis

        self.r = redis.Redis.from_url(url, socket_timeout=0.05)  # never let the cache slow a request
        self.ns = namespace
        self.hits = self.misses = 0

    def get(self, key: str) -> Any | None:
        try:
            raw = self.r.get(f"{self.ns}:{key}")
        except Exception as exc:
            log.warning("redis get failed: %s", exc)
            return None
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    def set(self, key: str, value: Any, ttl_s: int) -> None:
        try:
            self.r.set(f"{self.ns}:{key}", json.dumps(value), ex=ttl_s)
        except Exception as exc:
            log.warning("redis set failed: %s", exc)

    def delete_prefix(self, prefix: str) -> int:
        n = 0
        for k in self.r.scan_iter(f"{self.ns}:{prefix}*", count=1000):
            n += self.r.delete(k)
        return n


def build_cache(redis_url: str | None) -> Cache:
    if redis_url:
        try:
            c = RedisCache(redis_url)
            c.r.ping()
            return c
        except Exception as exc:  # pragma: no cover
            log.warning("redis unavailable (%s) - using in-memory cache", exc)
    return MemoryCache()
