"""
Thread-safe LRU cache for rendered images.

Keys are SHA-256 hashes of sorted, serialized render parameters.
Values are compressed image bytes (JPEG/WebP).
"""

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Optional


# Default limits
MAX_ENTRIES = 512
MAX_MEMORY_BYTES = 256 * 1024 * 1024  # 256 MB


def _param_hash(params: dict) -> str:
    """Compute a deterministic SHA-256 hash of render parameters.

    Args:
        params: Render parameters dict.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    # Sort keys for deterministic serialization
    serialized = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ImageCache:
    """Thread-safe LRU cache for rendered images.

    Evicts least-recently-used entries when either the max entry count
    or max memory limit is exceeded.
    """

    def __init__(
        self,
        max_entries: int = MAX_ENTRIES,
        max_memory_bytes: int = MAX_MEMORY_BYTES,
    ):
        self._max_entries = max_entries
        self._max_memory = max_memory_bytes
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._current_memory: int = 0
        self._hits: int = 0
        self._misses: int = 0

    def get(self, params: dict) -> Optional[bytes]:
        """Look up cached image by render parameters.

        Args:
            params: Render parameters dict.

        Returns:
            Compressed image bytes if found, None otherwise.
        """
        key = _param_hash(params)
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, params: dict, image_bytes: bytes) -> None:
        """Store a rendered image in the cache.

        Args:
            params: Render parameters dict (used as cache key).
            image_bytes: Compressed image bytes (JPEG/WebP).
        """
        key = _param_hash(params)
        size = len(image_bytes)

        with self._lock:
            # If key already exists, remove old entry first
            if key in self._cache:
                old_size = len(self._cache[key])
                self._current_memory -= old_size
                del self._cache[key]

            # Evict until we have room
            self._evict_if_needed(size)

            # Insert new entry at end (most recently used)
            self._cache[key] = image_bytes
            self._current_memory += size

    def _evict_if_needed(self, incoming_size: int) -> None:
        """Evict LRU entries until constraints are satisfied.

        Must be called with self._lock held.
        """
        # Evict by count
        while len(self._cache) >= self._max_entries:
            _, evicted = self._cache.popitem(last=False)
            self._current_memory -= len(evicted)

        # Evict by memory
        while (
            self._current_memory + incoming_size > self._max_memory
            and self._cache
        ):
            _, evicted = self._cache.popitem(last=False)
            self._current_memory -= len(evicted)

    def stats(self) -> dict:
        """Return cache statistics.

        Returns:
            Dict with entries, memory_bytes, max_entries, max_memory_bytes,
            hits, misses, hit_rate.
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._cache),
                "memory_bytes": self._current_memory,
                "max_entries": self._max_entries,
                "max_memory_bytes": self._max_memory,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }
