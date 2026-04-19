"""Test LRU cache behavior."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from cache import ImageCache

import pytest


def test_cache_hit():
    """Identical parameters return cached result."""
    cache = ImageCache(max_entries=10)
    params = {"spin": 0.6, "inclination": 80.0, "width": 320, "height": 180}
    data = b"fake_image_bytes"

    cache.put(params, data)
    result = cache.get(params)
    assert result == data


def test_cache_miss():
    """Different parameters return None."""
    cache = ImageCache(max_entries=10)
    params1 = {"spin": 0.6, "inclination": 80.0}
    params2 = {"spin": 0.7, "inclination": 80.0}
    data = b"fake_image_bytes"

    cache.put(params1, data)
    result = cache.get(params2)
    assert result is None


def test_cache_parameter_order_independent():
    """Cache key is independent of parameter insertion order."""
    cache = ImageCache(max_entries=10)
    params_a = {"spin": 0.6, "inclination": 80.0, "width": 320}
    params_b = {"width": 320, "spin": 0.6, "inclination": 80.0}
    data = b"fake_image_bytes"

    cache.put(params_a, data)
    result = cache.get(params_b)
    assert result == data


def test_cache_eviction_lru():
    """Least recently used entry is evicted when max_entries exceeded."""
    cache = ImageCache(max_entries=2)
    data1 = b"data1"
    data2 = b"data2"
    data3 = b"data3"

    cache.put({"id": 1}, data1)
    cache.put({"id": 2}, data2)
    # Access id=1 to make it recently used
    cache.get({"id": 1})
    # Add third entry, should evict id=2 (least recently used)
    cache.put({"id": 3}, data3)

    assert cache.get({"id": 2}) is None
    assert cache.get({"id": 1}) == data1
    assert cache.get({"id": 3}) == data3


def test_cache_stats():
    """Cache statistics are tracked correctly."""
    cache = ImageCache(max_entries=10)
    params = {"spin": 0.5, "width": 100, "height": 100}
    data = b"test_data"

    cache.put(params, data)
    # Hit
    cache.get(params)
    # Hit again
    cache.get(params)
    # Miss
    cache.get({"spin": 0.7})

    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["entries"] == 1
    assert stats["hit_rate"] == 2 / 3


def test_cache_clear():
    """Clear removes all entries."""
    cache = ImageCache(max_entries=10)
    cache.put({"a": 1}, b"x")
    cache.put({"b": 2}, b"y")
    count = cache.clear()
    assert count == 2
    assert cache.get({"a": 1}) is None
    assert cache.get({"b": 2}) is None
    assert cache.stats()["entries"] == 0
