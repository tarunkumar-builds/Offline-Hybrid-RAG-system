"""Small thread-safe LRU cache used by local retrieval components."""

from collections import OrderedDict
from threading import RLock
from typing import Generic, TypeVar


Key = TypeVar("Key", bound=object)
Value = TypeVar("Value")


class LRUCache(Generic[Key, Value]):
    """Bounded in-memory cache with least-recently-used eviction."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least one")
        self._capacity = capacity
        self._values: OrderedDict[Key, Value] = OrderedDict()
        self._lock = RLock()

    def get(self, key: Key) -> Value | None:
        """Return a cached value and promote it to most recently used."""
        with self._lock:
            if key not in self._values:
                return None
            value = self._values.pop(key)
            self._values[key] = value
            return value

    def put(self, key: Key, value: Value) -> None:
        """Store a value, evicting the least recently used value when full."""
        with self._lock:
            self._values.pop(key, None)
            self._values[key] = value
            if len(self._values) > self._capacity:
                self._values.popitem(last=False)
