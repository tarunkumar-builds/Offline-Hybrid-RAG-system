"""Thread-safe bounded cache for reranked responses."""

from collections import OrderedDict
from threading import RLock
from typing import Generic, TypeVar


Key = TypeVar("Key", bound=object)
Value = TypeVar("Value")


class RerankCache(Generic[Key, Value]):
    """A small LRU cache that avoids repeated local model inference."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._values: OrderedDict[Key, Value] = OrderedDict()
        self._lock = RLock()

    def get(self, key: Key) -> Value | None:
        """Return and promote a cached value when present."""
        with self._lock:
            if key not in self._values:
                return None
            value = self._values.pop(key)
            self._values[key] = value
            return value

    def put(self, key: Key, value: Value) -> None:
        """Store a value and evict the least-recently-used entry if needed."""
        with self._lock:
            self._values.pop(key, None)
            self._values[key] = value
            if len(self._values) > self._capacity:
                self._values.popitem(last=False)
