from __future__ import annotations

import threading
import time
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class LatestValue(Generic[T]):
    """Thread-safe latest-value slot.

    This is intentionally not a normal queue. Producers replace the current value;
    consumers always see the newest packet and backlog can never grow.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._value: Optional[T] = None
        self._seq = 0
        self._closed = False

    def set(self, value: T) -> int:
        with self._cond:
            self._value = value
            self._seq += 1
            self._cond.notify()  # P-05: only one consumer waits; notify() is sufficient
            return self._seq

    def get(self) -> tuple[Optional[T], int]:
        with self._cond:
            return self._value, self._seq

    def wait_next(self, last_seq: int = 0, timeout: float | None = None) -> tuple[Optional[T], int]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not self._closed and self._seq <= last_seq:
                if deadline is None:
                    self._cond.wait()
                else:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    self._cond.wait(left)
            return self._value, self._seq

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()  # wake all potential waiters on close
