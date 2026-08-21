"""教学对真实鼠标和键盘执行端的接管。"""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock


class TeachingGate:
    def __init__(self, release_actions: Callable[[], None]) -> None:
        self._release_actions = release_actions
        self._active = False
        self._lock = RLock()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def start(self) -> None:
        with self._lock:
            if self._active:
                return
            self._release_actions()
            self._active = True

    def stop(self) -> None:
        with self._lock:
            self._active = False

    def allow_artificial_action(self) -> bool:
        with self._lock:
            return not self._active

