"""只供控制台观察的顺序记录，不进入生命信号。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Any


def _observable(value: Any) -> Any:
    if isinstance(value, bytes):
        return {
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if is_dataclass(value):
        return _observable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _observable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        if len(value) > 256:
            return {"count": len(value), "first_values": [_observable(item) for item in value[:16]]}
        return [_observable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class ObservationLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def record(self, category: str, value: Any) -> None:
        record = {"category": str(category), "value": _observable(value)}
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line)
                stream.write("\n")

    def read_all(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        with self._lock:
            with self.path.open("r", encoding="utf-8") as stream:
                return tuple(json.loads(line) for line in stream if line.strip())

