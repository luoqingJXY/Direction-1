"""同一个人工生命的结构和路径状态永久存储。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StoredLife:
    life_structure: bytes
    path_state: bytes


class LifeStorage:
    STRUCTURE_FILE = "life_structure.bin"
    PATH_FILE = "path_state.bin"
    MANIFEST_FILE = "manifest.json"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def _atomic_write(self, destination: Path, data: bytes) -> None:
        temporary = destination.with_suffix(destination.suffix + ".new")
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)

    def save(self, life_structure: bytes, path_state: bytes) -> None:
        structure = bytes(life_structure)
        paths = bytes(path_state)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._atomic_write(self.directory / self.STRUCTURE_FILE, structure)
        self._atomic_write(self.directory / self.PATH_FILE, paths)
        manifest = {
            "format": 1,
            "life_structure": {
                "bytes": len(structure),
                "sha256": hashlib.sha256(structure).hexdigest(),
            },
            "path_state": {
                "bytes": len(paths),
                "sha256": hashlib.sha256(paths).hexdigest(),
            },
        }
        encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        self._atomic_write(self.directory / self.MANIFEST_FILE, encoded)

    def exists(self) -> bool:
        return all(
            (self.directory / name).is_file()
            for name in (self.STRUCTURE_FILE, self.PATH_FILE, self.MANIFEST_FILE)
        )

    def load(self) -> StoredLife:
        manifest = json.loads((self.directory / self.MANIFEST_FILE).read_text(encoding="utf-8"))
        if manifest.get("format") != 1:
            raise RuntimeError("人工生命存储格式无法识别")
        structure = (self.directory / self.STRUCTURE_FILE).read_bytes()
        paths = (self.directory / self.PATH_FILE).read_bytes()
        for name, data in (("life_structure", structure), ("path_state", paths)):
            expected = manifest.get(name, {})
            if expected.get("bytes") != len(data) or expected.get("sha256") != hashlib.sha256(data).hexdigest():
                raise RuntimeError(f"人工生命存储校验失败：{name}")
        return StoredLife(structure, paths)

