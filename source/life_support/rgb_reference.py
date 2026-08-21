"""工程观察用RGB参考文件，不进入人工生命信息流。"""

from __future__ import annotations

from pathlib import Path

from second_experiment.host_b.visual_updates import VisualFrame


def write_rgb_ppm(frame: VisualFrame, output: Path) -> Path:
    """无损保存RGB字节，不缩放、不改通道、不重新归一化。"""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = f"P6\n{frame.width} {frame.height}\n255\n".encode("ascii")
    destination.write_bytes(header + frame.rgb)
    return destination
