"""真实画面捕捉；尚未加入未确认的视觉变换。"""

from __future__ import annotations

from .minecraft_window import MinecraftWindow, MinecraftWindowLocator
from .visual_updates import VisualFrame


def visible_capture_box(
    window: MinecraftWindow,
    desktop: dict[str, int],
) -> dict[str, int]:
    """把客户区限制在真实可见桌面像素内。"""

    left = max(window.left, int(desktop["left"]))
    top = max(window.top, int(desktop["top"]))
    right = min(
        window.left + window.width,
        int(desktop["left"]) + int(desktop["width"]),
    )
    bottom = min(
        window.top + window.height,
        int(desktop["top"]) + int(desktop["height"]),
    )
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise RuntimeError("Minecraft客户区不在真实可见屏幕内")
    return {"left": left, "top": top, "width": width, "height": height}


class MssMinecraftCapture:
    """诊断用RGB捕捉，不改变分辨率，也不形成其他视觉活动。"""

    def __init__(self, locator: MinecraftWindowLocator) -> None:
        try:
            import mss
            import numpy
        except ImportError as exc:
            raise RuntimeError("画面捕捉需要安装 mss 和 numpy") from exc
        self._numpy = numpy
        self._capture = mss.mss()
        self.locator = locator

    def read(self) -> tuple[VisualFrame, MinecraftWindow]:
        window = self.locator.locate()
        if window is None:
            raise RuntimeError("没有找到唯一的Minecraft窗口")
        box = visible_capture_box(window, self._capture.monitors[0])
        bgra = self._numpy.asarray(self._capture.grab(box), dtype=self._numpy.uint8)
        rgb = self._numpy.ascontiguousarray(bgra[..., [2, 1, 0]])
        return VisualFrame(box["width"], box["height"], rgb.tobytes()), window

    def close(self) -> None:
        self._capture.close()
