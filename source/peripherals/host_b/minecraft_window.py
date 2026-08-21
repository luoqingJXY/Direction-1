"""在 Windows 上定位真实 Minecraft 客户区。"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
from pathlib import Path
import sys

from .config import MinecraftConfig


@dataclass(frozen=True, slots=True)
class MinecraftWindow:
    handle: int
    title: str
    process_name: str
    left: int
    top: int
    width: int
    height: int
    foreground: bool


def select_minecraft_window(
    candidates: list[MinecraftWindow],
    title_contains: str,
    process_names: set[str] | None = None,
) -> MinecraftWindow | None:
    """先按标题选择；无边框时在允许进程中选择唯一的主要窗口。"""

    title_key = title_contains.casefold()
    title_matches = [
        candidate
        for candidate in candidates
        if title_key in candidate.title.casefold()
    ]
    if title_matches:
        title_matches.sort(
            key=lambda item: (not item.foreground, -(item.width * item.height))
        )
        return title_matches[0]

    allowed_names = process_names or {"javaw.exe", "java.exe"}
    process_matches = [
        candidate
        for candidate in candidates
        if candidate.process_name.casefold() in allowed_names
    ]
    if len(process_matches) == 1:
        return process_matches[0]

    foreground_matches = [candidate for candidate in process_matches if candidate.foreground]
    if len(foreground_matches) == 1:
        return foreground_matches[0]

    if process_matches:
        largest_area = max(item.width * item.height for item in process_matches)
        largest = [
            item
            for item in process_matches
            if item.width * item.height == largest_area
        ]
        if len(largest) == 1:
            return largest[0]
    return None


class MinecraftWindowLocator:
    def __init__(self, config: MinecraftConfig) -> None:
        if sys.platform != "win32":
            raise RuntimeError("真实Minecraft窗口定位只支持Windows")
        self.title_contains = config.title_contains.casefold()
        self.process_names = {name.casefold() for name in config.process_names}
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._use_physical_screen_coordinates()

    def _use_physical_screen_coordinates(self) -> None:
        """让客户区坐标与真实画面像素保持同一尺度。"""

        try:
            function = self.user32.SetProcessDpiAwarenessContext
            function.argtypes = [wintypes.HANDLE]
            function.restype = wintypes.BOOL
            function(wintypes.HANDLE(-4))
        except (AttributeError, OSError, ValueError):
            try:
                self.user32.SetProcessDPIAware()
            except (AttributeError, OSError):
                pass

    def _title(self, handle: int) -> str:
        length = self.user32.GetWindowTextLengthW(wintypes.HWND(handle))
        buffer = ctypes.create_unicode_buffer(length + 1)
        self.user32.GetWindowTextW(wintypes.HWND(handle), buffer, length + 1)
        return buffer.value

    def _process_name(self, handle: int) -> str:
        process_id = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(wintypes.HWND(handle), ctypes.byref(process_id))
        process = self.kernel32.OpenProcess(0x1000, False, process_id.value)
        if not process:
            return ""
        try:
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not self.kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(size)):
                return ""
            return Path(buffer.value).name
        finally:
            self.kernel32.CloseHandle(process)

    def _client_area(self, handle: int) -> tuple[int, int, int, int] | None:
        rectangle = wintypes.RECT()
        if not self.user32.GetClientRect(wintypes.HWND(handle), ctypes.byref(rectangle)):
            return None
        origin = wintypes.POINT(0, 0)
        if not self.user32.ClientToScreen(wintypes.HWND(handle), ctypes.byref(origin)):
            return None
        width = rectangle.right - rectangle.left
        height = rectangle.bottom - rectangle.top
        if width <= 0 or height <= 0:
            return None
        return origin.x, origin.y, width, height

    def candidates(self) -> list[MinecraftWindow]:
        """列出可见且未最小化的顶层客户区，供定位和本机诊断共用。"""

        foreground = int(self.user32.GetForegroundWindow())
        candidates: list[MinecraftWindow] = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @callback_type
        def callback(window: int, _parameter: int) -> bool:
            handle = int(window)
            if not self.user32.IsWindowVisible(window) or self.user32.IsIconic(window):
                return True
            title = self._title(handle)
            process_name = self._process_name(handle)
            area = self._client_area(handle)
            if area is None:
                return True
            left, top, width, height = area
            candidates.append(
                MinecraftWindow(
                    handle,
                    title,
                    process_name,
                    left,
                    top,
                    width,
                    height,
                    handle == foreground,
                )
            )
            return True

        self.user32.EnumWindows(callback, 0)
        return candidates

    def locate(self) -> MinecraftWindow | None:
        return select_minecraft_window(
            self.candidates(),
            self.title_contains,
            self.process_names,
        )
