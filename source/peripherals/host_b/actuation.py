"""Windows鼠标和键盘的最终执行端。

这里只执行已经完成方向、阈值和键位解释的动作。活动到动作的解释规则尚未
确认，因此不在本文件中擅自实现。
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import sys
from threading import RLock

from .minecraft_window import MinecraftWindowLocator
from .teaching import TeachingGate


_ULONG_PTR = wintypes.WPARAM


# 即使上层配置或解码发生错误，这些物理按键也不能到达Windows输入接口。
FORBIDDEN_SCAN_CODES = frozenset(
    {
        0x01,       # Esc
        0x38,       # 左Alt
        0x57,       # F11
        0x58,       # F12紧急停止
        0xE038,     # 右Alt
        0xE05B,     # 左Win
        0xE05C,     # 右Win
        0xE05D,     # 系统菜单
        0xE05E,     # 电源
        0x54,
        0x55,
    }
)


class _MouseInput(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    )


class _KeyboardInput(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    )


class _InputUnion(ctypes.Union):
    _fields_ = (("mouse", _MouseInput), ("keyboard", _KeyboardInput))


class _Input(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = (("type", wintypes.DWORD), ("data", _InputUnion))


@dataclass(frozen=True, slots=True)
class ResolvedAction:
    """已经由确认规则形成的设备动作，不是脑的原始活动。"""

    mouse_dx: int = 0
    mouse_dy: int = 0
    # None表示本次真实发生没有更新键盘；空元组才表示明确松开全部键。
    # 鼠标和键盘是异步信息流，不能让一份鼠标活动隐式清空键盘状态。
    key_states: tuple[tuple[int, bool], ...] | None = None


class WindowsInputSink:
    KEY_EXTENDED = 0x0001
    MOUSE_MOVE = 0x0001
    KEY_UP = 0x0002
    KEY_SCANCODE = 0x0008

    def __init__(
        self,
        locator: MinecraftWindowLocator,
        require_foreground: bool,
    ) -> None:
        if sys.platform != "win32":
            raise RuntimeError("真实鼠标键盘执行只支持Windows")
        self.locator = locator
        self.require_foreground = bool(require_foreground)
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._held_scan_codes: set[int] = set()
        self._apply_lock = RLock()
        self.teaching = TeachingGate(self.release_all)

    def _send(self, event: _Input) -> None:
        if self.user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(_Input)) != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    def _send_key(self, scan_code: int, pressed: bool) -> None:
        encoded = int(scan_code)
        extended = (encoded & 0xFF00) == 0xE000
        code = encoded & 0xFF if extended else encoded
        if not 0 < code <= 0xFF:
            raise ValueError("键盘扫描码必须是普通码或E0扩展码")
        flags = self.KEY_SCANCODE | (self.KEY_EXTENDED if extended else 0)
        flags |= 0 if pressed else self.KEY_UP
        event = _Input(
            type=1,
            keyboard=_KeyboardInput(0, code, flags, 0, 0),
        )
        self._send(event)

    def _send_mouse_move(self, dx: int, dy: int) -> None:
        event = _Input(
            type=0,
            mouse=_MouseInput(dx, dy, 0, self.MOUSE_MOVE, 0, 0),
        )
        self._send(event)

    def _window_allows_action(self) -> bool:
        window = self.locator.locate()
        if window is None:
            return False
        return window.foreground or not self.require_foreground

    def apply(self, action: ResolvedAction) -> None:
        with self._apply_lock:
            if self.teaching.active:
                return
            if not -1 <= int(action.mouse_dx) <= 1 or not -1 <= int(action.mouse_dy) <= 1:
                raise ValueError("一次鼠标活动只能形成至多一个位置的实际位移")
            if action.key_states is not None:
                codes = tuple(int(scan_code) for scan_code, _pressed in action.key_states)
                if len(set(codes)) != len(codes):
                    raise ValueError("一次键盘动作不能重复控制同一个实际按键")
                if any(code in FORBIDDEN_SCAN_CODES for code in codes):
                    raise ValueError("实际执行端拒绝了锁定的系统按键")
            if not self._window_allows_action():
                self.release_all()
                raise RuntimeError("Minecraft窗口未处于允许执行动作的状态")
            if action.key_states is not None:
                desired = {scan_code for scan_code, pressed in action.key_states if pressed}
                for scan_code in self._held_scan_codes - desired:
                    self._send_key(scan_code, False)
                for scan_code in desired - self._held_scan_codes:
                    self._send_key(scan_code, True)
                self._held_scan_codes = desired
            if action.mouse_dx or action.mouse_dy:
                self._send_mouse_move(action.mouse_dx, action.mouse_dy)

    def release_all(self) -> None:
        with self._apply_lock:
            for scan_code in tuple(self._held_scan_codes):
                self._send_key(scan_code, False)
            self._held_scan_codes.clear()


def emergency_key_pressed() -> bool:
    """F12只作为设备安全中止键，不形成生命信号。"""

    if sys.platform != "win32":
        return False
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    return bool(user32.GetAsyncKeyState(0x7B) & 0x8000)
