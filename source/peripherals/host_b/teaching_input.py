"""把教学期间真实发生的Windows鼠标、键盘操作还原为器官活动。

窗口挂钩和队列只是客观器官边界的工程实现。发往甲电脑的内容
仍只有四项非负鼠标活动或108项非负键盘活动，没有时刻、标签或答案。
"""

from __future__ import annotations

from collections.abc import Callable
import ctypes
from ctypes import wintypes
from queue import Queue
import sys
from threading import Event, RLock, Thread

from second_experiment.common.activities import KeyboardActivities, MouseActivities

from .actuation_rules import SECOND_EXPERIMENT_KEYBOARD_SCAN_CODES


def mouse_activities_for_delta(dx: int, dy: int) -> tuple[MouseActivities, ...]:
    """按已确认的“强度一=一个真实位置”完整表达一次鼠标位移。"""

    remaining_x = abs(int(dx))
    remaining_y = abs(int(dy))
    count = max(remaining_x, remaining_y)
    result: list[MouseActivities] = []
    for _ in range(count):
        x = 1.0 if remaining_x else 0.0
        y = 1.0 if remaining_y else 0.0
        remaining_x = max(0, remaining_x - 1)
        remaining_y = max(0, remaining_y - 1)
        result.append(
            MouseActivities(
                x if dx > 0 else 0.0,
                y if dy > 0 else 0.0,
                x if dx < 0 else 0.0,
                y if dy < 0 else 0.0,
            )
        )
    return tuple(result)


class TeachingKeyboardState:
    """把客观按下/松开变化排成已冻结的108项器官顺序。"""

    def __init__(self) -> None:
        self._index = {
            int(scan_code): index
            for index, scan_code in enumerate(SECOND_EXPERIMENT_KEYBOARD_SCAN_CODES)
            if int(scan_code) != 0
        }
        self._values = [0.0] * 108

    def change(self, scan_code: int, pressed: bool) -> KeyboardActivities | None:
        index = self._index.get(int(scan_code))
        if index is None:
            return None
        value = 1.0 if pressed else 0.0
        if self._values[index] == value:
            return None
        self._values[index] = value
        return KeyboardActivities(self._values)

    def release_all(self) -> KeyboardActivities | None:
        if not any(self._values):
            return None
        self._values = [0.0] * 108
        return KeyboardActivities(self._values)

    def clear(self) -> None:
        self._values = [0.0] * 108


class _Point(ctypes.Structure):
    _fields_ = (("x", wintypes.LONG), ("y", wintypes.LONG))


class _MouseHookData(ctypes.Structure):
    _fields_ = (
        ("point", _Point),
        ("mouse_data", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("extra_info", wintypes.WPARAM),
    )


class _KeyboardHookData(ctypes.Structure):
    _fields_ = (
        ("virtual_key", wintypes.DWORD),
        ("scan_code", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("extra_info", wintypes.WPARAM),
    )


class WindowsTeachingInputCapture:
    """Windows低层输入事件到教学动作信息流的连续边界。"""

    WH_MOUSE_LL = 14
    WH_KEYBOARD_LL = 13
    WM_MOUSEMOVE = 0x0200
    WM_KEYDOWN = 0x0100
    WM_KEYUP = 0x0101
    WM_SYSKEYDOWN = 0x0104
    WM_SYSKEYUP = 0x0105
    WM_QUIT = 0x0012
    HOOK_INJECTED = 0x01
    KEY_EXTENDED = 0x01
    KEY_INJECTED = 0x10

    def __init__(
        self,
        *,
        teaching_active: Callable[[], bool],
        capture_allowed: Callable[[], bool],
        send_mouse: Callable[[MouseActivities], None],
        send_keyboard: Callable[[KeyboardActivities], None],
        report_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        if sys.platform != "win32":
            raise RuntimeError("教学键鼠捕捉只支持Windows")
        self.teaching_active = teaching_active
        self.capture_allowed = capture_allowed
        self.send_mouse = send_mouse
        self.send_keyboard = send_keyboard
        self.report_error = report_error or (lambda _error: None)
        self._mouse_queue: Queue[MouseActivities | object] = Queue()
        self._keyboard_queue: Queue[KeyboardActivities | object] = Queue()
        self._sentinel = object()
        self._keyboard = TeachingKeyboardState()
        self._last_mouse: tuple[int, int] | None = None
        self._state_lock = RLock()
        self._ready = Event()
        self._closed = Event()
        self._hook_thread: Thread | None = None
        self._sender_threads: list[Thread] = []
        self._hook_thread_id = 0
        self._startup_error: BaseException | None = None
        self._mouse_hook = None
        self._keyboard_hook = None
        self._mouse_callback = None
        self._keyboard_callback = None

    def begin_teaching(self) -> None:
        with self._state_lock:
            self._last_mouse = None
            self._keyboard.clear()

    def finish_teaching(self) -> None:
        with self._state_lock:
            release = self._keyboard.release_all()
            self._last_mouse = None
        if release is not None:
            self._keyboard_queue.put(release)

    def _send_forever(self, queue: Queue, sender: Callable[[object], None]) -> None:
        while True:
            value = queue.get()
            if value is self._sentinel:
                return
            try:
                sender(value)
            except BaseException as error:
                self.report_error(error)
                return

    def _allowed(self) -> bool:
        return bool(self.teaching_active()) and bool(self.capture_allowed())

    def _capture_mouse(self, data: _MouseHookData) -> None:
        point = (int(data.point.x), int(data.point.y))
        with self._state_lock:
            if not self._allowed():
                self._last_mouse = point
                return
            previous = self._last_mouse
            self._last_mouse = point
        if previous is None:
            return
        for activity in mouse_activities_for_delta(
            point[0] - previous[0], point[1] - previous[1]
        ):
            self._mouse_queue.put(activity)

    def _capture_keyboard(self, data: _KeyboardHookData, pressed: bool) -> None:
        if not self._allowed():
            return
        scan_code = int(data.scan_code)
        if int(data.flags) & self.KEY_EXTENDED:
            scan_code |= 0xE000
        with self._state_lock:
            activity = self._keyboard.change(scan_code, pressed)
        if activity is not None:
            self._keyboard_queue.put(activity)

    def _hook_loop(self) -> None:
        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            result_type = ctypes.c_ssize_t
            callback_type = ctypes.WINFUNCTYPE(
                result_type, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
            )
            user32.SetWindowsHookExW.argtypes = (
                ctypes.c_int,
                callback_type,
                wintypes.HINSTANCE,
                wintypes.DWORD,
            )
            user32.SetWindowsHookExW.restype = wintypes.HHOOK
            user32.CallNextHookEx.argtypes = (
                wintypes.HHOOK,
                ctypes.c_int,
                wintypes.WPARAM,
                wintypes.LPARAM,
            )
            user32.CallNextHookEx.restype = result_type
            user32.UnhookWindowsHookEx.argtypes = (wintypes.HHOOK,)
            user32.UnhookWindowsHookEx.restype = wintypes.BOOL
            kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
            kernel32.GetModuleHandleW.restype = wintypes.HMODULE

            def mouse_callback(code, message, pointer):
                if code >= 0 and int(message) == self.WM_MOUSEMOVE:
                    data = ctypes.cast(
                        pointer, ctypes.POINTER(_MouseHookData)
                    ).contents
                    if not (int(data.flags) & self.HOOK_INJECTED):
                        self._capture_mouse(data)
                return user32.CallNextHookEx(None, code, message, pointer)

            def keyboard_callback(code, message, pointer):
                if code >= 0 and int(message) in {
                    self.WM_KEYDOWN,
                    self.WM_KEYUP,
                    self.WM_SYSKEYDOWN,
                    self.WM_SYSKEYUP,
                }:
                    data = ctypes.cast(
                        pointer, ctypes.POINTER(_KeyboardHookData)
                    ).contents
                    if not (int(data.flags) & self.KEY_INJECTED):
                        self._capture_keyboard(
                            data,
                            int(message) in {self.WM_KEYDOWN, self.WM_SYSKEYDOWN},
                        )
                return user32.CallNextHookEx(None, code, message, pointer)

            self._mouse_callback = callback_type(mouse_callback)
            self._keyboard_callback = callback_type(keyboard_callback)
            self._hook_thread_id = int(kernel32.GetCurrentThreadId())
            module = kernel32.GetModuleHandleW(None)
            self._mouse_hook = user32.SetWindowsHookExW(
                self.WH_MOUSE_LL, self._mouse_callback, module, 0
            )
            self._keyboard_hook = user32.SetWindowsHookExW(
                self.WH_KEYBOARD_LL, self._keyboard_callback, module, 0
            )
            if not self._mouse_hook or not self._keyboard_hook:
                raise ctypes.WinError(ctypes.get_last_error())
            self._ready.set()
            message = wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
                user32.TranslateMessage(ctypes.byref(message))
                user32.DispatchMessageW(ctypes.byref(message))
        except BaseException as error:
            self._startup_error = error
            self._ready.set()
            self.report_error(error)
        finally:
            try:
                user32 = ctypes.WinDLL("user32", use_last_error=True)
                if self._mouse_hook:
                    user32.UnhookWindowsHookEx(self._mouse_hook)
                if self._keyboard_hook:
                    user32.UnhookWindowsHookEx(self._keyboard_hook)
            finally:
                self._closed.set()

    def start(self) -> None:
        if self._hook_thread is not None:
            return
        self._sender_threads = [
            Thread(
                target=self._send_forever,
                args=(self._mouse_queue, self.send_mouse),
                name="teaching-mouse-sender",
                daemon=True,
            ),
            Thread(
                target=self._send_forever,
                args=(self._keyboard_queue, self.send_keyboard),
                name="teaching-keyboard-sender",
                daemon=True,
            ),
        ]
        for thread in self._sender_threads:
            thread.start()
        self._hook_thread = Thread(
            target=self._hook_loop,
            name="teaching-windows-input",
            daemon=True,
        )
        self._hook_thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError("无法启动教学键鼠捕捉") from self._startup_error

    def close(self) -> None:
        if self._hook_thread is None:
            return
        if self._hook_thread_id:
            ctypes.WinDLL("user32", use_last_error=True).PostThreadMessageW(
                self._hook_thread_id, self.WM_QUIT, 0, 0
            )
        self._closed.wait(timeout=5.0)
        self._mouse_queue.put(self._sentinel)
        self._keyboard_queue.put(self._sentinel)
        for thread in self._sender_threads:
            thread.join(timeout=5.0)
