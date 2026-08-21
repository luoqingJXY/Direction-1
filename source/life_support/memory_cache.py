"""不建立时间片的记忆缓存。

不同信息流可以由不同线程同时送入。缓存只在当前调用中把活动送到对应
入口，不等待其他信息流对齐，也不记录时间。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Condition, RLock, Thread, get_ident
from typing import Any


class MemoryEntrance(str, Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    PREDICTED_VISUAL = "predicted_visual"
    PREDICTED_AUDITORY = "predicted_auditory"
    MOUSE_ACTION = "mouse_action"
    KEYBOARD_ACTION = "keyboard_action"
    VIEW_CENTER_ACTION = "view_center_action"
    VOICE_ACTION = "voice_action"
    BRAIN_OUTPUTS = "brain_outputs"


Receiver = Callable[[Any], None]


@dataclass(slots=True)
class _PendingOccurrence:
    """工程中的一项待发送内容；编号、时刻或标签都不进入 Brain。"""

    entrance: MemoryEntrance
    activity: Any
    complete: bool = False
    error: BaseException | None = None
    completed: Condition = field(default_factory=lambda: Condition(RLock()))


ACTION_ENTRANCES = frozenset(
    {
        MemoryEntrance.MOUSE_ACTION,
        MemoryEntrance.KEYBOARD_ACTION,
        MemoryEntrance.VIEW_CENTER_ACTION,
        MemoryEntrance.VOICE_ACTION,
    }
)


class MemoryCache:
    def __init__(self) -> None:
        self._receivers: dict[MemoryEntrance, list[Receiver]] = {}
        self._action_tendencies: dict[MemoryEntrance, Any] = {}
        self._sending = True
        self._lock = RLock()
        self._pending: deque[_PendingOccurrence] = deque()
        self._dispatching = False
        self._dispatch_thread: int | None = None
        self._background = False
        self._worker: Thread | None = None
        self._wake = Condition(self._lock)

    @property
    def sending(self) -> bool:
        with self._lock:
            return self._sending

    def register(self, entrance: MemoryEntrance, receiver: Receiver) -> None:
        with self._lock:
            self._receivers.setdefault(entrance, []).append(receiver)

    def accept(self, entrance: MemoryEntrance, activity: Any) -> bool:
        """按实际到达顺序发送一项完整信息流活动。

        不同线程可以异步调用本方法，但两项发生不能在接收端执行到一半时
        彼此穿插。第一个到达空闲缓存的调用负责依次送出已经进入容器的内容；
        同一条生命回环在接收过程中产生的新输出只排到当前发生之后，不会
        递归插进当前状态变化中间。这里不记录时刻，也不等待其他信息流同步。
        """

        return self._accept(entrance, activity, remember_action_tendency=True)

    def accept_actual_action(self, entrance: MemoryEntrance, activity: Any) -> bool:
        """发送已经实际发生的教学动作，不覆盖Brain自己的动作倾向。"""

        entrance = MemoryEntrance(entrance)
        if entrance not in {
            MemoryEntrance.MOUSE_ACTION,
            MemoryEntrance.KEYBOARD_ACTION,
        }:
            raise ValueError("教学接管只能送入已经实际发生的鼠标或键盘动作")
        return self._accept(entrance, activity, remember_action_tendency=False)

    def _accept(
        self,
        entrance: MemoryEntrance,
        activity: Any,
        *,
        remember_action_tendency: bool,
    ) -> bool:
        entrance = MemoryEntrance(entrance)
        pending = _PendingOccurrence(entrance, activity)
        caller = get_ident()
        with self._lock:
            if not self._sending:
                return False
            if remember_action_tendency and entrance in ACTION_ENTRANCES:
                self._action_tendencies[entrance] = activity
            self._pending.append(pending)
            if self._background:
                self._wake.notify_all()
                drives_dispatch = False
                recursive = self._dispatch_thread == caller
            elif not self._dispatching:
                self._dispatching = True
                self._dispatch_thread = caller
                drives_dispatch = True
            else:
                drives_dispatch = False
                recursive = self._dispatch_thread == caller

        if drives_dispatch:
            self._send_pending_in_order()
        elif recursive:
            # 当前接收关系产生的下一项生命活动只能在当前发生完成后发送。
            return True
        else:
            with pending.completed:
                while not pending.complete:
                    pending.completed.wait()

        if pending.error is not None:
            raise pending.error
        return True

    def start_background_dispatch(self) -> None:
        """让持续生命回环由一个工作线程推进，调用线程不被永久占住。"""

        with self._lock:
            if self._background:
                return
            if self._dispatching:
                raise RuntimeError("已有生命发生正在发送，不能中途切换发送方式")
            self._background = True
            self._start_worker_locked()

    def _start_worker_locked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = Thread(
            target=self._background_loop,
            name="life-occurrence-dispatch",
            daemon=True,
        )
        self._worker.start()

    def _background_loop(self) -> None:
        with self._lock:
            self._dispatching = True
            self._dispatch_thread = get_ident()
        while True:
            with self._lock:
                while not self._pending and self._sending:
                    self._wake.wait()
                if not self._pending and not self._sending:
                    self._dispatching = False
                    self._dispatch_thread = None
                    return
                pending = self._pending.popleft()
                receivers = tuple(self._receivers.get(pending.entrance, ()))
            try:
                for receiver in receivers:
                    receiver(pending.activity)
            except BaseException as error:
                pending.error = error
            finally:
                with pending.completed:
                    pending.complete = True
                    pending.completed.notify_all()

    def _send_pending_in_order(self) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    self._dispatching = False
                    self._dispatch_thread = None
                    break
                pending = self._pending.popleft()
                receivers = tuple(self._receivers.get(pending.entrance, ()))

            try:
                for receiver in receivers:
                    receiver(pending.activity)
            except BaseException as error:
                pending.error = error
            finally:
                with pending.completed:
                    pending.complete = True
                    pending.completed.notify_all()

        # 每个非递归调用会在 accept 中取得属于自己的错误，避免把一项
        # 信息流的接收异常错误地归给另一项信息流。

    def latest_action_tendency(self, entrance: MemoryEntrance) -> Any | None:
        if entrance not in ACTION_ENTRANCES:
            raise ValueError("该入口不是动作倾向")
        with self._lock:
            return self._action_tendencies.get(entrance)

    def remember_action_tendency(self, entrance: MemoryEntrance, activity: Any) -> None:
        if entrance not in ACTION_ENTRANCES:
            raise ValueError("该入口不是动作倾向")
        with self._lock:
            self._action_tendencies[entrance] = activity

    def stop_sending(self) -> None:
        with self._lock:
            self._sending = False
            self._wake.notify_all()

    def resume_sending(self) -> None:
        with self._lock:
            self._sending = True
            if self._background:
                self._start_worker_locked()
            self._wake.notify_all()
