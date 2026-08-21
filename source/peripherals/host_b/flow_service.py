"""乙电脑已确认信息流的独立持续执行。"""

from __future__ import annotations

from collections.abc import Callable
from threading import Event, RLock, Thread, current_thread

from .actuation import ResolvedAction
from .actuation_rules import (
    KeyboardActuationRule,
    MouseActuationRule,
    VisualCenterActuationRule,
)
from .auditory_flow import ContinuousAuditoryFlow
from .runtime import HostBRuntime
from .visual_flow import ContinuousVisualFlow
from .teaching_input import WindowsTeachingInputCapture


class HostBFlowService:
    """七条通道并行维持视觉、听觉、动作与工程控制。"""

    def __init__(
        self,
        runtime: HostBRuntime,
        visual_flow: ContinuousVisualFlow,
        auditory_flow: ContinuousAuditoryFlow | None = None,
        *,
        visual_center_rule: VisualCenterActuationRule | None = None,
        mouse_rule: MouseActuationRule | None = None,
        keyboard_rule: KeyboardActuationRule | None = None,
        apply_input_action: Callable[[ResolvedAction], None] | None = None,
        teaching_input: WindowsTeachingInputCapture | None = None,
        external_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.runtime = runtime
        self.visual_flow = visual_flow
        self.auditory_flow = auditory_flow
        self.visual_center_rule = visual_center_rule
        self.mouse_rule = mouse_rule
        self.keyboard_rule = keyboard_rule
        self.apply_input_action = apply_input_action
        self.teaching_input = teaching_input
        if (mouse_rule is not None or keyboard_rule is not None) and apply_input_action is None:
            raise ValueError("启用鼠标或键盘接收时必须连接实际执行端")
        self.external_stop = external_stop or (lambda: False)
        self._stop = Event()
        self._workers: list[Thread] = []
        self._errors: list[BaseException] = []
        self._state_lock = RLock()
        self._started = False

    def _should_continue(self) -> bool:
        return not self._stop.is_set() and not self.external_stop()

    def _remember_error(self, error: BaseException) -> None:
        with self._state_lock:
            if not self._stop.is_set():
                self._errors.append(error)
        self._stop.set()
        self.runtime.close()

    def _run_visual(self) -> None:
        try:
            self.visual_flow.run(self._should_continue)
            self._stop.set()
            self.runtime.close()
        except (ConnectionError, OSError) as exc:
            if not self._stop.is_set():
                self._remember_error(exc)
        except BaseException as exc:
            self._remember_error(exc)

    def _run_control(self) -> None:
        try:
            while not self._stop.is_set():
                self.runtime.receive_and_apply_control()
                if self.runtime.closed:
                    self._stop.set()
                    return
        except (ConnectionError, OSError) as exc:
            if not self._stop.is_set():
                self._remember_error(exc)
        except BaseException as exc:
            self._remember_error(exc)

    def _run_computer_audio(self) -> None:
        try:
            while self._should_continue():
                assert self.auditory_flow is not None
                self.auditory_flow.process_computer_next()
        except (ConnectionError, OSError) as exc:
            if not self._stop.is_set():
                self._remember_error(exc)
        except BaseException as exc:
            self._remember_error(exc)

    def _run_microphone(self) -> None:
        try:
            while self._should_continue():
                assert self.auditory_flow is not None
                self.auditory_flow.process_microphone_next()
        except (ConnectionError, OSError) as exc:
            if not self._stop.is_set():
                self._remember_error(exc)
        except BaseException as exc:
            self._remember_error(exc)

    def _run_view_center(self) -> None:
        try:
            while self._should_continue():
                assert self.visual_center_rule is not None
                self.runtime.receive_and_apply_view_center(
                    self.visual_center_rule,
                    self.visual_flow.center,
                )
        except (ConnectionError, OSError) as exc:
            if not self._stop.is_set():
                self._remember_error(exc)
        except BaseException as exc:
            self._remember_error(exc)

    def _run_mouse(self) -> None:
        try:
            while self._should_continue():
                assert self.mouse_rule is not None
                assert self.apply_input_action is not None
                self.runtime.receive_and_apply_mouse(
                    self.mouse_rule,
                    self.apply_input_action,
                )
        except (ConnectionError, OSError) as exc:
            if not self._stop.is_set():
                self._remember_error(exc)
        except BaseException as exc:
            self._remember_error(exc)

    def _run_keyboard(self) -> None:
        try:
            while self._should_continue():
                assert self.keyboard_rule is not None
                assert self.apply_input_action is not None
                self.runtime.receive_and_apply_keyboard(
                    self.keyboard_rule,
                    self.apply_input_action,
                )
        except (ConnectionError, OSError) as exc:
            if not self._stop.is_set():
                self._remember_error(exc)
        except BaseException as exc:
            self._remember_error(exc)

    def start(self) -> None:
        with self._state_lock:
            if self._started:
                return
            self._started = True
        try:
            self.runtime.connect_all()
            if self.teaching_input is not None:
                self.teaching_input.start()
        except BaseException:
            with self._state_lock:
                self._started = False
            raise
        self._workers = [
            Thread(target=self._run_visual, name="host-b-visual"),
            Thread(target=self._run_control, name="host-b-control"),
        ]
        if self.auditory_flow is not None:
            self._workers.extend(
                (
                    Thread(target=self._run_computer_audio, name="host-b-computer-audio"),
                    Thread(target=self._run_microphone, name="host-b-microphone"),
                )
            )
        if self.visual_center_rule is not None:
            self._workers.append(
                Thread(target=self._run_view_center, name="host-b-view-center")
            )
        if self.mouse_rule is not None:
            self._workers.append(Thread(target=self._run_mouse, name="host-b-mouse"))
        if self.keyboard_rule is not None:
            self._workers.append(
                Thread(target=self._run_keyboard, name="host-b-keyboard")
            )
        for worker in self._workers:
            worker.start()

    def wait(self) -> None:
        for worker in self._workers:
            worker.join()
        with self._state_lock:
            if self._errors:
                raise RuntimeError("乙电脑信息流执行已经中止") from self._errors[0]

    def close(self) -> None:
        self._stop.set()
        try:
            self.visual_flow.close()
        finally:
            try:
                if self.teaching_input is not None:
                    self.teaching_input.close()
                if self.auditory_flow is not None:
                    self.auditory_flow.close()
            finally:
                self.runtime.close()
        for worker in list(self._workers):
            if worker is not current_thread():
                worker.join()

    def run(self) -> None:
        self.start()
        try:
            self.wait()
        finally:
            self.close()
