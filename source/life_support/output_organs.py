"""脑输出到客观器官切入点的传递。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from second_experiment.common.activities import (
    KeyboardActivities,
    MouseActivities,
    PositiveActivityField,
    VisualPredictionActivities,
    encode_positive_float32,
)
from second_experiment.host_b.entry_points import Channel

from .memory_cache import MemoryCache, MemoryEntrance


SendToHostB = Callable[[Channel, bytes], None]


@dataclass(frozen=True, slots=True)
class BrainOutputOccurrence:
    predicted_visual: VisualPredictionActivities
    predicted_auditory: PositiveActivityField
    mouse: MouseActivities
    keyboard: KeyboardActivities
    view_center: MouseActivities
    # 这是工程路由状态，不进入Brain。教学接管时，Brain的两组
    # 倾向仍保留在Action，但不冒充“已经实际发生”的器官反回。
    motor_feedback_included: bool = True


class OutputOrgans:
    def __init__(
        self,
        memory_cache: MemoryCache,
        send_to_host_b: SendToHostB,
        teaching_active: Callable[[], bool] | None = None,
    ) -> None:
        self.memory_cache = memory_cache
        self.send_to_host_b = send_to_host_b
        self.teaching_active = teaching_active or (lambda: False)

    def mouse(self, activities: MouseActivities) -> bool:
        self.memory_cache.remember_action_tendency(MemoryEntrance.MOUSE_ACTION, activities)
        if self.teaching_active():
            return True
        accepted = self.memory_cache.accept_actual_action(
            MemoryEntrance.MOUSE_ACTION, activities
        )
        if accepted:
            self.send_to_host_b(
                Channel.MOUSE_ACTIVITIES,
                encode_positive_float32(activities.values, 4),
            )
        return accepted

    def keyboard(self, activities: KeyboardActivities) -> bool:
        self.memory_cache.remember_action_tendency(
            MemoryEntrance.KEYBOARD_ACTION, activities
        )
        if self.teaching_active():
            return True
        accepted = self.memory_cache.accept_actual_action(
            MemoryEntrance.KEYBOARD_ACTION, activities
        )
        if accepted:
            self.send_to_host_b(
                Channel.KEYBOARD_ACTIVITIES,
                encode_positive_float32(activities.values, 108),
            )
        return accepted

    def view_center(self, activities: MouseActivities) -> bool:
        accepted = self.memory_cache.accept(MemoryEntrance.VIEW_CENTER_ACTION, activities)
        if accepted:
            self.send_to_host_b(
                Channel.VIEW_CENTER_ACTIVITIES,
                encode_positive_float32(activities.values, 4),
            )
        return accepted

    def predicted_visual(self, activities: VisualPredictionActivities) -> bool:
        return self.memory_cache.accept(MemoryEntrance.PREDICTED_VISUAL, activities)

    def predicted_auditory(self, activities: PositiveActivityField) -> bool:
        return self.memory_cache.accept(MemoryEntrance.PREDICTED_AUDITORY, activities)

    def complete_brain_outputs(
        self,
        occurrence: BrainOutputOccurrence,
    ) -> bool:
        """同一次脑状态变化形成的五组输出只返回为下一项完整发生一次。"""

        self.memory_cache.remember_action_tendency(
            MemoryEntrance.MOUSE_ACTION, occurrence.mouse
        )
        self.memory_cache.remember_action_tendency(
            MemoryEntrance.KEYBOARD_ACTION, occurrence.keyboard
        )
        self.memory_cache.remember_action_tendency(
            MemoryEntrance.VIEW_CENTER_ACTION, occurrence.view_center
        )
        teaching = self.teaching_active()
        feedback = occurrence
        if teaching:
            # 预测和视野中心仍属于同一次Brain输出；真实鼠标、键盘
            # 则由乙电脑上的教学操作异步返回，本次不伪造一份零动作。
            feedback = BrainOutputOccurrence(
                predicted_visual=occurrence.predicted_visual,
                predicted_auditory=occurrence.predicted_auditory,
                mouse=occurrence.mouse,
                keyboard=occurrence.keyboard,
                view_center=occurrence.view_center,
                motor_feedback_included=False,
            )
        accepted = self.memory_cache.accept(MemoryEntrance.BRAIN_OUTPUTS, feedback)
        if accepted:
            if not teaching:
                self.send_to_host_b(
                    Channel.MOUSE_ACTIVITIES,
                    encode_positive_float32(occurrence.mouse.values, 4),
                )
                self.send_to_host_b(
                    Channel.KEYBOARD_ACTIVITIES,
                    encode_positive_float32(occurrence.keyboard.values, 108),
                )
            self.send_to_host_b(
                Channel.VIEW_CENTER_ACTIVITIES,
                encode_positive_float32(occurrence.view_center.values, 4),
            )
        return accepted

    def voice_waveform(self, resolved_waveform: bytes) -> bool:
        """声音器官形成最终波形后才可调用；本方法不解释声音活动。"""

        accepted = self.memory_cache.accept(MemoryEntrance.VOICE_ACTION, bytes(resolved_waveform))
        if accepted:
            self.send_to_host_b(Channel.VOICE_OUTPUT, bytes(resolved_waveform))
        return accepted
