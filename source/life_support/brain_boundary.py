"""脑与已完成外围组件之间的固定切入点。

这里没有脑实现，也没有人工DNA实现。以后真正的脑必须满足这些连接方法，
但方法内部的区域、神经元和路径由确认后的人工DNA与脑规则决定。
"""

from __future__ import annotations

from typing import Protocol

from second_experiment.common.activities import (
    KeyboardActivities,
    MouseActivities,
    PositiveActivityField,
)
from second_experiment.host_b.auditory_field import AuditoryUpdate
from second_experiment.host_b.visual_updates import VisualFrame, VisualUpdate


class BrainBoundary(Protocol):
    def receive_visual(self, update: VisualUpdate, current_state: VisualFrame) -> None: ...

    def receive_auditory(
        self,
        update: AuditoryUpdate,
        current_state: PositiveActivityField,
    ) -> None: ...

    def receive_predicted_visual(self, activities: PositiveActivityField) -> None: ...

    def receive_predicted_auditory(self, activities: PositiveActivityField) -> None: ...

    def receive_mouse_action(self, activities: MouseActivities) -> None: ...

    def receive_keyboard_action(self, activities: KeyboardActivities) -> None: ...

    def receive_view_center_action(self, activities: MouseActivities) -> None: ...

    def weaken_plastic_paths_for_sleep(self) -> None: ...

    def export_life_structure(self) -> bytes: ...

    def export_path_state(self) -> bytes: ...

    def restore_existing_life(self, life_structure: bytes, path_state: bytes) -> None: ...

    def shutdown(self) -> None: ...
