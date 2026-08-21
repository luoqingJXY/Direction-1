"""甲电脑中持续存在的真实受体状态。"""

from __future__ import annotations

from collections.abc import Callable

from second_experiment.common.activities import PositiveActivityField
from second_experiment.host_b.auditory_field import (
    ACTIVITIES_PER_STREAM,
    AuditoryUpdate,
    TOTAL_ACTIVITY_COUNT,
)

from second_experiment.host_b.visual_updates import (
    VisualFrame,
    VisualUpdate,
    apply_visual_update,
)


class VisualReceptors:
    def __init__(self, on_activity: Callable[[VisualUpdate, VisualFrame], None] | None = None) -> None:
        self._state: VisualFrame | None = None
        self._on_activity = on_activity

    @property
    def initialized(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> VisualFrame:
        if self._state is None:
            raise RuntimeError("视觉受体尚未收到完整状态")
        return self._state

    def accept(self, update: VisualUpdate) -> None:
        self._state = apply_visual_update(self._state, update)
        if self._on_activity is not None:
            self._on_activity(update, self._state)


class AuditoryReceptors:
    """保存三份异步到达的当前听觉受体活动。"""

    def __init__(
        self,
        on_activity: Callable[[AuditoryUpdate, PositiveActivityField], None] | None = None,
    ) -> None:
        self._values = [0.0] * TOTAL_ACTIVITY_COUNT
        self._on_activity = on_activity

    @property
    def state(self) -> PositiveActivityField:
        return PositiveActivityField((3, 1025, 3), self._values)

    def accept(self, update: AuditoryUpdate) -> None:
        start = int(update.stream) * ACTIVITIES_PER_STREAM
        self._values[start : start + ACTIVITIES_PER_STREAM] = update.activities.values
        if self._on_activity is not None:
            self._on_activity(update, self.state)
