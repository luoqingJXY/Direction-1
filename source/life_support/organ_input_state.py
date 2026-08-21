"""统一公式中 U 的当前器官输入状态。

这个文件不是新的脑组件。它只让已经由乙电脑形成、并已通过 Memory Cache
异步送达的器官活动，保留在各自固定接收神经元的位置。它不修改 Neuron
当前活动 A，不决定阈值，不形成 Path 当前传播 Q，也不保存发生时刻。
"""

from __future__ import annotations

import numpy

from second_experiment.common.activities import (
    KeyboardActivities,
    MouseActivities,
    PositiveActivityField,
    VisualPredictionActivities,
)
from second_experiment.host_b.auditory_field import AuditoryUpdate
from second_experiment.host_b.visual_updates import VisualFrame, VisualUpdate

from .fixed_receiver_arrivals import FixedReceiverArrivalBuilder, FixedReceiverArrivals


class OrganInputState:
    """以神经组织地址保存当前 U，直到对应器官下一次活动到达。

    ``values`` 必须由 Runtime 在内存中提供。Sleep 的“全部关闭”不应把它
    当作后天 Path 状态写入硬盘；重新启动后，它只会由重新到达的真实器官
    活动再次形成。
    """

    def __init__(self, values: numpy.ndarray) -> None:
        state = numpy.asarray(values)
        if state.ndim != 1:
            raise ValueError("器官当前输入状态必须是一维神经组织排列")
        if not numpy.issubdtype(state.dtype, numpy.floating):
            raise ValueError("器官当前输入状态必须使用连续数值")
        if not numpy.isfinite(state).all() or numpy.any(state < 0.0):
            raise ValueError("器官当前输入状态必须是有限非负活动")
        self._values = state

    @classmethod
    def empty(cls, neuron_count: int) -> "OrganInputState":
        if int(neuron_count) <= 0:
            raise ValueError("神经组织容量必须大于零")
        return cls(numpy.zeros(int(neuron_count), dtype=numpy.float32))

    @property
    def values(self) -> numpy.ndarray:
        """当前 U；调用方只读后再与同一位置的 Q 直接相加形成 S。"""

        return self._values

    def accept(self, arrivals: FixedReceiverArrivals) -> int:
        """逐项更新一个器官实际发生带来的当前活动。

        同一批中的每项活动已经有唯一固定接收端，所以这里不平均、相加或
        解释。不同器官或异步声音流未被本次发生触及的位置保持原活动。
        """

        indexes = arrivals.receiver_indices
        if indexes.size and int(indexes.max()) >= self._values.size:
            raise ValueError("器官固定接收神经元超出当前神经组织容量")
        self._values[indexes] = arrivals.activities
        return arrivals.count


class OrganInputReceiver:
    """把 Memory Cache 已发送的器官活动精确落到 U。

    这是 Brain 内部对器官入口的实现切面，不是新的器官或新的区域。视觉
    更新先由 Runtime 重建为乙电脑已经形成的完整感受场，随后才进入这里；
    所以网络的变化段格式不会成为 Brain 的输入结构。
    """

    def __init__(
        self,
        state: OrganInputState,
        *,
        builder: FixedReceiverArrivalBuilder | None = None,
    ) -> None:
        self.state = state
        self.builder = builder or FixedReceiverArrivalBuilder()

    def receive_visual(self, _update: VisualUpdate, current_state: VisualFrame) -> int:
        return self.state.accept(self.builder.visual_field(current_state))

    def receive_auditory(
        self,
        update: AuditoryUpdate,
        _current_state: PositiveActivityField,
    ) -> int:
        # 一次声音发生只更新实际到达的那一条独立声音流。
        return self.state.accept(self.builder.auditory(update))

    def receive_predicted_visual(self, activities: VisualPredictionActivities) -> int:
        return self.state.accept(self.builder.predicted_visual(activities))

    def receive_predicted_auditory(self, activities: PositiveActivityField) -> int:
        return self.state.accept(self.builder.predicted_auditory(activities))

    def receive_mouse_action(self, activities: MouseActivities) -> int:
        return self.state.accept(self.builder.mouse(activities))

    def receive_keyboard_action(self, activities: KeyboardActivities) -> int:
        return self.state.accept(self.builder.keyboard(activities))

    def receive_view_center_action(self, activities: MouseActivities) -> int:
        return self.state.accept(self.builder.view_center(activities))
