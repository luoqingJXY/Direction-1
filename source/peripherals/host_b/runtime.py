"""乙电脑外围运行编排；未确认的活动解释不会在此发生。"""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock

from second_experiment.common.activities import (
    KeyboardActivities,
    MouseActivities,
    decode_positive_float32,
    encode_positive_float32,
)
from second_experiment.host_a.control import ControlCommand, decode_control

from .actuation import ResolvedAction
from .actuation_rules import (
    KeyboardActuationRule,
    MouseActuationRule,
    VisualCenterActuationRule,
)
from .auditory_updates import AuditoryTransmission
from .config import HostBConfig
from .entry_points import Channel
from .transport import ChannelConnection
from .visual_processing import VisualCenterState
from .visual_updates import VisualUpdate


class HostBRuntime:
    def __init__(
        self,
        config: HostBConfig,
        *,
        start_teaching: Callable[[], None],
        stop_teaching: Callable[[], None],
        close_components: Callable[[], None],
    ) -> None:
        self.config = config
        self.start_teaching = start_teaching
        self.stop_teaching = stop_teaching
        self.close_components = close_components
        self.connections = {
            channel: ChannelConnection(config.network, channel) for channel in Channel
        }
        self.closed = False
        self._close_lock = RLock()

    def connect_all(self) -> None:
        connected: list[ChannelConnection] = []
        try:
            for connection in self.connections.values():
                connection.connect()
                connected.append(connection)
        except BaseException:
            for connection in connected:
                connection.close()
            raise

    @property
    def fully_connected(self) -> bool:
        return all(connection.socket is not None for connection in self.connections.values())

    def send_visual_update(self, update: VisualUpdate) -> None:
        if not self.config.visual.nonlinear_mapping_confirmed:
            raise RuntimeError("视野中心非线性空间变换的具体函数尚未确认")
        if not self.config.visual.low_frequency_operation_confirmed:
            raise RuntimeError("去低频操作的具体规则尚未确认")
        if (update.width, update.height) != (
            self.config.visual.width,
            self.config.visual.height,
        ):
            raise RuntimeError("视觉变化尺寸与第二次实验正式视觉尺寸不一致")
        self.connections[Channel.VISUAL_RECEPTOR_UPDATES].send(update.encode())

    def send_auditory_update(self, update: AuditoryTransmission) -> None:
        if not self.config.audio.auditory_field_confirmed:
            raise RuntimeError("当前配置没有启用已经确认的听觉感受场")
        self.connections[Channel.AUDITORY_RECEPTOR_UPDATES].send(update.encode())

    def send_teacher_mouse(self, activities: MouseActivities) -> None:
        self.connections[Channel.MOUSE_ACTIVITIES].send(
            encode_positive_float32(activities.values, 4)
        )

    def send_teacher_keyboard(self, activities: KeyboardActivities) -> None:
        self.connections[Channel.KEYBOARD_ACTIVITIES].send(
            encode_positive_float32(activities.values, 108)
        )

    def receive_and_apply_control(self) -> ControlCommand:
        payload = self.connections[Channel.CONTROL].receive()
        command = decode_control(payload)
        if command is ControlCommand.START_TEACHING:
            self.start_teaching()
        elif command is ControlCommand.STOP_TEACHING:
            self.stop_teaching()
        elif command is ControlCommand.CLOSE_AFTER_SLEEP:
            self.close()
        return command

    def receive_and_apply_view_center(
        self,
        rule: VisualCenterActuationRule,
        state: VisualCenterState,
    ) -> MouseActivities:
        """接收一次四路活动并让同一个视觉中心状态发生一次变化。"""

        payload = self.connections[Channel.VIEW_CENTER_ACTIVITIES].receive()
        values = decode_positive_float32(payload, 4)
        activities = MouseActivities(*values)
        rule.apply(activities, state)
        return activities

    def receive_and_apply_mouse(
        self,
        rule: MouseActuationRule,
        apply_action: Callable[[ResolvedAction], None],
    ) -> MouseActivities:
        payload = self.connections[Channel.MOUSE_ACTIVITIES].receive()
        values = decode_positive_float32(payload, 4)
        activities = MouseActivities(*values)
        apply_action(rule.resolve(activities))
        return activities

    def receive_and_apply_keyboard(
        self,
        rule: KeyboardActuationRule,
        apply_action: Callable[[ResolvedAction], None],
    ) -> KeyboardActivities:
        payload = self.connections[Channel.KEYBOARD_ACTIVITIES].receive()
        values = decode_positive_float32(payload, 108)
        activities = KeyboardActivities(values)
        apply_action(rule.resolve(activities))
        return activities

    def close(self) -> None:
        with self._close_lock:
            if self.closed:
                return
            try:
                self.close_components()
            finally:
                for connection in self.connections.values():
                    connection.close()
                self.closed = True
