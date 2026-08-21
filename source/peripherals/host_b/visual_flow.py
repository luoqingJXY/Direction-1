"""乙电脑真实视觉从捕捉到发送的发生顺序编排。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .config import HostBConfig
from .minecraft_window import MinecraftWindow
from .minecraft_window import MinecraftWindowLocator
from .visual_capture import MssMinecraftCapture
from .visual_processing import VisualCenterState, build_confirmed_visual_processing
from .visual_updates import ReferenceVisualChangeEncoder, VisualFrame, VisualUpdate


class VisualCapture(Protocol):
    def read(self) -> tuple[VisualFrame, MinecraftWindow]: ...

    def close(self) -> None: ...


class VisualProcessing(Protocol):
    def process(self, raw_frame: VisualFrame, center: VisualCenterState) -> VisualFrame: ...


class ContinuousVisualFlow:
    """逐份完成捕捉、处理、变化形成和同步发送。

    本类不解释视觉内容，也不引入生命时间。只有上一份变化已经成功发送后，
    当前完整状态才成为下一次比较的依据。
    """

    def __init__(
        self,
        capture: VisualCapture,
        processing: VisualProcessing,
        center: VisualCenterState,
        send_visual_update: Callable[[VisualUpdate], None],
        *,
        encoder: ReferenceVisualChangeEncoder | None = None,
    ) -> None:
        self.capture = capture
        self.processing = processing
        self.center = center
        self.send_visual_update = send_visual_update
        self.encoder = encoder or ReferenceVisualChangeEncoder()
        self.last_sent_frame: VisualFrame | None = None

    def process_next(self) -> bool:
        """处理下一份真实画面；返回本次是否实际发送。"""

        raw_frame, _window = self.capture.read()
        current_frame = self.processing.process(raw_frame, self.center)
        update = self.encoder.build(self.last_sent_frame, current_frame)

        should_send = update.is_full or update.changed_pixel_count > 0
        if not should_send:
            return False

        self.send_visual_update(update)
        self.last_sent_frame = current_frame
        return True

    def run(self, should_continue: Callable[[], bool]) -> None:
        """按实际发生顺序连续处理；不依据时间或固定轮次运行。"""

        while should_continue():
            self.process_next()

    def close(self) -> None:
        self.capture.close()


def build_confirmed_continuous_visual_flow(
    config: HostBConfig,
    center: VisualCenterState,
    send_visual_update: Callable[[VisualUpdate], None],
) -> ContinuousVisualFlow:
    """组装乙电脑已经确认的正式视觉链路。"""

    if not config.visual.nonlinear_mapping_confirmed:
        raise RuntimeError("视野中心非线性空间变换的具体函数尚未确认")
    if not config.visual.low_frequency_operation_confirmed:
        raise RuntimeError("去低频操作的具体规则尚未确认")
    locator = MinecraftWindowLocator(config.minecraft)
    capture = MssMinecraftCapture(locator)
    return ContinuousVisualFlow(
        capture,
        build_confirmed_visual_processing(
            config.visual.width,
            config.visual.height,
        ),
        center,
        send_visual_update,
    )
