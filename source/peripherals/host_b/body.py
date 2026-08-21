"""乙电脑上真实人工身体的工程容器。"""

from __future__ import annotations

from collections.abc import Callable

from .actuation import ResolvedAction, WindowsInputSink
from .visual_processing import VisualCenterState


class ArtificialBody:
    """承载已确认的实际执行端，不另外产生一种身体动作。"""

    def __init__(
        self,
        input_sink: WindowsInputSink,
        visual_center: VisualCenterState,
        play_voice_waveform: Callable[[bytes], None] | None = None,
    ) -> None:
        self.input_sink = input_sink
        self.visual_center = visual_center
        self.play_voice_waveform = play_voice_waveform
        self.closed = False

    @property
    def teaching_active(self) -> bool:
        return self.input_sink.teaching.active

    def start_teaching(self) -> None:
        self.input_sink.teaching.start()

    def stop_teaching(self) -> None:
        self.input_sink.teaching.stop()

    def apply_mouse_and_keyboard(self, action: ResolvedAction) -> None:
        if self.closed:
            raise RuntimeError("人工身体已经关闭")
        self.input_sink.apply(action)

    def output_voice(self, resolved_waveform: bytes) -> None:
        if self.closed:
            raise RuntimeError("人工身体已经关闭")
        if self.play_voice_waveform is None:
            raise RuntimeError("声音实际播放端尚未连接")
        self.play_voice_waveform(bytes(resolved_waveform))

    def close(self) -> None:
        if self.closed:
            return
        self.input_sink.release_all()
        self.closed = True

