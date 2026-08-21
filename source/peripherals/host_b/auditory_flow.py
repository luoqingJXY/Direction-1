"""电脑左右声道与麦克风三份听觉活动的异步形成。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .audio_capture import Float32AudioCapture
from .auditory_field import AuditoryUpdate, ConfirmedAuditoryField
from .auditory_updates import AuditoryTransmission, ExactAuditoryChangeEncoder
from .config import HostBConfig


class RawAudioCapture(Protocol):
    channels: int
    sample_rate: int

    def read(self) -> bytes: ...

    def close(self) -> None: ...


class ContinuousAuditoryFlow:
    def __init__(
        self,
        computer_output: RawAudioCapture,
        microphone: RawAudioCapture,
        field: ConfirmedAuditoryField,
        send_update: Callable[[AuditoryTransmission], None],
    ) -> None:
        self.computer_output = computer_output
        self.microphone = microphone
        self.field = field
        self.send_update = send_update
        self.encoder = ExactAuditoryChangeEncoder()

    def _send_changed(self, update: AuditoryUpdate) -> bool:
        transmission = self.encoder.build(update)
        if transmission is None:
            return False
        self.send_update(transmission)
        return True

    def process_computer_next(self) -> int:
        updates = self.field.accept_computer_output(
            self.computer_output.read(),
            self.computer_output.channels,
            self.computer_output.sample_rate,
        )
        return sum(self._send_changed(update) for update in updates)

    def process_microphone_next(self) -> int:
        updates = self.field.accept_microphone(
            self.microphone.read(),
            self.microphone.channels,
            self.microphone.sample_rate,
        )
        return sum(self._send_changed(update) for update in updates)

    def close(self) -> None:
        self.computer_output.close()
        self.microphone.close()


def build_confirmed_continuous_auditory_flow(
    config: HostBConfig,
    send_update: Callable[[AuditoryTransmission], None],
) -> ContinuousAuditoryFlow:
    if not config.audio.auditory_field_confirmed:
        raise RuntimeError("当前配置没有启用已经确认的听觉感受场")
    return ContinuousAuditoryFlow(
        Float32AudioCapture(
            config.audio.computer_output_device,
            channels=2,
            require_loopback=True,
        ),
        Float32AudioCapture(
            config.audio.microphone_device,
            channels=1,
            require_loopback=False,
        ),
        ConfirmedAuditoryField(),
        send_update,
    )
