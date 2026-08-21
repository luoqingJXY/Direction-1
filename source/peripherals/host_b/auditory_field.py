"""冻结理论已经确认的三份听觉感受活动。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import struct

import numpy

from second_experiment.common.activities import PositiveActivityField
from .errors import ProtocolError


SAMPLE_RATE = 48_000
NEW_SAMPLE_COUNT = 192
WINDOW_SAMPLE_COUNT = 2_048
FREQUENCY_POSITION_COUNT = WINDOW_SAMPLE_COUNT // 2 + 1
ACTIVITIES_PER_POSITION = 3
ACTIVITIES_PER_STREAM = FREQUENCY_POSITION_COUNT * ACTIVITIES_PER_POSITION
TOTAL_ACTIVITY_COUNT = 3 * ACTIVITIES_PER_STREAM


class AuditoryStream(IntEnum):
    COMPUTER_LEFT = 0
    COMPUTER_RIGHT = 1
    MICROPHONE = 2


@dataclass(frozen=True, slots=True)
class AuditoryUpdate:
    """一份声音流刚形成的1,025×3项活动及其工程入口地址。"""

    stream: AuditoryStream
    activities: PositiveActivityField

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream", AuditoryStream(self.stream))
        if self.activities.shape != (FREQUENCY_POSITION_COUNT, ACTIVITIES_PER_POSITION):
            raise ValueError("每份听觉活动必须为1,025×3")
        if any(value > 1.0 for value in self.activities.values):
            raise ValueError("听觉活动必须处于[0,1]")

    def encode(self) -> bytes:
        return bytes((int(self.stream),)) + struct.pack(
            f"!{ACTIVITIES_PER_STREAM}f",
            *self.activities.values,
        )

    @classmethod
    def decode(cls, payload: bytes) -> "AuditoryUpdate":
        expected = 1 + ACTIVITIES_PER_STREAM * 4
        if len(payload) != expected:
            raise ProtocolError(f"一份听觉活动字节数必须为{expected}")
        try:
            stream = AuditoryStream(payload[0])
        except ValueError as exc:
            raise ProtocolError("听觉活动入口地址无效") from exc
        values = struct.unpack(f"!{ACTIVITIES_PER_STREAM}f", payload[1:])
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ProtocolError("听觉活动必须是[0,1]内的有限数")
        return cls(
            stream,
            PositiveActivityField(
                (FREQUENCY_POSITION_COUNT, ACTIVITIES_PER_POSITION),
                values,
            ),
        )


class ConfirmedAuditoryField:
    """把三份原始声音分别变成冻结理论规定的非负听觉活动。"""

    def __init__(self) -> None:
        self._windows = numpy.zeros(
            (len(AuditoryStream), WINDOW_SAMPLE_COUNT),
            dtype=numpy.float32,
        )
        self._pending = {
            stream: numpy.empty(0, dtype=numpy.float32) for stream in AuditoryStream
        }

    @property
    def activity_count(self) -> int:
        return TOTAL_ACTIVITY_COUNT

    @staticmethod
    def _decode(raw_float32: bytes, channels: int) -> numpy.ndarray:
        if channels <= 0:
            raise ValueError("原始声音通道数量必须大于零")
        values = numpy.frombuffer(raw_float32, dtype="<f4")
        if values.size % channels:
            raise ValueError("原始声音字节没有形成完整取样")
        if not numpy.isfinite(values).all():
            raise ValueError("原始声音包含无效数值")
        if values.size and (values.min() < -1.0 or values.max() > 1.0):
            raise ValueError("32位浮点原始声音必须处于[-1,1]")
        return values.reshape(-1, channels)

    @staticmethod
    def _activities(window: numpy.ndarray) -> PositiveActivityField:
        # 除以固定窗口长度，不根据当前声音内容重新归一化。
        response = numpy.fft.rfft(window.astype(numpy.float64)) / WINDOW_SAMPLE_COUNT
        x = response.real
        y = response.imag
        radius = numpy.abs(response)
        activities = numpy.stack(
            (radius, (radius + x) / 2.0, (radius + y) / 2.0),
            axis=1,
        )
        activities = numpy.clip(activities, 0.0, 1.0)
        return PositiveActivityField(
            (FREQUENCY_POSITION_COUNT, ACTIVITIES_PER_POSITION),
            activities.reshape(-1),
        )

    def _accept_stream(
        self,
        stream: AuditoryStream,
        samples: numpy.ndarray,
    ) -> tuple[AuditoryUpdate, ...]:
        pending = numpy.concatenate((self._pending[stream], samples.astype(numpy.float32)))
        updates: list[AuditoryUpdate] = []
        while pending.size >= NEW_SAMPLE_COUNT:
            new = pending[:NEW_SAMPLE_COUNT]
            pending = pending[NEW_SAMPLE_COUNT:]
            window = self._windows[int(stream)]
            window[:-NEW_SAMPLE_COUNT] = window[NEW_SAMPLE_COUNT:]
            window[-NEW_SAMPLE_COUNT:] = new
            updates.append(AuditoryUpdate(stream, self._activities(window)))
        self._pending[stream] = pending
        return tuple(updates)

    def accept_computer_output(
        self,
        raw_float32: bytes,
        channels: int,
        sample_rate: int,
    ) -> tuple[AuditoryUpdate, ...]:
        if sample_rate != SAMPLE_RATE:
            raise ValueError("电脑声音采集频率必须为48,000")
        values = self._decode(raw_float32, channels)
        if channels != 2:
            raise ValueError("电脑声音必须提供独立的左、右两个声道")
        left = self._accept_stream(AuditoryStream.COMPUTER_LEFT, values[:, 0])
        right = self._accept_stream(AuditoryStream.COMPUTER_RIGHT, values[:, 1])
        result: list[AuditoryUpdate] = []
        for index in range(max(len(left), len(right))):
            if index < len(left):
                result.append(left[index])
            if index < len(right):
                result.append(right[index])
        return tuple(result)

    def accept_microphone(
        self,
        raw_float32: bytes,
        channels: int,
        sample_rate: int,
    ) -> tuple[AuditoryUpdate, ...]:
        if sample_rate != SAMPLE_RATE:
            raise ValueError("麦克风采集频率必须为48,000")
        values = self._decode(raw_float32, channels)
        if channels != 1:
            raise ValueError("麦克风必须作为一份独立单声道活动采集")
        return self._accept_stream(AuditoryStream.MICROPHONE, values[:, 0])
