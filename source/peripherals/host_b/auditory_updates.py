"""三份听觉感受场的无损变化传递。

比较、地址和变化段只用于减少局域网字节。甲电脑恢复出的仍是当前一份
1,025×3非负听觉活动；这些工程信息不会进入Brain。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import numpy

from second_experiment.common.activities import PositiveActivityField

from .auditory_field import (
    ACTIVITIES_PER_POSITION,
    FREQUENCY_POSITION_COUNT,
    AuditoryStream,
    AuditoryUpdate,
)
from .errors import ProtocolError


FULL_STATE = 0
CHANGED_SPANS = 1
UPDATE_HEADER = struct.Struct("!BBH")
SPAN_HEADER = struct.Struct("!HH")
FLOAT32 = struct.Struct("!f")


@dataclass(frozen=True, slots=True)
class AuditoryChangedSpan:
    start_position: int
    activities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not 0 <= int(self.start_position) < FREQUENCY_POSITION_COUNT:
            raise ValueError("听觉变化段起点超出1,025个频率位置")
        values = tuple(float(value) for value in self.activities)
        object.__setattr__(self, "activities", values)
        if not values or len(values) % ACTIVITIES_PER_POSITION:
            raise ValueError("听觉变化段必须包含完整的三项位置活动")
        if self.start_position + self.position_count > FREQUENCY_POSITION_COUNT:
            raise ValueError("听觉变化段超出1,025个频率位置")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("听觉变化活动必须是[0,1]内的有限值")

    @property
    def position_count(self) -> int:
        return len(self.activities) // ACTIVITIES_PER_POSITION


@dataclass(frozen=True, slots=True)
class AuditoryTransmission:
    stream: AuditoryStream
    full_activities: tuple[float, ...] | None = None
    spans: tuple[AuditoryChangedSpan, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream", AuditoryStream(self.stream))
        if self.full_activities is not None:
            values = tuple(float(value) for value in self.full_activities)
            object.__setattr__(self, "full_activities", values)
            if self.spans:
                raise ValueError("完整听觉状态不能同时携带变化段")
            if len(values) != FREQUENCY_POSITION_COUNT * ACTIVITIES_PER_POSITION:
                raise ValueError("完整听觉状态数量不正确")
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
                raise ValueError("完整听觉活动必须是[0,1]内的有限值")
        else:
            previous_stop = 0
            for span in self.spans:
                if span.start_position < previous_stop:
                    raise ValueError("听觉变化段必须按位置递增且不能重叠")
                previous_stop = span.start_position + span.position_count

    @property
    def is_full(self) -> bool:
        return self.full_activities is not None

    @property
    def changed_position_count(self) -> int:
        if self.full_activities is not None:
            return FREQUENCY_POSITION_COUNT
        return sum(span.position_count for span in self.spans)

    def encode(self) -> bytes:
        if self.full_activities is not None:
            return UPDATE_HEADER.pack(FULL_STATE, int(self.stream), 0) + struct.pack(
                f"!{len(self.full_activities)}f",
                *self.full_activities,
            )
        chunks = [UPDATE_HEADER.pack(CHANGED_SPANS, int(self.stream), len(self.spans))]
        for span in self.spans:
            chunks.append(SPAN_HEADER.pack(span.start_position, span.position_count))
            chunks.append(struct.pack(f"!{len(span.activities)}f", *span.activities))
        return b"".join(chunks)

    @classmethod
    def decode(cls, payload: bytes) -> "AuditoryTransmission":
        if len(payload) < UPDATE_HEADER.size:
            raise ProtocolError("听觉变化数据不完整")
        kind, stream_number, span_count = UPDATE_HEADER.unpack(
            payload[: UPDATE_HEADER.size]
        )
        try:
            stream = AuditoryStream(stream_number)
        except ValueError as exc:
            raise ProtocolError("听觉变化的声音流地址无效") from exc
        position = UPDATE_HEADER.size
        if kind == FULL_STATE:
            if span_count != 0:
                raise ProtocolError("完整听觉状态包含无效变化段数量")
            count = FREQUENCY_POSITION_COUNT * ACTIVITIES_PER_POSITION
            if len(payload) != position + count * FLOAT32.size:
                raise ProtocolError("完整听觉状态字节数不正确")
            values = struct.unpack(f"!{count}f", payload[position:])
            try:
                return cls(stream, full_activities=values)
            except ValueError as exc:
                raise ProtocolError(str(exc)) from exc
        if kind != CHANGED_SPANS:
            raise ProtocolError("未知听觉变化类型")
        spans: list[AuditoryChangedSpan] = []
        try:
            for _ in range(span_count):
                if position + SPAN_HEADER.size > len(payload):
                    raise ProtocolError("听觉变化段头不完整")
                start, count = SPAN_HEADER.unpack(
                    payload[position : position + SPAN_HEADER.size]
                )
                position += SPAN_HEADER.size
                value_count = count * ACTIVITIES_PER_POSITION
                byte_count = value_count * FLOAT32.size
                if count == 0 or position + byte_count > len(payload):
                    raise ProtocolError("听觉变化段范围或字节数无效")
                values = struct.unpack(
                    f"!{value_count}f",
                    payload[position : position + byte_count],
                )
                spans.append(AuditoryChangedSpan(start, values))
                position += byte_count
            if position != len(payload):
                raise ProtocolError("听觉变化数据尾部存在多余字节")
            return cls(stream, spans=tuple(spans))
        except ValueError as exc:
            raise ProtocolError(str(exc)) from exc


def _float32_values(update: AuditoryUpdate) -> numpy.ndarray:
    return numpy.asarray(update.activities.values, dtype="<f4").reshape(
        FREQUENCY_POSITION_COUNT,
        ACTIVITIES_PER_POSITION,
    )


class ExactAuditoryChangeEncoder:
    """逐频率位置精确比较，不添加变化阈值或内容判断。"""

    def __init__(self) -> None:
        self._previous: dict[AuditoryStream, numpy.ndarray] = {}

    def build(self, update: AuditoryUpdate) -> AuditoryTransmission | None:
        current = _float32_values(update)
        previous = self._previous.get(update.stream)
        if previous is None:
            transmission = AuditoryTransmission(
                update.stream,
                full_activities=tuple(float(value) for value in current.reshape(-1)),
            )
        else:
            changed = numpy.any(previous != current, axis=1)
            spans: list[AuditoryChangedSpan] = []
            position = 0
            while position < FREQUENCY_POSITION_COUNT:
                if not changed[position]:
                    position += 1
                    continue
                start = position
                while position < FREQUENCY_POSITION_COUNT and changed[position]:
                    position += 1
                values = current[start:position].reshape(-1)
                spans.append(
                    AuditoryChangedSpan(
                        start,
                        tuple(float(value) for value in values),
                    )
                )
            if not spans:
                return None
            transmission = AuditoryTransmission(update.stream, spans=tuple(spans))
        self._previous[update.stream] = current.copy()
        return transmission


class AuditoryTransmissionReconstructor:
    """在甲电脑恢复三份当前完整听觉活动。"""

    def __init__(self) -> None:
        self._states: dict[AuditoryStream, numpy.ndarray] = {}

    def apply(self, transmission: AuditoryTransmission) -> AuditoryUpdate:
        if transmission.full_activities is not None:
            values = numpy.asarray(transmission.full_activities, dtype="<f4").reshape(
                FREQUENCY_POSITION_COUNT,
                ACTIVITIES_PER_POSITION,
            )
        else:
            previous = self._states.get(transmission.stream)
            if previous is None:
                raise ValueError("一份声音流的首次听觉传递必须是完整状态")
            values = previous.copy()
            for span in transmission.spans:
                start = span.start_position
                stop = start + span.position_count
                values[start:stop] = numpy.asarray(
                    span.activities,
                    dtype="<f4",
                ).reshape(span.position_count, ACTIVITIES_PER_POSITION)
        self._states[transmission.stream] = values.copy()
        return AuditoryUpdate(
            transmission.stream,
            PositiveActivityField(
                (FREQUENCY_POSITION_COUNT, ACTIVITIES_PER_POSITION),
                values.reshape(-1),
            ),
        )
