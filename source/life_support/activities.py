"""已经确认的数据量所对应的非负活动。

这些类型只检查活动数量和非负性，不解释活动代表的目标、答案或意义。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterable

from second_experiment.host_b.errors import ProtocolError


def _positive_values(values: Iterable[float], expected_count: int | None = None) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if expected_count is not None and len(result) != expected_count:
        raise ValueError(f"活动数量必须为{expected_count}")
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError("所有生命活动必须是有限非负数")
    return result


@dataclass(frozen=True, slots=True)
class MouseActivities:
    x_positive: float
    y_positive: float
    x_negative: float
    y_negative: float

    def __post_init__(self) -> None:
        _positive_values(self.values, 4)

    @property
    def values(self) -> tuple[float, float, float, float]:
        return self.x_positive, self.y_positive, self.x_negative, self.y_negative


@dataclass(frozen=True, slots=True)
class KeyboardActivities:
    values: tuple[float, ...]

    def __init__(self, values: Iterable[float]) -> None:
        object.__setattr__(self, "values", _positive_values(values, 108))


@dataclass(frozen=True, slots=True)
class PositiveActivityField:
    shape: tuple[int, ...]
    values: tuple[float, ...]

    def __init__(self, shape: Iterable[int], values: Iterable[float]) -> None:
        normalized_shape = tuple(int(item) for item in shape)
        if not normalized_shape or any(item <= 0 for item in normalized_shape):
            raise ValueError("活动场每个尺寸必须大于零")
        count = math.prod(normalized_shape)
        object.__setattr__(self, "shape", normalized_shape)
        object.__setattr__(self, "values", _positive_values(values, count))


class VisualPredictionActivities(PositiveActivityField):
    def __init__(self, values: Iterable[float]) -> None:
        normalized = tuple(float(value) for value in values)
        if any(value > 1.0 for value in normalized):
            raise ValueError("预测视觉活动必须处于[0,1]")
        super().__init__((512, 512), normalized)


def encode_positive_float32(values: Iterable[float], expected_count: int | None = None) -> bytes:
    normalized = _positive_values(values, expected_count)
    if not normalized:
        return b""
    return struct.pack(f"!{len(normalized)}f", *normalized)


def decode_positive_float32(payload: bytes, expected_count: int | None = None) -> tuple[float, ...]:
    if len(payload) % 4:
        raise ProtocolError("活动字节长度必须是4的倍数")
    count = len(payload) // 4
    if expected_count is not None and count != expected_count:
        raise ProtocolError(f"活动数量必须为{expected_count}")
    if count == 0:
        return ()
    values = struct.unpack(f"!{count}f", payload)
    try:
        return _positive_values(values, expected_count)
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc

