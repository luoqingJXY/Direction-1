"""共同结晶视觉位置到视觉第二还原来源的出生神经元数值。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy

from .brain_address_plan import VISUAL_SOURCE_START_Z
from .brain_geometry import TISSUE_PLANE
from .fixed_path_topology import VISUAL_SOURCE_PER_SIDE
from .fixed_receiver_birth_values import NEURON_BIRTH_VALUE_DTYPE
from .visual_grayscale_admission_birth_values import (
    visual_source_local_inbound_counts,
)


VISUAL_SECOND_SOURCE_THRESHOLD = 0.0
CONFIRMED_VISUAL_SECOND_SOURCE_NEURON_BIRTH_VALUE_COUNT = VISUAL_SOURCE_PER_SIDE


@dataclass(frozen=True, slots=True)
class VisualSecondSourceBirthValueBatch:
    source_offset: int
    values: numpy.ndarray

    @property
    def count(self) -> int:
        return int(self.values.size)


def iter_visual_second_source_birth_value_batches(
    *, batch_size: int = 262_144
) -> Iterator[VisualSecondSourceBirthValueBatch]:
    """逐地址生成第二来源；一条固定汇入加全部可能相邻汇入共同定界。"""

    size = int(batch_size)
    if size <= 0:
        raise ValueError("视觉第二来源出生值批次大小必须大于零")
    start_neuron = (VISUAL_SOURCE_START_Z + 1) * TISSUE_PLANE
    local_inbound = visual_source_local_inbound_counts()
    produced = 0
    for start in range(0, VISUAL_SOURCE_PER_SIDE, size):
        stop = min(VISUAL_SOURCE_PER_SIDE, start + size)
        offsets = numpy.arange(start, stop, dtype="<u4")
        values = numpy.empty(offsets.size, dtype=NEURON_BIRTH_VALUE_DTYPE)
        values["neuron"] = numpy.uint32(start_neuron) + offsets
        total_inbound = (
            numpy.uint16(1)
            + local_inbound[start:stop].astype("<u2")
        )
        values["response_gain"] = 1.0 / total_inbound.astype(numpy.float32)
        values["threshold"] = VISUAL_SECOND_SOURCE_THRESHOLD
        yield VisualSecondSourceBirthValueBatch(start, values)
        produced += int(values.size)
    if produced != CONFIRMED_VISUAL_SECOND_SOURCE_NEURON_BIRTH_VALUE_COUNT:
        raise RuntimeError("视觉第二来源出生值没有覆盖全部结晶返回位置")


def validate_visual_second_source_birth_values() -> None:
    if CONFIRMED_VISUAL_SECOND_SOURCE_NEURON_BIRTH_VALUE_COUNT != 611_668:
        raise RuntimeError("视觉第二来源位置数量发生变化")
    if VISUAL_SECOND_SOURCE_THRESHOLD != 0.0:
        raise RuntimeError("视觉第二来源不再传播任意正连续活动")


validate_visual_second_source_birth_values()
