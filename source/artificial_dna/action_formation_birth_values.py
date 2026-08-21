"""三段动作形成组织逐神经元出生响应值。

三段只是已确认微观通道的三节，不是三个独立能力模块。第一段
逐项接收共同结晶位置；第二段逐项接收视觉和听觉还原位置；
第三段的1224028个视觉和听觉位置接收前两段直接相加，
末端580个动作返回位置只接收第一段。每段内部仍由普通神经元
之间的直接相邻可变路径形成后天干涉；三段之间不因硬盘上物理
相邻就自动形成路径，只经过已明确的固定路径组装。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy

from .brain_address_plan import (
    ACTION_FORMATION_SECTION_COUNT,
    ACTION_FORMATION_START_Z,
)
from .brain_dna_layout import SENSORY_ACTION_FORMATION_COUNT
from .brain_geometry import TISSUE_HEIGHT, TISSUE_PLANE, TISSUE_WIDTH
from .fixed_receiver_birth_values import NEURON_BIRTH_VALUE_DTYPE
from .organization_local_inbound import organization_local_inbound_counts


ACTION_FORMATION_THRESHOLD = 0.0
CONFIRMED_ACTION_FORMATION_NEURON_BIRTH_VALUE_COUNT = (
    3 * ACTION_FORMATION_SECTION_COUNT
)


@dataclass(frozen=True, slots=True)
class ActionFormationBirthValueBatch:
    section: int
    position_offset: int
    values: numpy.ndarray

    @property
    def count(self) -> int:
        return int(self.values.size)


def action_formation_local_inbound_counts() -> numpy.ndarray:
    """一个独立动作形成段内每个位置的最多相邻汇入数。"""

    mask = numpy.zeros((2, TISSUE_HEIGHT, TISSUE_WIDTH), dtype=numpy.bool_)
    mask.reshape(-1)[:ACTION_FORMATION_SECTION_COUNT] = True
    counts = organization_local_inbound_counts(mask)
    return counts.reshape(-1)[:ACTION_FORMATION_SECTION_COUNT]


def action_formation_fixed_inbound_counts(section: int) -> numpy.ndarray:
    part = int(section)
    if part not in (0, 1, 2):
        raise ValueError("动作形成组织段只能是0、1或2")
    if part == 0:
        return numpy.ones(ACTION_FORMATION_SECTION_COUNT, dtype="u1")
    if part == 1:
        counts = numpy.zeros(ACTION_FORMATION_SECTION_COUNT, dtype="u1")
        counts[:SENSORY_ACTION_FORMATION_COUNT] = 1
        return counts
    counts = numpy.ones(ACTION_FORMATION_SECTION_COUNT, dtype="u1")
    counts[:SENSORY_ACTION_FORMATION_COUNT] = 2
    return counts


def iter_action_formation_birth_value_batches(
    *,
    batch_size: int = 262_144,
) -> Iterator[ActionFormationBirthValueBatch]:
    size = int(batch_size)
    if size <= 0:
        raise ValueError("动作形成出生值批次大小必须大于零")
    local = action_formation_local_inbound_counts()
    produced = 0
    for section in range(3):
        fixed = action_formation_fixed_inbound_counts(section)
        base = (ACTION_FORMATION_START_Z + section * 2) * TISSUE_PLANE
        for start in range(0, ACTION_FORMATION_SECTION_COUNT, size):
            stop = min(ACTION_FORMATION_SECTION_COUNT, start + size)
            offsets = numpy.arange(start, stop, dtype="<u4")
            values = numpy.empty(offsets.size, dtype=NEURON_BIRTH_VALUE_DTYPE)
            values["neuron"] = numpy.uint32(base) + offsets
            total = (
                local[start:stop].astype("<u2")
                + fixed[start:stop].astype("<u2")
            )
            if numpy.any(total == 0):
                raise RuntimeError("动作形成神经元没有任何已定或相邻汇入")
            values["response_gain"] = 1.0 / total.astype(numpy.float32)
            if section in (0, 2):
                action_positions = offsets >= SENSORY_ACTION_FORMATION_COUNT
                values["response_gain"][action_positions] = 1.0
            values["threshold"] = ACTION_FORMATION_THRESHOLD
            yield ActionFormationBirthValueBatch(section, start, values)
            produced += int(values.size)
    if produced != CONFIRMED_ACTION_FORMATION_NEURON_BIRTH_VALUE_COUNT:
        raise RuntimeError("动作形成出生值没有覆盖完整三段组织")


def validate_action_formation_birth_values() -> None:
    if CONFIRMED_ACTION_FORMATION_NEURON_BIRTH_VALUE_COUNT != 3_673_824:
        raise RuntimeError("三段动作形成神经元数量发生变化")
    local = action_formation_local_inbound_counts()
    if (int(local.min()), int(local.max())) != (3, 17):
        raise RuntimeError("动作形成组织直接相邻汇入范围不正确")
    if int(numpy.count_nonzero(action_formation_fixed_inbound_counts(1) == 0)) != 580:
        raise RuntimeError("第二段不应为580个动作返回位置伪造感受来源")
    if ACTION_FORMATION_THRESHOLD != 0.0:
        raise RuntimeError("动作形成组织不再保留全部连续正活动")


validate_action_formation_birth_values()
