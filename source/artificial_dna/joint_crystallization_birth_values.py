"""共同结晶组织逐神经元出生数值。

共同结晶不是单独能力模块。该组织只是让预测视觉、预测听觉、动作回流和
真实视觉灰度来源在同一片普通组织内活动，并允许后天相邻路径形成。所有
到达仍按统一公式直接相加，不比较真实与预测，也不生成正确或错误。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy

from .brain_address_plan import (
    JOINT_CRYSTALLIZATION_COUNT,
    JOINT_CRYSTALLIZATION_START_Z,
)
from .brain_dna_layout import (
    ACTION_JOINT_COUNT,
    SENSORY_ACTION_FORMATION_COUNT,
    VISUAL_JOINT_COUNT,
)
from .brain_geometry import TISSUE_HEIGHT, TISSUE_PLANE, TISSUE_WIDTH
from .fixed_receiver_birth_values import NEURON_BIRTH_VALUE_DTYPE
from .organization_local_inbound import organization_local_inbound_counts


JOINT_CRYSTALLIZATION_THRESHOLD = 0.0
CONFIRMED_JOINT_CRYSTALLIZATION_NEURON_BIRTH_VALUE_COUNT = (
    JOINT_CRYSTALLIZATION_COUNT
)


@dataclass(frozen=True, slots=True)
class JointCrystallizationBirthValueBatch:
    position_offset: int
    values: numpy.ndarray

    @property
    def count(self) -> int:
        return int(self.values.size)


def joint_crystallization_local_inbound_counts() -> numpy.ndarray:
    mask = numpy.zeros((2, TISSUE_HEIGHT, TISSUE_WIDTH), dtype=numpy.bool_)
    mask.reshape(-1)[:JOINT_CRYSTALLIZATION_COUNT] = True
    counts = organization_local_inbound_counts(mask)
    return counts.reshape(-1)[:JOINT_CRYSTALLIZATION_COUNT]


def joint_crystallization_fixed_inbound_counts() -> numpy.ndarray:
    """视觉位置五路汇入；其余既有位置和真实听觉接触各一路汇入。"""

    counts = numpy.ones(JOINT_CRYSTALLIZATION_COUNT, dtype="u1")
    counts[: 2 * VISUAL_JOINT_COUNT : 2] = 5
    return counts


def iter_joint_crystallization_birth_value_batches(
    *, batch_size: int = 262_144
) -> Iterator[JointCrystallizationBirthValueBatch]:
    size = int(batch_size)
    if size <= 0:
        raise ValueError("共同结晶出生值批次大小必须大于零")
    local_inbound = joint_crystallization_local_inbound_counts()
    fixed_inbound = joint_crystallization_fixed_inbound_counts()
    base = JOINT_CRYSTALLIZATION_START_Z * TISSUE_PLANE
    produced = 0
    for start in range(0, JOINT_CRYSTALLIZATION_COUNT, size):
        stop = min(JOINT_CRYSTALLIZATION_COUNT, start + size)
        offsets = numpy.arange(start, stop, dtype="<u4")
        values = numpy.empty(offsets.size, dtype=NEURON_BIRTH_VALUE_DTYPE)
        values["neuron"] = numpy.uint32(base) + offsets
        total_inbound = (
            fixed_inbound[start:stop].astype("<u2")
            + local_inbound[start:stop].astype("<u2")
        )
        values["response_gain"] = 1.0 / total_inbound.astype(numpy.float32)
        action_stop = SENSORY_ACTION_FORMATION_COUNT + ACTION_JOINT_COUNT
        action_positions = (
            (offsets >= SENSORY_ACTION_FORMATION_COUNT)
            & (offsets < action_stop)
        )
        values["response_gain"][action_positions] = 1.0
        values["threshold"] = JOINT_CRYSTALLIZATION_THRESHOLD
        yield JointCrystallizationBirthValueBatch(start, values)
        produced += int(values.size)
    if produced != CONFIRMED_JOINT_CRYSTALLIZATION_NEURON_BIRTH_VALUE_COUNT:
        raise RuntimeError("共同结晶出生值没有覆盖视觉、听觉与动作接触位置")


def validate_joint_crystallization_birth_values() -> None:
    if CONFIRMED_JOINT_CRYSTALLIZATION_NEURON_BIRTH_VALUE_COUNT != 1_233_833:
        raise RuntimeError("共同结晶神经元数量发生变化")
    local = joint_crystallization_local_inbound_counts()
    if int(local.min()) != 3 or int(local.max()) != 17:
        raise RuntimeError("共同结晶组织的相邻可形成汇入范围发生变化")
    fixed = joint_crystallization_fixed_inbound_counts()
    if int(numpy.count_nonzero(fixed == 5)) != VISUAL_JOINT_COUNT:
        raise RuntimeError("真实灰度、真实RGB与预测视觉没有逐项到达同一结晶位置")
    if JOINT_CRYSTALLIZATION_THRESHOLD != 0.0:
        raise RuntimeError("共同结晶组织不再保留全部连续正活动")


validate_joint_crystallization_birth_values()
