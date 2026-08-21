"""真实视觉和真实听觉两条逐项分支的透明神经元出生值。

每项已经独立继续的真实感受活动在这里分成两路：一路继续进入同模态
还原通道，一路继续进入跨组织通道。分支只复制传播关系，不改变活动
强度、不比较内容，也不提取特征。因此两条固定路径强度均为1，两端
分支神经元的活动响应强度为1、阈值为0。

分支编号和批次偏移只用于人工出生结构展开，不进入 Brain 的 Signal。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

import numpy

from .brain_address_plan import (
    AUDITORY_FIRST_BRANCH_START_Z,
    AUDITORY_SECOND_BRANCH_START_Z,
    VISUAL_FIRST_BRANCH_START_Z,
    VISUAL_SECOND_BRANCH_START_Z,
)
from .brain_geometry import TISSUE_PLANE
from .birth_value_constraints import derive_exact_contact_neuron_values
from .fixed_receiver_birth_values import NEURON_BIRTH_VALUE_DTYPE
from .organ_entrances import AUDITORY_ACTIVITY_COUNT, VISUAL_ACTIVITY_COUNT


IDENTITY_BRANCH_PATH_STRENGTH = 1.0
IDENTITY_BRANCH_RESPONSE_GAIN, IDENTITY_BRANCH_THRESHOLD = (
    derive_exact_contact_neuron_values(IDENTITY_BRANCH_PATH_STRENGTH)
)
CONFIRMED_SENSORY_IDENTITY_BRANCH_PATH_COUNT = 2 * (
    VISUAL_ACTIVITY_COUNT + AUDITORY_ACTIVITY_COUNT
)
CONFIRMED_SENSORY_IDENTITY_BRANCH_NEURON_BIRTH_VALUE_COUNT = (
    CONFIRMED_SENSORY_IDENTITY_BRANCH_PATH_COUNT
)


class SensoryIdentityBranch(IntEnum):
    VISUAL_RECONSTRUCTION = 0
    VISUAL_CROSS_ORGANIZATION = 1
    AUDITORY_RECONSTRUCTION = 2
    AUDITORY_CROSS_ORGANIZATION = 3


@dataclass(frozen=True, slots=True)
class SensoryIdentityBranchBirthValueBatch:
    branch: SensoryIdentityBranch
    activity_offset: int
    values: numpy.ndarray

    def __post_init__(self) -> None:
        if int(self.activity_offset) < 0:
            raise ValueError("真实感受分支活动起点不能小于零")
        if self.values.dtype != NEURON_BIRTH_VALUE_DTYPE:
            raise ValueError("真实感受分支神经元出生值记录格式不正确")
        if self.values.ndim != 1:
            raise ValueError("真实感受分支神经元出生值必须是一维排列")

    @property
    def count(self) -> int:
        return int(self.values.size)


def _visual_branch_indices(activities: numpy.ndarray, start_z: int) -> numpy.ndarray:
    pixels, channels = numpy.divmod(activities, numpy.uint32(3))
    y, x = numpy.divmod(pixels, numpy.uint32(1280))
    tiles, local_x = numpy.divmod(x, numpy.uint32(256))
    return (
        (numpy.uint32(start_z) + tiles * numpy.uint32(2))
        * numpy.uint32(TISSUE_PLANE)
        + y * numpy.uint32(800)
        + local_x * numpy.uint32(3)
        + channels
    ).astype("<u4", copy=False)


def _auditory_branch_indices(activities: numpy.ndarray, start_z: int) -> numpy.ndarray:
    streams, remainder = numpy.divmod(activities, numpy.uint32(1025 * 3))
    frequencies, components = numpy.divmod(remainder, numpy.uint32(3))
    tiles, local_frequency = numpy.divmod(frequencies, numpy.uint32(256))
    return (
        numpy.uint32(start_z * TISSUE_PLANE)
        + (streams * numpy.uint32(6) + tiles) * numpy.uint32(800)
        + local_frequency * numpy.uint32(3)
        + components
    ).astype("<u4", copy=False)


def _birth_values(neurons: numpy.ndarray) -> numpy.ndarray:
    values = numpy.empty(neurons.size, dtype=NEURON_BIRTH_VALUE_DTYPE)
    values["neuron"] = neurons
    values["response_gain"] = IDENTITY_BRANCH_RESPONSE_GAIN
    values["threshold"] = IDENTITY_BRANCH_THRESHOLD
    return values


def iter_sensory_identity_branch_birth_value_batches(
    *,
    batch_size: int = 262_144,
) -> Iterator[SensoryIdentityBranchBirthValueBatch]:
    """逐项展开四个同形分支神经元，不一次占用全部内存。"""

    size = int(batch_size)
    if size <= 0:
        raise ValueError("真实感受分支出生值批次大小必须大于零")
    definitions = (
        (
            SensoryIdentityBranch.VISUAL_RECONSTRUCTION,
            VISUAL_ACTIVITY_COUNT,
            VISUAL_FIRST_BRANCH_START_Z,
            _visual_branch_indices,
        ),
        (
            SensoryIdentityBranch.VISUAL_CROSS_ORGANIZATION,
            VISUAL_ACTIVITY_COUNT,
            VISUAL_SECOND_BRANCH_START_Z,
            _visual_branch_indices,
        ),
        (
            SensoryIdentityBranch.AUDITORY_RECONSTRUCTION,
            AUDITORY_ACTIVITY_COUNT,
            AUDITORY_FIRST_BRANCH_START_Z,
            _auditory_branch_indices,
        ),
        (
            SensoryIdentityBranch.AUDITORY_CROSS_ORGANIZATION,
            AUDITORY_ACTIVITY_COUNT,
            AUDITORY_SECOND_BRANCH_START_Z,
            _auditory_branch_indices,
        ),
    )
    produced = 0
    for branch, count, start_z, mapper in definitions:
        for start in range(0, count, size):
            stop = min(count, start + size)
            activities = numpy.arange(start, stop, dtype="<u4")
            neurons = mapper(activities, start_z)
            yield SensoryIdentityBranchBirthValueBatch(
                branch,
                start,
                _birth_values(neurons),
            )
            produced += int(neurons.size)
    if produced != CONFIRMED_SENSORY_IDENTITY_BRANCH_NEURON_BIRTH_VALUE_COUNT:
        raise RuntimeError("真实感受分支神经元出生值没有完整覆盖两条逐项分支")


def validate_sensory_identity_branch_birth_values() -> None:
    if IDENTITY_BRANCH_RESPONSE_GAIN != 1.0:
        raise RuntimeError("真实感受分支不再原强度响应")
    if IDENTITY_BRANCH_THRESHOLD != 0.0:
        raise RuntimeError("真实感受分支不再传播全部正活动")
    if CONFIRMED_SENSORY_IDENTITY_BRANCH_PATH_COUNT != 5_064_210:
        raise RuntimeError("真实视觉和听觉双分支数量与器官活动总数不一致")


validate_sensory_identity_branch_birth_values()
