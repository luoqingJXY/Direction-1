"""全部客观器官固定接收神经元的已确定出生值。

本文件只机械展开冻结理论已经唯一确定的入口关系。批次编号和神经元地址
只用于把人工出生结构写入硬盘，不进入 Brain 的 Signal。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy

from .birth_value_constraints import derive_exact_contact_neuron_values
from .organ_entrances import SECOND_EXPERIMENT_ENTRANCES, OrganEntranceLayout


FIXED_RECEIVER_RESPONSE_GAIN, FIXED_RECEIVER_THRESHOLD = (
    derive_exact_contact_neuron_values(1.0)
)
CONFIRMED_FIXED_RECEIVER_BIRTH_VALUE_COUNT = (
    SECOND_EXPERIMENT_ENTRANCES.activity_count
)

NEURON_BIRTH_VALUE_DTYPE = numpy.dtype(
    [
        ("neuron", "<u4"),
        ("response_gain", "<f4"),
        ("threshold", "<f4"),
    ]
)
# 保留原名称，避免已经存在的出生值读取器失去兼容性。
FIXED_RECEIVER_BIRTH_VALUE_DTYPE = NEURON_BIRTH_VALUE_DTYPE


@dataclass(frozen=True, slots=True)
class FixedReceiverBirthValueBatch:
    entrance: str
    activity_offset: int
    values: numpy.ndarray

    def __post_init__(self) -> None:
        if not self.entrance:
            raise ValueError("固定接收神经元出生值批次必须指明器官入口")
        if int(self.activity_offset) < 0:
            raise ValueError("器官活动批次起点不能小于零")
        if self.values.dtype != FIXED_RECEIVER_BIRTH_VALUE_DTYPE:
            raise ValueError("固定接收神经元出生值记录格式不正确")
        if self.values.ndim != 1:
            raise ValueError("固定接收神经元出生值必须是一维排列")

    @property
    def count(self) -> int:
        return int(self.values.size)


def iter_fixed_receiver_birth_value_batches(
    *,
    entrances: OrganEntranceLayout = SECOND_EXPERIMENT_ENTRANCES,
    batch_size: int = 262_144,
) -> Iterator[FixedReceiverBirthValueBatch]:
    """逐器官、逐活动生成，不一次在内存中容纳三百多万条记录。"""

    size = int(batch_size)
    if size <= 0:
        raise ValueError("固定接收神经元出生值批次大小必须大于零")
    produced = 0
    for group in entrances.ranges:
        for start in range(0, group.activity_count, size):
            stop = min(group.activity_count, start + size)
            activities = numpy.arange(start, stop, dtype="<u4")
            values = numpy.empty(activities.size, dtype=FIXED_RECEIVER_BIRTH_VALUE_DTYPE)
            values["neuron"] = entrances.receiver_indices(group.name, activities)
            values["response_gain"] = FIXED_RECEIVER_RESPONSE_GAIN
            values["threshold"] = FIXED_RECEIVER_THRESHOLD
            yield FixedReceiverBirthValueBatch(group.name, start, values)
            produced += int(values.size)
    if produced != entrances.activity_count:
        raise RuntimeError("固定接收神经元出生值没有覆盖全部器官活动")


def validate_fixed_receiver_birth_values() -> None:
    if FIXED_RECEIVER_RESPONSE_GAIN != 1.0:
        raise RuntimeError("器官固定接收神经元不再完整响应客观活动")
    if FIXED_RECEIVER_THRESHOLD != 0.0:
        raise RuntimeError("器官固定接收神经元不再传播全部正活动")
    if CONFIRMED_FIXED_RECEIVER_BIRTH_VALUE_COUNT != 3_055_995:
        raise RuntimeError("器官固定接收神经元出生值数量与入口总数不一致")


validate_fixed_receiver_birth_values()
