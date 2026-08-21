"""器官普通入口与逐项独立继续位置的透明出生值。

这一段只保留每项器官活动原有的连续强度和内部差异。路径不能放大，
而这里也不是后续用于恢复活动的放大神经元段，因此两处神经元均以
响应强度1、阈值0运行，中间固定路径以强度1完整传播。

批次名称、器官名称和地址只用于人工出生结构写入与核对，不进入Brain
的Signal。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

import numpy

from .brain_address_plan import ORGAN_CONTINUATION_Z_OFFSET
from .brain_geometry import TISSUE_PLANE
from .birth_value_constraints import derive_exact_contact_neuron_values
from .fixed_receiver_birth_values import NEURON_BIRTH_VALUE_DTYPE
from .organ_entrances import SECOND_EXPERIMENT_ENTRANCES, OrganEntranceLayout


TRANSPARENT_ORGAN_RESPONSE_GAIN, TRANSPARENT_ORGAN_THRESHOLD = (
    derive_exact_contact_neuron_values(1.0)
)
CONFIRMED_ORDINARY_ENTRANCE_BIRTH_VALUE_COUNT = (
    SECOND_EXPERIMENT_ENTRANCES.activity_count
)
CONFIRMED_INDEPENDENT_CONTINUATION_BIRTH_VALUE_COUNT = (
    SECOND_EXPERIMENT_ENTRANCES.activity_count
)
CONFIRMED_TRANSPARENT_ORGAN_NEURON_BIRTH_VALUE_COUNT = (
    CONFIRMED_ORDINARY_ENTRANCE_BIRTH_VALUE_COUNT
    + CONFIRMED_INDEPENDENT_CONTINUATION_BIRTH_VALUE_COUNT
)


class TransparentOrganStage(IntEnum):
    """只用于出生文件核对的两处实际神经元位置。"""

    ORDINARY_ENTRANCE = 0
    INDEPENDENT_CONTINUATION = 1


@dataclass(frozen=True, slots=True)
class TransparentOrganBirthValueBatch:
    stage: TransparentOrganStage
    entrance: str
    activity_offset: int
    values: numpy.ndarray

    def __post_init__(self) -> None:
        if not self.entrance:
            raise ValueError("透明器官通道出生值批次必须指明器官入口")
        if int(self.activity_offset) < 0:
            raise ValueError("器官活动批次起点不能小于零")
        if self.values.dtype != NEURON_BIRTH_VALUE_DTYPE:
            raise ValueError("透明器官通道神经元出生值记录格式不正确")
        if self.values.ndim != 1:
            raise ValueError("透明器官通道神经元出生值必须是一维排列")

    @property
    def count(self) -> int:
        return int(self.values.size)


def _birth_values(neurons: numpy.ndarray) -> numpy.ndarray:
    values = numpy.empty(neurons.size, dtype=NEURON_BIRTH_VALUE_DTYPE)
    values["neuron"] = neurons
    values["response_gain"] = TRANSPARENT_ORGAN_RESPONSE_GAIN
    values["threshold"] = TRANSPARENT_ORGAN_THRESHOLD
    return values


def iter_transparent_organ_birth_value_batches(
    *,
    entrances: OrganEntranceLayout = SECOND_EXPERIMENT_ENTRANCES,
    batch_size: int = 262_144,
) -> Iterator[TransparentOrganBirthValueBatch]:
    """逐器官展开普通入口和独立继续位置，不一次占用全部内存。"""

    size = int(batch_size)
    if size <= 0:
        raise ValueError("透明器官通道出生值批次大小必须大于零")
    produced = {stage: 0 for stage in TransparentOrganStage}
    continuation_offset = numpy.uint32(
        ORGAN_CONTINUATION_Z_OFFSET * TISSUE_PLANE
    )
    for group in entrances.ranges:
        for start in range(0, group.activity_count, size):
            stop = min(group.activity_count, start + size)
            activities = numpy.arange(start, stop, dtype="<u4")
            ordinary = entrances.ordinary_indices(group.name, activities)
            continuation = (ordinary + continuation_offset).astype(
                numpy.uint32,
                copy=False,
            )
            for stage, neurons in (
                (TransparentOrganStage.ORDINARY_ENTRANCE, ordinary),
                (TransparentOrganStage.INDEPENDENT_CONTINUATION, continuation),
            ):
                values = _birth_values(neurons)
                yield TransparentOrganBirthValueBatch(
                    stage,
                    group.name,
                    start,
                    values,
                )
                produced[stage] += int(values.size)
    expected = entrances.activity_count
    if any(count != expected for count in produced.values()):
        raise RuntimeError("透明器官通道出生值没有覆盖全部器官活动")


def validate_transparent_organ_birth_values() -> None:
    if TRANSPARENT_ORGAN_RESPONSE_GAIN != 1.0:
        raise RuntimeError("透明器官通道神经元不再原强度响应")
    if TRANSPARENT_ORGAN_THRESHOLD != 0.0:
        raise RuntimeError("透明器官通道神经元不再传播全部正活动")
    if CONFIRMED_TRANSPARENT_ORGAN_NEURON_BIRTH_VALUE_COUNT != 6_111_990:
        raise RuntimeError("透明器官通道神经元出生值数量与入口总数不一致")


validate_transparent_organ_birth_values()
