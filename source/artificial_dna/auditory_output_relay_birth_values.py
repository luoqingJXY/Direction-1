"""听觉还原末段到预测听觉输出神经元的透明出生值。

最后一段听觉还原活动与预测听觉器官控制入口的形状完全相同。专用输出
神经元只把每项还原活动交给随后的一对一器官固定路径，不再次压缩、汇合
或改变强度。因此该段固定路径强度为1，目标神经元响应强度为1、阈值为0。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy

from .brain_address_plan import PREDICTED_AUDITORY_OUTPUT_Z
from .brain_geometry import TISSUE_PLANE
from .birth_value_constraints import derive_exact_contact_neuron_values
from .fixed_receiver_birth_values import NEURON_BIRTH_VALUE_DTYPE
from .organ_entrances import PREDICTED_AUDITORY_ACTIVITY_COUNT


AUDITORY_OUTPUT_RELAY_PATH_STRENGTH = 1.0
AUDITORY_OUTPUT_RELAY_RESPONSE_GAIN, AUDITORY_OUTPUT_RELAY_THRESHOLD = (
    derive_exact_contact_neuron_values(AUDITORY_OUTPUT_RELAY_PATH_STRENGTH)
)
CONFIRMED_AUDITORY_OUTPUT_RELAY_COUNT = PREDICTED_AUDITORY_ACTIVITY_COUNT


@dataclass(frozen=True, slots=True)
class AuditoryOutputRelayBirthValueBatch:
    activity_offset: int
    values: numpy.ndarray

    def __post_init__(self) -> None:
        if int(self.activity_offset) < 0:
            raise ValueError("预测听觉输出活动起点不能小于零")
        if self.values.dtype != NEURON_BIRTH_VALUE_DTYPE:
            raise ValueError("预测听觉输出神经元出生值记录格式不正确")
        if self.values.ndim != 1:
            raise ValueError("预测听觉输出神经元出生值必须是一维排列")

    @property
    def count(self) -> int:
        return int(self.values.size)


def _output_neuron_indices(activities: numpy.ndarray) -> numpy.ndarray:
    streams, remainder = numpy.divmod(
        activities,
        numpy.uint32(342 * 85 * 3),
    )
    frequencies, remainder = numpy.divmod(remainder, numpy.uint32(85 * 3))
    sequences, components = numpy.divmod(remainder, numpy.uint32(3))
    tiles, local_frequency = numpy.divmod(frequencies, numpy.uint32(256))
    x = local_frequency * numpy.uint32(3) + components
    y = (streams * numpy.uint32(2) + tiles) * numpy.uint32(85) + sequences
    return (
        numpy.uint32(PREDICTED_AUDITORY_OUTPUT_Z * TISSUE_PLANE)
        + y * numpy.uint32(800)
        + x
    ).astype("<u4", copy=False)


def iter_auditory_output_relay_birth_value_batches(
    *,
    batch_size: int = 262_144,
) -> Iterator[AuditoryOutputRelayBirthValueBatch]:
    size = int(batch_size)
    if size <= 0:
        raise ValueError("预测听觉输出出生值批次大小必须大于零")
    produced = 0
    for start in range(0, CONFIRMED_AUDITORY_OUTPUT_RELAY_COUNT, size):
        stop = min(CONFIRMED_AUDITORY_OUTPUT_RELAY_COUNT, start + size)
        activities = numpy.arange(start, stop, dtype="<u4")
        neurons = _output_neuron_indices(activities)
        values = numpy.empty(neurons.size, dtype=NEURON_BIRTH_VALUE_DTYPE)
        values["neuron"] = neurons
        values["response_gain"] = AUDITORY_OUTPUT_RELAY_RESPONSE_GAIN
        values["threshold"] = AUDITORY_OUTPUT_RELAY_THRESHOLD
        yield AuditoryOutputRelayBirthValueBatch(start, values)
        produced += int(neurons.size)
    if produced != CONFIRMED_AUDITORY_OUTPUT_RELAY_COUNT:
        raise RuntimeError("预测听觉输出神经元出生值没有覆盖完整器官形状")


def validate_auditory_output_relay_birth_values() -> None:
    if AUDITORY_OUTPUT_RELAY_RESPONSE_GAIN != 1.0:
        raise RuntimeError("预测听觉输出神经元不再原强度响应")
    if AUDITORY_OUTPUT_RELAY_THRESHOLD != 0.0:
        raise RuntimeError("预测听觉输出神经元不再传播全部正活动")
    if CONFIRMED_AUDITORY_OUTPUT_RELAY_COUNT != 261_630:
        raise RuntimeError("预测听觉输出神经元数量与器官控制入口不一致")


validate_auditory_output_relay_birth_values()
