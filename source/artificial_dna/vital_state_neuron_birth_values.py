"""生命状态本能内部固定神经元的已确认出生值。

这些神经元的阈值由进入它们的连续活动直接相加关系计算。为了让保存的
阈值仍具有该确定含义，其出生活动响应强度必须为1。器官固定接收端、普通
入口和逐项独立继续神经元已经由统一入口出生值覆盖，本文件只输出本能
片段自身的160个固定神经元，避免重复写入同一地址。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from .birth_structure import FixedBirthFragment, NeuronNature
from .brain_geometry import linear_index
from .fixed_receiver_birth_values import NEURON_BIRTH_VALUE_DTYPE


CONFIRMED_VITAL_STATE_INTERNAL_NEURON_BIRTH_VALUE_COUNT = 160


@dataclass(frozen=True, slots=True)
class VitalStateNeuronBirthValueBatch:
    values: numpy.ndarray

    def __post_init__(self) -> None:
        if self.values.dtype != NEURON_BIRTH_VALUE_DTYPE:
            raise ValueError("生命状态本能神经元出生值记录格式不正确")
        if self.values.ndim != 1:
            raise ValueError("生命状态本能神经元出生值必须是一维排列")

    @property
    def count(self) -> int:
        return int(self.values.size)


def build_vital_state_internal_neuron_birth_values(
    fragment: FixedBirthFragment,
) -> VitalStateNeuronBirthValueBatch:
    """从已经确认的本能片段提取其自身固定神经元出生值。"""

    entrance_neurons = {
        name
        for binding in fragment.visual_bindings
        for name in (
            binding.receiver_neuron,
            binding.ordinary_neuron,
            binding.continuation_neuron,
        )
    }
    internal = tuple(
        neuron for neuron in fragment.neurons if neuron.name not in entrance_neurons
    )
    if len(internal) != CONFIRMED_VITAL_STATE_INTERNAL_NEURON_BIRTH_VALUE_COUNT:
        raise ValueError("生命状态本能内部固定神经元数量与出生总账不一致")
    if any(neuron.nature is not NeuronNature.FIXED for neuron in internal):
        raise ValueError("生命状态本能内部通道混入了普通神经元")
    if any(float(neuron.response_gain) != 1.0 for neuron in internal):
        raise ValueError("生命状态本能阈值不是按原活动直接相加关系形成")

    values = numpy.empty(len(internal), dtype=NEURON_BIRTH_VALUE_DTYPE)
    values["neuron"] = numpy.asarray(
        [
            linear_index(neuron.address.x, neuron.address.y, neuron.address.z)
            for neuron in internal
        ],
        dtype="<u4",
    )
    values["response_gain"] = numpy.asarray(
        [neuron.response_gain for neuron in internal],
        dtype="<f4",
    )
    values["threshold"] = numpy.asarray(
        [neuron.threshold for neuron in internal],
        dtype="<f4",
    )
    if numpy.unique(values["neuron"]).size != values.size:
        raise ValueError("生命状态本能内部神经元地址发生重复")
    return VitalStateNeuronBirthValueBatch(values)
