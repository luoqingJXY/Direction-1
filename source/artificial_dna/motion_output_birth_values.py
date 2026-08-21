"""116个鼠标、键盘和视野中心输出神经元的出生响应值。"""

from __future__ import annotations

import numpy

from .brain_address_plan import MOTION_OUTPUT_COUNT
from .brain_dna_layout import motion_output_address
from .brain_geometry import linear_index
from .fixed_receiver_birth_values import NEURON_BIRTH_VALUE_DTYPE


MOTION_OUTPUT_FIXED_INBOUND_COUNT = 5
MOTION_OUTPUT_THRESHOLD = 0.0
CONFIRMED_MOTION_OUTPUT_NEURON_BIRTH_VALUE_COUNT = MOTION_OUTPUT_COUNT


def build_motion_output_neuron_birth_values() -> numpy.ndarray:
    values = numpy.empty(MOTION_OUTPUT_COUNT, dtype=NEURON_BIRTH_VALUE_DTYPE)
    local_inbound = numpy.full(MOTION_OUTPUT_COUNT, 2, dtype="u1")
    local_inbound[0] = 1
    local_inbound[-1] = 1
    for action in range(MOTION_OUTPUT_COUNT):
        address = motion_output_address(action)
        values["neuron"][action] = linear_index(address.x, address.y, address.z)
    values["response_gain"] = 1.0 / (
        local_inbound.astype(numpy.float32) + MOTION_OUTPUT_FIXED_INBOUND_COUNT
    )
    values["threshold"] = MOTION_OUTPUT_THRESHOLD
    return values


def validate_motion_output_neuron_birth_values() -> None:
    values = build_motion_output_neuron_birth_values()
    if values.size != 116:
        raise RuntimeError("动作输出神经元出生值数量不正确")
    if numpy.unique(values["neuron"]).size != values.size:
        raise RuntimeError("动作输出神经元地址不再逐项唯一")
    if MOTION_OUTPUT_THRESHOLD != 0.0:
        raise RuntimeError("动作输出神经元不再保留连续动作倾向")


validate_motion_output_neuron_birth_values()
