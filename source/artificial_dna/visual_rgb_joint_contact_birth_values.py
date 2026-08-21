"""共同结晶组织中真实RGB逐项接触位置的神经元出生值。

每一项真实RGB活动都由视觉第二支路通过一条强度1的固定
路径到达唯一同形位置。这些位置是共同结晶普通组织的一部分，
所以同一物理平面内的直接相邻普通神经元之间仍保留从零强度
形成可变路径的空间。

响应强度使用“一条已定固定汇入＋本组织最多直接相邻汇入”
的倒数，阈值为零。这不添加裁切或选择机制，只把当前已明确
连接空间内可能同时到达的普通活动放回[0,1]容纳范围。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy

from .brain_address_plan import (
    VISUAL_RGB_JOINT_CONTACT_COUNT,
    VISUAL_RGB_JOINT_CONTACT_START_Z,
)
from .brain_geometry import TISSUE_PLANE
from .fixed_receiver_birth_values import NEURON_BIRTH_VALUE_DTYPE


VISUAL_RGB_JOINT_CONTACT_THRESHOLD = 0.0
CONFIRMED_VISUAL_RGB_JOINT_CONTACT_NEURON_BIRTH_VALUE_COUNT = (
    VISUAL_RGB_JOINT_CONTACT_COUNT
)


@dataclass(frozen=True, slots=True)
class VisualRgbJointContactBirthValueBatch:
    activity_offset: int
    values: numpy.ndarray

    @property
    def count(self) -> int:
        return int(self.values.size)


def _addresses_and_local_inbound(
    activities: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    pixels, channels = numpy.divmod(activities, numpy.uint32(3))
    y, x = numpy.divmod(pixels, numpy.uint32(1280))
    tiles, local_pixel_x = numpy.divmod(x, numpy.uint32(256))
    local_x = local_pixel_x * numpy.uint32(3) + channels
    addresses = (
        (numpy.uint32(VISUAL_RGB_JOINT_CONTACT_START_Z) + tiles * numpy.uint32(2))
        * numpy.uint32(TISSUE_PLANE)
        + y * numpy.uint32(800)
        + local_x
    ).astype("<u4", copy=False)

    x_span = numpy.where(
        (local_x == 0) | (local_x == 767),
        numpy.uint8(2),
        numpy.uint8(3),
    )
    y_span = numpy.where(
        (y == 0) | (y == 656),
        numpy.uint8(2),
        numpy.uint8(3),
    )
    local_inbound = (x_span * y_span - numpy.uint8(1)).astype("u1", copy=False)
    return addresses, local_inbound


def iter_visual_rgb_joint_contact_birth_value_batches(
    *,
    batch_size: int = 262_144,
) -> Iterator[VisualRgbJointContactBirthValueBatch]:
    size = int(batch_size)
    if size <= 0:
        raise ValueError("真实RGB接触组织出生值批次大小必须大于零")
    produced = 0
    for start in range(0, VISUAL_RGB_JOINT_CONTACT_COUNT, size):
        stop = min(VISUAL_RGB_JOINT_CONTACT_COUNT, start + size)
        activities = numpy.arange(start, stop, dtype="<u4")
        addresses, local_inbound = _addresses_and_local_inbound(activities)
        values = numpy.empty(addresses.size, dtype=NEURON_BIRTH_VALUE_DTYPE)
        values["neuron"] = addresses
        total_inbound = local_inbound.astype("<u2") + numpy.uint16(1)
        values["response_gain"] = 1.0 / total_inbound.astype(numpy.float32)
        values["threshold"] = VISUAL_RGB_JOINT_CONTACT_THRESHOLD
        yield VisualRgbJointContactBirthValueBatch(start, values)
        produced += int(values.size)
    if produced != CONFIRMED_VISUAL_RGB_JOINT_CONTACT_NEURON_BIRTH_VALUE_COUNT:
        raise RuntimeError("真实RGB逐项接触组织出生值没有完整覆盖")


def validate_visual_rgb_joint_contact_birth_values() -> None:
    probe = numpy.asarray(
        [0, 1, (656 * 1280 + 1279) * 3 + 2],
        dtype="<u4",
    )
    addresses, local = _addresses_and_local_inbound(probe)
    if numpy.unique(addresses).size != probe.size:
        raise RuntimeError("真实RGB接触地址不再逐项独立")
    if int(local.min()) < 3 or int(local.max()) > 8:
        raise RuntimeError("真实RGB接触组织直接相邻汇入范围不正确")
    if VISUAL_RGB_JOINT_CONTACT_THRESHOLD != 0.0:
        raise RuntimeError("真实RGB接触组织不再保留全部连续正活动")


validate_visual_rgb_joint_contact_birth_values()
