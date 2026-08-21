"""真实视觉第一支路进入灰度还原第一来源组织的神经元出生值。

完整1280×657×RGB视觉活动已经逐项进入Brain并分成两条保真支路。
本文件只落实其中通往视觉还原的一条支路：人工出生连接把完整视野固定
对应到512×512位置，同一位置的R、G、B三项活动直接相加到同一神经元。
这些位置属于普通组织，周围零强度相邻路径以后可能形成；因此出生响应
强度必须同时为已经存在的固定汇入和该位置最多可形成的相邻汇入留出容纳：

    g_i = 1 / (d_i^固定 + d_i^可形成相邻)

这不是器官预处理、识别或内容比较；另一条同形支路仍保留全部RGB活动。
基础灰度位置向其余九个尺度来源位置一对一复制时采用同一条边界。这个
数值不是运行时求平均；它是人工DNA为每个具体位置保存的出生属性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy

from .brain_address_plan import VISUAL_SOURCE_START_Z
from .brain_geometry import TISSUE_HEIGHT, TISSUE_PLANE, TISSUE_WIDTH
from .fixed_receiver_birth_values import NEURON_BIRTH_VALUE_DTYPE
from .fixed_path_topology import VISUAL_SOURCE_PER_SIDE
from .organization_local_inbound import organization_local_inbound_counts


VISUAL_GRAYSCALE_BASE_COUNT = 512 * 512
VISUAL_GRAYSCALE_THRESHOLD = 0.0
VISUAL_MULTISCALE_COPY_THRESHOLD = 0.0

VISUAL_GRAYSCALE_RGB_PATH_COUNT = VISUAL_GRAYSCALE_BASE_COUNT * 3
VISUAL_GRAYSCALE_MULTISCALE_COPY_PATH_COUNT = (
    VISUAL_SOURCE_PER_SIDE - VISUAL_GRAYSCALE_BASE_COUNT
)
VISUAL_FIRST_SOURCE_TO_RECONSTRUCTION_PATH_COUNT = VISUAL_SOURCE_PER_SIDE
CONFIRMED_VISUAL_FIRST_SOURCE_CHANNEL_PATH_COUNT = (
    VISUAL_GRAYSCALE_RGB_PATH_COUNT
    + VISUAL_GRAYSCALE_MULTISCALE_COPY_PATH_COUNT
    + VISUAL_FIRST_SOURCE_TO_RECONSTRUCTION_PATH_COUNT
)
CONFIRMED_VISUAL_FIRST_SOURCE_NEURON_BIRTH_VALUE_COUNT = VISUAL_SOURCE_PER_SIDE


def visual_source_local_inbound_counts() -> numpy.ndarray:
    """视觉双来源组织每一侧对应位置的最大相邻汇入数。"""

    mask = numpy.zeros((2, TISSUE_HEIGHT, TISSUE_WIDTH), dtype=numpy.bool_)
    mask.reshape(2, TISSUE_PLANE)[:, :VISUAL_SOURCE_PER_SIDE] = True
    counts = organization_local_inbound_counts(mask)
    first = counts[0].reshape(-1)[:VISUAL_SOURCE_PER_SIDE]
    second = counts[1].reshape(-1)[:VISUAL_SOURCE_PER_SIDE]
    if not numpy.array_equal(first, second):  # pragma: no cover
        raise RuntimeError("视觉双来源两侧的物理相邻关系不一致")
    return first


@dataclass(frozen=True, slots=True)
class VisualGrayscaleAdmissionBirthValueBatch:
    source_offset: int
    values: numpy.ndarray

    def __post_init__(self) -> None:
        if int(self.source_offset) < 0:
            raise ValueError("视觉第一来源出生值起点不能小于零")
        if self.values.dtype != NEURON_BIRTH_VALUE_DTYPE:
            raise ValueError("视觉第一来源神经元出生值记录格式不正确")
        if self.values.ndim != 1:
            raise ValueError("视觉第一来源神经元出生值必须是一维排列")

    @property
    def count(self) -> int:
        return int(self.values.size)


def iter_visual_grayscale_admission_birth_value_batches(
    *,
    batch_size: int = 262_144,
) -> Iterator[VisualGrayscaleAdmissionBirthValueBatch]:
    """流式给出视觉第一来源全部611668个神经元的出生值。"""

    size = int(batch_size)
    if size <= 0:
        raise ValueError("视觉第一来源出生值批次大小必须大于零")
    produced = 0
    source_start = VISUAL_SOURCE_START_Z * TISSUE_PLANE
    local_inbound = visual_source_local_inbound_counts()
    for start in range(0, VISUAL_SOURCE_PER_SIDE, size):
        stop = min(VISUAL_SOURCE_PER_SIDE, start + size)
        offsets = numpy.arange(start, stop, dtype="<u4")
        values = numpy.empty(offsets.size, dtype=NEURON_BIRTH_VALUE_DTYPE)
        values["neuron"] = numpy.uint32(source_start) + offsets

        base_mask = offsets < numpy.uint32(VISUAL_GRAYSCALE_BASE_COUNT)
        fixed_inbound = numpy.ones(offsets.size, dtype="u1")
        values["threshold"] = VISUAL_MULTISCALE_COPY_THRESHOLD
        fixed_inbound[base_mask] = 3
        total_inbound = (
            fixed_inbound.astype("<u2")
            + local_inbound[start:stop].astype("<u2")
        )
        values["response_gain"] = 1.0 / total_inbound.astype(numpy.float32)
        values["threshold"][base_mask] = VISUAL_GRAYSCALE_THRESHOLD

        yield VisualGrayscaleAdmissionBirthValueBatch(start, values)
        produced += int(values.size)
    if produced != CONFIRMED_VISUAL_FIRST_SOURCE_NEURON_BIRTH_VALUE_COUNT:
        raise RuntimeError("视觉第一来源出生值没有覆盖完整来源组织")


def validate_visual_grayscale_admission_birth_values() -> None:
    if VISUAL_GRAYSCALE_BASE_COUNT != 262_144:
        raise RuntimeError("视觉基础灰度位置不再是512×512")
    if VISUAL_GRAYSCALE_RGB_PATH_COUNT != 786_432:
        raise RuntimeError("每个视觉灰度位置不再逐项接收R、G、B三项活动")
    if VISUAL_GRAYSCALE_MULTISCALE_COPY_PATH_COUNT != 349_524:
        raise RuntimeError("视觉灰度基础位置到其余尺度的复制数量不一致")
    if CONFIRMED_VISUAL_FIRST_SOURCE_CHANNEL_PATH_COUNT != 1_747_624:
        raise RuntimeError("视觉第一来源完整固定通道数量不一致")
    if VISUAL_GRAYSCALE_THRESHOLD != 0.0:
        raise RuntimeError("视觉灰度接入不再传播全部正活动")
    if VISUAL_MULTISCALE_COPY_THRESHOLD != 0.0:
        raise RuntimeError("视觉多尺度来源复制不再传播全部正活动")
    local = visual_source_local_inbound_counts()
    if int(local.min()) != 7 or int(local.max()) != 17:
        raise RuntimeError("视觉双来源组织的相邻可形成汇入范围发生变化")


validate_visual_grayscale_admission_birth_values()
