"""十段视觉还原组织逐神经元的出生响应值。

预测视觉器官要求每项输出处于[0,1]，输出阶段又不能截断或重新归一化。
因此本组织直接依据冻结统一公式确定每个位置的安全响应强度：

    g_i = 1 / (d_i^固定 + d_i^可形成相邻)

``d_i^固定`` 是人工出生结构中最多能够到达该位置的固定路径数，包括为
第二来源预留的路径；``d_i^可形成相邻`` 是同一视觉还原组织中最多能够
形成并到达该位置的直接相邻可变路径数。若所有来源活动不超过1且所有
Path强度不超过1，则直接相加后乘以该固定响应强度仍不超过1。这个关系
不按当前实际到达数动态平均，也不在器官处截断。

阈值为0，使任意正的连续还原活动都能继续传播。十段内部固定路径只负责
把一个位置的活动完整送到其出生指定位置，出生强度为1；Path不放大。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy

from .brain_address_plan import VISUAL_RECONSTRUCTION_COUNT
from .brain_dna_layout import visual_reconstruction_address
from .brain_geometry import linear_index
from .fixed_path_topology import VISUAL_RECONSTRUCTION_SPACINGS
from .fixed_receiver_birth_values import NEURON_BIRTH_VALUE_DTYPE


VISUAL_RECONSTRUCTION_INTERNAL_PATH_STRENGTH = 1.0
VISUAL_RECONSTRUCTION_THRESHOLD = 0.0
CONFIRMED_VISUAL_RECONSTRUCTION_NEURON_BIRTH_VALUE_COUNT = (
    VISUAL_RECONSTRUCTION_COUNT
)


@dataclass(frozen=True, slots=True)
class VisualReconstructionBirthValueBatch:
    section: int
    position_offset: int
    values: numpy.ndarray

    def __post_init__(self) -> None:
        if not 0 <= int(self.section) < 10:
            raise ValueError("视觉还原出生值段编号必须处于0到9")
        if int(self.position_offset) < 0:
            raise ValueError("视觉还原出生值位置起点不能小于零")
        if self.values.dtype != NEURON_BIRTH_VALUE_DTYPE:
            raise ValueError("视觉还原神经元出生值记录格式不正确")
        if self.values.ndim != 1:
            raise ValueError("视觉还原神经元出生值必须是一维排列")

    @property
    def count(self) -> int:
        return int(self.values.size)


def _axis_neighbor_counts(axis: numpy.ndarray) -> numpy.ndarray:
    return numpy.where((axis == 0) | (axis == 511), 2, 3).astype("u1")


def visual_reconstruction_possible_inbound_counts(
    section: int,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """给出该段每个位置的固定、可形成相邻和合计最大汇入数量。"""

    value = int(section)
    if not 0 <= value < 10:
        raise ValueError("视觉还原段编号必须处于0到9")
    axis = numpy.arange(512, dtype="<u2")
    x = axis[None, :]
    y = axis[:, None]
    x_neighbors = _axis_neighbor_counts(x)
    y_neighbors = _axis_neighbor_counts(y)
    z_neighbors = 2 if value in (0, 9) else 3
    local = (
        x_neighbors.astype("<u2")
        * y_neighbors.astype("<u2")
        * numpy.uint16(z_neighbors)
        - numpy.uint16(1)
    ).astype("u1")

    if value < 9:
        distance = 1 << value
        internal = (1 + (x >= distance)) * (1 + (y >= distance))
    else:
        internal = numpy.zeros((512, 512), dtype="u1")
    spacing = VISUAL_RECONSTRUCTION_SPACINGS[value]
    two_sources = (
        ((x % spacing) == 0) & ((y % spacing) == 0)
    ).astype("u1") * numpy.uint8(2)
    fixed = internal.astype("u1", copy=False) + two_sources
    total = fixed.astype("<u2") + local.astype("<u2")
    if numpy.any(total == 0):  # pragma: no cover - 每个位置至少有组织内邻接
        raise RuntimeError("视觉还原神经元没有任何可能汇入关系")
    return fixed, local, total


def iter_visual_reconstruction_birth_value_batches(
    *,
    batch_size: int = 262_144,
) -> Iterator[VisualReconstructionBirthValueBatch]:
    """逐段、逐地址给出全部十段视觉还原神经元出生值。"""

    size = int(batch_size)
    if size <= 0:
        raise ValueError("视觉还原出生值批次大小必须大于零")
    produced = 0
    for section in range(10):
        _fixed, _local, total = visual_reconstruction_possible_inbound_counts(
            section
        )
        flat_total = total.reshape(-1)
        for start in range(0, 512 * 512, size):
            stop = min(512 * 512, start + size)
            positions = numpy.arange(start, stop, dtype="<u4")
            y, x = numpy.divmod(positions, numpy.uint32(512))
            base = visual_reconstruction_address(section, 0, 0)
            base_index = linear_index(base.x, base.y, base.z)
            neurons = (
                numpy.uint32(base_index)
                + y * numpy.uint32(800)
                + x
            ).astype("<u4", copy=False)
            values = numpy.empty(neurons.size, dtype=NEURON_BIRTH_VALUE_DTYPE)
            values["neuron"] = neurons
            values["response_gain"] = (
                1.0 / flat_total[start:stop].astype(numpy.float32)
            )
            values["threshold"] = VISUAL_RECONSTRUCTION_THRESHOLD
            yield VisualReconstructionBirthValueBatch(section, start, values)
            produced += int(values.size)
    if produced != CONFIRMED_VISUAL_RECONSTRUCTION_NEURON_BIRTH_VALUE_COUNT:
        raise RuntimeError("视觉还原出生值没有覆盖完整十段组织")


def validate_visual_reconstruction_birth_values() -> None:
    if CONFIRMED_VISUAL_RECONSTRUCTION_NEURON_BIRTH_VALUE_COUNT != 2_621_440:
        raise RuntimeError("视觉还原神经元出生值数量不再是十个512×512")
    if VISUAL_RECONSTRUCTION_INTERNAL_PATH_STRENGTH != 1.0:
        raise RuntimeError("视觉还原内部固定路径不再完整传播位置活动")
    if VISUAL_RECONSTRUCTION_THRESHOLD != 0.0:
        raise RuntimeError("视觉还原神经元不再传播任意正连续活动")
    minimum_total = 2**16
    maximum_total = 0
    for section in range(10):
        _fixed, _local, total = visual_reconstruction_possible_inbound_counts(
            section
        )
        minimum_total = min(minimum_total, int(total.min()))
        maximum_total = max(maximum_total, int(total.max()))
    if minimum_total != 7 or maximum_total != 32:
        raise RuntimeError("视觉还原逐位置最大可能汇入数量发生变化")


validate_visual_reconstruction_birth_values()
