"""已确定普通神经元组织中的直接相邻可塑路径物理候选。

冻结理论中的全部26方向底层路径槽位仍由P、Q硬盘数组保留。本文件并不
删除固定神经元或未分配位置周围的底层槽位，也不填写任何路径强度；它只
按当前人工出生结构标出“两端均已经确定为普通神经元”的直接相邻方向。
这部分只表示本个体当前已确定组织中的真实普通—普通物理邻接候选。人工
出生结构如何按区域让这些候选路径形成或保持为零，冻结理论仍明确为待补充，
因此本文件不擅自把候选关系写成“允许”或“禁止”。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy

from .brain_geometry import (
    DIRECT_NEIGHBOR_OFFSETS,
    TISSUE_DEPTH,
    TISSUE_HEIGHT,
    TISSUE_NEURON_COUNT,
    TISSUE_PLANE,
    TISSUE_WIDTH,
)
from .neuron_nature_topology import NeuronNatureTopology, StoredNeuronNature


LOCAL_PATH_MASK_DTYPE = numpy.dtype("<u4")
LOCAL_PATH_CHUNK_DEPTH = 32
# 由当前已确认物理排列导出；测试会重新遍历掩码，防止地址布局变化后总账失效。
CURRENT_ORDINARY_NEIGHBOR_CANDIDATE_COUNT = 401_761_080
_BYTE_BIT_COUNT = numpy.asarray(
    [value.bit_count() for value in range(256)],
    dtype="u1",
)


@dataclass(frozen=True, slots=True)
class OrdinaryLocalPathMaskBatch:
    """连续深度块中每个来源位置通往普通邻居的26位物理候选掩码。"""

    start_z: int
    stop_z: int
    masks: numpy.ndarray

    def __post_init__(self) -> None:
        if not 0 <= self.start_z < self.stop_z <= TISSUE_DEPTH:
            raise ValueError("普通邻接路径块深度超出完整组织")
        expected = (self.stop_z - self.start_z, TISSUE_HEIGHT, TISSUE_WIDTH)
        if self.masks.shape != expected or self.masks.dtype != LOCAL_PATH_MASK_DTYPE:
            raise ValueError("普通邻接路径掩码形状或类型不正确")

    @property
    def start_index(self) -> int:
        return self.start_z * TISSUE_PLANE

    @property
    def stop_index(self) -> int:
        return self.stop_z * TISSUE_PLANE

    @property
    def directed_path_count(self) -> int:
        return int(_BYTE_BIT_COUNT[self.masks.view("u1")].sum(dtype=numpy.uint64))


def _normal_pair_slice(length: int, delta: int) -> tuple[slice, slice]:
    """同一方向上来源与到达位置均在范围内的两个切片。"""

    if delta >= 0:
        return slice(0, length - delta), slice(delta, length)
    return slice(-delta, length), slice(0, length + delta)


class OrdinaryLocalPathSpace:
    """按26方向生成当前已确定普通组织的物理邻接候选。"""

    def __init__(self, nature_topology: NeuronNatureTopology) -> None:
        self.nature_topology = nature_topology

    def build_known_nature_map(self) -> numpy.ndarray:
        """仅在临时内存建立已确定性质；0仍表示未分配，而非一种神经元。"""

        values = numpy.zeros(TISSUE_NEURON_COUNT, dtype="u1")
        self.nature_topology.apply_to(values)
        return values.reshape(TISSUE_DEPTH, TISSUE_HEIGHT, TISSUE_WIDTH)

    @staticmethod
    def _make_batch(
        ordinary: numpy.ndarray,
        start_z: int,
        stop_z: int,
    ) -> OrdinaryLocalPathMaskBatch:
        masks = numpy.zeros(
            (stop_z - start_z, TISSUE_HEIGHT, TISSUE_WIDTH),
            dtype=LOCAL_PATH_MASK_DTYPE,
        )
        for direction, (dx, dy, dz) in enumerate(DIRECT_NEIGHBOR_OFFSETS):
            source_z_start = max(start_z, -dz)
            source_z_stop = min(stop_z, TISSUE_DEPTH - dz)
            if source_z_start >= source_z_stop:
                continue
            source_y, target_y = _normal_pair_slice(TISSUE_HEIGHT, dy)
            source_x, target_x = _normal_pair_slice(TISSUE_WIDTH, dx)
            source = ordinary[
                source_z_start:source_z_stop,
                source_y,
                source_x,
            ]
            target = ordinary[
                source_z_start + dz : source_z_stop + dz,
                target_y,
                target_x,
            ]
            linked = source & target
            masks[
                source_z_start - start_z : source_z_stop - start_z,
                source_y,
                source_x,
            ] |= numpy.where(linked, numpy.uint32(1 << direction), numpy.uint32(0))
        return OrdinaryLocalPathMaskBatch(start_z, stop_z, masks)

    def iter_batches(
        self,
        *,
        chunk_depth: int = LOCAL_PATH_CHUNK_DEPTH,
    ) -> Iterator[OrdinaryLocalPathMaskBatch]:
        """按块产生掩码，不保存全部409.6MB掩码结果。"""

        depth = int(chunk_depth)
        if depth <= 0:
            raise ValueError("普通邻接路径块深度必须大于零")
        known = self.build_known_nature_map()
        ordinary = known == int(StoredNeuronNature.ORDINARY)
        for start_z in range(0, TISSUE_DEPTH, depth):
            yield self._make_batch(ordinary, start_z, min(start_z + depth, TISSUE_DEPTH))

    def directed_path_count(self) -> int:
        """逐块统计当前已确定普通组织的有方向物理邻接候选。"""

        return sum(batch.directed_path_count for batch in self.iter_batches())


def validate_current_candidate_count(space: OrdinaryLocalPathSpace) -> int:
    count = space.directed_path_count()
    if count != CURRENT_ORDINARY_NEIGHBOR_CANDIDATE_COUNT:
        raise RuntimeError("当前普通神经元直接相邻候选数量与出生结构总账不一致")
    return count
