"""已分配普通组织内部的直接相邻可变路径形成许可。

三维硬盘排列会让若干不同组织在物理边界上碰巧直接相邻。人工出生结构
不能把这种工程接触自动解释成跨组织传播，因此当前只允许同一已分配普通
组织内部的26方向零强度路径后天形成。不同组织之间即使物理相邻也保持
形成许可关闭；以后需要跨组织传播时，必须由明确的固定路径或另行确认的
出生许可实现。

许可掩码只是人工出生结构，不进入 Brain 的 Signal，也不改变统一公式。
所有被允许的位置第一次出生时路径强度仍为零。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy

from .brain_geometry import (
    DIRECT_NEIGHBOR_OFFSETS,
    TISSUE_DEPTH,
    TISSUE_HEIGHT,
    TISSUE_PLANE,
    TISSUE_WIDTH,
)
from .ordinary_local_path_space import (
    LOCAL_PATH_CHUNK_DEPTH,
    LOCAL_PATH_MASK_DTYPE,
    _BYTE_BIT_COUNT,
    _normal_pair_slice,
)
from .ordinary_organization_contacts import OrdinaryOrganizationContacts


CONFIRMED_INTERNAL_FORMATION_PERMISSION_COUNT = 360_488_844
CONFIRMED_CROSS_ORGANIZATION_HELD_ZERO_COUNT = 41_272_236
CONFIRMED_CLASSIFIED_ORDINARY_NEIGHBOR_COUNT = (
    CONFIRMED_INTERNAL_FORMATION_PERMISSION_COUNT
    + CONFIRMED_CROSS_ORGANIZATION_HELD_ZERO_COUNT
)


@dataclass(frozen=True, slots=True)
class OrdinaryLocalFormationPermissionBatch:
    start_z: int
    stop_z: int
    masks: numpy.ndarray

    def __post_init__(self) -> None:
        if not 0 <= self.start_z < self.stop_z <= TISSUE_DEPTH:
            raise ValueError("可变路径形成许可块深度超出完整组织")
        expected = (self.stop_z - self.start_z, TISSUE_HEIGHT, TISSUE_WIDTH)
        if self.masks.shape != expected or self.masks.dtype != LOCAL_PATH_MASK_DTYPE:
            raise ValueError("可变路径形成许可掩码形状或类型不正确")

    @property
    def start_index(self) -> int:
        return self.start_z * TISSUE_PLANE

    @property
    def stop_index(self) -> int:
        return self.stop_z * TISSUE_PLANE

    @property
    def directed_permission_count(self) -> int:
        return int(_BYTE_BIT_COUNT[self.masks.view("u1")].sum(dtype=numpy.uint64))


class OrdinaryLocalFormationSpace:
    """按组织身份和26方向展开当前人工出生许可。"""

    def __init__(self, contacts: OrdinaryOrganizationContacts) -> None:
        self.contacts = contacts

    @staticmethod
    def _make_batch(
        organizations: numpy.ndarray,
        start_z: int,
        stop_z: int,
    ) -> OrdinaryLocalFormationPermissionBatch:
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
            source = organizations[
                source_z_start:source_z_stop,
                source_y,
                source_x,
            ]
            target = organizations[
                source_z_start + dz : source_z_stop + dz,
                target_y,
                target_x,
            ]
            permitted = (source != 0) & (source == target)
            masks[
                source_z_start - start_z : source_z_stop - start_z,
                source_y,
                source_x,
            ] |= numpy.where(
                permitted,
                numpy.uint32(1 << direction),
                numpy.uint32(0),
            )
        return OrdinaryLocalFormationPermissionBatch(start_z, stop_z, masks)

    def iter_batches(
        self,
        *,
        chunk_depth: int = LOCAL_PATH_CHUNK_DEPTH,
    ) -> Iterator[OrdinaryLocalFormationPermissionBatch]:
        depth = int(chunk_depth)
        if depth <= 0:
            raise ValueError("可变路径形成许可块深度必须大于零")
        organizations = self.contacts.build_organization_map()
        for start_z in range(0, TISSUE_DEPTH, depth):
            yield self._make_batch(
                organizations,
                start_z,
                min(start_z + depth, TISSUE_DEPTH),
            )

    def directed_permission_count(self) -> int:
        return sum(
            batch.directed_permission_count
            for batch in self.iter_batches()
        )


def validate_formation_permission_ledger(
    space: OrdinaryLocalFormationSpace,
) -> int:
    count = space.directed_permission_count()
    if count != CONFIRMED_INTERNAL_FORMATION_PERMISSION_COUNT:
        raise RuntimeError("组织内部可变路径形成许可数量与人工出生总账不一致")
    return count


def validate_formation_permission_constants() -> None:
    if CONFIRMED_CLASSIFIED_ORDINARY_NEIGHBOR_COUNT != 401_761_080:
        raise RuntimeError("已分类普通相邻关系没有覆盖全部当前物理候选")


validate_formation_permission_constants()
