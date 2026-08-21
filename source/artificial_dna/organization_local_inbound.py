"""按真实三维组织掩码计算每个位置最多具有的相邻可变路径汇入数。"""

from __future__ import annotations

import numpy

from .brain_geometry import DIRECT_NEIGHBOR_OFFSETS


def _normal_slice(length: int, delta: int) -> tuple[slice, slice]:
    if delta < 0:
        return slice(-delta, length), slice(0, length + delta)
    if delta > 0:
        return slice(0, length - delta), slice(delta, length)
    return slice(0, length), slice(0, length)


def organization_local_inbound_counts(mask: numpy.ndarray) -> numpy.ndarray:
    """给出掩码内每个位置能够从同一组织直接到达的相邻位置数。"""

    organization = numpy.asarray(mask, dtype=numpy.bool_)
    if organization.ndim != 3:
        raise ValueError("普通组织掩码必须按深度、高度、宽度三维排列")
    counts = numpy.zeros(organization.shape, dtype="u1")
    depth, height, width = organization.shape
    for dx, dy, dz in DIRECT_NEIGHBOR_OFFSETS:
        source_z, target_z = _normal_slice(depth, dz)
        source_y, target_y = _normal_slice(height, dy)
        source_x, target_x = _normal_slice(width, dx)
        source_exists = organization[source_z, source_y, source_x]
        target_exists = organization[target_z, target_y, target_x]
        counts[source_z, source_y, source_x] += (
            source_exists & target_exists
        ).astype("u1")
    counts[~organization] = 0
    return counts
