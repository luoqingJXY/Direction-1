"""视觉和听觉还原组织每个位置的固定路径汇入数量总账。

该总账只展开当前人工出生结构草案已经存在的端点关系，回答每个神经元
会有多少条普通活动直接相加。它不选择路径强度、响应强度或阈值，也不把
汇入数量当作 Brain 的 Signal。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from .fixed_path_topology import (
    AUDITORY_RECONSTRUCTION_SPACINGS,
    FIXED_PATH_FAMILY_COUNTS,
    VISUAL_RECONSTRUCTION_SPACINGS,
    FixedPathFamily,
)


@dataclass(frozen=True, slots=True)
class ReconstructionSectionInboundLedger:
    modality: str
    section: int
    neuron_count: int
    internal_path_count: int
    source_path_count: int
    no_fixed_inbound_count: int
    minimum_inbound_count: int
    maximum_inbound_count: int
    degree_histogram: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.modality not in {"visual", "auditory"}:
            raise ValueError("还原组织总账只允许视觉或听觉")
        if not 0 <= int(self.section) < 10:
            raise ValueError("还原组织段编号必须处于0到9")
        if int(self.neuron_count) <= 0:
            raise ValueError("还原组织神经元数量必须大于零")
        if sum(count for _, count in self.degree_histogram) != self.neuron_count:
            raise ValueError("还原组织汇入度分布没有覆盖全部神经元")
        if sum(degree * count for degree, count in self.degree_histogram) != (
            self.internal_path_count + self.source_path_count
        ):
            raise ValueError("还原组织汇入度分布与固定路径数量不一致")


@dataclass(frozen=True, slots=True)
class VisualReconstructionReachabilityLedger:
    """视觉来源出生时可到达的位置，以及留给后天活动的其余位置。"""

    section: int
    direct_source_position_count: int
    birth_reachable_neuron_count: int
    latent_neuron_count: int
    outgoing_birth_reachable_path_count: int
    outgoing_latent_path_count: int

    def __post_init__(self) -> None:
        if not 0 <= int(self.section) < 10:
            raise ValueError("视觉还原可到达总账段编号必须处于0到9")
        if self.birth_reachable_neuron_count + self.latent_neuron_count != 512 * 512:
            raise ValueError("视觉还原可到达位置与潜在位置没有覆盖完整一段")
        if self.direct_source_position_count > self.birth_reachable_neuron_count:
            raise ValueError("视觉直接来源位置不能多于出生可到达位置")


def _histogram(values: numpy.ndarray) -> tuple[tuple[int, int], ...]:
    counts = numpy.bincount(values.reshape(-1), minlength=int(values.max()) + 1)
    return tuple(
        (degree, int(count))
        for degree, count in enumerate(counts)
        if count
    )


def build_visual_reconstruction_inbound_ledgers(
) -> tuple[ReconstructionSectionInboundLedger, ...]:
    ledgers: list[ReconstructionSectionInboundLedger] = []
    x = numpy.arange(512, dtype="<u2")[None, :]
    y = numpy.arange(512, dtype="<u2")[:, None]
    internal_total = 0
    source_total = 0
    for section, spacing in enumerate(VISUAL_RECONSTRUCTION_SPACINGS):
        if section < 9:
            distance = 1 << section
            internal = (1 + (x >= distance)) * (1 + (y >= distance))
        else:
            internal = numpy.zeros((512, 512), dtype="u1")
        selected = ((x % spacing) == 0) & ((y % spacing) == 0)
        source = selected.astype("u1") * numpy.uint8(2)
        degree = internal.astype("u1", copy=False) + source
        internal_count = int(internal.sum(dtype=numpy.uint64))
        source_count = int(source.sum(dtype=numpy.uint64))
        histogram = _histogram(degree)
        ledgers.append(
            ReconstructionSectionInboundLedger(
                "visual",
                section,
                512 * 512,
                internal_count,
                source_count,
                int(numpy.count_nonzero(degree == 0)),
                int(degree.min()),
                int(degree.max()),
                histogram,
            )
        )
        internal_total += internal_count
        source_total += source_count
    if internal_total != FIXED_PATH_FAMILY_COUNTS[FixedPathFamily.VISUAL_RECONSTRUCTION]:
        raise RuntimeError("视觉还原内部汇入数量与固定路径总账不一致")
    if source_total != FIXED_PATH_FAMILY_COUNTS[FixedPathFamily.VISUAL_SOURCES]:
        raise RuntimeError("视觉还原来源汇入数量与固定路径总账不一致")
    return tuple(ledgers)


def _visual_internal_outgoing_count(mask: numpy.ndarray, distance: int) -> int:
    y, x = numpy.nonzero(mask)
    return sum(
        int(numpy.count_nonzero((x + dx < 512) & (y + dy < 512)))
        for dx, dy in (
            (0, 0),
            (distance, 0),
            (0, distance),
            (distance, distance),
        )
    )


def build_visual_reconstruction_reachability_ledgers(
) -> tuple[VisualReconstructionReachabilityLedger, ...]:
    """按固定来源和逐节端点推出出生可到达骨架，不模拟生命运行。"""

    axis = numpy.arange(512, dtype="<u2")
    x = axis[None, :]
    y = axis[:, None]
    direct_masks = [
        ((x % spacing) == 0) & ((y % spacing) == 0)
        for spacing in VISUAL_RECONSTRUCTION_SPACINGS
    ]
    reachable_masks: list[numpy.ndarray] = [
        numpy.zeros((512, 512), dtype=numpy.bool_) for _ in range(10)
    ]
    reachable_masks[9] = direct_masks[9].copy()
    for section in range(8, -1, -1):
        reachable = direct_masks[section].copy()
        distance = 1 << section
        source_y, source_x = numpy.nonzero(reachable_masks[section + 1])
        for dx, dy in (
            (0, 0),
            (distance, 0),
            (0, distance),
            (distance, distance),
        ):
            target_x = source_x + dx
            target_y = source_y + dy
            valid = (target_x < 512) & (target_y < 512)
            reachable[target_y[valid], target_x[valid]] = True
        reachable_masks[section] = reachable

    ledgers: list[VisualReconstructionReachabilityLedger] = []
    all_positions = numpy.ones((512, 512), dtype=numpy.bool_)
    reachable_path_total = 0
    latent_path_total = 0
    for section in range(10):
        reachable = reachable_masks[section]
        if section == 0:
            reachable_paths = 0
            latent_paths = 0
        else:
            distance = 1 << (section - 1)
            reachable_paths = _visual_internal_outgoing_count(
                reachable,
                distance,
            )
            latent_paths = _visual_internal_outgoing_count(
                all_positions & ~reachable,
                distance,
            )
        ledgers.append(
            VisualReconstructionReachabilityLedger(
                section,
                int(numpy.count_nonzero(direct_masks[section])),
                int(numpy.count_nonzero(reachable)),
                int(numpy.count_nonzero(~reachable)),
                reachable_paths,
                latent_paths,
            )
        )
        reachable_path_total += reachable_paths
        latent_path_total += latent_paths
    expected = FIXED_PATH_FAMILY_COUNTS[FixedPathFamily.VISUAL_RECONSTRUCTION]
    if reachable_path_total + latent_path_total != expected:
        raise RuntimeError("视觉出生可到达和潜在展开路径没有覆盖十段内部端点")
    return tuple(ledgers)


def build_auditory_reconstruction_inbound_ledgers(
) -> tuple[ReconstructionSectionInboundLedger, ...]:
    ledgers: list[ReconstructionSectionInboundLedger] = []
    frequency = numpy.arange(342, dtype="<u2")[None, :, None]
    sequence = numpy.arange(85, dtype="<u2")[None, None, :]
    internal_degree = (
        numpy.ones((1, 342, 85), dtype="u1")
        + (frequency > 0)
        + (frequency < 341)
        + (sequence > 0)
        + (sequence < 84)
    ).astype("u1", copy=False)
    internal_total = 0
    source_total = 0
    for section, spacing in enumerate(AUDITORY_RECONSTRUCTION_SPACINGS):
        if section == 0:
            one_plane_internal = numpy.zeros((1, 342, 85), dtype="u1")
        else:
            one_plane_internal = internal_degree
        selected = ((frequency % spacing) == 0) & ((sequence % spacing) == 0)
        one_plane_source = selected.astype("u1") * numpy.uint8(2)
        one_plane_degree = one_plane_internal + one_plane_source
        # 三条声音流、每个频率位置的三项非负活动具有相同汇入排列。
        multiplicity = 3 * 3
        histogram = tuple(
            (degree, count * multiplicity)
            for degree, count in _histogram(one_plane_degree)
        )
        internal_count = int(
            one_plane_internal.sum(dtype=numpy.uint64)
        ) * multiplicity
        source_count = int(
            one_plane_source.sum(dtype=numpy.uint64)
        ) * multiplicity
        ledgers.append(
            ReconstructionSectionInboundLedger(
                "auditory",
                section,
                3 * 342 * 85 * 3,
                internal_count,
                source_count,
                next((count for degree, count in histogram if degree == 0), 0),
                histogram[0][0],
                histogram[-1][0],
                histogram,
            )
        )
        internal_total += internal_count
        source_total += source_count
    if internal_total != FIXED_PATH_FAMILY_COUNTS[FixedPathFamily.AUDITORY_RECONSTRUCTION]:
        raise RuntimeError("听觉还原内部汇入数量与固定路径总账不一致")
    if source_total != FIXED_PATH_FAMILY_COUNTS[FixedPathFamily.AUDITORY_SOURCES]:
        raise RuntimeError("听觉还原来源汇入数量与固定路径总账不一致")
    return tuple(ledgers)


VISUAL_RECONSTRUCTION_INBOUND_LEDGERS = build_visual_reconstruction_inbound_ledgers()
AUDITORY_RECONSTRUCTION_INBOUND_LEDGERS = build_auditory_reconstruction_inbound_ledgers()
VISUAL_RECONSTRUCTION_REACHABILITY_LEDGERS = (
    build_visual_reconstruction_reachability_ledgers()
)
VISUAL_BIRTH_REACHABLE_INTERNAL_PATH_COUNT = sum(
    value.outgoing_birth_reachable_path_count
    for value in VISUAL_RECONSTRUCTION_REACHABILITY_LEDGERS
)
VISUAL_LATENT_INTERNAL_PATH_COUNT = sum(
    value.outgoing_latent_path_count
    for value in VISUAL_RECONSTRUCTION_REACHABILITY_LEDGERS
)

if VISUAL_BIRTH_REACHABLE_INTERNAL_PATH_COUNT != 1_394_017:
    raise RuntimeError("视觉出生可到达还原骨架路径数量不一致")
if VISUAL_LATENT_INTERNAL_PATH_COUNT != 7_084_020:
    raise RuntimeError("视觉后天潜在展开路径数量不一致")
