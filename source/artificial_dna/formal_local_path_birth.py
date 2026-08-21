"""完整普通神经组织的26向可变路径出生值与70路物质空间分布。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy

from .birth_structure import FixedBirthFragment
from .brain_geometry import (
    DIRECT_NEIGHBOR_OFFSETS,
    DIRECTED_LOCAL_PATH_COUNT,
    TISSUE_DEPTH,
    TISSUE_HEIGHT,
    TISSUE_NEURON_COUNT,
    TISSUE_PLANE,
    TISSUE_WIDTH,
)
from .formal_brain_topology import FormalBrainTopology
from .modulation_records import PATH_MODULATION_RESPONSE_DTYPE
from .ordinary_local_path_space import (
    LOCAL_PATH_CHUNK_DEPTH,
    LOCAL_PATH_MASK_DTYPE,
    _BYTE_BIT_COUNT,
    _normal_pair_slice,
)


MATERIAL_ACTIVITY_COUNT = 70
MATERIAL_COLUMNS = 10
MATERIAL_ROWS = 7
# 一次完整发生中，同一来源最多有26个直接相邻方向。把一次最大物质响应
# 均分到26条可能关系，是由既有物理邻接数推出的微变尺度，不是奖励率。
FORMAL_LOCAL_PATH_CHANGE_RATE = numpy.float32(1.0 / len(DIRECT_NEIGHBOR_OFFSETS))
# 一次睡眠采用同一个出生微变尺度：一次偶然形成可以被一次睡眠削到零，
# 经历反复形成的关系则按已经积累的路径强度继续保留。
FORMAL_SLEEP_PATH_WEAKENING = numpy.float32(1.0 / len(DIRECT_NEIGHBOR_OFFSETS))
# 物质活动本身已经由固定本能通道的实际到达形成；这里不再添加第二个判断器。
FORMAL_LOCAL_PATH_FORMATION_THRESHOLD = numpy.float32(0.0)
FORMAL_LOCAL_FORMATION_PERMISSION_COUNT = 2_289_069_086
FORMAL_LOCAL_CROSS_ORGANIZATION_HELD_ZERO_COUNT = 231_935_610
FORMAL_LOCAL_FIXED_ENDPOINT_HELD_ZERO_COUNT = 125_288_416


@dataclass(frozen=True, slots=True)
class FormalLocalPathBirthBatch:
    start_z: int
    stop_z: int
    formation_masks: numpy.ndarray

    def __post_init__(self) -> None:
        expected = (self.stop_z - self.start_z, TISSUE_HEIGHT, TISSUE_WIDTH)
        if (
            not 0 <= self.start_z < self.stop_z <= TISSUE_DEPTH
            or self.formation_masks.shape != expected
            or self.formation_masks.dtype != LOCAL_PATH_MASK_DTYPE
        ):
            raise ValueError("完整可变路径出生块的范围、形状或类型不正确")

    @property
    def start_index(self) -> int:
        return self.start_z * TISSUE_PLANE

    @property
    def stop_index(self) -> int:
        return self.stop_z * TISSUE_PLANE

    @property
    def permitted_count(self) -> int:
        return int(
            _BYTE_BIT_COUNT[self.formation_masks.view("u1")].sum(dtype=numpy.uint64)
        )


@dataclass(frozen=True, slots=True)
class FormalLocalPathLedger:
    formation_permitted: int
    ordinary_cross_organization_held_zero: int
    fixed_endpoint_held_zero: int

    @property
    def classified_count(self) -> int:
        return (
            self.formation_permitted
            + self.ordinary_cross_organization_held_zero
            + self.fixed_endpoint_held_zero
        )

    def require_complete(self) -> None:
        if self.classified_count != DIRECTED_LOCAL_PATH_COUNT:
            raise RuntimeError("完整26向相邻路径分类没有覆盖全部有效物理关系")


def _formation_masks(
    organizations: numpy.ndarray,
    start_z: int,
    stop_z: int,
) -> numpy.ndarray:
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
        source = organizations[source_z_start:source_z_stop, source_y, source_x]
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
    return masks


class FormalLocalPathBirth:
    def __init__(self, topology: FormalBrainTopology) -> None:
        self.topology = topology

    def iter_batches(
        self,
        *,
        chunk_depth: int = LOCAL_PATH_CHUNK_DEPTH,
    ) -> Iterator[FormalLocalPathBirthBatch]:
        depth = int(chunk_depth)
        if depth <= 0:
            raise ValueError("完整可变路径出生块深度必须大于零")
        organizations = self.topology.build_organization_map()
        for start_z in range(0, TISSUE_DEPTH, depth):
            stop_z = min(start_z + depth, TISSUE_DEPTH)
            yield FormalLocalPathBirthBatch(
                start_z,
                stop_z,
                _formation_masks(organizations, start_z, stop_z),
            )

    def build_ledger(self) -> FormalLocalPathLedger:
        organizations = self.topology.build_organization_map()
        permitted = 0
        ordinary_neighbors = 0
        for start_z in range(0, TISSUE_DEPTH, LOCAL_PATH_CHUNK_DEPTH):
            stop_z = min(start_z + LOCAL_PATH_CHUNK_DEPTH, TISSUE_DEPTH)
            masks = _formation_masks(organizations, start_z, stop_z)
            permitted += int(
                _BYTE_BIT_COUNT[masks.view("u1")].sum(dtype=numpy.uint64)
            )
            for dx, dy, dz in DIRECT_NEIGHBOR_OFFSETS:
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
                ordinary_neighbors += int(
                    ((source != 0) & (target != 0)).sum(dtype=numpy.uint64)
                )
        ledger = FormalLocalPathLedger(
            formation_permitted=permitted,
            ordinary_cross_organization_held_zero=ordinary_neighbors - permitted,
            fixed_endpoint_held_zero=DIRECTED_LOCAL_PATH_COUNT - ordinary_neighbors,
        )
        ledger.require_complete()
        expected = (
            FORMAL_LOCAL_FORMATION_PERMISSION_COUNT,
            FORMAL_LOCAL_CROSS_ORGANIZATION_HELD_ZERO_COUNT,
            FORMAL_LOCAL_FIXED_ENDPOINT_HELD_ZERO_COUNT,
        )
        actual = (
            ledger.formation_permitted,
            ledger.ordinary_cross_organization_held_zero,
            ledger.fixed_endpoint_held_zero,
        )
        if actual != expected:
            raise RuntimeError("完整可变路径分类与正式人工出生总账不一致")
        return ledger


def material_channel_for_sources(source_indexes: numpy.ndarray) -> numpy.ndarray:
    """把来源位置分到10列×7行的70条独立空间物质通道。"""

    sources = numpy.asarray(source_indexes)
    if sources.ndim != 1 or not numpy.issubdtype(sources.dtype, numpy.integer):
        raise ValueError("物质空间通道来源必须是一维整数神经元地址")
    if sources.size and (numpy.any(sources < 0) or int(sources.max()) >= TISSUE_NEURON_COUNT):
        raise ValueError("物质空间通道来源超出完整神经组织")
    x = sources.astype(numpy.uint64, copy=False) % TISSUE_WIDTH
    y = (sources.astype(numpy.uint64, copy=False) // TISSUE_WIDTH) % TISSUE_HEIGHT
    column = x * MATERIAL_COLUMNS // TISSUE_WIDTH
    row = y * MATERIAL_ROWS // TISSUE_HEIGHT
    return (row * MATERIAL_COLUMNS + column).astype("<u4", copy=False)


def derive_material_response_normalizers(
    instinct: FixedBirthFragment,
) -> numpy.ndarray:
    """由70条固定本能通道各自可达到的最大活动推出受体归一系数。"""

    neurons = {value.name: value for value in instinct.neurons}
    incoming: dict[str, list[tuple[str, float]]] = {}
    for path in instinct.paths:
        incoming.setdefault(path.target_neuron, []).append(
            (path.source_neuron, float(path.path_strength))
        )

    maxima: dict[str, float] = {}
    # 器官三段入口的每项真实视觉活动均在[0,1]中，最大活动为1。
    for binding in instinct.visual_bindings:
        maxima[binding.receiver_neuron] = 1.0
        maxima[binding.ordinary_neuron] = 1.0
        maxima[binding.continuation_neuron] = 1.0

    unresolved = set(neurons).difference(maxima)
    while unresolved:
        progressed = False
        for name in tuple(unresolved):
            arrivals = incoming.get(name, ())
            if not arrivals or any(source not in maxima for source, _ in arrivals):
                continue
            maximum = float(neurons[name].response_gain) * sum(
                maxima[source] * strength for source, strength in arrivals
            )
            if not numpy.isfinite(maximum) or maximum <= 0.0:
                raise ValueError("情绪物质固定来源没有有限正最大活动")
            maxima[name] = maximum
            unresolved.remove(name)
            progressed = True
        if not progressed:
            raise ValueError("生命状态固定通道不是可由器官入口展开的有向出生结构")

    if len(instinct.emotion_bindings) != MATERIAL_ACTIVITY_COUNT:
        raise ValueError("第二次实验生命状态本能必须恰好形成70路物质活动")
    result = numpy.empty(MATERIAL_ACTIVITY_COUNT, dtype="<f4")
    seen: set[int] = set()
    for binding in instinct.emotion_bindings:
        entrance = int(binding.entrance)
        if entrance in seen or not 0 <= entrance < MATERIAL_ACTIVITY_COUNT:
            raise ValueError("情绪器官控制入口编号重复或越界")
        seen.add(entrance)
        maximum = maxima[binding.source_neuron] * float(binding.path_strength)
        if not numpy.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("情绪器官物质来源没有有限正最大活动")
        result[entrance] = numpy.float32(1.0 / maximum)
    return result


def path_response_records_for_batch(
    batch: FormalLocalPathBirthBatch,
    material_normalizers: numpy.ndarray,
) -> numpy.ndarray:
    normalizers = numpy.asarray(material_normalizers, dtype="<f4")
    if normalizers.shape != (MATERIAL_ACTIVITY_COUNT,) or numpy.any(normalizers <= 0.0):
        raise ValueError("70路物质受体归一系数不完整")
    flat_masks = batch.formation_masks.reshape(-1)
    rows, directions = numpy.nonzero(
        (flat_masks[:, None] & (numpy.uint32(1) << numpy.arange(26, dtype="<u4")))
        != 0
    )
    sources = rows.astype(numpy.uint64) + numpy.uint64(batch.start_index)
    records = numpy.empty(rows.size, dtype=PATH_MODULATION_RESPONSE_DTYPE)
    records["path"] = (
        sources * numpy.uint64(26) + directions.astype(numpy.uint64)
    ).astype("<u4")
    materials = material_channel_for_sources(sources)
    records["material"] = materials
    records["response"] = normalizers[materials]
    if records.size > 1 and numpy.any(records["path"][1:] <= records["path"][:-1]):
        raise RuntimeError("可变路径物质受体记录没有按路径地址严格递增")
    return records


def iter_path_response_record_batches(
    birth: FormalLocalPathBirth,
    instinct: FixedBirthFragment,
    *,
    chunk_depth: int = 1,
) -> Iterator[numpy.ndarray]:
    normalizers = derive_material_response_normalizers(instinct)
    for batch in birth.iter_batches(chunk_depth=chunk_depth):
        yield path_response_records_for_batch(batch, normalizers)
