"""第二次实验完整1.024亿神经元的响应强度与阈值出生值。"""

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
    linear_index,
)
from .brain_dna_layout import (
    action_formation_address,
    joint_action_address,
    returned_action_formation_index,
)
from .fixed_path_topology import FixedPathTopology
from .fixed_receiver_birth_values import NEURON_BIRTH_VALUE_DTYPE
from .formal_brain_topology import FormalBrainTopology
from .neuron_nature_topology import StoredNeuronNature
from .ordinary_local_path_space import _normal_pair_slice
from .organ_continuation_birth_values import (
    iter_transparent_organ_birth_value_batches,
)
from .sensory_identity_branch_birth_values import (
    iter_sensory_identity_branch_birth_value_batches,
)
from .vital_state_neuron_birth_values import (
    build_vital_state_internal_neuron_birth_values,
)


FORMAL_NEURON_THRESHOLD = numpy.float32(0.0)


@dataclass(frozen=True, slots=True)
class FormalNeuronBirthValueBatch:
    start: int
    stop: int
    values: numpy.ndarray

    def __post_init__(self) -> None:
        if (
            not 0 <= int(self.start) < int(self.stop) <= TISSUE_NEURON_COUNT
            or self.values.shape != (self.stop - self.start,)
            or self.values.dtype != NEURON_BIRTH_VALUE_DTYPE
        ):
            raise ValueError("正式神经元出生值批次范围、形状或类型不正确")


def same_organization_local_inbound_counts(
    organizations: numpy.ndarray,
) -> numpy.ndarray:
    labels = numpy.asarray(organizations)
    expected = (TISSUE_DEPTH, TISSUE_HEIGHT, TISSUE_WIDTH)
    if labels.shape != expected or labels.dtype != numpy.dtype("u1"):
        raise ValueError("完整普通组织图形状或类型不正确")
    counts = numpy.zeros(expected, dtype="u1")
    for dx, dy, dz in DIRECT_NEIGHBOR_OFFSETS:
        source_z, target_z = _normal_pair_slice(TISSUE_DEPTH, dz)
        source_y, target_y = _normal_pair_slice(TISSUE_HEIGHT, dy)
        source_x, target_x = _normal_pair_slice(TISSUE_WIDTH, dx)
        source = labels[source_z, source_y, source_x]
        target = labels[target_z, target_y, target_x]
        counts[source_z, source_y, source_x] += (
            (source != 0) & (source == target)
        ).astype("u1")
    return counts.reshape(-1)


def active_fixed_inbound_counts(topology: FixedPathTopology) -> numpy.ndarray:
    counts = numpy.zeros(TISSUE_NEURON_COUNT, dtype="<u2")
    produced = 0
    for batch in topology.iter_second_experiment_batches():
        numpy.add.at(counts, batch.endpoints["target_neuron"], numpy.uint16(1))
        produced += int(batch.endpoints.size)
    if produced != topology.second_experiment_path_count:
        raise RuntimeError("正式神经元汇入计数没有读取完整的本次固定路径")
    return counts


def neutral_transfer_mask(topology: FixedPathTopology) -> numpy.ndarray:
    """保留已经确定必须原强度传播的固定微观通道。"""

    mask = numpy.zeros(TISSUE_NEURON_COUNT, dtype=numpy.bool_)
    for batch in iter_transparent_organ_birth_value_batches(
        entrances=topology.entrances,
    ):
        mask[batch.values["neuron"]] = True
    for batch in iter_sensory_identity_branch_birth_value_batches():
        mask[batch.values["neuron"]] = True
    # 每项实际鼠标、键盘和视野中心活动在独立继续位置之后分成五条
    # 动作接触支路。共同接触、动作形成第一段和第三段都只有在实际
    # 到达发生时才应响应；不能把尚为零的相邻可塑路径预先算作实际
    # 汇入并削弱这条固定回流。五路最终仍在运动输出神经元按其既有
    # 出生响应汇合，因此强度1会形成0.5到0.625的客观输出。
    for action in range(116):
        for repetition in range(5):
            returned = returned_action_formation_index(action, repetition)
            for address in (
                joint_action_address(action, repetition),
                action_formation_address(0, returned),
                action_formation_address(2, returned),
            ):
                mask[linear_index(address.x, address.y, address.z)] = True
    return mask


class FormalNeuronBirthValues:
    """同一公式下按完整连接上限生成逐神经元出生参数。"""

    def __init__(
        self,
        brain_topology: FormalBrainTopology,
        fixed_paths: FixedPathTopology,
    ) -> None:
        self.brain_topology = brain_topology
        self.fixed_paths = fixed_paths

    def iter_batches(
        self,
        *,
        batch_size: int = 262_144,
    ) -> Iterator[FormalNeuronBirthValueBatch]:
        size = int(batch_size)
        if size <= 0:
            raise ValueError("正式神经元出生值批次大小必须大于零")

        nature = self.brain_topology.build_nature_map().reshape(-1)
        organizations = self.brain_topology.build_organization_map()
        local_inbound = same_organization_local_inbound_counts(organizations)
        fixed_inbound = active_fixed_inbound_counts(self.fixed_paths)
        neutral = neutral_transfer_mask(self.fixed_paths)

        vital = build_vital_state_internal_neuron_birth_values(
            self.fixed_paths.instinct
        ).values
        vital_thresholds = {
            int(neuron): float(threshold)
            for neuron, threshold in zip(vital["neuron"], vital["threshold"])
        }

        produced = 0
        for start in range(0, TISSUE_NEURON_COUNT, size):
            stop = min(start + size, TISSUE_NEURON_COUNT)
            indexes = numpy.arange(start, stop, dtype="<u4")
            values = numpy.empty(indexes.size, dtype=NEURON_BIRTH_VALUE_DTYPE)
            values["neuron"] = indexes
            total = (
                local_inbound[start:stop].astype("<u4")
                + fixed_inbound[start:stop].astype("<u4")
            )
            total[total == 0] = 1
            values["response_gain"] = 1.0 / total.astype(numpy.float32)
            values["threshold"] = FORMAL_NEURON_THRESHOLD

            fixed = nature[start:stop] == int(StoredNeuronNature.FIXED)
            values["response_gain"][fixed] = 1.0
            values["response_gain"][neutral[start:stop]] = 1.0
            for neuron, threshold in vital_thresholds.items():
                if start <= neuron < stop:
                    values["threshold"][neuron - start] = threshold

            if numpy.any(values["response_gain"] <= 0.0):
                raise RuntimeError("正式神经元出现非正响应强度")
            yield FormalNeuronBirthValueBatch(start, stop, values)
            produced += int(values.size)

        if produced != TISSUE_NEURON_COUNT:
            raise RuntimeError("正式神经元出生值没有覆盖完整1.024亿位置")
