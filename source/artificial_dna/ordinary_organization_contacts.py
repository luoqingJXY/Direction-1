"""当前已定普通组织之间的直接相邻物理接触账。

组织名称和接触统计只服务于人工DNA核对，不作为Signal进入Brain。接触只表示
两端处于直接相邻位置；是否允许可塑路径在该处形成，仍由待补充的区域生效
关系决定。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

import numpy

from .brain_geometry import (
    DIRECT_NEIGHBOR_OFFSETS,
    TISSUE_DEPTH,
    TISSUE_HEIGHT,
    TISSUE_WIDTH,
)
from .neuron_nature_topology import NeuronNatureBatch, NeuronNatureTopology, StoredNeuronNature
from .ordinary_local_path_space import (
    CURRENT_ORDINARY_NEIGHBOR_CANDIDATE_COUNT,
    _normal_pair_slice,
)


class OrdinaryOrganization(IntEnum):
    VISUAL_INPUT = 1
    AUDITORY_INPUT = 2
    PREDICTED_VISUAL_RETURN = 3
    PREDICTED_AUDITORY_RETURN = 4
    ACTION_RETURN = 5
    ORGAN_CONTINUATION = 6
    VISUAL_RECONSTRUCTION = 7
    VISUAL_SOURCE = 8
    PREDICTED_AUDITORY_OUTPUT = 9
    MOTION_OUTPUT = 10
    AUDITORY_RECONSTRUCTION = 11
    AUDITORY_SOURCE = 12
    JOINT_CRYSTALLIZATION = 13
    ACTION_FORMATION_SECTION_0 = 14
    VISUAL_IDENTITY_BRANCH = 15
    AUDITORY_IDENTITY_BRANCH = 16
    ACTION_FORMATION_SECTION_1 = 17
    ACTION_FORMATION_SECTION_2 = 18
    VITAL_STATE_INSTINCT = 19
    VISUAL_IDENTITY_BRANCH_2 = 20
    AUDITORY_IDENTITY_BRANCH_2 = 21


@dataclass(frozen=True, slots=True)
class OrdinaryOrganizationContactLedger:
    """有方向接触数量矩阵；矩阵下标不是生命中的类别标签。"""

    counts: numpy.ndarray

    def __post_init__(self) -> None:
        expected = (len(OrdinaryOrganization) + 1, len(OrdinaryOrganization) + 1)
        if self.counts.shape != expected or self.counts.dtype != numpy.dtype("<u8"):
            raise ValueError("普通组织接触账形状或类型不正确")

    @property
    def directed_total(self) -> int:
        return int(self.counts.sum(dtype=numpy.uint64))

    def count(self, source: OrdinaryOrganization, target: OrdinaryOrganization) -> int:
        return int(self.counts[int(source), int(target)])

    def cross_organization_counts(self) -> dict[tuple[OrdinaryOrganization, OrdinaryOrganization], int]:
        result: dict[tuple[OrdinaryOrganization, OrdinaryOrganization], int] = {}
        for source in OrdinaryOrganization:
            for target in OrdinaryOrganization:
                if source is target:
                    continue
                value = self.count(source, target)
                if value:
                    result[source, target] = value
        return result


def _organization_for(batch: NeuronNatureBatch) -> OrdinaryOrganization:
    name = batch.organization
    if name == "真实视觉器官入口":
        return OrdinaryOrganization.VISUAL_INPUT
    if name == "真实听觉器官入口":
        return OrdinaryOrganization.AUDITORY_INPUT
    if name == "预测视觉回流入口":
        return OrdinaryOrganization.PREDICTED_VISUAL_RETURN
    if name == "预测听觉回流入口":
        return OrdinaryOrganization.PREDICTED_AUDITORY_RETURN
    if name == "鼠标键盘视野中心回流入口":
        return OrdinaryOrganization.ACTION_RETURN
    if name == "器官活动逐项独立继续组织":
        return OrdinaryOrganization.ORGAN_CONTINUATION
    if name == "视觉还原组织":
        return OrdinaryOrganization.VISUAL_RECONSTRUCTION
    if name == "视觉双来源组织":
        return OrdinaryOrganization.VISUAL_SOURCE
    if name == "预测听觉输出":
        return OrdinaryOrganization.PREDICTED_AUDITORY_OUTPUT
    if name == "鼠标键盘视野中心输出":
        return OrdinaryOrganization.MOTION_OUTPUT
    if name == "听觉还原组织":
        return OrdinaryOrganization.AUDITORY_RECONSTRUCTION
    if name == "听觉双来源组织":
        return OrdinaryOrganization.AUDITORY_SOURCE
    if name.startswith("共同结晶组织"):
        return OrdinaryOrganization.JOINT_CRYSTALLIZATION
    if name == "动作形成组织第1段":
        return OrdinaryOrganization.ACTION_FORMATION_SECTION_0
    if name == "动作形成组织第2段":
        return OrdinaryOrganization.ACTION_FORMATION_SECTION_1
    if name == "动作形成组织第3段":
        return OrdinaryOrganization.ACTION_FORMATION_SECTION_2
    if name.startswith("真实视觉逐项独立支路"):
        return OrdinaryOrganization.VISUAL_IDENTITY_BRANCH
    if name.startswith("真实听觉逐项独立支路"):
        return OrdinaryOrganization.AUDITORY_IDENTITY_BRANCH
    raise ValueError(f"未知普通神经元出生组织：{name}")


class OrdinaryOrganizationContacts:
    """从已定普通神经元性质映射重建组织接触账。"""

    def __init__(self, nature_topology: NeuronNatureTopology) -> None:
        self.nature_topology = nature_topology

    def build_organization_map(self) -> numpy.ndarray:
        labels = numpy.zeros(TISSUE_DEPTH * TISSUE_HEIGHT * TISSUE_WIDTH, dtype="u1")
        for batch in self.nature_topology.iter_batches():
            if batch.nature is not StoredNeuronNature.ORDINARY:
                continue
            label = int(_organization_for(batch))
            previous = labels[batch.indexes]
            if numpy.any(previous != 0):
                raise ValueError("两个普通出生组织占用了同一个神经元地址")
            labels[batch.indexes] = label
        return labels.reshape(TISSUE_DEPTH, TISSUE_HEIGHT, TISSUE_WIDTH)

    def build_ledger(self) -> OrdinaryOrganizationContactLedger:
        labels = self.build_organization_map()
        size = len(OrdinaryOrganization) + 1
        counts = numpy.zeros((size, size), dtype="<u8")
        for dx, dy, dz in DIRECT_NEIGHBOR_OFFSETS:
            source_z, target_z = _normal_pair_slice(TISSUE_DEPTH, dz)
            source_y, target_y = _normal_pair_slice(TISSUE_HEIGHT, dy)
            source_x, target_x = _normal_pair_slice(TISSUE_WIDTH, dx)
            source = labels[source_z, source_y, source_x]
            target = labels[target_z, target_y, target_x]
            linked = (source != 0) & (target != 0)
            pairs = source[linked].astype("<u2") * size + target[linked]
            direction_counts = numpy.bincount(
                pairs,
                minlength=size * size,
            ).astype("<u8", copy=False).reshape(size, size)
            counts += direction_counts
        ledger = OrdinaryOrganizationContactLedger(counts)
        if ledger.directed_total != CURRENT_ORDINARY_NEIGHBOR_CANDIDATE_COUNT:
            raise RuntimeError("普通组织接触账与普通邻接候选总数不一致")
        return ledger
