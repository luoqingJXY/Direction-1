"""第二次实验完整1.024亿神经元的出生性质与组织底图。

此前逐项生成器只覆盖已经画出固定端点的29,610,434个位置。正式个体不能
把其余位置的硬盘零值误当成未出生，也不能把不同组织之间的物理贴边误当成
可形成关系。本文件把同一800×800×160物理组织一次补全：明确固定的位置
保持固定，其余位置均为普通神经元；每个普通位置只继承它所在物理组织带的
身份。组织身份只用于出生时生成路径，不进入生命Signal。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from .brain_geometry import (
    TISSUE_DEPTH,
    TISSUE_HEIGHT,
    TISSUE_NEURON_COUNT,
    TISSUE_WIDTH,
)
from .neuron_nature_topology import NeuronNatureTopology, StoredNeuronNature
from .ordinary_organization_contacts import OrdinaryOrganization


@dataclass(frozen=True, slots=True)
class OrganizationDepthBand:
    start_z: int
    stop_z: int
    organization: OrdinaryOrganization

    def __post_init__(self) -> None:
        if not 0 <= int(self.start_z) < int(self.stop_z) <= TISSUE_DEPTH:
            raise ValueError("完整组织深度带超出800×800×160神经组织")


# 这些边界全部沿用已经确认的全脑地址计划。108层以后的容量接在共同结晶
# 组织之后，使完整脑保留跨感觉、预测和动作共同变化的最大空间；它不是新增
# Region，也不产生新的传播规则。
FORMAL_ORGANIZATION_DEPTH_BANDS = (
    OrganizationDepthBand(0, 10, OrdinaryOrganization.VISUAL_INPUT),
    OrganizationDepthBand(10, 12, OrdinaryOrganization.AUDITORY_INPUT),
    OrganizationDepthBand(12, 14, OrdinaryOrganization.PREDICTED_VISUAL_RETURN),
    OrganizationDepthBand(14, 16, OrdinaryOrganization.PREDICTED_AUDITORY_RETURN),
    OrganizationDepthBand(16, 18, OrdinaryOrganization.ACTION_RETURN),
    OrganizationDepthBand(18, 19, OrdinaryOrganization.VITAL_STATE_INSTINCT),
    OrganizationDepthBand(19, 37, OrdinaryOrganization.ORGAN_CONTINUATION),
    OrganizationDepthBand(37, 47, OrdinaryOrganization.VISUAL_RECONSTRUCTION),
    OrganizationDepthBand(47, 51, OrdinaryOrganization.VISUAL_SOURCE),
    OrganizationDepthBand(51, 52, OrdinaryOrganization.PREDICTED_AUDITORY_OUTPUT),
    OrganizationDepthBand(52, 53, OrdinaryOrganization.MOTION_OUTPUT),
    OrganizationDepthBand(53, 63, OrdinaryOrganization.AUDITORY_RECONSTRUCTION),
    OrganizationDepthBand(63, 67, OrdinaryOrganization.AUDITORY_SOURCE),
    OrganizationDepthBand(67, 69, OrdinaryOrganization.JOINT_CRYSTALLIZATION),
    OrganizationDepthBand(69, 71, OrdinaryOrganization.ACTION_FORMATION_SECTION_0),
    OrganizationDepthBand(71, 73, OrdinaryOrganization.ACTION_FORMATION_SECTION_1),
    OrganizationDepthBand(73, 76, OrdinaryOrganization.ACTION_FORMATION_SECTION_2),
    OrganizationDepthBand(76, 86, OrdinaryOrganization.VISUAL_IDENTITY_BRANCH),
    OrganizationDepthBand(86, 96, OrdinaryOrganization.VISUAL_IDENTITY_BRANCH_2),
    OrganizationDepthBand(96, 98, OrdinaryOrganization.AUDITORY_IDENTITY_BRANCH),
    OrganizationDepthBand(98, 99, OrdinaryOrganization.AUDITORY_IDENTITY_BRANCH_2),
    OrganizationDepthBand(99, 160, OrdinaryOrganization.JOINT_CRYSTALLIZATION),
)


def validate_formal_organization_depth_bands() -> None:
    expected = 0
    for band in FORMAL_ORGANIZATION_DEPTH_BANDS:
        if band.start_z != expected:
            raise RuntimeError("完整组织深度带存在空洞或重叠")
        expected = band.stop_z
    if expected != TISSUE_DEPTH:
        raise RuntimeError("完整组织深度带没有覆盖全部160层")


class FormalBrainTopology:
    """由既有固定位置和完整物理组织带生成正式出生底图。"""

    def __init__(self, known: NeuronNatureTopology) -> None:
        self.known = known

    @property
    def fixed_count(self) -> int:
        return self.known.fixed_count

    @property
    def ordinary_count(self) -> int:
        return TISSUE_NEURON_COUNT - self.fixed_count

    @property
    def assigned_count(self) -> int:
        return TISSUE_NEURON_COUNT

    def fixed_mask(self) -> numpy.ndarray:
        mask = numpy.zeros(TISSUE_NEURON_COUNT, dtype=numpy.bool_)
        for batch in self.known.iter_batches():
            if batch.nature is StoredNeuronNature.FIXED:
                if numpy.any(mask[batch.indexes]):
                    raise ValueError("固定神经元地址在既有出生结构中重复")
                mask[batch.indexes] = True
        if int(mask.sum(dtype=numpy.uint64)) != self.fixed_count:
            raise RuntimeError("完整组织固定神经元数量与器官及本能总账不一致")
        return mask.reshape(TISSUE_DEPTH, TISSUE_HEIGHT, TISSUE_WIDTH)

    def build_nature_map(self) -> numpy.ndarray:
        nature = numpy.full(
            (TISSUE_DEPTH, TISSUE_HEIGHT, TISSUE_WIDTH),
            int(StoredNeuronNature.ORDINARY),
            dtype="u1",
        )
        nature[self.fixed_mask()] = int(StoredNeuronNature.FIXED)
        return nature

    def build_organization_map(self) -> numpy.ndarray:
        labels = numpy.empty(
            (TISSUE_DEPTH, TISSUE_HEIGHT, TISSUE_WIDTH),
            dtype="u1",
        )
        for band in FORMAL_ORGANIZATION_DEPTH_BANDS:
            labels[band.start_z : band.stop_z] = int(band.organization)
        # 固定神经元不属于可变路径形成空间。零在这里仅表示没有普通组织身份。
        labels[self.fixed_mask()] = 0
        return labels

    def apply_nature_to(self, target: numpy.ndarray) -> int:
        if target.shape != (TISSUE_NEURON_COUNT,) or target.dtype != numpy.dtype("u1"):
            raise ValueError("完整神经元性质目标必须是一维1.024亿项无符号字节数组")
        target[:] = self.build_nature_map().reshape(-1)
        return TISSUE_NEURON_COUNT


def validate_formal_neuron_counts(topology: FormalBrainTopology) -> None:
    if topology.fixed_count != 3_056_155:
        raise RuntimeError("正式固定神经元数量与器官及生命状态本能不一致")
    if topology.ordinary_count != 99_343_845:
        raise RuntimeError("正式普通神经元数量没有填满固定位置以外的完整组织")
    if topology.assigned_count != TISSUE_NEURON_COUNT:
        raise RuntimeError("正式神经元性质没有覆盖完整组织")


validate_formal_organization_depth_bands()
