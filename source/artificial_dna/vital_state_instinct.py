"""生命值与饱食度画面接入情绪器官的固定出生通道。

这里形成的是人工DNA片段。运行时仍由完整生命公式中的 U、A、Q、M、P
共同变化；本文件不读取游戏数值，也不在脑外计算生命值或饱食度。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .birth_structure import (
    EmotionControlBinding,
    FixedBirthFragment,
    FixedPathGene,
    NeuronGene,
    NeuronNature,
    TissueAllocator,
    VisualOrganBinding,
    VisualReceptorAddress,
)
from .brain_dna_layout import organ_continuation_address
from .organ_entrances import SECOND_EXPERIMENT_ENTRANCES


HEALTH_SLOT_COUNT = 10
HUNGER_SLOT_COUNT = 10
# 乙电脑用八位数传送视觉活动；进入 Brain 的统一强度始终已经还原为[0,1]。
VISUAL_ACTIVITY_MAX = 1.0


class VitalRow(IntEnum):
    HEALTH = 0
    HUNGER = 1


@dataclass(frozen=True, slots=True)
class PositiveEvidenceGroup:
    """一组必须以正活动实际到达的RGB受体地址。"""

    receptors: tuple[VisualReceptorAddress, ...]
    required_fraction: float = 0.65

    def __post_init__(self) -> None:
        if not self.receptors:
            raise ValueError("固定活动组合不能是空的")
        if len(set(self.receptors)) != len(self.receptors):
            raise ValueError("同一固定活动组合中不能重复使用一个RGB受体")
        if not 0.0 < float(self.required_fraction) <= 1.0:
            raise ValueError("固定活动组合阈值比例必须处于(0,1]")


@dataclass(frozen=True, slots=True)
class VitalSlotEvidence:
    row: VitalRow
    slot: int
    presence: tuple[PositiveEvidenceGroup, ...]
    complete: tuple[PositiveEvidenceGroup, ...]
    missing: tuple[PositiveEvidenceGroup, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "row", VitalRow(self.row))
        limit = HEALTH_SLOT_COUNT if self.row is VitalRow.HEALTH else HUNGER_SLOT_COUNT
        if not 0 <= self.slot < limit:
            raise ValueError("生命值或饱食度位置编号无效")
        if not self.presence or not self.complete or not self.missing:
            raise ValueError("每个位置都必须有HUD存在、完整和缺失的正活动组合")


@dataclass(frozen=True, slots=True)
class VitalVisualBranch:
    """某一种视野中心映射下的二十个固定活动组合。"""

    branch: int
    slots: tuple[VitalSlotEvidence, ...]
    supplemental_stable: tuple[PositiveEvidenceGroup, ...] = ()

    def __post_init__(self) -> None:
        if self.branch < 0:
            raise ValueError("视野中心映射分支编号不能小于零")
        expected = {
            *((VitalRow.HEALTH, slot) for slot in range(HEALTH_SLOT_COUNT)),
            *((VitalRow.HUNGER, slot) for slot in range(HUNGER_SLOT_COUNT)),
        }
        actual = {(slot.row, slot.slot) for slot in self.slots}
        if actual != expected or len(self.slots) != len(expected):
            raise ValueError("每个视野中心映射分支必须恰好包含十个生命值和十个饱食度位置")


@dataclass(frozen=True, slots=True)
class VitalStateInstinctPlan:
    branches: tuple[VitalVisualBranch, ...]

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("生命稳定状态本能至少需要一个真实视觉映射分支")
        branch_ids = [branch.branch for branch in self.branches]
        if len(set(branch_ids)) != len(branch_ids):
            raise ValueError("视野中心映射分支编号不能重复")


class _FragmentBuilder:
    def __init__(self, allocator: TissueAllocator) -> None:
        self.allocator = allocator
        self.neurons: list[NeuronGene] = []
        self.visual_bindings: list[VisualOrganBinding] = []
        self.paths: list[FixedPathGene] = []
        self.emotion_bindings: list[EmotionControlBinding] = []
        self._visual_entries: dict[VisualReceptorAddress, str] = {}
        self._next_emotion_entrance = 0

    def neuron(
        self,
        name: str,
        threshold: float,
        nature: NeuronNature = NeuronNature.FIXED,
    ) -> str:
        self.neurons.append(
            NeuronGene(
                name=name,
                address=self.allocator.take(),
                response_gain=1.0,
                threshold=threshold,
                nature=nature,
            )
        )
        return name

    def visual_entry(self, receptor: VisualReceptorAddress) -> str:
        """建立受体到独立继续位置的完整逐项入口通道。"""

        existing = self._visual_entries.get(receptor)
        if existing is not None:
            return existing
        channel = int(receptor.channel)
        prefix = f"visual.x{receptor.x}.y{receptor.y}.c{channel}"
        addresses = SECOND_EXPERIMENT_ENTRANCES.visual_pair(receptor)
        activity = (receptor.y * 1280 + receptor.x) * 3 + channel
        receiver = f"{prefix}.receiver"
        ordinary = f"{prefix}.ordinary"
        continuation = f"{prefix}.continuation"
        self.neurons.append(
            NeuronGene(
                name=receiver,
                address=addresses.receiver,
                response_gain=1.0,
                threshold=0.0,
                nature=NeuronNature.FIXED,
            )
        )
        self.neurons.append(
            NeuronGene(
                name=ordinary,
                address=addresses.ordinary,
                response_gain=1.0,
                threshold=0.0,
                nature=NeuronNature.ORDINARY,
            )
        )
        self.neurons.append(
            NeuronGene(
                name=continuation,
                address=organ_continuation_address(activity),
                response_gain=1.0,
                threshold=0.0,
                nature=NeuronNature.ORDINARY,
            )
        )
        self.visual_bindings.append(
            VisualOrganBinding(
                f"{prefix}.organ",
                receptor,
                receiver,
                ordinary,
                continuation,
            )
        )
        self.paths.extend(
            (
                FixedPathGene(f"{prefix}.entrance", receiver, ordinary, 1.0),
                FixedPathGene(
                    f"{prefix}.continuation",
                    ordinary,
                    continuation,
                    1.0,
                ),
            )
        )
        self._visual_entries[receptor] = continuation
        return continuation

    @dataclass(frozen=True, slots=True)
    class _BoundedSource:
        name: str
        minimum_active: float
        maximum_active: float

    def evidence(
        self,
        name: str,
        group: PositiveEvidenceGroup,
    ) -> _BoundedSource:
        # 每项视觉活动先到达自己的独立继续位置，再分别经过固定路径，随后在目标神经元
        # 直接相加。这里不按到达路径数量平均，也不设置全局归一化规则。
        threshold = group.required_fraction * len(group.receptors)
        target = self.neuron(name, threshold)
        # 每项视觉活动已是[0,1]强度。一对一入口完整传递后，这里也不能
        # 因八位传输表示再额外削弱255倍。
        attenuation = 1.0 / VISUAL_ACTIVITY_MAX
        for index, receptor in enumerate(group.receptors):
            continuation = self.visual_entry(receptor)
            self.paths.append(
                FixedPathGene(
                    f"{name}.visual.{index}",
                    continuation,
                    target,
                    attenuation,
                )
            )
        return self._BoundedSource(
            target,
            float(threshold),
            float(len(group.receptors)),
        )

    def require_both(
        self,
        name: str,
        first: _BoundedSource,
        second: _BoundedSource,
    ) -> _BoundedSource:
        """只用连续活动之和形成两路共同到达，不假定阈值后输出变成1。"""

        largest_single = max(first.maximum_active, second.maximum_active)
        smallest_joint = first.minimum_active + second.minimum_active
        if smallest_joint <= largest_single:
            raise ValueError("两组正活动之间没有形成可实现的共同到达阈值间隔")
        threshold = (largest_single + smallest_joint) / 2.0
        target = self.neuron(name, threshold)
        self.paths.extend(
            (
                FixedPathGene(f"{name}.path.0", first.name, target, 1.0),
                FixedPathGene(f"{name}.path.1", second.name, target, 1.0),
            )
        )
        return self._BoundedSource(
            target,
            threshold,
            first.maximum_active + second.maximum_active,
        )

    def emotion(self, name: str, source: str, strength: float = 1.0) -> None:
        """让每条微观固定通道保留自己的情绪器官控制入口。"""

        entrance = self._next_emotion_entrance
        self._next_emotion_entrance += 1
        self.emotion_bindings.append(
            EmotionControlBinding(name, source, entrance, strength)
        )

    def finish(self) -> FixedBirthFragment:
        return FixedBirthFragment(
            tuple(self.neurons),
            tuple(self.visual_bindings),
            tuple(self.paths),
            tuple(self.emotion_bindings),
        )


def _slot_prefix(branch: int, evidence: VitalSlotEvidence) -> str:
    return f"vital.b{branch}.r{int(evidence.row)}.s{evidence.slot}"


def _build_slot(
    builder: _FragmentBuilder,
    branch: int,
    evidence: VitalSlotEvidence,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    prefix = _slot_prefix(branch, evidence)
    presence_groups = tuple(
        builder.evidence(f"{prefix}.presence.g{index}", group)
        for index, group in enumerate(evidence.presence)
    )
    complete_groups = tuple(
        builder.evidence(f"{prefix}.complete.g{index}", group)
        for index, group in enumerate(evidence.complete)
    )
    missing_groups = tuple(
        builder.evidence(f"{prefix}.missing.g{index}", group)
        for index, group in enumerate(evidence.missing)
    )
    complete = tuple(
        builder.require_both(
            f"{prefix}.complete.p{presence_index}.g{content_index}",
            presence,
            content,
        ).name
        for presence_index, presence in enumerate(presence_groups)
        for content_index, content in enumerate(complete_groups)
    )
    missing = tuple(
        builder.require_both(
            f"{prefix}.missing.p{presence_index}.g{content_index}",
            presence,
            content,
        ).name
        for presence_index, presence in enumerate(presence_groups)
        for content_index, content in enumerate(missing_groups)
    )
    return complete, missing


def build_vital_state_instinct(
    plan: VitalStateInstinctPlan,
    *,
    tissue_start_index: int,
) -> FixedBirthFragment:
    """形成可并入完整人工DNA的固定本能通道。

    每条特殊短路径保留独立的控制入口工程地址。它们没有把“满足”或“不安”
    这样的名称送入脑；正式全脑端点生成还会把这里复用的视觉普通入口来源
    接到该视觉活动的独立继续位置，确保本能分支不会绕过统一继续通道。
    """

    if tissue_start_index < SECOND_EXPERIMENT_ENTRANCES.next_free_index:
        raise ValueError("生命状态本能组织与已经确认的器官入口神经元重叠")
    builder = _FragmentBuilder(TissueAllocator(tissue_start_index))

    for branch in plan.branches:
        for slot in sorted(branch.slots, key=lambda value: (int(value.row), value.slot)):
            complete_sources, missing_sources = _build_slot(
                builder,
                branch.branch,
                slot,
            )
            for index, source in enumerate(complete_sources):
                builder.emotion(
                    f"{_slot_prefix(branch.branch, slot)}.complete.emotion.{index}",
                    source,
                )
            for index, source in enumerate(missing_sources):
                builder.emotion(
                    f"{_slot_prefix(branch.branch, slot)}.missing.emotion.{index}",
                    source,
                )

        for index, group in enumerate(branch.supplemental_stable):
            source = builder.evidence(
                f"vital.b{branch.branch}.supplemental.g{index}",
                group,
            )
            builder.emotion(
                f"vital.b{branch.branch}.supplemental.g{index}.emotion",
                source.name,
                0.25,
            )

    return builder.finish()
