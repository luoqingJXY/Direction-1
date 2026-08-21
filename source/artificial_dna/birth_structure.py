"""可并入完整人工DNA的固定出生结构片段。

本文件只描述神经元、器官入口连接、固定路径和情绪器官控制入口连接。
它不运行识别、不比较答案，也不在脑外形成任何生命活动。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
import math

from .brain_geometry import (
    TISSUE_DEPTH,
    TISSUE_HEIGHT,
    TISSUE_NEURON_COUNT,
    TISSUE_WIDTH,
    coordinates,
    validate_coordinates,
)


VISUAL_WIDTH = 1280
VISUAL_HEIGHT = 657


class VisualChannel(IntEnum):
    R = 0
    G = 1
    B = 2


@dataclass(frozen=True, slots=True, order=True)
class VisualReceptorAddress:
    x: int
    y: int
    channel: VisualChannel

    def __post_init__(self) -> None:
        if not 0 <= self.x < VISUAL_WIDTH or not 0 <= self.y < VISUAL_HEIGHT:
            raise ValueError("视觉受体地址超出1280×657感受场")
        object.__setattr__(self, "channel", VisualChannel(self.channel))


@dataclass(frozen=True, slots=True, order=True)
class TissueAddress:
    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        validate_coordinates(self.x, self.y, self.z)


def _positive_attenuation(value: float, name: str) -> float:
    normalized = float(value)
    if not 0.0 < normalized <= 1.0:
        raise ValueError(f"{name}必须处于(0,1]，路径不能放大活动")
    return normalized


class NeuronNature(str, Enum):
    FIXED = "fixed"
    ORDINARY = "ordinary"


@dataclass(frozen=True, slots=True)
class NeuronGene:
    name: str
    address: TissueAddress
    response_gain: float
    threshold: float
    nature: NeuronNature

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("神经元必须有工程地址名")
        gain = float(self.response_gain)
        threshold = float(self.threshold)
        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError("神经元活动响应强度必须是有限非负数")
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("神经元阈值必须是有限非负数")
        object.__setattr__(self, "nature", NeuronNature(self.nature))


@dataclass(frozen=True, slots=True)
class VisualOrganBinding:
    name: str
    receptor: VisualReceptorAddress
    receiver_neuron: str
    ordinary_neuron: str
    continuation_neuron: str


@dataclass(frozen=True, slots=True)
class FixedPathGene:
    name: str
    source_neuron: str
    target_neuron: str
    path_strength: float

    def __post_init__(self) -> None:
        _positive_attenuation(self.path_strength, "固定路径强度")
        if self.source_neuron == self.target_neuron:
            raise ValueError("固定路径不能把同一个神经元直接连回自身")


@dataclass(frozen=True, slots=True)
class EmotionControlBinding:
    """从固定神经通道到情绪器官控制入口的特殊短路径。"""

    name: str
    source_neuron: str
    entrance: int
    path_strength: float

    def __post_init__(self) -> None:
        if self.entrance < 0:
            raise ValueError("情绪器官控制入口编号不能小于零")
        _positive_attenuation(self.path_strength, "情绪器官固定路径强度")


@dataclass(frozen=True, slots=True)
class FixedBirthFragment:
    """人工DNA中的一段固定结构，不是可独立运行的脑。"""

    neurons: tuple[NeuronGene, ...]
    visual_bindings: tuple[VisualOrganBinding, ...]
    paths: tuple[FixedPathGene, ...]
    emotion_bindings: tuple[EmotionControlBinding, ...]

    def __post_init__(self) -> None:
        names = [neuron.name for neuron in self.neurons]
        if len(set(names)) != len(names):
            raise ValueError("固定出生结构中存在重复神经元名称")
        addresses = [neuron.address for neuron in self.neurons]
        if len(set(addresses)) != len(addresses):
            raise ValueError("固定出生结构中存在重复神经组织地址")
        known = set(names)
        by_name = {neuron.name: neuron for neuron in self.neurons}
        bound_receptors: set[VisualReceptorAddress] = set()
        bound_receivers: set[str] = set()
        bound_ordinaries: set[str] = set()
        bound_continuations: set[str] = set()
        for binding in self.visual_bindings:
            chain = (
                binding.receiver_neuron,
                binding.ordinary_neuron,
                binding.continuation_neuron,
            )
            if any(name not in known for name in chain):
                raise ValueError("视觉器官连接的完整入口通道存在缺失神经元")
            if len(set(chain)) != len(chain):
                raise ValueError("视觉器官入口通道不能复用同一个神经元位置")
            if by_name[binding.receiver_neuron].nature is not NeuronNature.FIXED:
                raise ValueError("视觉器官活动必须先独立进入固定接收神经元")
            if by_name[binding.ordinary_neuron].nature is not NeuronNature.ORDINARY:
                raise ValueError("视觉器官固定接收端必须到达唯一普通入口")
            if by_name[binding.continuation_neuron].nature is not NeuronNature.ORDINARY:
                raise ValueError("视觉器官普通入口必须到达普通独立继续神经元")
            if binding.receptor in bound_receptors:
                raise ValueError("同一项视觉器官活动只能进入一个固定接收神经元")
            if binding.receiver_neuron in bound_receivers:
                raise ValueError("一个固定接收神经元只能接收一项视觉器官活动")
            if binding.ordinary_neuron in bound_ordinaries:
                raise ValueError("一个视觉普通入口只能承载一项器官活动")
            if binding.continuation_neuron in bound_continuations:
                raise ValueError("一个视觉独立继续神经元只能承载一项器官活动")
            bound_receptors.add(binding.receptor)
            bound_receivers.add(binding.receiver_neuron)
            bound_ordinaries.add(binding.ordinary_neuron)
            bound_continuations.add(binding.continuation_neuron)
        for path in self.paths:
            if path.source_neuron not in known or path.target_neuron not in known:
                raise ValueError("固定路径端点不在本出生结构片段中")
        for binding in self.emotion_bindings:
            if binding.source_neuron not in known:
                raise ValueError("情绪器官固定路径来源不存在")

        outgoing: dict[str, list[FixedPathGene]] = {
            source: [] for source in bound_receivers | bound_ordinaries
        }
        for path in self.paths:
            if path.source_neuron in outgoing:
                outgoing[path.source_neuron].append(path)
        for binding in self.visual_bindings:
            receiver_paths = outgoing[binding.receiver_neuron]
            if len(receiver_paths) != 1:
                raise ValueError("每个固定接收神经元必须只有一条一对一固定入口路径")
            receiver_path = receiver_paths[0]
            if receiver_path.path_strength != 1.0:
                raise ValueError("器官入口的一对一固定路径必须完整传递活动")
            if receiver_path.target_neuron != binding.ordinary_neuron:
                raise ValueError("固定接收神经元的一对一固定路径必须到达普通神经元")

            ordinary_paths = outgoing[binding.ordinary_neuron]
            if len(ordinary_paths) != 1:
                raise ValueError("每个普通入口必须先且只能到达自己的独立继续位置")
            continuation_path = ordinary_paths[0]
            if continuation_path.path_strength != 1.0:
                raise ValueError("普通入口到独立继续位置必须完整传播活动")
            if continuation_path.target_neuron != binding.continuation_neuron:
                raise ValueError("普通入口固定路径没有到达自己的独立继续神经元")

    @property
    def neuron_count(self) -> int:
        return len(self.neurons)

    @property
    def fixed_path_count(self) -> int:
        return len(self.visual_bindings) + len(self.paths) + len(self.emotion_bindings)


class TissueAllocator:
    """把固定出生通道逐个放入唯一的完整神经组织地址空间。"""

    def __init__(self, start_index: int = 0) -> None:
        if not 0 <= start_index < TISSUE_NEURON_COUNT:
            raise ValueError("出生结构起始位置超出神经组织容量")
        self._next = int(start_index)

    def take(self) -> TissueAddress:
        if self._next >= TISSUE_NEURON_COUNT:
            raise MemoryError("固定出生结构超过1.024亿神经元逻辑容量")
        index = self._next
        self._next += 1
        return TissueAddress(*coordinates(index))

    @property
    def next_free_index(self) -> int:
        return self._next
