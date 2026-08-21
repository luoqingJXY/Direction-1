"""已确定人工出生结构的神经元固定/普通性质映射。

这里仅生成已经拥有明确组织用途的神经元位置。未分配用途的物理容量不写入
任何默认性质；磁盘数组中的零值不能被理解为固定或普通神经元。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

import numpy

from .birth_structure import FixedBirthFragment, NeuronNature
from .brain_address_plan import (
    ACTION_FORMATION_SECTION_COUNT,
    ACTION_FORMATION_START_Z,
    AUDITORY_FIRST_BRANCH_START_Z,
    AUDITORY_RECONSTRUCTION_START_Z,
    AUDITORY_SECOND_BRANCH_START_Z,
    AUDITORY_SOURCE_START_Z,
    JOINT_CRYSTALLIZATION_START_Z,
    MOTION_OUTPUT_Z,
    ORGAN_CONTINUATION_COUNT,
    ORGAN_CONTINUATION_START_Z,
    ORGAN_CONTINUATION_Z_OFFSET,
    PREDICTED_AUDITORY_OUTPUT_Z,
    REAL_AUDITORY_ASSOCIATION_CONTACT_COUNT,
    VISUAL_FIRST_BRANCH_START_Z,
    VISUAL_RGB_JOINT_CONTACT_COUNT,
    VISUAL_RGB_JOINT_CONTACT_START_Z,
    VISUAL_RECONSTRUCTION_START_Z,
    VISUAL_SECOND_BRANCH_START_Z,
    VISUAL_SOURCE_START_Z,
)
from .brain_dna_layout import (
    AUDITORY_JOINT_COUNT,
    AUDITORY_SOURCE_PER_SIDE,
    VISUAL_JOINT_COUNT,
    VISUAL_SOURCE_PER_SIDE,
)
from .brain_geometry import TISSUE_NEURON_COUNT, TISSUE_PLANE
from .organ_entrances import SECOND_EXPERIMENT_ENTRANCES, OrganEntranceLayout


class StoredNeuronNature(IntEnum):
    """硬盘性质值；零刻意不代表任何神经元性质。"""

    FIXED = 1
    ORDINARY = 2


@dataclass(frozen=True, slots=True)
class NeuronNatureBatch:
    """同一性质的一批唯一线性地址，不携带生命Signal。"""

    nature: StoredNeuronNature
    indexes: numpy.ndarray
    organization: str

    def __post_init__(self) -> None:
        if self.indexes.dtype != numpy.dtype("<u4"):
            raise ValueError("神经元性质地址必须使用32位无符号整数")
        if self.indexes.ndim != 1:
            raise ValueError("神经元性质地址必须是一维排列")
        if not self.organization:
            raise ValueError("神经元性质批次必须标明所属出生组织")

    @property
    def count(self) -> int:
        return int(self.indexes.size)


def _indexes(values: numpy.ndarray | list[int]) -> numpy.ndarray:
    return numpy.asarray(values, dtype="<u4").reshape(-1)


def _grid_indexes(z: int, width: int, height: int, *, x: int = 0, y: int = 0) -> numpy.ndarray:
    """800宽物理平面上的左上矩形，不虚构不存在的右侧位置。"""

    if not (0 <= int(z) < 160 and 0 <= int(x) < 800 and 0 <= int(y) < 800):
        raise ValueError("神经元性质矩形起点超出完整组织")
    if not (0 < int(width) <= 800 - int(x) and 0 < int(height) <= 800 - int(y)):
        raise ValueError("神经元性质矩形超出完整组织")
    rows = numpy.arange(int(y), int(y) + int(height), dtype="<u4")[:, None]
    columns = numpy.arange(int(x), int(x) + int(width), dtype="<u4")[None, :]
    return (int(z) * TISSUE_PLANE + rows * 800 + columns).reshape(-1)


def _linear_span(start: int, count: int) -> numpy.ndarray:
    if not (0 <= int(start) <= int(start) + int(count) <= TISSUE_NEURON_COUNT):
        raise ValueError("神经元性质连续区间超出完整组织")
    return numpy.arange(int(start), int(start) + int(count), dtype="<u4")


class NeuronNatureTopology:
    """把已确认组织逐块转换为固定或普通神经元地址。"""

    def __init__(
        self,
        instinct: FixedBirthFragment,
        *,
        entrances: OrganEntranceLayout = SECOND_EXPERIMENT_ENTRANCES,
    ) -> None:
        self.instinct = instinct
        self.entrances = entrances
        self._vital_fixed_indexes = self._find_vital_fixed_indexes()

    @property
    def fixed_count(self) -> int:
        return self.entrances.activity_count + int(self._vital_fixed_indexes.size)

    @property
    def ordinary_count(self) -> int:
        return (
            self.entrances.activity_count
            + ORGAN_CONTINUATION_COUNT
            + 12_845_974
            + 5_045_760
            + 18_450
            + VISUAL_RGB_JOINT_CONTACT_COUNT
            + REAL_AUDITORY_ASSOCIATION_CONTACT_COUNT
        )

    @property
    def assigned_count(self) -> int:
        return self.fixed_count + self.ordinary_count

    def _find_vital_fixed_indexes(self) -> numpy.ndarray:
        shared_organ_neurons = {
            name
            for binding in self.instinct.visual_bindings
            for name in (
                binding.receiver_neuron,
                binding.ordinary_neuron,
                binding.continuation_neuron,
            )
        }
        indexes: list[int] = []
        for neuron in self.instinct.neurons:
            if neuron.name in shared_organ_neurons:
                continue
            index = (
                neuron.address.z * TISSUE_PLANE
                + neuron.address.y * 800
                + neuron.address.x
            )
            if neuron.nature is not NeuronNature.FIXED:
                raise ValueError("生命状态本能在器官入口之后只能新增固定神经元")
            indexes.append(index)
        result = _indexes(sorted(indexes))
        if result.size != 160:
            raise ValueError("生命状态本能新增固定神经元数量不再是160")
        if result.size and not numpy.array_equal(
            result,
            _linear_span(self.entrances.next_free_index, result.size),
        ):
            raise ValueError("生命状态本能新增固定神经元地址不连续")
        return result

    def _visual_entrance_batches(
        self,
        nature: StoredNeuronNature,
    ) -> Iterator[NeuronNatureBatch]:
        depth_offset = 0 if nature is StoredNeuronNature.FIXED else 1
        for tile in range(5):
            yield NeuronNatureBatch(
                nature,
                _grid_indexes(tile * 2 + depth_offset, 768, 657),
                "真实视觉器官入口",
            )

    def _auditory_entrance_batches(
        self,
        nature: StoredNeuronNature,
    ) -> Iterator[NeuronNatureBatch]:
        depth = 10 if nature is StoredNeuronNature.FIXED else 11
        for stream in range(3):
            for tile in range(5):
                frequency_count = min(256, 1025 - tile * 256)
                yield NeuronNatureBatch(
                    nature,
                    _grid_indexes(
                        depth,
                        frequency_count * 3,
                        1,
                        y=stream * 6 + tile,
                    ),
                    "真实听觉器官入口",
                )

    def _predicted_visual_entrance_batches(
        self,
        nature: StoredNeuronNature,
    ) -> Iterator[NeuronNatureBatch]:
        depth = 12 if nature is StoredNeuronNature.FIXED else 13
        yield NeuronNatureBatch(
            nature,
            _grid_indexes(depth, 512, 512),
            "预测视觉回流入口",
        )

    def _predicted_auditory_entrance_batches(
        self,
        nature: StoredNeuronNature,
    ) -> Iterator[NeuronNatureBatch]:
        depth = 14 if nature is StoredNeuronNature.FIXED else 15
        for stream in range(3):
            for tile in range(2):
                frequency_count = min(256, 342 - tile * 256)
                yield NeuronNatureBatch(
                    nature,
                    _grid_indexes(
                        depth,
                        frequency_count * 3,
                        85,
                        y=(stream * 2 + tile) * 85,
                    ),
                    "预测听觉回流入口",
                )

    def _action_return_entrance_batches(
        self,
        nature: StoredNeuronNature,
    ) -> Iterator[NeuronNatureBatch]:
        depth = 16 if nature is StoredNeuronNature.FIXED else 17
        for y, width in ((0, 4), (1, 108), (2, 4)):
            yield NeuronNatureBatch(
                nature,
                _grid_indexes(depth, width, 1, y=y),
                "鼠标键盘视野中心回流入口",
            )

    def _organ_batches(self, nature: StoredNeuronNature) -> Iterator[NeuronNatureBatch]:
        yield from self._visual_entrance_batches(nature)
        yield from self._auditory_entrance_batches(nature)
        yield from self._predicted_visual_entrance_batches(nature)
        yield from self._predicted_auditory_entrance_batches(nature)
        yield from self._action_return_entrance_batches(nature)

    def _draft_ordinary_batches(self) -> Iterator[NeuronNatureBatch]:
        ordinary = StoredNeuronNature.ORDINARY
        for section in range(10):
            yield NeuronNatureBatch(
                ordinary,
                _grid_indexes(
                    VISUAL_RECONSTRUCTION_START_Z + section,
                    512,
                    512,
                ),
                "视觉还原组织",
            )
        for side in range(2):
            yield NeuronNatureBatch(
                ordinary,
                _linear_span(
                    (VISUAL_SOURCE_START_Z + side) * TISSUE_PLANE,
                    VISUAL_SOURCE_PER_SIDE,
                ),
                "视觉双来源组织",
            )
        for stream in range(3):
            for tile in range(2):
                frequency_count = min(256, 342 - tile * 256)
                yield NeuronNatureBatch(
                    ordinary,
                    _grid_indexes(
                        PREDICTED_AUDITORY_OUTPUT_Z,
                        frequency_count * 3,
                        85,
                        y=(stream * 2 + tile) * 85,
                    ),
                    "预测听觉输出",
                )
        yield NeuronNatureBatch(
            ordinary,
            _grid_indexes(MOTION_OUTPUT_Z, 116, 1),
            "鼠标键盘视野中心输出",
        )
        for section in range(10):
            for stream in range(3):
                for tile in range(2):
                    frequency_count = min(256, 342 - tile * 256)
                    yield NeuronNatureBatch(
                        ordinary,
                        _grid_indexes(
                            AUDITORY_RECONSTRUCTION_START_Z + section,
                            frequency_count * 3,
                            85,
                            y=(stream * 2 + tile) * 85,
                        ),
                        "听觉还原组织",
                    )
        for side in range(2):
            yield NeuronNatureBatch(
                ordinary,
                _linear_span(
                    (AUDITORY_SOURCE_START_Z + side) * TISSUE_PLANE,
                    AUDITORY_SOURCE_PER_SIDE,
                ),
                "听觉双来源组织",
            )

        joint_start = JOINT_CRYSTALLIZATION_START_Z * TISSUE_PLANE
        yield NeuronNatureBatch(
            ordinary,
            joint_start + numpy.arange(VISUAL_JOINT_COUNT, dtype="<u4") * 2,
            "共同结晶组织（视觉位置）",
        )
        yield NeuronNatureBatch(
            ordinary,
            joint_start + numpy.arange(VISUAL_JOINT_COUNT, dtype="<u4") * 2 + 1,
            "共同结晶组织（听觉配对位置）",
        )
        auditory_extra = AUDITORY_JOINT_COUNT - VISUAL_JOINT_COUNT
        yield NeuronNatureBatch(
            ordinary,
            _linear_span(
                joint_start + 2 * VISUAL_JOINT_COUNT,
                auditory_extra,
            ),
            "共同结晶组织（听觉扩展位置）",
        )
        yield NeuronNatureBatch(
            ordinary,
            _linear_span(
                joint_start + 2 * VISUAL_JOINT_COUNT + auditory_extra,
                580,
            ),
            "共同结晶组织（动作接触位置）",
        )
        yield NeuronNatureBatch(
            ordinary,
            _linear_span(
                joint_start + ACTION_FORMATION_SECTION_COUNT,
                REAL_AUDITORY_ASSOCIATION_CONTACT_COUNT,
            ),
            "共同结晶组织（真实听觉逐项接触位置）",
        )
        for section in range(3):
            yield NeuronNatureBatch(
                ordinary,
                _linear_span(
                    (ACTION_FORMATION_START_Z + section * 2) * TISSUE_PLANE,
                    ACTION_FORMATION_SECTION_COUNT,
                ),
                f"动作形成组织第{section + 1}段",
            )
        for branch, start_z in enumerate(
            (VISUAL_FIRST_BRANCH_START_Z, VISUAL_SECOND_BRANCH_START_Z),
            start=1,
        ):
            for tile in range(5):
                yield NeuronNatureBatch(
                    ordinary,
                    _grid_indexes(start_z + tile * 2, 768, 657),
                    f"真实视觉逐项独立支路{branch}",
                )
        for tile in range(5):
            yield NeuronNatureBatch(
                ordinary,
                _grid_indexes(
                    VISUAL_RGB_JOINT_CONTACT_START_Z + tile * 2,
                    768,
                    657,
                ),
                "共同结晶组织（真实RGB逐项接触位置）",
            )
        for branch, start_z in enumerate(
            (AUDITORY_FIRST_BRANCH_START_Z, AUDITORY_SECOND_BRANCH_START_Z),
            start=1,
        ):
            for stream in range(3):
                for tile in range(5):
                    frequency_count = min(256, 1025 - tile * 256)
                    yield NeuronNatureBatch(
                        ordinary,
                        _grid_indexes(
                            start_z,
                            frequency_count * 3,
                            1,
                            y=stream * 6 + tile,
                        ),
                        f"真实听觉逐项独立支路{branch}",
                    )

    def iter_batches(self) -> Iterator[NeuronNatureBatch]:
        """以出生组织为单位生成，不为未分配容量产生任何条目。"""

        yield from self._organ_batches(StoredNeuronNature.FIXED)
        yield from self._organ_batches(StoredNeuronNature.ORDINARY)
        yield NeuronNatureBatch(
            StoredNeuronNature.FIXED,
            self._vital_fixed_indexes,
            "生命状态本能固定通道",
        )
        for batch in self._organ_batches(StoredNeuronNature.ORDINARY):
            yield NeuronNatureBatch(
                StoredNeuronNature.ORDINARY,
                (
                    batch.indexes
                    + numpy.uint32(ORGAN_CONTINUATION_Z_OFFSET * TISSUE_PLANE)
                ).astype("<u4", copy=False),
                "器官活动逐项独立继续组织",
            )
        yield from self._draft_ordinary_batches()

    def apply_to(self, target: numpy.ndarray) -> int:
        """把已确认性质写入目标数组；未分配位置保持调用者原有值。"""

        if target.shape != (TISSUE_NEURON_COUNT,):
            raise ValueError("神经元性质目标必须覆盖完整800×800×160组织")
        if target.dtype != numpy.dtype("u1"):
            raise ValueError("神经元性质目标必须使用8位无符号整数")
        written = 0
        for batch in self.iter_batches():
            target[batch.indexes] = int(batch.nature)
            written += batch.count
        if written != self.assigned_count:
            raise RuntimeError("已确认神经元性质映射数量与出生总账不一致")
        return written


def validate_neuron_nature_counts(topology: NeuronNatureTopology) -> None:
    if topology.fixed_count != 3_056_155:
        raise RuntimeError("固定神经元数量与器官入口及本能通道不一致")
    if topology.ordinary_count != 26_554_279:
        raise RuntimeError("普通神经元数量与已推进组织不一致")
    if topology.assigned_count != 29_610_434:
        raise RuntimeError("神经元性质映射数量与全脑地址总账不一致")
