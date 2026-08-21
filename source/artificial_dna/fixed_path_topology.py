"""第二次实验人工出生结构草案的固定路径端点。

本文件只把已经推进的固定连接拓扑落实为稳定的来源神经元地址和到达
神经元地址。它不填写具体路径强度、当前传播活动、神经元出生属性或调制
受体分布；这些出生值不能由端点排列反推或代替。

连接类别和批次编号只用于生成时核对硬盘记录，不进入 Brain 的 Signal。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

import numpy

from .birth_structure import FixedBirthFragment, TissueAddress
from .brain_address_plan import (
    ACTION_FORMATION_SECTION_COUNT,
    REALLOCATED_FIXED_NEURON_PATH_COUNT,
)
from .brain_dna_layout import (
    ACTION_JOINT_COUNT,
    AUDITORY_JOINT_COUNT,
    AUDITORY_SOURCE_PER_SIDE,
    SENSORY_ACTION_FORMATION_COUNT,
    VISUAL_JOINT_COUNT,
    VISUAL_SOURCE_PER_SIDE,
    action_formation_address,
    auditory_action_formation_index,
    auditory_continuation_address,
    auditory_identity_branch_address,
    auditory_reconstruction_address,
    auditory_source_address,
    joint_action_address,
    joint_auditory_address,
    joint_real_auditory_contact_address,
    joint_visual_address,
    organ_continuation_address,
    predicted_auditory_output_address,
    returned_action_formation_index,
    visual_reconstruction_address,
    visual_continuation_address,
    visual_identity_branch_address,
    visual_action_formation_index,
    visual_full_field_sample_coordinate,
    visual_source_address,
    visual_rgb_joint_contact_address,
    motion_output_address,
)
from .brain_geometry import TISSUE_NEURON_COUNT, linear_index
from .organ_entrances import SECOND_EXPERIMENT_ENTRANCES, OrganEntranceLayout


VISUAL_RECONSTRUCTION_SPACINGS = (1, 1, 2, 4, 8, 16, 32, 64, 128, 256)
AUDITORY_RECONSTRUCTION_SPACINGS = VISUAL_RECONSTRUCTION_SPACINGS

VISUAL_SELECTED_COUNTS = tuple(
    ((512 + spacing - 1) // spacing) ** 2
    for spacing in VISUAL_RECONSTRUCTION_SPACINGS
)
AUDITORY_SELECTED_COUNTS = tuple(
    3
    * ((342 + spacing - 1) // spacing)
    * ((85 + spacing - 1) // spacing)
    * 3
    for spacing in AUDITORY_RECONSTRUCTION_SPACINGS
)


class FixedPathFamily(IntEnum):
    """只用于工程核对的固定连接类别，不是 Brain 内的标签。"""

    ORGAN_ENTRANCE = 0
    VITAL_STATE = 1
    ORGAN_CONTINUATION = 2
    VISUAL_RECONSTRUCTION = 3
    VISUAL_SOURCES = 4
    AUDITORY_RECONSTRUCTION = 5
    AUDITORY_OUTPUT = 6
    AUDITORY_SOURCES = 7
    REAL_VISUAL_TO_RECONSTRUCTION_BRANCH = 8
    REAL_VISUAL_TO_CROSS_ORGANIZATION_BRANCH = 9
    PREDICTED_VISUAL_TO_JOINT = 10
    VISUAL_JOINT_TO_SECOND_SOURCE = 11
    REAL_AUDITORY_TO_RECONSTRUCTION_BRANCH = 12
    REAL_AUDITORY_TO_CROSS_ORGANIZATION_BRANCH = 13
    PREDICTED_AUDITORY_TO_JOINT = 14
    AUDITORY_JOINT_TO_SECOND_SOURCE = 15
    ACTION_RETURN_TO_JOINT = 16
    JOINT_TO_ACTION_ZERO = 17
    RECONSTRUCTION_TO_ACTION_ONE = 18
    ACTION_ZERO_TO_ACTION_TWO = 19
    ACTION_ONE_TO_ACTION_TWO = 20
    VISUAL_GRAYSCALE_ADMISSION = 21
    VISUAL_FIRST_SOURCE_TO_JOINT = 22
    REAL_VISUAL_CROSS_BRANCH_TO_RGB_CONTACT = 23
    VISUAL_RGB_CONTACT_TO_JOINT = 24
    ACTION_FORMATION_TO_MOTION_OUTPUT = 25
    REAL_AUDITORY_TO_ASSOCIATION_CONTACT = 26


FIXED_PATH_FAMILY_COUNTS: dict[FixedPathFamily, int] = {
    FixedPathFamily.ORGAN_ENTRANCE: 3_055_995,
    FixedPathFamily.VITAL_STATE: 1_760,
    FixedPathFamily.ORGAN_CONTINUATION: 3_055_995,
    FixedPathFamily.VISUAL_RECONSTRUCTION: 8_478_037,
    FixedPathFamily.VISUAL_SOURCES: 1_223_336,
    FixedPathFamily.AUDITORY_RECONSTRUCTION: 11_704_176,
    FixedPathFamily.AUDITORY_OUTPUT: 261_630,
    FixedPathFamily.AUDITORY_SOURCES: 1_224_720,
    FixedPathFamily.REAL_VISUAL_TO_RECONSTRUCTION_BRANCH: 2_522_880,
    FixedPathFamily.REAL_VISUAL_TO_CROSS_ORGANIZATION_BRANCH: 2_522_880,
    FixedPathFamily.PREDICTED_VISUAL_TO_JOINT: 611_668,
    FixedPathFamily.VISUAL_JOINT_TO_SECOND_SOURCE: 611_668,
    FixedPathFamily.REAL_AUDITORY_TO_RECONSTRUCTION_BRANCH: 9_225,
    FixedPathFamily.REAL_AUDITORY_TO_CROSS_ORGANIZATION_BRANCH: 9_225,
    FixedPathFamily.PREDICTED_AUDITORY_TO_JOINT: 612_360,
    FixedPathFamily.AUDITORY_JOINT_TO_SECOND_SOURCE: 612_360,
    FixedPathFamily.ACTION_RETURN_TO_JOINT: 580,
    FixedPathFamily.JOINT_TO_ACTION_ZERO: 1_224_608,
    FixedPathFamily.RECONSTRUCTION_TO_ACTION_ONE: 1_224_028,
    FixedPathFamily.ACTION_ZERO_TO_ACTION_TWO: 1_224_608,
    FixedPathFamily.ACTION_ONE_TO_ACTION_TWO: 1_224_028,
    FixedPathFamily.VISUAL_GRAYSCALE_ADMISSION: 1_135_956,
    FixedPathFamily.VISUAL_FIRST_SOURCE_TO_JOINT: 611_668,
    FixedPathFamily.REAL_VISUAL_CROSS_BRANCH_TO_RGB_CONTACT: 2_522_880,
    FixedPathFamily.VISUAL_RGB_CONTACT_TO_JOINT: 3 * 611_668,
    FixedPathFamily.ACTION_FORMATION_TO_MOTION_OUTPUT: 580,
    FixedPathFamily.REAL_AUDITORY_TO_ASSOCIATION_CONTACT: 9_225,
}

# 当前理论能够直接确定端点的包括统一器官入口、生命值/饱食度本能通道、
# 真实视觉和听觉从逐项独立继续位置形成的两条同形分支，以及已经继续
# 批准的十段视觉还原内部排列。其余类别
# 保存的是此前推进出的分节通道草案；它们具有真实、稳定、可观测的端点，
# 但其中的轮流分配、最近位置对应、等序号配对等具体连接规则尚未由
# 冻结理论唯一确定。
# “已生成”不能在正式出生检查中被误写成“已经确认”。
CONFIRMED_FIXED_PATH_FAMILIES = (
    FixedPathFamily.ORGAN_ENTRANCE,
    FixedPathFamily.VITAL_STATE,
    FixedPathFamily.ORGAN_CONTINUATION,
    FixedPathFamily.VISUAL_RECONSTRUCTION,
    FixedPathFamily.AUDITORY_OUTPUT,
    FixedPathFamily.REAL_VISUAL_TO_RECONSTRUCTION_BRANCH,
    FixedPathFamily.REAL_VISUAL_TO_CROSS_ORGANIZATION_BRANCH,
    FixedPathFamily.REAL_AUDITORY_TO_RECONSTRUCTION_BRANCH,
    FixedPathFamily.REAL_AUDITORY_TO_CROSS_ORGANIZATION_BRANCH,
    FixedPathFamily.VISUAL_GRAYSCALE_ADMISSION,
    FixedPathFamily.VISUAL_SOURCES,
    FixedPathFamily.PREDICTED_VISUAL_TO_JOINT,
    FixedPathFamily.VISUAL_JOINT_TO_SECOND_SOURCE,
    FixedPathFamily.PREDICTED_AUDITORY_TO_JOINT,
    FixedPathFamily.ACTION_RETURN_TO_JOINT,
    FixedPathFamily.VISUAL_FIRST_SOURCE_TO_JOINT,
    FixedPathFamily.REAL_VISUAL_CROSS_BRANCH_TO_RGB_CONTACT,
    FixedPathFamily.VISUAL_RGB_CONTACT_TO_JOINT,
    FixedPathFamily.JOINT_TO_ACTION_ZERO,
    FixedPathFamily.RECONSTRUCTION_TO_ACTION_ONE,
    FixedPathFamily.ACTION_ZERO_TO_ACTION_TWO,
    FixedPathFamily.ACTION_ONE_TO_ACTION_TWO,
    FixedPathFamily.ACTION_FORMATION_TO_MOTION_OUTPUT,
    FixedPathFamily.REAL_AUDITORY_TO_ASSOCIATION_CONTACT,
)
CONFIRMED_PARTIAL_FIXED_PATH_ENDPOINT_COUNTS: dict[FixedPathFamily, int] = {}
DRAFT_FIXED_PATH_FAMILIES = tuple(
    family for family in FixedPathFamily if family not in CONFIRMED_FIXED_PATH_FAMILIES
)
CONFIRMED_FIXED_PATH_ENDPOINT_COUNT = sum(
    FIXED_PATH_FAMILY_COUNTS[family] for family in CONFIRMED_FIXED_PATH_FAMILIES
) + sum(CONFIRMED_PARTIAL_FIXED_PATH_ENDPOINT_COUNTS.values())
DRAFT_FIXED_PATH_ENDPOINT_COUNT = sum(
    FIXED_PATH_FAMILY_COUNTS[family] for family in DRAFT_FIXED_PATH_FAMILIES
) - sum(CONFIRMED_PARTIAL_FIXED_PATH_ENDPOINT_COUNTS.values())

# 第二次实验只让真实听觉活动到达联系组织。下面六类属于以后完整模型的
# 听觉还原/预测草案，本次个体出生时不编译，也不要求声音输出器官存在。
# 这只是本次实验的出生范围，不修改完整项目理论。
SECOND_EXPERIMENT_INACTIVE_AUDITORY_FAMILIES = (
    FixedPathFamily.AUDITORY_RECONSTRUCTION,
    FixedPathFamily.AUDITORY_OUTPUT,
    FixedPathFamily.AUDITORY_SOURCES,
    FixedPathFamily.REAL_AUDITORY_TO_RECONSTRUCTION_BRANCH,
    FixedPathFamily.PREDICTED_AUDITORY_TO_JOINT,
    FixedPathFamily.AUDITORY_JOINT_TO_SECOND_SOURCE,
)
SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_FAMILIES = tuple(
    family
    for family in FixedPathFamily
    if family not in SECOND_EXPERIMENT_INACTIVE_AUDITORY_FAMILIES
)
SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_COUNT = sum(
    FIXED_PATH_FAMILY_COUNTS[family]
    for family in SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_FAMILIES
)

# 器官固定接收端、普通入口、逐项独立继续位置和两条真实感受同形分支
# 均属于透明传播段，必须完整传播；生命状态本能片段中的固定路径也已经
# 逐条保存强度。
CONFIRMED_FIXED_PATH_STRENGTH_FAMILIES = (
    FixedPathFamily.ORGAN_ENTRANCE,
    FixedPathFamily.VITAL_STATE,
    FixedPathFamily.ORGAN_CONTINUATION,
    FixedPathFamily.VISUAL_RECONSTRUCTION,
    FixedPathFamily.AUDITORY_OUTPUT,
    FixedPathFamily.REAL_VISUAL_TO_RECONSTRUCTION_BRANCH,
    FixedPathFamily.REAL_VISUAL_TO_CROSS_ORGANIZATION_BRANCH,
    FixedPathFamily.REAL_AUDITORY_TO_RECONSTRUCTION_BRANCH,
    FixedPathFamily.REAL_AUDITORY_TO_CROSS_ORGANIZATION_BRANCH,
    FixedPathFamily.VISUAL_GRAYSCALE_ADMISSION,
    FixedPathFamily.VISUAL_SOURCES,
    FixedPathFamily.PREDICTED_VISUAL_TO_JOINT,
    FixedPathFamily.VISUAL_JOINT_TO_SECOND_SOURCE,
    FixedPathFamily.PREDICTED_AUDITORY_TO_JOINT,
    FixedPathFamily.ACTION_RETURN_TO_JOINT,
    FixedPathFamily.VISUAL_FIRST_SOURCE_TO_JOINT,
    FixedPathFamily.REAL_VISUAL_CROSS_BRANCH_TO_RGB_CONTACT,
    FixedPathFamily.VISUAL_RGB_CONTACT_TO_JOINT,
    FixedPathFamily.JOINT_TO_ACTION_ZERO,
    FixedPathFamily.RECONSTRUCTION_TO_ACTION_ONE,
    FixedPathFamily.ACTION_ZERO_TO_ACTION_TWO,
    FixedPathFamily.ACTION_ONE_TO_ACTION_TWO,
    FixedPathFamily.ACTION_FORMATION_TO_MOTION_OUTPUT,
    FixedPathFamily.REAL_AUDITORY_TO_ASSOCIATION_CONTACT,
)
CONFIRMED_PARTIAL_FIXED_PATH_STRENGTH_COUNTS: dict[FixedPathFamily, int] = {}
CONFIRMED_FIXED_PATH_STRENGTH_COUNT = sum(
    FIXED_PATH_FAMILY_COUNTS[family]
    for family in CONFIRMED_FIXED_PATH_STRENGTH_FAMILIES
) + sum(CONFIRMED_PARTIAL_FIXED_PATH_STRENGTH_COUNTS.values())


# 这是出生结构生成阶段的临时端点格式，不是正式 Path 状态格式。
FIXED_PATH_ENDPOINT_DTYPE = numpy.dtype(
    [("source_neuron", "<u4"), ("target_neuron", "<u4")]
)


@dataclass(frozen=True, slots=True)
class FixedPathEndpointBatch:
    family: FixedPathFamily
    family_offset: int
    global_offset: int
    endpoints: numpy.ndarray


def _index(address: TissueAddress) -> int:
    return linear_index(address.x, address.y, address.z)


def _prefixes(counts: tuple[int, ...]) -> tuple[int, ...]:
    result: list[int] = []
    current = 0
    for count in counts:
        result.append(current)
        current += count
    return tuple(result)


VISUAL_SELECTED_PREFIXES = _prefixes(VISUAL_SELECTED_COUNTS)
AUDITORY_SELECTED_PREFIXES = _prefixes(AUDITORY_SELECTED_COUNTS)


def iter_visual_selected_positions() -> Iterator[tuple[int, int, int, int]]:
    """按段、纵向、横向顺序给出(总编号、段、x、y)。"""

    selected_index = 0
    for section, spacing in enumerate(VISUAL_RECONSTRUCTION_SPACINGS):
        for y in range(0, 512, spacing):
            for x in range(0, 512, spacing):
                yield selected_index, section, x, y
                selected_index += 1


def visual_selected_index(section: int, x: int, y: int) -> int:
    section = int(section)
    x = int(x)
    y = int(y)
    if not 0 <= section < len(VISUAL_RECONSTRUCTION_SPACINGS):
        raise ValueError("视觉接入段编号必须处于0到9")
    spacing = VISUAL_RECONSTRUCTION_SPACINGS[section]
    if not 0 <= x < 512 or not 0 <= y < 512:
        raise ValueError("视觉接入位置超出512×512")
    if x % spacing or y % spacing:
        raise ValueError("视觉位置不属于该还原段的接入位置")
    width = (512 + spacing - 1) // spacing
    return VISUAL_SELECTED_PREFIXES[section] + (y // spacing) * width + x // spacing


def iter_auditory_selected_positions(
) -> Iterator[tuple[int, int, int, int, int, int]]:
    """按段、声音流、频率、排列位置、三项活动给出选定位置。"""

    selected_index = 0
    for section, spacing in enumerate(AUDITORY_RECONSTRUCTION_SPACINGS):
        for stream in range(3):
            for frequency in range(0, 342, spacing):
                for sequence in range(0, 85, spacing):
                    for component in range(3):
                        yield (
                            selected_index,
                            section,
                            stream,
                            frequency,
                            sequence,
                            component,
                        )
                        selected_index += 1


def auditory_selected_index(
    section: int,
    stream: int,
    frequency: int,
    sequence: int,
    component: int,
) -> int:
    section = int(section)
    stream = int(stream)
    frequency = int(frequency)
    sequence = int(sequence)
    component = int(component)
    if not 0 <= section < len(AUDITORY_RECONSTRUCTION_SPACINGS):
        raise ValueError("听觉接入段编号必须处于0到9")
    spacing = AUDITORY_RECONSTRUCTION_SPACINGS[section]
    if not 0 <= stream < 3 or not 0 <= component < 3:
        raise ValueError("听觉流或三项非负活动编号无效")
    if not 0 <= frequency < 342 or not 0 <= sequence < 85:
        raise ValueError("听觉还原位置超出范围")
    if frequency % spacing or sequence % spacing:
        raise ValueError("听觉位置不属于该还原段的接入位置")
    frequency_count = (342 + spacing - 1) // spacing
    sequence_count = (85 + spacing - 1) // spacing
    local = (
        ((stream * frequency_count + frequency // spacing) * sequence_count
         + sequence // spacing)
        * 3
        + component
    )
    return AUDITORY_SELECTED_PREFIXES[section] + local


def _joint_order_address(index: int) -> TissueAddress:
    value = int(index)
    if not 0 <= value < ACTION_FORMATION_SECTION_COUNT:
        raise ValueError("共同结晶顺序编号超出范围")
    if value < VISUAL_JOINT_COUNT:
        return joint_visual_address(visual_action_formation_index(value))
    if value < SENSORY_ACTION_FORMATION_COUNT:
        return joint_auditory_address(value - VISUAL_JOINT_COUNT)
    repetition, action = divmod(value - SENSORY_ACTION_FORMATION_COUNT, 116)
    if returned_action_formation_index(action, repetition) != value:
        raise AssertionError("动作返回编号与共同结晶位置不一致")
    return joint_action_address(action, repetition)


class FixedPathTopology:
    """以流式方式产生全部固定神经元路径的来源—到达端点。"""

    def __init__(
        self,
        instinct: FixedBirthFragment,
        *,
        entrances: OrganEntranceLayout = SECOND_EXPERIMENT_ENTRANCES,
    ) -> None:
        self.instinct = instinct
        self.entrances = entrances
        self._instinct_by_name = {value.name: value for value in instinct.neurons}
        self._instinct_shared_edges = {
            edge
            for binding in instinct.visual_bindings
            for edge in (
                (binding.receiver_neuron, binding.ordinary_neuron),
                (binding.ordinary_neuron, binding.continuation_neuron),
            )
        }
        self._validate_instinct()

    @property
    def path_count(self) -> int:
        return sum(FIXED_PATH_FAMILY_COUNTS.values())

    @property
    def second_experiment_path_count(self) -> int:
        """当前实验真正出生的固定路径数，不含听觉还原和发声前置草案。"""

        return SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_COUNT

    @property
    def emotion_control_path_count(self) -> int:
        """到情绪器官控制入口的固定短路径数。

        这批路径的到达端不是神经元地址，而是情绪器官的控制入口，因此不能
        混进神经元—神经元固定路径中；它们仍属于同一个出生
        结构，并且必须和其他 Path 一起保存当前传播活动。
        """

        return len(self.instinct.emotion_bindings)

    def _validate_instinct(self) -> None:
        remaining = sum(
            (path.source_neuron, path.target_neuron)
            not in self._instinct_shared_edges
            for path in self.instinct.paths
        )
        if remaining != FIXED_PATH_FAMILY_COUNTS[FixedPathFamily.VITAL_STATE]:
            raise ValueError("生命状态本能的非入口固定路径数量与全脑总账不一致")

    def _organ_entrances(self) -> Iterator[tuple[int, int]]:
        for group in self.entrances.ranges:
            for activity_index in range(group.activity_count):
                pair = self.entrances.pair(group.name, activity_index)
                yield _index(pair.receiver), _index(pair.ordinary)

    def iter_confirmed_strengths(
        self,
        family: FixedPathFamily,
    ) -> Iterator[float]:
        """给出已经由出生结构明确的神经元—神经元固定路径强度。

        返回顺序与对应连接类别的端点顺序一致。未确定类别不会用默认值
        填充，而是直接拒绝读取。
        """

        family = FixedPathFamily(family)
        if family is FixedPathFamily.ORGAN_ENTRANCE:
            return (
                1.0
                for _ in range(
                    FIXED_PATH_FAMILY_COUNTS[FixedPathFamily.ORGAN_ENTRANCE]
                )
            )
        if family is FixedPathFamily.VITAL_STATE:
            return (
                float(path.path_strength)
                for path in self.instinct.paths
                if (path.source_neuron, path.target_neuron)
                not in self._instinct_shared_edges
            )
        if family is FixedPathFamily.ORGAN_CONTINUATION:
            return (
                1.0
                for _ in range(
                    FIXED_PATH_FAMILY_COUNTS[FixedPathFamily.ORGAN_CONTINUATION]
                )
            )
        if family is FixedPathFamily.AUDITORY_OUTPUT:
            return (
                1.0
                for _ in range(
                    FIXED_PATH_FAMILY_COUNTS[FixedPathFamily.AUDITORY_OUTPUT]
                )
            )
        if family is FixedPathFamily.VISUAL_RECONSTRUCTION:
            return (
                1.0
                for _ in range(
                    FIXED_PATH_FAMILY_COUNTS[
                        FixedPathFamily.VISUAL_RECONSTRUCTION
                    ]
                )
            )
        if family in {
            FixedPathFamily.REAL_VISUAL_TO_RECONSTRUCTION_BRANCH,
            FixedPathFamily.REAL_VISUAL_TO_CROSS_ORGANIZATION_BRANCH,
            FixedPathFamily.REAL_AUDITORY_TO_RECONSTRUCTION_BRANCH,
            FixedPathFamily.REAL_AUDITORY_TO_CROSS_ORGANIZATION_BRANCH,
        }:
            return (1.0 for _ in range(FIXED_PATH_FAMILY_COUNTS[family]))
        if family is FixedPathFamily.VISUAL_GRAYSCALE_ADMISSION:
            return (
                1.0
                for _ in range(
                    FIXED_PATH_FAMILY_COUNTS[
                        FixedPathFamily.VISUAL_GRAYSCALE_ADMISSION
                    ]
                )
            )
        if family in {
            FixedPathFamily.VISUAL_SOURCES,
            FixedPathFamily.PREDICTED_VISUAL_TO_JOINT,
            FixedPathFamily.VISUAL_JOINT_TO_SECOND_SOURCE,
            FixedPathFamily.PREDICTED_AUDITORY_TO_JOINT,
            FixedPathFamily.ACTION_RETURN_TO_JOINT,
            FixedPathFamily.VISUAL_FIRST_SOURCE_TO_JOINT,
            FixedPathFamily.REAL_VISUAL_CROSS_BRANCH_TO_RGB_CONTACT,
            FixedPathFamily.VISUAL_RGB_CONTACT_TO_JOINT,
            FixedPathFamily.JOINT_TO_ACTION_ZERO,
            FixedPathFamily.RECONSTRUCTION_TO_ACTION_ONE,
            FixedPathFamily.ACTION_ZERO_TO_ACTION_TWO,
            FixedPathFamily.ACTION_ONE_TO_ACTION_TWO,
            FixedPathFamily.ACTION_FORMATION_TO_MOTION_OUTPUT,
            FixedPathFamily.REAL_AUDITORY_TO_ASSOCIATION_CONTACT,
        }:
            return (1.0 for _ in range(FIXED_PATH_FAMILY_COUNTS[family]))
        raise ValueError("该固定连接类别的路径强度尚未由人工出生结构确定")

    def _vital_state(self) -> Iterator[tuple[int, int]]:
        for path in self.instinct.paths:
            if (
                path.source_neuron,
                path.target_neuron,
            ) in self._instinct_shared_edges:
                continue
            source = self._instinct_by_name[path.source_neuron].address
            target = self._instinct_by_name[path.target_neuron].address
            yield _index(source), _index(target)

    def _organ_continuations(self) -> Iterator[tuple[int, int]]:
        for group in self.entrances.ranges:
            for activity_index in range(group.activity_count):
                ordinary = self.entrances.pair(group.name, activity_index).ordinary
                continuation = organ_continuation_address(
                    group.activity_offset + activity_index
                )
                yield _index(ordinary), _index(continuation)

    def iter_emotion_control_paths(self) -> Iterator[tuple[int, int, float]]:
        """按出生片段原有顺序给出（来源神经元、控制入口、固定强度）。"""

        for binding in self.instinct.emotion_bindings:
            source = self._instinct_by_name[binding.source_neuron].address
            yield _index(source), int(binding.entrance), float(binding.path_strength)

    @staticmethod
    def _visual_reconstruction() -> Iterator[tuple[int, int]]:
        for section in range(1, 10):
            distance = 1 << (section - 1)
            for y in range(512):
                for x in range(512):
                    source = _index(visual_reconstruction_address(section, x, y))
                    for dx, dy in (
                        (0, 0),
                        (distance, 0),
                        (0, distance),
                        (distance, distance),
                    ):
                        target_x = x + dx
                        target_y = y + dy
                        if target_x < 512 and target_y < 512:
                            yield source, _index(
                                visual_reconstruction_address(
                                    section - 1,
                                    target_x,
                                    target_y,
                                )
                            )

    @staticmethod
    def _visual_sources() -> Iterator[tuple[int, int]]:
        for selected, section, x, y in iter_visual_selected_positions():
            target = _index(visual_reconstruction_address(section, x, y))
            yield _index(visual_source_address(0, selected)), target
            yield _index(visual_source_address(1, selected)), target

    @staticmethod
    def _visual_grayscale_admission() -> Iterator[tuple[int, int]]:
        # 每个512×512位置先由完整视觉还原分支中的R、G、B三项活动汇入
        # 第一来源组织的基础灰度位置。
        for y in range(512):
            for x in range(512):
                source_x, source_y = visual_full_field_sample_coordinate(x, y)
                target = _index(visual_source_address(0, y * 512 + x))
                pixel = source_y * 1280 + source_x
                for channel in range(3):
                    activity = pixel * 3 + channel
                    yield (
                        _index(visual_identity_branch_address(0, activity)),
                        target,
                    )

        # 基础灰度位置已经形成后，再逐项复制到其余九段实际需要的来源
        # 位置；这里没有第二次灰度汇合，也不重新读取原始RGB。
        for selected, section, x, y in iter_visual_selected_positions():
            if section == 0:
                continue
            base = y * 512 + x
            yield (
                _index(visual_source_address(0, base)),
                _index(visual_source_address(0, selected)),
            )

    @staticmethod
    def _auditory_reconstruction() -> Iterator[tuple[int, int]]:
        for section in range(9):
            for stream in range(3):
                for frequency in range(342):
                    for sequence in range(85):
                        for component in range(3):
                            source = _index(
                                auditory_reconstruction_address(
                                    section,
                                    stream,
                                    frequency,
                                    sequence,
                                    component,
                                )
                            )
                            targets = [(frequency, sequence)]
                            if frequency > 0:
                                targets.append((frequency - 1, sequence))
                            if frequency < 341:
                                targets.append((frequency + 1, sequence))
                            if sequence > 0:
                                targets.append((frequency, sequence - 1))
                            if sequence < 84:
                                targets.append((frequency, sequence + 1))
                            for target_frequency, target_sequence in targets:
                                yield source, _index(
                                    auditory_reconstruction_address(
                                        section + 1,
                                        stream,
                                        target_frequency,
                                        target_sequence,
                                        component,
                                    )
                                )

    @staticmethod
    def _auditory_output() -> Iterator[tuple[int, int]]:
        for stream in range(3):
            for frequency in range(342):
                for sequence in range(85):
                    for component in range(3):
                        yield (
                            _index(
                                auditory_reconstruction_address(
                                    9, stream, frequency, sequence, component
                                )
                            ),
                            _index(
                                predicted_auditory_output_address(
                                    stream, frequency, sequence, component
                                )
                            ),
                        )

    @staticmethod
    def _auditory_sources() -> Iterator[tuple[int, int]]:
        for selected, section, stream, frequency, sequence, component in (
            iter_auditory_selected_positions()
        ):
            target = _index(
                auditory_reconstruction_address(
                    section,
                    stream,
                    frequency,
                    sequence,
                    component,
                )
            )
            yield _index(auditory_source_address(0, selected)), target
            yield _index(auditory_source_address(1, selected)), target

    @staticmethod
    def _real_visual_identity_branch(branch: int) -> Iterator[tuple[int, int]]:
        for activity in range(2_522_880):
            yield (
                _index(visual_continuation_address(activity)),
                _index(visual_identity_branch_address(branch, activity)),
            )

    def _predicted_visual_to_joint(self) -> Iterator[tuple[int, int]]:
        for selected, _section, x, y in iter_visual_selected_positions():
            activity = y * 512 + x
            group = next(
                value for value in self.entrances.ranges
                if value.name == "predicted_visual"
            )
            source = _index(
                organ_continuation_address(group.activity_offset + activity)
            )
            yield source, _index(joint_visual_address(selected))

    @staticmethod
    def _visual_joint_to_second_source() -> Iterator[tuple[int, int]]:
        for selected in range(VISUAL_JOINT_COUNT):
            yield (
                _index(joint_visual_address(selected)),
                _index(visual_source_address(1, selected)),
            )

    @staticmethod
    def _visual_first_source_to_joint() -> Iterator[tuple[int, int]]:
        """真实灰度来源与预测回流在同一视觉结晶位置直接相加。"""

        for selected in range(VISUAL_JOINT_COUNT):
            yield (
                _index(visual_source_address(0, selected)),
                _index(joint_visual_address(selected)),
            )

    @staticmethod
    def _real_visual_cross_branch_to_rgb_contact() -> Iterator[tuple[int, int]]:
        """全部RGB第二支路逐项跨入共同结晶的同形接触部分。"""

        for activity in range(2_522_880):
            yield (
                _index(visual_identity_branch_address(1, activity)),
                _index(visual_rgb_joint_contact_address(activity)),
            )

    @staticmethod
    def _visual_rgb_contact_to_joint() -> Iterator[tuple[int, int]]:
        """已明确几何位置的R、G、B分别到达同一视觉结晶位置。"""

        for selected, _section, x, y in iter_visual_selected_positions():
            source_x, source_y = visual_full_field_sample_coordinate(x, y)
            first_activity = (source_y * 1280 + source_x) * 3
            target = _index(joint_visual_address(selected))
            for channel in range(3):
                yield (
                    _index(
                        visual_rgb_joint_contact_address(
                            first_activity + channel
                        )
                    ),
                    target,
                )

    @staticmethod
    def _real_auditory_identity_branch(branch: int) -> Iterator[tuple[int, int]]:
        for activity in range(9_225):
            yield (
                _index(auditory_continuation_address(activity)),
                _index(auditory_identity_branch_address(branch, activity)),
            )

    @staticmethod
    def _real_auditory_to_association_contact() -> Iterator[tuple[int, int]]:
        """第二条真实听觉支路逐项进入联系组织，不缩并或复制。"""

        for activity in range(9_225):
            yield (
                _index(auditory_identity_branch_address(1, activity)),
                _index(joint_real_auditory_contact_address(activity)),
            )

    def _predicted_auditory_to_joint(self) -> Iterator[tuple[int, int]]:
        for selected, _section, stream, frequency, sequence, component in (
            iter_auditory_selected_positions()
        ):
            activity = (
                ((stream * 342 + frequency) * 85 + sequence) * 3 + component
            )
            group = next(
                value for value in self.entrances.ranges
                if value.name == "predicted_auditory"
            )
            source = _index(
                organ_continuation_address(group.activity_offset + activity)
            )
            yield source, _index(joint_auditory_address(selected))

    @staticmethod
    def _auditory_joint_to_second_source() -> Iterator[tuple[int, int]]:
        for selected in range(AUDITORY_JOINT_COUNT):
            yield (
                _index(joint_auditory_address(selected)),
                _index(auditory_source_address(1, selected)),
            )

    def _action_return_to_joint(self) -> Iterator[tuple[int, int]]:
        action = 0
        for entrance in ("mouse", "keyboard", "view_center"):
            group = next(value for value in self.entrances.ranges if value.name == entrance)
            for activity in range(group.activity_count):
                source = _index(
                    organ_continuation_address(group.activity_offset + activity)
                )
                for repetition in range(5):
                    yield source, _index(joint_action_address(action, repetition))
                action += 1

    @staticmethod
    def _joint_to_action_zero() -> Iterator[tuple[int, int]]:
        for index in range(ACTION_FORMATION_SECTION_COUNT):
            yield (
                _index(_joint_order_address(index)),
                _index(action_formation_address(0, index)),
            )

    @staticmethod
    def _reconstruction_to_action_one() -> Iterator[tuple[int, int]]:
        for selected, visual_section, x, y in iter_visual_selected_positions():
            yield (
                _index(visual_reconstruction_address(visual_section, x, y)),
                _index(
                    action_formation_address(
                        1,
                        visual_action_formation_index(selected),
                    )
                ),
            )

        for selected, auditory_section, stream, frequency, sequence, component in (
            iter_auditory_selected_positions()
        ):
            yield (
                _index(
                    auditory_reconstruction_address(
                        auditory_section,
                        stream,
                        frequency,
                        sequence,
                        component,
                    )
                ),
                _index(
                    action_formation_address(
                        1,
                        auditory_action_formation_index(selected),
                    )
                ),
            )

    @staticmethod
    def _action_to_action(
        source_section: int,
        target_section: int,
        count: int,
    ) -> Iterator[tuple[int, int]]:
        for index in range(int(count)):
            yield (
                _index(action_formation_address(source_section, index)),
                _index(action_formation_address(target_section, index)),
            )

    @staticmethod
    def _action_formation_to_motion_output() -> Iterator[tuple[int, int]]:
        """116项动作的五个独立末端位置直接汇入对应输出神经元。"""

        for action in range(116):
            target = _index(motion_output_address(action))
            for repetition in range(5):
                source_index = returned_action_formation_index(action, repetition)
                yield (
                    _index(action_formation_address(2, source_index)),
                    target,
                )

    def iter_family(self, family: FixedPathFamily) -> Iterator[tuple[int, int]]:
        family = FixedPathFamily(family)
        if family is FixedPathFamily.ORGAN_ENTRANCE:
            return self._organ_entrances()
        if family is FixedPathFamily.VITAL_STATE:
            return self._vital_state()
        if family is FixedPathFamily.ORGAN_CONTINUATION:
            return self._organ_continuations()
        if family is FixedPathFamily.VISUAL_RECONSTRUCTION:
            return self._visual_reconstruction()
        if family is FixedPathFamily.VISUAL_SOURCES:
            return self._visual_sources()
        if family is FixedPathFamily.AUDITORY_RECONSTRUCTION:
            return self._auditory_reconstruction()
        if family is FixedPathFamily.AUDITORY_OUTPUT:
            return self._auditory_output()
        if family is FixedPathFamily.AUDITORY_SOURCES:
            return self._auditory_sources()
        if family is FixedPathFamily.REAL_VISUAL_TO_RECONSTRUCTION_BRANCH:
            return self._real_visual_identity_branch(0)
        if family is FixedPathFamily.REAL_VISUAL_TO_CROSS_ORGANIZATION_BRANCH:
            return self._real_visual_identity_branch(1)
        if family is FixedPathFamily.PREDICTED_VISUAL_TO_JOINT:
            return self._predicted_visual_to_joint()
        if family is FixedPathFamily.VISUAL_JOINT_TO_SECOND_SOURCE:
            return self._visual_joint_to_second_source()
        if family is FixedPathFamily.REAL_AUDITORY_TO_RECONSTRUCTION_BRANCH:
            return self._real_auditory_identity_branch(0)
        if family is FixedPathFamily.REAL_AUDITORY_TO_CROSS_ORGANIZATION_BRANCH:
            return self._real_auditory_identity_branch(1)
        if family is FixedPathFamily.PREDICTED_AUDITORY_TO_JOINT:
            return self._predicted_auditory_to_joint()
        if family is FixedPathFamily.AUDITORY_JOINT_TO_SECOND_SOURCE:
            return self._auditory_joint_to_second_source()
        if family is FixedPathFamily.ACTION_RETURN_TO_JOINT:
            return self._action_return_to_joint()
        if family is FixedPathFamily.JOINT_TO_ACTION_ZERO:
            return self._joint_to_action_zero()
        if family is FixedPathFamily.RECONSTRUCTION_TO_ACTION_ONE:
            return self._reconstruction_to_action_one()
        if family is FixedPathFamily.ACTION_ZERO_TO_ACTION_TWO:
            return self._action_to_action(0, 2, ACTION_FORMATION_SECTION_COUNT)
        if family is FixedPathFamily.ACTION_ONE_TO_ACTION_TWO:
            return self._action_to_action(
                1,
                2,
                SENSORY_ACTION_FORMATION_COUNT,
            )
        if family is FixedPathFamily.VISUAL_GRAYSCALE_ADMISSION:
            return self._visual_grayscale_admission()
        if family is FixedPathFamily.VISUAL_FIRST_SOURCE_TO_JOINT:
            return self._visual_first_source_to_joint()
        if family is FixedPathFamily.REAL_VISUAL_CROSS_BRANCH_TO_RGB_CONTACT:
            return self._real_visual_cross_branch_to_rgb_contact()
        if family is FixedPathFamily.VISUAL_RGB_CONTACT_TO_JOINT:
            return self._visual_rgb_contact_to_joint()
        if family is FixedPathFamily.ACTION_FORMATION_TO_MOTION_OUTPUT:
            return self._action_formation_to_motion_output()
        if family is FixedPathFamily.REAL_AUDITORY_TO_ASSOCIATION_CONTACT:
            return self._real_auditory_to_association_contact()
        raise AssertionError("固定连接类别分派不完整")  # pragma: no cover

    def _iter_selected_batches(
        self,
        families: tuple[FixedPathFamily, ...],
        *,
        batch_size: int,
    ) -> Iterator[FixedPathEndpointBatch]:
        """按选定出生范围流式给出端点，不把未选草案混入正式个体。"""

        size = int(batch_size)
        if size <= 0:
            raise ValueError("固定路径端点批次大小必须大于零")
        global_offset = 0
        for family in families:
            family_offset = 0
            buffer = numpy.empty(size, dtype=FIXED_PATH_ENDPOINT_DTYPE)
            used = 0
            for source, target in self.iter_family(family):
                if not (
                    0 <= source < TISSUE_NEURON_COUNT
                    and 0 <= target < TISSUE_NEURON_COUNT
                ):
                    raise ValueError("固定路径端点超出完整神经组织")
                if source == target:
                    raise ValueError("固定路径不能把同一神经元直接连回自身")
                buffer[used] = source, target
                used += 1
                if used == size:
                    yield FixedPathEndpointBatch(
                        family, family_offset, global_offset, buffer
                    )
                    family_offset += used
                    global_offset += used
                    buffer = numpy.empty(size, dtype=FIXED_PATH_ENDPOINT_DTYPE)
                    used = 0
            if used:
                yield FixedPathEndpointBatch(
                    family,
                    family_offset,
                    global_offset,
                    buffer[:used].copy(),
                )
                family_offset += used
                global_offset += used
            expected = FIXED_PATH_FAMILY_COUNTS[family]
            if family_offset != expected:
                raise RuntimeError(
                    f"固定连接类别{family.value}实际产生{family_offset}条，"
                    f"但总账为{expected}条"
                )
        expected_total = sum(FIXED_PATH_FAMILY_COUNTS[family] for family in families)
        if global_offset != expected_total:
            raise RuntimeError("选定出生范围的固定路径端点数量与总账不一致")

    def iter_batches(self, batch_size: int = 262_144) -> Iterator[FixedPathEndpointBatch]:
        """按连接类别流式给出全部端点，包括为以后保留的结构草案。"""

        yield from self._iter_selected_batches(
            tuple(FixedPathFamily),
            batch_size=batch_size,
        )
        if self.path_count != REALLOCATED_FIXED_NEURON_PATH_COUNT:
            raise RuntimeError("全部固定路径端点数量与全脑总账不一致")

    def iter_second_experiment_batches(
        self,
        batch_size: int = 262_144,
    ) -> Iterator[FixedPathEndpointBatch]:
        """只产生第二次实验个体实际出生的固定路径端点。"""

        yield from self._iter_selected_batches(
            SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_FAMILIES,
            batch_size=batch_size,
        )


def validate_fixed_path_count_ledger() -> None:
    if sum(VISUAL_SELECTED_COUNTS) != VISUAL_SOURCE_PER_SIDE:
        raise RuntimeError("视觉还原接入位置数量与双来源组织不一致")
    if sum(AUDITORY_SELECTED_COUNTS) != AUDITORY_SOURCE_PER_SIDE:
        raise RuntimeError("听觉还原接入位置数量与双来源组织不一致")
    if VISUAL_JOINT_COUNT + AUDITORY_JOINT_COUNT + ACTION_JOINT_COUNT != (
        ACTION_FORMATION_SECTION_COUNT
    ):
        raise RuntimeError("共同结晶组织与动作形成段大小不一致")
    if FIXED_PATH_FAMILY_COUNTS[FixedPathFamily.JOINT_TO_ACTION_ZERO] != (
        ACTION_FORMATION_SECTION_COUNT
    ):
        raise RuntimeError("第一动作形成组织没有逐项接收全部共同结晶位置")
    if FIXED_PATH_FAMILY_COUNTS[
        FixedPathFamily.RECONSTRUCTION_TO_ACTION_ONE
    ] != SENSORY_ACTION_FORMATION_COUNT:
        raise RuntimeError("第二动作形成组织的视觉听觉来源数量不一致")
    if FIXED_PATH_FAMILY_COUNTS[
        FixedPathFamily.ACTION_ZERO_TO_ACTION_TWO
    ] != ACTION_FORMATION_SECTION_COUNT:
        raise RuntimeError("第一动作形成组织没有逐项到达第三组织")
    if FIXED_PATH_FAMILY_COUNTS[
        FixedPathFamily.ACTION_ONE_TO_ACTION_TWO
    ] != SENSORY_ACTION_FORMATION_COUNT:
        raise RuntimeError("第二动作形成组织不得为580项动作返回伪造来源")
    if sum(FIXED_PATH_FAMILY_COUNTS.values()) != (
        REALLOCATED_FIXED_NEURON_PATH_COUNT
    ):
        raise RuntimeError("固定路径分项与全脑总账不一致")
    if CONFIRMED_FIXED_PATH_STRENGTH_COUNT != 33_989_824:
        raise RuntimeError(
            "已确定固定路径强度数量与透明器官通道、本能、真实感受分支和视觉还原不一致"
        )
    if CONFIRMED_FIXED_PATH_ENDPOINT_COUNT != 33_989_824:
        raise RuntimeError("已确认固定路径端点没有包含完整视觉还原与共同结晶入口")
    if CONFIRMED_FIXED_PATH_ENDPOINT_COUNT + DRAFT_FIXED_PATH_ENDPOINT_COUNT != (
        REALLOCATED_FIXED_NEURON_PATH_COUNT
    ):
        raise RuntimeError("已确认和待确认固定路径端点没有恰好分割完整总账")
    if SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_COUNT != 33_106_609:
        raise RuntimeError("第二次实验固定路径出生范围数量发生变化")
    if not set(SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_FAMILIES).issubset(
        CONFIRMED_FIXED_PATH_STRENGTH_FAMILIES
    ):
        raise RuntimeError("第二次实验出生范围中混入尚未确认强度的固定路径")


validate_fixed_path_count_ledger()
