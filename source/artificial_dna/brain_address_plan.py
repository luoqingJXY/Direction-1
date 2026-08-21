"""第二次实验已经确认部分的唯一全脑地址分配。"""

from __future__ import annotations

from dataclasses import dataclass

from .birth_structure import FixedBirthFragment
from .brain_geometry import (
    LinearAddressRange,
    TISSUE_NEURON_COUNT,
    TISSUE_PLANE,
    linear_index,
)
from .organ_entrances import OrganEntranceLayout, SECOND_EXPERIMENT_ENTRANCES


VISUAL_RECONSTRUCTION_COUNT = 10 * 512 * 512
VISUAL_RECONSTRUCTION_SOURCE_COUNT = 2 * 611_668
PREDICTED_AUDITORY_OUTPUT_COUNT = 261_630
MOTION_OUTPUT_COUNT = 4 + 108 + 4
AUDITORY_RECONSTRUCTION_COUNT = 10 * 261_630
AUDITORY_RECONSTRUCTION_SOURCE_COUNT = 2 * 612_360
REAL_AUDITORY_ASSOCIATION_CONTACT_COUNT = 9_225
# 联系组织保留原有视觉、预测听觉和动作接触位置，并追加9,225个真实
# 听觉逐项接触位置。动作形成三段仍只覆盖此前已经确认的1,224,608项；
# 新听觉接触依靠同一普通组织内的相邻可变路径参与后天联合，不伪造一条
# 先天动作含义路径。
ACTION_FORMATION_SECTION_COUNT = 611_668 + 612_360 + 580
JOINT_CRYSTALLIZATION_COUNT = (
    ACTION_FORMATION_SECTION_COUNT + REAL_AUDITORY_ASSOCIATION_CONTACT_COUNT
)
ORGAN_CONTINUATION_COUNT = 3_055_995
VISUAL_IDENTITY_BRANCH_COUNT = 2 * 2_522_880
AUDITORY_IDENTITY_BRANCH_COUNT = 2 * 9_225
VISUAL_RGB_JOINT_CONTACT_COUNT = 2_522_880

# 地址层次只承担物理落位。器官普通入口之后先保留一层逐项独立继续组织，
# 原有还原、结晶与动作草案整体后移，不能覆盖或绕过这层关系。
ORGAN_CONTINUATION_START_Z = 19
ORGAN_CONTINUATION_Z_OFFSET = 18
VISUAL_RECONSTRUCTION_START_Z = 37
VISUAL_SOURCE_START_Z = 47
PREDICTED_AUDITORY_OUTPUT_Z = 51
MOTION_OUTPUT_Z = 52
AUDITORY_RECONSTRUCTION_START_Z = 53
AUDITORY_SOURCE_START_Z = 63
JOINT_CRYSTALLIZATION_START_Z = 67
ACTION_FORMATION_START_Z = 69
VISUAL_FIRST_BRANCH_START_Z = 76
VISUAL_SECOND_BRANCH_START_Z = 86
AUDITORY_FIRST_BRANCH_START_Z = 96
AUDITORY_SECOND_BRANCH_START_Z = 98
VISUAL_RGB_JOINT_CONTACT_START_Z = 99
UNASSIGNED_START_Z = 108

# 只计算神经元到神经元的固定路径记录。器官绑定另存，不混入该文件。
# 旧500宽组织为了折叠视觉和听觉而添加的183720条视觉片缝、61560条
# 听觉分页缝在800宽新落位中不再存在，不能重复计入。
REALLOCATED_FIXED_NEURON_PATH_COUNT = (
    3_055_995  # 全部器官固定接收端到普通入口
    + 1_760  # 生命状态本能在复用视觉双入口之后的固定路径
    + ORGAN_CONTINUATION_COUNT  # 每项器官活动从普通入口逐项独立继续
    + 8_478_037  # 十段视觉还原之间
    + 1_223_336  # 视觉双来源到还原接入位置
    + 11_704_176  # 十段听觉还原之间
    + 261_630  # 最后一段听觉还原到预测听觉输出
    + 1_224_720  # 听觉双来源到还原接入位置
    + 7_512_846  # 真实视觉双支路及预测、听觉、动作回流的草案关系
    + 4_897_272  # 同模态结晶与动作形成；无对应来源的580项不伪造第二输入
    + 1_135_956  # 真实RGB还原分支到视觉第一来源组织的灰度接入和多尺度复制
    + 611_668  # 视觉第一来源逐项到达共同结晶视觉位置
    + VISUAL_RGB_JOINT_CONTACT_COUNT  # 视觉第二支路逐项进入共同结晶RGB接触部分
    + 3 * 611_668  # 已选中视觉位置的R、G、B三路到达对应共同结晶位置
    + 580  # 116项鼠标、键盘和视野中心动作各五个末端位置到输出神经元
    + REAL_AUDITORY_ASSOCIATION_CONTACT_COUNT  # 真实听觉第二支路逐项进入联系组织
)


@dataclass(frozen=True, slots=True)
class BrainAddressReservation:
    """一段互不重叠的物理组织体积，以及其中已经明确用途的位置数。"""

    name: str
    start: int
    stop: int
    neuron_count: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("全脑物理区间必须有名称")
        if not 0 <= self.start < self.stop <= TISSUE_NEURON_COUNT:
            raise ValueError("全脑物理区间超出完整神经组织")
        if not 0 <= self.neuron_count <= self.capacity_count:
            raise ValueError("结构神经元数量超过其物理组织体积")

    @property
    def count(self) -> int:
        return self.neuron_count

    @property
    def capacity_count(self) -> int:
        return self.stop - self.start

    @property
    def uncommitted_count(self) -> int:
        return self.capacity_count - self.neuron_count


@dataclass(frozen=True, slots=True)
class BrainAddressPlan:
    """区分冻结边界与已推进但仍需微观参数确认的人工出生结构草案。"""

    confirmed: tuple[BrainAddressReservation, ...]
    dna_draft: tuple[BrainAddressReservation, ...]
    unassigned: LinearAddressRange

    def __post_init__(self) -> None:
        allocated = (*self.confirmed, *self.dna_draft)
        ordered = sorted(allocated, key=lambda value: value.start)
        if tuple(ordered) != allocated:
            raise ValueError("全脑地址区间必须按线性地址递增排列")
        expected_start = 0
        names: set[str] = set()
        for value in ordered:
            if value.name in names:
                raise ValueError("全脑地址分配中存在重复名称")
            names.add(value.name)
            if value.start != expected_start:
                raise ValueError("已确认地址之间存在空洞或重叠")
            expected_start = value.stop
        if self.unassigned.start != expected_start:
            raise ValueError("未分配空间必须紧接全部已确认结构")
        if self.unassigned.stop != TISSUE_NEURON_COUNT:
            raise ValueError("全脑地址分配没有覆盖完整神经组织")

    @property
    def confirmed_neuron_count(self) -> int:
        return sum(value.neuron_count for value in self.confirmed)

    @property
    def dna_draft_neuron_count(self) -> int:
        return sum(value.neuron_count for value in self.dna_draft)

    @property
    def allocated_neuron_count(self) -> int:
        return self.confirmed_neuron_count + self.dna_draft_neuron_count

    @property
    def next_unreserved_index(self) -> int:
        return self.unassigned.start

    @property
    def reserved_capacity_count(self) -> int:
        return sum(value.capacity_count for value in (*self.confirmed, *self.dna_draft))

    @property
    def uncommitted_inside_reserved_count(self) -> int:
        return sum(value.uncommitted_count for value in (*self.confirmed, *self.dna_draft))

    @property
    def future_neuron_count(self) -> int:
        return self.uncommitted_inside_reserved_count + self.unassigned.count

    def range_named(self, name: str) -> BrainAddressReservation | LinearAddressRange:
        for value in (*self.confirmed, *self.dna_draft, self.unassigned):
            if value.name == name:
                return value
        raise ValueError("未知全脑地址区间")


def _z_reservation(
    name: str,
    start_z: int,
    stop_z: int,
    neuron_count: int,
) -> BrainAddressReservation:
    return BrainAddressReservation(
        name,
        start_z * TISSUE_PLANE,
        stop_z * TISSUE_PLANE,
        neuron_count,
    )


def _organ_ranges(layout: OrganEntranceLayout) -> list[BrainAddressReservation]:
    by_name = {value.name: value.activity_count for value in layout.ranges}
    return [
        _z_reservation("organ.visual.volume", 0, 10, 2 * by_name["visual"]),
        _z_reservation("organ.auditory.volume", 10, 12, 2 * by_name["auditory"]),
        _z_reservation(
            "organ.predicted_visual.volume",
            12,
            14,
            2 * by_name["predicted_visual"],
        ),
        _z_reservation(
            "organ.predicted_auditory.volume",
            14,
            16,
            2 * by_name["predicted_auditory"],
        ),
        _z_reservation(
            "organ.mouse_keyboard_view_center.volume",
            16,
            18,
            2 * (by_name["mouse"] + by_name["keyboard"] + by_name["view_center"]),
        ),
    ]


def build_brain_address_plan(
    instinct: FixedBirthFragment,
    *,
    entrances: OrganEntranceLayout = SECOND_EXPERIMENT_ENTRANCES,
) -> BrainAddressPlan:
    """把器官双入口和生命状态本能固定通道排入同一地址空间。"""

    organ_ranges = _organ_ranges(entrances)
    if organ_ranges[-1].stop != entrances.next_free_index:
        raise ValueError("器官入口区间与器官入口总量不一致")

    # 本能片段会引用已经存在的视觉双入口；这里逐项反查，防止片段内部
    # 另外分配一套同名入口，或把固定接收端与普通入口端接反。
    by_name = {neuron.name: neuron for neuron in instinct.neurons}
    path_by_edge = {
        (path.source_neuron, path.target_neuron): path for path in instinct.paths
    }
    shared_organ_neurons: set[str] = set()
    for binding in instinct.visual_bindings:
        expected_pair = entrances.visual_pair(binding.receptor)
        receiver = by_name[binding.receiver_neuron]
        ordinary = by_name[binding.ordinary_neuron]
        continuation = by_name[binding.continuation_neuron]
        shared_organ_neurons.update(
            (
                binding.receiver_neuron,
                binding.ordinary_neuron,
                binding.continuation_neuron,
            )
        )
        receiver_index = linear_index(
            receiver.address.x,
            receiver.address.y,
            receiver.address.z,
        )
        if receiver_index != linear_index(
            expected_pair.receiver.x,
            expected_pair.receiver.y,
            expected_pair.receiver.z,
        ):
            raise ValueError("本能片段复用的视觉固定接收地址与全脑入口不一致")
        ordinary_index = linear_index(
            ordinary.address.x,
            ordinary.address.y,
            ordinary.address.z,
        )
        if ordinary_index != linear_index(
            expected_pair.ordinary.x,
            expected_pair.ordinary.y,
            expected_pair.ordinary.z,
        ):
            raise ValueError("本能片段复用的视觉普通入口地址与全脑入口不一致")
        continuation_index = linear_index(
            continuation.address.x,
            continuation.address.y,
            continuation.address.z,
        )
        expected_continuation_index = (
            ordinary_index + ORGAN_CONTINUATION_Z_OFFSET * TISSUE_PLANE
        )
        if continuation_index != expected_continuation_index:
            raise ValueError("本能片段复用的视觉独立继续地址与全脑结构不一致")
        for edge in (
            (binding.receiver_neuron, binding.ordinary_neuron),
            (binding.ordinary_neuron, binding.continuation_neuron),
        ):
            try:
                path = path_by_edge[edge]
            except KeyError as exc:  # pragma: no cover - 片段自身已经验证
                raise ValueError("本能片段没有保存完整器官入口固定路径") from exc
            if float(path.path_strength) != 1.0:
                raise ValueError("本能片段复用的器官入口通道没有完整传播活动")

    allocated = sorted(
        {
            linear_index(neuron.address.x, neuron.address.y, neuron.address.z)
            for neuron in instinct.neurons
            if neuron.name not in shared_organ_neurons
        }
    )
    if not allocated:
        raise ValueError("生命状态本能片段没有自己的固定通道神经元")
    expected = list(range(entrances.next_free_index, allocated[-1] + 1))
    if allocated != expected:
        raise ValueError("生命状态本能固定通道没有连续占用分配给它的地址")

    instinct_range = BrainAddressReservation(
        "instinct.vital_state.fixed_circuits",
        entrances.next_free_index,
        19 * TISSUE_PLANE,
        len(allocated),
    )
    if allocated != list(range(instinct_range.start, instinct_range.start + len(allocated))):
        raise ValueError("生命状态本能固定通道没有从自己的物理体积起点连续排列")
    continuation_range = _z_reservation(
        "dna.organ_activity.independent_continuation",
        ORGAN_CONTINUATION_START_Z,
        VISUAL_RECONSTRUCTION_START_Z,
        ORGAN_CONTINUATION_COUNT,
    )
    confirmed = tuple((*organ_ranges, instinct_range, continuation_range))

    # 这些数量来自本任务此前已经逐段推进的人工出生结构。它们没有被
    # 后来的微观公式审计撤销；但具体路径强度、阈值出生值和调制分布仍待补充，
    # 所以与冻结边界分开登记，不能冒充已经完成的正式DNA。
    draft_counts = (
        (
            "dna.visual_reconstruction.ten_sections",
            VISUAL_RECONSTRUCTION_START_Z,
            VISUAL_SOURCE_START_Z,
            VISUAL_RECONSTRUCTION_COUNT,
        ),
        (
            "dna.visual_reconstruction.two_source_organizations",
            VISUAL_SOURCE_START_Z,
            PREDICTED_AUDITORY_OUTPUT_Z,
            VISUAL_RECONSTRUCTION_SOURCE_COUNT,
        ),
        (
            "dna.predicted_auditory.output",
            PREDICTED_AUDITORY_OUTPUT_Z,
            MOTION_OUTPUT_Z,
            PREDICTED_AUDITORY_OUTPUT_COUNT,
        ),
        (
            "dna.mouse_keyboard_view_center.output",
            MOTION_OUTPUT_Z,
            AUDITORY_RECONSTRUCTION_START_Z,
            MOTION_OUTPUT_COUNT,
        ),
        (
            "dna.auditory_reconstruction.ten_sections",
            AUDITORY_RECONSTRUCTION_START_Z,
            AUDITORY_SOURCE_START_Z,
            AUDITORY_RECONSTRUCTION_COUNT,
        ),
        (
            "dna.auditory_reconstruction.two_source_organizations",
            AUDITORY_SOURCE_START_Z,
            JOINT_CRYSTALLIZATION_START_Z,
            AUDITORY_RECONSTRUCTION_SOURCE_COUNT,
        ),
        (
            "dna.joint_crystallization",
            JOINT_CRYSTALLIZATION_START_Z,
            ACTION_FORMATION_START_Z,
            JOINT_CRYSTALLIZATION_COUNT,
        ),
        (
            "dna.action_formation.section_0",
            ACTION_FORMATION_START_Z,
            ACTION_FORMATION_START_Z + 2,
            ACTION_FORMATION_SECTION_COUNT,
        ),
        (
            "dna.action_formation.section_1",
            ACTION_FORMATION_START_Z + 2,
            ACTION_FORMATION_START_Z + 4,
            ACTION_FORMATION_SECTION_COUNT,
        ),
        (
            "dna.action_formation.section_2",
            ACTION_FORMATION_START_Z + 4,
            VISUAL_FIRST_BRANCH_START_Z,
            ACTION_FORMATION_SECTION_COUNT,
        ),
        (
            "dna.visual_identity_branch.reconstruction_side",
            VISUAL_FIRST_BRANCH_START_Z,
            VISUAL_SECOND_BRANCH_START_Z,
            VISUAL_IDENTITY_BRANCH_COUNT // 2,
        ),
        (
            "dna.visual_identity_branch.cross_organization_side",
            VISUAL_SECOND_BRANCH_START_Z,
            AUDITORY_FIRST_BRANCH_START_Z,
            VISUAL_IDENTITY_BRANCH_COUNT // 2,
        ),
        (
            "dna.auditory_identity_branch.reconstruction_side",
            AUDITORY_FIRST_BRANCH_START_Z,
            AUDITORY_SECOND_BRANCH_START_Z,
            AUDITORY_IDENTITY_BRANCH_COUNT // 2,
        ),
        (
            "dna.auditory_identity_branch.cross_organization_side",
            AUDITORY_SECOND_BRANCH_START_Z,
            VISUAL_RGB_JOINT_CONTACT_START_Z,
            AUDITORY_IDENTITY_BRANCH_COUNT // 2,
        ),
        (
            "dna.joint_crystallization.real_visual_rgb_contacts",
            VISUAL_RGB_JOINT_CONTACT_START_Z,
            UNASSIGNED_START_Z,
            VISUAL_RGB_JOINT_CONTACT_COUNT,
        ),
    )
    dna_draft = tuple(
        _z_reservation(name, start_z, stop_z, count)
        for name, start_z, stop_z, count in draft_counts
    )

    unassigned = LinearAddressRange(
        "brain_structure.unassigned",
        UNASSIGNED_START_Z * TISSUE_PLANE,
        TISSUE_NEURON_COUNT,
    )
    return BrainAddressPlan(confirmed, dna_draft, unassigned)
