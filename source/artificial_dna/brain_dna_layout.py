"""此前已推进人工出生结构在800×800×160组织中的物理落位公式。

本文件只确定地址，不填写仍待补充的具体路径强度、神经元出生属性或调制
受体分布；已经闭合的统一微观变化关系由共同状态变化实现。
"""

from __future__ import annotations

from .birth_structure import TissueAddress
from .brain_address_plan import (
    ACTION_FORMATION_SECTION_COUNT,
    ACTION_FORMATION_START_Z,
    AUDITORY_FIRST_BRANCH_START_Z,
    AUDITORY_RECONSTRUCTION_START_Z,
    AUDITORY_RECONSTRUCTION_SOURCE_COUNT,
    AUDITORY_SECOND_BRANCH_START_Z,
    AUDITORY_SOURCE_START_Z,
    JOINT_CRYSTALLIZATION_START_Z,
    JOINT_CRYSTALLIZATION_COUNT,
    REAL_AUDITORY_ASSOCIATION_CONTACT_COUNT,
    MOTION_OUTPUT_Z,
    MOTION_OUTPUT_COUNT,
    ORGAN_CONTINUATION_COUNT,
    ORGAN_CONTINUATION_START_Z,
    ORGAN_CONTINUATION_Z_OFFSET,
    PREDICTED_AUDITORY_OUTPUT_Z,
    PREDICTED_AUDITORY_OUTPUT_COUNT,
    VISUAL_RGB_JOINT_CONTACT_START_Z,
    VISUAL_RECONSTRUCTION_START_Z,
    VISUAL_RECONSTRUCTION_SOURCE_COUNT,
    VISUAL_FIRST_BRANCH_START_Z,
    VISUAL_SECOND_BRANCH_START_Z,
    VISUAL_SOURCE_START_Z,
)
from .brain_geometry import TISSUE_PLANE, coordinates, linear_index
from .organ_entrances import AUDITORY_ACTIVITY_COUNT, SECOND_EXPERIMENT_ENTRANCES


VISUAL_RECONSTRUCTION_SECTION_COUNT = 10
AUDITORY_RECONSTRUCTION_SECTION_COUNT = 10
VISUAL_SOURCE_PER_SIDE = VISUAL_RECONSTRUCTION_SOURCE_COUNT // 2
AUDITORY_SOURCE_PER_SIDE = AUDITORY_RECONSTRUCTION_SOURCE_COUNT // 2
VISUAL_JOINT_COUNT = 611_668
AUDITORY_JOINT_COUNT = 612_360
ACTION_JOINT_COUNT = 580
SENSORY_ACTION_FORMATION_COUNT = VISUAL_JOINT_COUNT + AUDITORY_JOINT_COUNT


def _at_linear(index: int) -> TissueAddress:
    return TissueAddress(*coordinates(index))


def organ_continuation_address(activity_index: int) -> TissueAddress:
    """每一项器官活动在普通入口之后的唯一独立继续位置。"""

    value = int(activity_index)
    if not 0 <= value < ORGAN_CONTINUATION_COUNT:
        raise ValueError("器官活动全局编号超出独立继续组织")
    visual = SECOND_EXPERIMENT_ENTRANCES.ranges[0]
    if visual.name != "visual":  # pragma: no cover - 出生入口顺序的结构保护
        raise AssertionError("真实视觉不再是第一个器官入口")
    if value < visual.activity_count:
        return visual_continuation_address(value)
    auditory = SECOND_EXPERIMENT_ENTRANCES.ranges[1]
    if auditory.name != "auditory":  # pragma: no cover - 出生入口顺序的结构保护
        raise AssertionError("真实听觉不再是第二个器官入口")
    if auditory.activity_offset <= value < auditory.activity_offset + auditory.activity_count:
        return auditory_continuation_address(value - auditory.activity_offset)
    for group in SECOND_EXPERIMENT_ENTRANCES.ranges:
        if group.activity_offset <= value < group.activity_offset + group.activity_count:
            pair = SECOND_EXPERIMENT_ENTRANCES.pair(
                group.name,
                value - group.activity_offset,
            )
            return TissueAddress(
                pair.ordinary.x,
                pair.ordinary.y,
                pair.ordinary.z + ORGAN_CONTINUATION_Z_OFFSET,
            )
    raise AssertionError("器官活动全局编号没有对应入口")  # pragma: no cover


def _visual_same_shape_address(activity_index: int, start_z: int) -> TissueAddress:
    """把1280×657×RGB活动原样放入五块768×657物理平面。"""

    value = int(activity_index)
    visual_count = 1280 * 657 * 3
    if not 0 <= value < visual_count:
        raise ValueError("真实视觉活动编号超出同形组织")
    pixel, channel = divmod(value, 3)
    y, x = divmod(pixel, 1280)
    tile, local_x = divmod(x, 256)
    return TissueAddress(local_x * 3 + channel, y, int(start_z) + tile * 2)


def visual_continuation_address(activity_index: int) -> TissueAddress:
    """真实视觉在逐项独立继续组织中的同形位置。"""

    return _visual_same_shape_address(activity_index, ORGAN_CONTINUATION_START_Z)


def visual_identity_branch_address(branch: int, activity_index: int) -> TissueAddress:
    """真实视觉活动分支后的唯一同形位置；不进行缩放、混色或汇合。"""

    side = int(branch)
    value = int(activity_index)
    if side not in (0, 1):
        raise ValueError("真实视觉独立支路只能是0或1")
    start_z = (
        VISUAL_FIRST_BRANCH_START_Z
        if side == 0
        else VISUAL_SECOND_BRANCH_START_Z
    )
    return _visual_same_shape_address(value, start_z)


def visual_rgb_joint_contact_address(activity_index: int) -> TissueAddress:
    """真实视觉第二支路在共同结晶组织中的逐项同形位置。

    这一段不缩放、不混色、不汇合；它只是跨组织固定通道
    后的独立继续位置。
    """

    return _visual_same_shape_address(
        activity_index,
        VISUAL_RGB_JOINT_CONTACT_START_Z,
    )


def _auditory_same_shape_address(activity_index: int, z: int) -> TissueAddress:
    """把3×1025×3项听觉活动原样放入声音感受场的物理排列。"""

    value = int(activity_index)
    if not 0 <= value < AUDITORY_ACTIVITY_COUNT:
        raise ValueError("真实听觉活动编号超出同形组织")
    stream, remainder = divmod(value, 1025 * 3)
    frequency, component = divmod(remainder, 3)
    tile, local_frequency = divmod(frequency, 256)
    return TissueAddress(
        local_frequency * 3 + component,
        stream * 6 + tile,
        int(z),
    )


def auditory_continuation_address(activity_index: int) -> TissueAddress:
    """真实听觉在逐项独立继续组织中的同形位置。"""

    return _auditory_same_shape_address(
        activity_index,
        ORGAN_CONTINUATION_START_Z + 10,
    )


def auditory_identity_branch_address(branch: int, activity_index: int) -> TissueAddress:
    """真实听觉活动分支后的唯一同形位置；不缩频率或指定排列位置。"""

    side = int(branch)
    if side not in (0, 1):
        raise ValueError("真实听觉独立支路只能是0或1")
    start_z = (
        AUDITORY_FIRST_BRANCH_START_Z
        if side == 0
        else AUDITORY_SECOND_BRANCH_START_Z
    )
    return _auditory_same_shape_address(activity_index, start_z)


def _predicted_auditory_xy(
    stream: int,
    frequency: int,
    sequence: int,
    component: int,
) -> tuple[int, int]:
    if not 0 <= int(stream) < 3:
        raise ValueError("预测声音流编号必须处于0到2")
    if not 0 <= int(frequency) < 342:
        raise ValueError("预测声音频率位置必须处于0到341")
    if not 0 <= int(sequence) < 85:
        raise ValueError("预测声音排列位置必须处于0到84")
    if not 0 <= int(component) < 3:
        raise ValueError("预测声音非负活动编号必须处于0到2")
    tile, local_frequency = divmod(int(frequency), 256)
    x = local_frequency * 3 + int(component)
    y = (int(stream) * 2 + tile) * 85 + int(sequence)
    return x, y


def visual_reconstruction_address(section: int, x: int, y: int) -> TissueAddress:
    if not 0 <= int(section) < VISUAL_RECONSTRUCTION_SECTION_COUNT:
        raise ValueError("视觉还原段编号必须处于0到9")
    if not 0 <= int(x) < 512 or not 0 <= int(y) < 512:
        raise ValueError("视觉还原位置超出512×512")
    return TissueAddress(
        int(x),
        int(y),
        VISUAL_RECONSTRUCTION_START_Z + int(section),
    )


def visual_source_address(side: int, index: int) -> TissueAddress:
    if int(side) not in (0, 1):
        raise ValueError("视觉还原来源面只能是0或1")
    if not 0 <= int(index) < VISUAL_SOURCE_PER_SIDE:
        raise ValueError("视觉还原来源编号超出611668个位置")
    y, x = divmod(int(index), 800)
    return TissueAddress(x, y, VISUAL_SOURCE_START_Z + int(side))


def visual_full_field_sample_coordinate(x: int, y: int) -> tuple[int, int]:
    """把512×512灰度还原位置对应到完整1280×657视觉感受场。

    这是已经进入 Brain 后的人工出生连接公式。四边、四角和中心按完整
    归一化视野对应；取最近位置时使用整数关系固定结果，不按画面内容变化。
    """

    target_x = int(x)
    target_y = int(y)
    if not 0 <= target_x < 512 or not 0 <= target_y < 512:
        raise ValueError("灰度还原位置超出512×512")
    source_x = (target_x * 1279 + 255) // 511
    source_y = (target_y * 656 + 255) // 511
    return source_x, source_y


def predicted_auditory_output_address(
    stream: int,
    frequency: int,
    sequence: int,
    component: int,
) -> TissueAddress:
    x, y = _predicted_auditory_xy(stream, frequency, sequence, component)
    return TissueAddress(x, y, PREDICTED_AUDITORY_OUTPUT_Z)


def motion_output_address(index: int) -> TissueAddress:
    if not 0 <= int(index) < MOTION_OUTPUT_COUNT:
        raise ValueError("鼠标、键盘和视野中心输出编号超出116项")
    return TissueAddress(int(index), 0, MOTION_OUTPUT_Z)


def auditory_reconstruction_address(
    section: int,
    stream: int,
    frequency: int,
    sequence: int,
    component: int,
) -> TissueAddress:
    if not 0 <= int(section) < AUDITORY_RECONSTRUCTION_SECTION_COUNT:
        raise ValueError("听觉还原段编号必须处于0到9")
    x, y = _predicted_auditory_xy(stream, frequency, sequence, component)
    return TissueAddress(x, y, AUDITORY_RECONSTRUCTION_START_Z + int(section))


def auditory_source_address(side: int, index: int) -> TissueAddress:
    if int(side) not in (0, 1):
        raise ValueError("听觉还原来源面只能是0或1")
    if not 0 <= int(index) < AUDITORY_SOURCE_PER_SIDE:
        raise ValueError("听觉还原来源编号超出612360个位置")
    y, x = divmod(int(index), 800)
    return TissueAddress(x, y, AUDITORY_SOURCE_START_Z + int(side))


def _joint_paired_address(index: int, auditory: bool) -> TissueAddress:
    if not 0 <= int(index) < VISUAL_JOINT_COUNT:
        raise ValueError("视觉与听觉共同结晶配对编号超出611668")
    absolute = (
        JOINT_CRYSTALLIZATION_START_Z * TISSUE_PLANE
        + int(index) * 2
        + int(auditory)
    )
    return _at_linear(absolute)


def joint_visual_address(index: int) -> TissueAddress:
    return _joint_paired_address(index, False)


def joint_auditory_address(index: int) -> TissueAddress:
    value = int(index)
    if not 0 <= value < AUDITORY_JOINT_COUNT:
        raise ValueError("听觉共同结晶编号超出612360")
    if value < VISUAL_JOINT_COUNT:
        return _joint_paired_address(value, True)
    extra = value - VISUAL_JOINT_COUNT
    absolute = (
        JOINT_CRYSTALLIZATION_START_Z * TISSUE_PLANE
        + 2 * VISUAL_JOINT_COUNT
        + extra
    )
    return _at_linear(absolute)


def joint_action_address(action: int, repetition: int) -> TissueAddress:
    if not 0 <= int(action) < MOTION_OUTPUT_COUNT:
        raise ValueError("动作结晶接触编号超出116项")
    if not 0 <= int(repetition) < 5:
        raise ValueError("每项动作必须恰好具有五个结晶接触位置")
    index = int(repetition) * MOTION_OUTPUT_COUNT + int(action)
    absolute = (
        JOINT_CRYSTALLIZATION_START_Z * TISSUE_PLANE
        + 2 * VISUAL_JOINT_COUNT
        + (AUDITORY_JOINT_COUNT - VISUAL_JOINT_COUNT)
        + index
    )
    return _at_linear(absolute)


def joint_real_auditory_contact_address(activity: int) -> TissueAddress:
    """真实听觉9,225项在同一联系组织中的逐项独立接触位置。"""

    value = int(activity)
    if not 0 <= value < REAL_AUDITORY_ASSOCIATION_CONTACT_COUNT:
        raise ValueError("真实听觉联系接触编号超出9,225项")
    absolute = (
        JOINT_CRYSTALLIZATION_START_Z * TISSUE_PLANE
        + ACTION_FORMATION_SECTION_COUNT
        + value
    )
    return _at_linear(absolute)


def visual_action_formation_index(index: int) -> int:
    """视觉同位置活动在三个动作形成组织中的共同编号。"""

    value = int(index)
    if not 0 <= value < VISUAL_JOINT_COUNT:
        raise ValueError("视觉动作形成编号超出611668项")
    return value


def auditory_action_formation_index(index: int) -> int:
    """听觉同位置活动在三个动作形成组织中的共同编号。"""

    value = int(index)
    if not 0 <= value < AUDITORY_JOINT_COUNT:
        raise ValueError("听觉动作形成编号超出612360项")
    return VISUAL_JOINT_COUNT + value


def returned_action_formation_index(action: int, repetition: int) -> int:
    """116项真实动作倾向的五个独立返回位置在动作形成组织中的编号。"""

    if not 0 <= int(action) < 116:
        raise ValueError("动作返回编号必须处于0到115")
    if not 0 <= int(repetition) < 5:
        raise ValueError("每项动作的返回位置编号必须处于0到4")
    return (
        SENSORY_ACTION_FORMATION_COUNT
        + int(repetition) * 116
        + int(action)
    )


def action_formation_address(section: int, index: int) -> TissueAddress:
    if not 0 <= int(section) < 3:
        raise ValueError("动作形成组织段编号必须处于0到2")
    if not 0 <= int(index) < ACTION_FORMATION_SECTION_COUNT:
        raise ValueError("动作形成组织编号超出1224608")
    start = (ACTION_FORMATION_START_Z + int(section) * 2) * TISSUE_PLANE
    return _at_linear(start + int(index))


def validate_layout_counts() -> None:
    if PREDICTED_AUDITORY_OUTPUT_COUNT != 3 * 342 * 85 * 3:
        raise RuntimeError("预测听觉输出总数与其物理排列不一致")
    if JOINT_CRYSTALLIZATION_COUNT != (
        VISUAL_JOINT_COUNT
        + AUDITORY_JOINT_COUNT
        + ACTION_JOINT_COUNT
        + REAL_AUDITORY_ASSOCIATION_CONTACT_COUNT
    ):
        raise RuntimeError("共同结晶组织总数与三类接触位置不一致")
    if SENSORY_ACTION_FORMATION_COUNT + ACTION_JOINT_COUNT != (
        ACTION_FORMATION_SECTION_COUNT
    ):
        raise RuntimeError("感受活动与动作返回活动没有完整占满动作形成组织")
    if visual_action_formation_index(VISUAL_JOINT_COUNT - 1) + 1 != (
        auditory_action_formation_index(0)
    ):
        raise RuntimeError("视觉与听觉在动作形成组织中的边界不连续")
    if auditory_action_formation_index(AUDITORY_JOINT_COUNT - 1) + 1 != (
        returned_action_formation_index(0, 0)
    ):
        raise RuntimeError("听觉与动作返回在动作形成组织中的边界不连续")
    if returned_action_formation_index(115, 4) + 1 != (
        ACTION_FORMATION_SECTION_COUNT
    ):
        raise RuntimeError("动作返回活动没有在动作形成组织末端完整结束")
    first_contact = joint_real_auditory_contact_address(0)
    last_contact = joint_real_auditory_contact_address(
        REAL_AUDITORY_ASSOCIATION_CONTACT_COUNT - 1
    )
    if linear_index(first_contact.x, first_contact.y, first_contact.z) != (
        JOINT_CRYSTALLIZATION_START_Z * TISSUE_PLANE
        + ACTION_FORMATION_SECTION_COUNT
    ):
        raise RuntimeError("真实听觉联系接触没有紧接原联系组织")
    if linear_index(last_contact.x, last_contact.y, last_contact.z) + 1 != (
        JOINT_CRYSTALLIZATION_START_Z * TISSUE_PLANE
        + JOINT_CRYSTALLIZATION_COUNT
    ):
        raise RuntimeError("真实听觉联系接触没有完整占满新增位置")


validate_layout_counts()
