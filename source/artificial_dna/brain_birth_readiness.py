"""第二次实验正式个体出生前的完整性边界。

这里不替待补充关系选默认值，只把已经可以编译的出生结构与仍缺少的
统一公式关系精确分开。正式出生必须在全部必需项明确以后进行，避免把
文件初始零值误当成人工DNA或生命初始状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .birth_structure import FixedBirthFragment
from .auditory_output_relay_birth_values import (
    CONFIRMED_AUDITORY_OUTPUT_RELAY_COUNT,
)
from .action_formation_birth_values import (
    CONFIRMED_ACTION_FORMATION_NEURON_BIRTH_VALUE_COUNT,
)
from .brain_address_plan import BrainAddressPlan, REALLOCATED_FIXED_NEURON_PATH_COUNT
from .brain_geometry import DIRECTED_LOCAL_PATH_COUNT, TISSUE_NEURON_COUNT
from .fixed_path_topology import (
    FIXED_PATH_FAMILY_COUNTS,
    SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_COUNT,
    FixedPathTopology,
)
from .fixed_receiver_birth_values import (
    CONFIRMED_FIXED_RECEIVER_BIRTH_VALUE_COUNT,
)
from .formal_brain_topology import FormalBrainTopology, validate_formal_neuron_counts
from .formal_local_path_birth import (
    FORMAL_LOCAL_CROSS_ORGANIZATION_HELD_ZERO_COUNT,
    FORMAL_LOCAL_FIXED_ENDPOINT_HELD_ZERO_COUNT,
    FORMAL_LOCAL_FORMATION_PERMISSION_COUNT,
)
from .first_birth_state import FIRST_BIRTH_RESOLVED_RELATION_COUNT
from .organ_continuation_birth_values import (
    CONFIRMED_TRANSPARENT_ORGAN_NEURON_BIRTH_VALUE_COUNT,
)
from .sensory_identity_branch_birth_values import (
    CONFIRMED_SENSORY_IDENTITY_BRANCH_NEURON_BIRTH_VALUE_COUNT,
)
from .vital_state_neuron_birth_values import (
    CONFIRMED_VITAL_STATE_INTERNAL_NEURON_BIRTH_VALUE_COUNT,
)
from .visual_grayscale_admission_birth_values import (
    CONFIRMED_VISUAL_FIRST_SOURCE_NEURON_BIRTH_VALUE_COUNT,
)
from .visual_reconstruction_birth_values import (
    CONFIRMED_VISUAL_RECONSTRUCTION_NEURON_BIRTH_VALUE_COUNT,
)
from .visual_second_source_birth_values import (
    CONFIRMED_VISUAL_SECOND_SOURCE_NEURON_BIRTH_VALUE_COUNT,
)
from .visual_rgb_joint_contact_birth_values import (
    CONFIRMED_VISUAL_RGB_JOINT_CONTACT_NEURON_BIRTH_VALUE_COUNT,
)
from .joint_crystallization_birth_values import (
    CONFIRMED_JOINT_CRYSTALLIZATION_NEURON_BIRTH_VALUE_COUNT,
)
from .motion_output_birth_values import (
    CONFIRMED_MOTION_OUTPUT_NEURON_BIRTH_VALUE_COUNT,
)
from .neuron_nature_topology import (
    NeuronNatureTopology,
    validate_neuron_nature_counts,
)
from .ordinary_local_path_space import CURRENT_ORDINARY_NEIGHBOR_CANDIDATE_COUNT
from .ordinary_local_formation_space import (
    CONFIRMED_CLASSIFIED_ORDINARY_NEIGHBOR_COUNT,
    CONFIRMED_CROSS_ORGANIZATION_HELD_ZERO_COUNT,
    CONFIRMED_INTERNAL_FORMATION_PERMISSION_COUNT,
)
from .organ_entrances import OrganEntranceLayout
from .output_control_paths import (
    KNOWN_OUTPUT_CONTROL_PATH_COUNT,
    build_known_output_control_path_genes_by_source,
)


class BirthRequirementState(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class BirthRequirement:
    """工程完整性项目；名称和状态不进入Brain。"""

    name: str
    state: BirthRequirementState
    resolved_count: int | None
    required_count: int | None
    detail: str

    @property
    def complete(self) -> bool:
        return self.state is BirthRequirementState.COMPLETE


@dataclass(frozen=True, slots=True)
class BrainBirthReadiness:
    requirements: tuple[BirthRequirement, ...]

    @property
    def ready(self) -> bool:
        return all(value.complete for value in self.requirements)

    @property
    def incomplete(self) -> tuple[BirthRequirement, ...]:
        return tuple(value for value in self.requirements if not value.complete)

    def require_ready(self) -> None:
        if self.ready:
            return
        names = "、".join(value.name for value in self.incomplete)
        raise RuntimeError(f"正式人工生命尚不能出生，以下关系仍未闭合：{names}")


def build_brain_birth_readiness(
    *,
    entrances: OrganEntranceLayout,
    instinct: FixedBirthFragment,
    addresses: BrainAddressPlan,
    topology: FixedPathTopology,
) -> BrainBirthReadiness:
    """依据当前真实工程状态形成出生完整性报告。"""

    if topology.path_count != REALLOCATED_FIXED_NEURON_PATH_COUNT:
        raise ValueError("固定路径端点数量与地址总账不一致")
    if sum(FIXED_PATH_FAMILY_COUNTS.values()) != topology.path_count:
        raise ValueError("固定路径分类数量与端点生成器不一致")
    if topology.emotion_control_path_count != len(instinct.emotion_bindings):
        raise ValueError("情绪器官固定短路径数量与出生片段不一致")

    nature_topology = NeuronNatureTopology(instinct, entrances=entrances)
    validate_neuron_nature_counts(nature_topology)
    known_nature_count = nature_topology.assigned_count
    if known_nature_count != addresses.allocated_neuron_count:
        raise ValueError("神经元性质映射与已推进人工出生结构地址数量不一致")
    formal_topology = FormalBrainTopology(nature_topology)
    validate_formal_neuron_counts(formal_topology)
    known_output_path_count = int(
        build_known_output_control_path_genes_by_source().size
    )
    if known_output_path_count != KNOWN_OUTPUT_CONTROL_PATH_COUNT:
        raise ValueError("已确认输出器官固定出生路径没有覆盖全部控制入口")

    requirements = (
        BirthRequirement(
            "完整神经组织地址",
            BirthRequirementState.COMPLETE,
            TISSUE_NEURON_COUNT,
            TISSUE_NEURON_COUNT,
            "800×800×160位置及26个直接相邻方向已经固定。",
        ),
        BirthRequirement(
            "器官入口连接",
            BirthRequirementState.COMPLETE,
            entrances.activity_count,
            entrances.activity_count,
            "每项已确认器官活动都有独立固定接收端和唯一普通入口。",
        ),
        BirthRequirement(
            "固定路径来源与到达端点",
            BirthRequirementState.COMPLETE,
            SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_COUNT,
            SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_COUNT,
            "第二次实验实际出生的33106609条固定路径端点已经全部确定。"
            "真实听觉只保留入口、逐项继续、单条同形支路和到联系组织的9225条"
            "一对一接触；完整项目中为以后保留的听觉还原、听觉预测和双来源"
            "草案不编译进本次个体。",
        ),
        BirthRequirement(
            "生命状态到情绪器官控制入口",
            BirthRequirementState.COMPLETE,
            len(instinct.emotion_bindings),
            len(instinct.emotion_bindings),
            "中央视野映射下70条特殊短路径的来源和控制入口编号已经确定。",
        ),
        BirthRequirement(
            "生命状态本能与视野中心运动的联合覆盖",
            BirthRequirementState.COMPLETE,
            2,
            2,
            "出生时视野中心固定为(0.5,0.5)，生命状态固定通道由该位置的真实"
            "视觉活动形成；四路视野中心输出现在也已经按一次活动一次发生、强度1"
            "移动一个3840×2160原始位置的关系接入同一个实际中心状态。出生结构"
            "只固定中央实测本能，不预装其他视野中心的同一画面答案；移动后的后续"
            "联系由返回Brain的四路动作活动、真实视觉变化、普通视觉相邻可变路径"
            "和70路物质活动在生命中形成。这已经完整定义出生关系，移动后能否形成"
            "连续覆盖属于本次实验实际观察结果，不再误列为出生前缺项。",
        ),
        BirthRequirement(
            "情绪器官固定短路径状态承载",
            BirthRequirementState.COMPLETE,
            topology.emotion_control_path_count,
            topology.emotion_control_path_count,
            "每条特殊短路径均单独保存来源、控制入口、固定强度与当前传播活动。",
        ),
        BirthRequirement(
            "已确定输出器官的固定到达路径",
            BirthRequirementState.COMPLETE,
            known_output_path_count,
            KNOWN_OUTPUT_CONTROL_PATH_COUNT,
            "五组输出器官的523890个控制入口与末端来源逐项一对一接触；"
            "边界固定路径以强度1原样传递末端活动，并按来源地址编成正式出生记录。"
            "末端神经元所在组织的其他出生属性仍由各自项目单独核对。",
        ),
        BirthRequirement(
            "神经元固定或可塑性质",
            BirthRequirementState.COMPLETE,
            formal_topology.assigned_count,
            TISSUE_NEURON_COUNT,
            "3056155个器官接收和生命状态本能位置保持固定；其余99343845个位置"
            "全部按所在视觉、听觉、预测回流、动作回流、逐项继续、还原、联系、"
            "动作形成和本能物理组织带归为普通神经元。两条视觉及听觉独立支路保持"
            "不同组织身份；第108至159层顺接既有共同联系组织。全部1.024亿位置"
            "都有且只有一种出生性质。",
        ),
        BirthRequirement(
            "已确定普通组织的直接相邻候选",
            BirthRequirementState.COMPLETE,
            CURRENT_ORDINARY_NEIGHBOR_CANDIDATE_COUNT,
            CURRENT_ORDINARY_NEIGHBOR_CANDIDATE_COUNT,
            "401761080条普通—普通物理邻接已逐地址生成；这不是形成许可或路径强度。",
        ),
        BirthRequirement(
            "统一神经元和路径变化关系",
            BirthRequirementState.COMPLETE,
            7,
            7,
            "S、A、Q、M、非零可变路径增强、零强度相邻路径形成和睡眠削弱"
            "已经由同一完整发生顺序定义。",
        ),
        BirthRequirement(
            "神经元响应增益与阈值出生值",
            BirthRequirementState.COMPLETE,
            TISSUE_NEURON_COUNT,
            TISSUE_NEURON_COUNT,
            "器官固定接收、普通入口、逐项继续和真实感受同形分支保持响应强度1、"
            "阈值0，因而不改变原始连续活动。生命状态本能160个固定神经元保持"
            "响应强度1并使用已经逐项推导的阈值。其余普通神经元统一按本次实际"
            "固定汇入数与同组织26向全部可能相邻汇入数之和的倒数形成响应强度，"
            "没有汇入的位置以1作为中性响应，阈值均为0。数值来自同一S、A、Q公式"
            "和完整物理连接上限，不再由各组件分别猜一套参数。",
        ),
        BirthRequirement(
            "固定路径强度",
            BirthRequirementState.COMPLETE,
            SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_COUNT,
            SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_COUNT,
            "本次出生范围内每条固定路径的强度均已确定。真实听觉入口到联系"
            "位置全程采用强度1逐项传播，不加入频率合并、音节判断或听觉预测；"
            "未参加本次实验的听觉草案也没有获得伪造的默认强度。",
        ),
        BirthRequirement(
            "直接相邻可塑路径出生状态",
            BirthRequirementState.COMPLETE,
            DIRECTED_LOCAL_PATH_COUNT,
            DIRECTED_LOCAL_PATH_COUNT,
            f"{FORMAL_LOCAL_FORMATION_PERMISSION_COUNT}条普通神经元同组织有向相邻"
            "关系以强度0出生并允许后天形成；"
            f"{FORMAL_LOCAL_CROSS_ORGANIZATION_HELD_ZERO_COUNT}条普通神经元跨组织"
            "关系保持0且关闭形成；"
            f"{FORMAL_LOCAL_FIXED_ENDPOINT_HELD_ZERO_COUNT}条至少一端固定的关系也"
            "保持0且关闭形成。全部关系的变化系数为1/26、形成阈值为0，三类合计"
            "恰好覆盖2646293112条有效26向物理关系。",
        ),
        BirthRequirement(
            "神经元与路径的调制响应分布",
            BirthRequirementState.COMPLETE,
            FORMAL_LOCAL_FORMATION_PERMISSION_COUNT,
            FORMAL_LOCAL_FORMATION_PERMISSION_COUNT,
            "本次实验的神经元物质响应记录明确为0条，70路物质不直接改变神经元"
            "响应强度或阈值。每条允许形成的可变路径恰有一条物质受体记录；来源"
            "位置按10列×7行分到70条贯穿全部组织的空间通道。70路本能固定来源的"
            "最大活动均为36，所以各自受体系数为1/36；不把70路压成正负两类，"
            "也不增加奖励、比较或选择关系。",
        ),
        BirthRequirement(
            "情绪器官物质值形成关系",
            BirthRequirementState.COMPLETE,
            len(instinct.emotion_bindings),
            len(instinct.emotion_bindings),
            "70个控制入口保持独立；到达同一入口的特殊短路径活动直接相加形成该路当前物质活动。",
        ),
        BirthRequirement(
            "声音输出器官活动数量与入口",
            BirthRequirementState.COMPLETE,
            0,
            0,
            "第二次实验不检验说话能力，因此本次个体没有声音输出入口。完整项目"
            "中声音器官的待定规格保持待定，不被本次实验反向冻结。",
        ),
        BirthRequirement(
            "第一次出生的当前生命状态",
            BirthRequirementState.COMPLETE,
            FIRST_BIRTH_RESOLVED_RELATION_COUNT,
            FIRST_BIRTH_RESOLVED_RELATION_COUNT,
            "第一次出生尚未发生任何输入、神经元变化、路径传播、物质形成或器官输出，"
            "所以U、A、所有Q、M和Y从零开始；没有后天经历的可变路径P从零开始。"
            "固定路径P不清零，而是逐条读取人工出生结构的强度。重新启动已有个体"
            "直接恢复储存状态，不再次套用第一次出生关系。",
        ),
    )
    return BrainBirthReadiness(requirements)
