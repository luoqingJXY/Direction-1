# 人工出生结构代码

这里保存第二次实验实际使用的人工出生结构生成代码。它负责决定“出生时有什么”，不保存后天记忆，也不在个体出生后改变。

## 组织和地址

- `brain_geometry.py`：`800 × 800 × 160` 三维神经组织和地址换算。
- `brain_address_plan.py`：器官入口、组织、区域和输出位置的统一地址安排。
- `brain_dna_layout.py`：把已经确认的组织分段组合成完整出生布局。
- `formal_brain_topology.py`：正式结构的整体拓扑描述。
- `neuron_nature_topology.py`：固定神经元与普通神经元性质的出生排列。

## 器官入口与独立继续

- `organ_entrances.py`：视觉、听觉、预测、动作和情绪等器官活动进入大脑的位置。
- `fixed_receiver_birth_values.py`：固定接收神经元的出生响应和阈值。
- `organ_continuation_birth_values.py`：器官活动进入普通组织后的独立继续位置。
- `sensory_identity_branch_birth_values.py`：感受活动在脑内形成先天分支时的出生属性。

## 视觉路径通道

- `visual_rgb_joint_contact_birth_values.py`：同一视觉位置三项颜色活动的具体汇合接触。
- `visual_grayscale_admission_birth_values.py`：灰度还原通道入口的出生属性。
- `visual_reconstruction_birth_values.py`：预测视觉还原通道的出生属性。
- `visual_second_source_birth_values.py`：视觉第二来源接触的出生属性。
- `joint_crystallization_birth_values.py`：跨来源汇合和结晶段的出生属性。

## 动作与预测输出

- `action_formation_birth_values.py`：动作倾向形成段的出生属性。
- `motion_output_birth_values.py`：鼠标、键盘和视野中心输出末端的出生属性。
- `auditory_output_relay_birth_values.py`：预测听觉输出中继段。
- `output_control_paths.py`：从大脑末端到客观输出器官控制入口的固定路径。

## 情绪本能

- `vital_state_reference_plan.py`：生命值、饱食度和死亡参考活动的结构计划。
- `vital_state_neuron_birth_values.py`：生命状态本能神经元的出生属性。
- `vital_state_instinct.py`：参考受体到70个情绪器官控制入口的具体先天短路径结构。
- `modulation_records.py`：神经元和可塑路径的调制受体记录格式。

## 固定路径和可塑空间

- `fixed_path_topology.py`：跨组织固定传播端点的生成关系。
- `fixed_path_compilation.py`：把确定的端点编译为存储结构。
- `confirmed_fixed_path_birth_strengths.py`：已确认固定路径的出生强度。
- `ordinary_local_path_space.py`：普通神经元26方向直接相邻路径槽位。
- `ordinary_local_formation_space.py`：哪些局部相邻关系允许形成可塑路径。
- `formal_local_path_birth.py`：局部路径状态、变化系数、形成阈值和许可的出生展开。

## 出生执行

- `formal_neuron_birth_values.py`：全部神经元响应强度、阈值和性质的正式出生展开。
- `formal_birth_compilation.py`：把出生结构写入大型永久状态文件。
- `birth_value_constraints.py`：出生数值的统一合法范围。
- `birth_structure.py`：出生结构的基础记录形式。

这些文件共同描述人工出生结构，不是多个可独立训练的模型。实际能力只能在完整生命回环中由器官、神经元、路径、情绪器官和后天可塑状态联合形成。
