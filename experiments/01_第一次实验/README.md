# 历史实验报告

> 本报告形成于最终理论冻结之前，不符合当前完整理论，不得作为理论依据或有效验证结果。当前唯一理论正文为 [`../../docs/01_冻结理论.md`](../../docs/01_冻结理论.md)。

总判定：**通过**

这份实验不使用目标、奖励、损失、预测误差、反向传播或 Path 权重训练。清醒期只积累活动历史和可塑痕迹；稳定传播结构只在睡眠阶段结算。

## 自动理论判据

- ✅ 所有冻结组件进入同一生命回环
- ✅ 每个回环和睡眠周期都观测全部Neuron与Path
- ✅ 所有六类生命Signal逐通道可观测
- ✅ 清醒期只积累痕迹而不永久修改Path结构
- ✅ 情绪器官实时调节活跃PlasticPath痕迹强度
- ✅ Memory以完整FIFO时间窗持续参与脑活动
- ✅ Teaching只覆盖动作执行且脑内倾向仍被保存
- ✅ Sleep只注入0.5信号且不消费MemoryCache
- ✅ FixedPath在Sleep后完全不变
- ✅ PlasticPath只在Sleep中发生结构结算
- ✅ PredictionRegion产生连续而非答案式活动
- ✅ 全部世界实体运动均逐回环可观测
- ✅ 动作结果通过Motor到身体感觉的跨时Path形成关系
- ✅ 特定实体不可见时PredictionField仍连续移动

## 出生结构

| Region | Neuron 数量 |
|---|---:|
| visual | 180 |
| audio | 16 |
| interoceptive | 6 |
| association | 24 |
| prediction | 36 |
| emotion_control | 6 |
| motor | 4 |

- Neuron 总数：272
- Path 总数：864
- Fixed Path：177
- Plastic Path 形成空间：687

## 情绪调制证据

- 真实情绪回环后的清醒期 Plastic Trace：21.572826
- 相同 DNA、世界和教学但关闭情绪调制后的 Trace：2.081613
- 调制倍率：10.364
- 清醒期永久 Path 变化：0 条

情绪器官不选择 Path。画面、音频、身体状态先形成活动，只有当时参与传播的 Plastic Path 才按接收到的调制波形留下不同深度的痕迹。

## 世界实体与动作—结果

- World 实体数量：5（人工身体也是其中一个实体）
- `entity_1` 本次清醒生命中的运动距离：162.400
- Teaching 覆盖期间脑内动作倾向保留：True
- Motor → Interoceptive Plastic Path：12 条
- 其中经睡眠发生结算：11 条
- 其中形成稳定关系：11 条

## 睡眠证据

- 人工睡眠信号强度：0.5
- 睡眠传播周期：24
- 发生结算的 Plastic Path：235
- Fixed Path 变化：0
- 睡眠前后缓存帧数：12 → 12

睡眠函数没有 Memory Cache 输入。World、感觉器官与 Motor Organ 在此阶段都没有运行。

## 特定实体与预测连续性

- 外部观察对象：`entity_1`（该 ID 不进入任何感觉 Signal 或 Brain）
- 暂时不可见期间 Prediction Field 有效输出率：100.0%
- 不可见期间预测场移动量：1.550
- 学成结构的不可见期误差：9.217
- 消融通往 Prediction Region 的 Plastic Path 后误差：11.558

## 速度

- 单个清醒回环平均：3.121 ms
- 本机每秒约：320.5 个完整回环

## 逐项观测文件

- `structure.json`：DNA、所有 Neuron 和所有 Path 的出生结构
- `timeline.csv`：每个生命回环的身体、情绪、预测、动作和痕迹总量
- `world_entities.csv`：每个实体在每个回环的位置、速度和可见状态
- `entity_continuity.csv`：特定实体暂时不可见时的 Prediction Field 与消融对照
- `signals.csv`：每个 Signal 的每个通道
- `neurons.csv`：每个回环中每个 Neuron 的活动、阈值和调制
- `paths.csv`：每个清醒回环及睡眠周期中每一条 Path 的完整状态
- `sleep_settlement.csv`：每一条 Path 的睡眠结算明细
- `memory_cache.json`：FIFO 缓存中最终完整 TimeFrame
- `path_changes.csv`：所有发生及未发生结算变化的 Path

## 边界

这次验证的是冻结理论中的组件和因果顺序已经形成一个可运行、可逐项审计的小闭环。它不证明语言理解、社会性或通用智能已经出现；也不以外部任务成绩替代理论机制证据。
