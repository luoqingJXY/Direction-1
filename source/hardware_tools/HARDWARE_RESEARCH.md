# CPU、GPU、内存与显存联合运行研究

> 历史性能研究，不是理论依据。文中的结构名称和测试假设形成于最终理论冻结之前；只能保留为硬件测量材料，不能用于解释或补全当前理论。唯一理论正文为 [`THEORY_FREEZE.md`](THEORY_FREEZE.md)。

## 本机与当前状态

- CPU：Intel Core i9-13900KF，32 个逻辑处理器；
- GPU：AMD Radeon RX 7900 XTX，24 GB GDDR6；
- 系统内存：约 32 GB，检查时约 18.8 GB 可用；
- Python：3.12；
- 已安装计算库：NumPy 2.3.5；
- 项目隔离环境已安装：PyTorch ROCm 7.2.1；CuPy、Numba未安装，也不再是当前方案所需；
- 当前严格理论实验：约使用一个 CPU 核心，没有 GPU 或显存参与。

当前完整实验的 Python 分配峰值约 145 MB；大部分来自逐 Path、逐 Neuron 观测字典，而不是生命结构本身。

## 已完成的打包基准

打包后端把 272 个 Neuron 和 864 条 Path 的完整数值状态压成 56,448 字节的连续数组，同时保留稳定 ID。一次传播的 NumPy 结果已经与 Python 对象版逐值对照通过。

本机实测 5,000 轮传播：

- Python 对象版：1.729 秒；
- NumPy 连续数组版：0.100 秒；
- 传播核心加速：17.32 倍。

纯传播核心的合成规模测试：

| Neuron | Path | Path传播/秒 | 四轮传播理论回环/秒 |
|---:|---:|---:|---:|
| 2,500 | 30,000 | 2.52 亿 | 2,098 |
| 50,000 | 1,000,000 | 1.04 亿 | 26.0 |

这些数字只测 Path 聚合与 Neuron 积分，不含 World、器官、Emotion Trace、Sleep、Memory 和完整观测，不能直接当作最终生命回环速度。

## ROCm 实装结果

已经在项目独立环境 `.venv-rocm` 中安装并实际运行：

- PyTorch `2.9.1+rocm7.2.1`；
- HIP runtime `7.2.53211-158bd99533`；
- NumPy `2.3.5`；
- GPU：AMD Radeon RX 7900 XTX；
- PyTorch 报告总显存：25,753,026,560 字节；
- Path target 的 `index_add` 聚合测试通过；
- CPU/GPU 单轮传播最大绝对差异：`2.98e-8`。

256 MiB RAM/显存传输实测：

- RAM → VRAM：11.00 GiB/s；
- VRAM → RAM：5.23 GiB/s；
- 往返内容完全一致。

GPU规模实测：

| Neuron | Path | GPU Path传播/秒 | 四轮传播核心/秒 | 峰值显存 |
|---:|---:|---:|---:|---:|
| 2,500 | 30,000 | 0.60 亿 | 501 | 1.09 MB |
| 50,000 | 1,000,000 | 17.91 亿 | 448 | 35.86 MB |
| 200,000 | 10,000,000 | 72.95 亿 | 182 | 345.64 MB |

本机多次测量得到的交叉区约在10万至15万条活跃 Path，具体位置会随图分布、批次形状和预热状态波动。为避免临界规模下频繁迁移，联合调度器保守地以18万活跃Path作为初始GPU迁移阈值，后续再按真实Region分布自动校准。

## 联合架构

```text
SSD / structure checkpoint
        ↓ 按 Region 和 Path 邻接块读取
系统内存（完整结构）
        ├─ DNA、Fixed/Plastic Path 全状态
        ├─ 冷 Region、Memory Cache、完整观测日志
        └─ 活动索引与显存镜像版本号
                    ↓ 只迁移热子图
显存（当前活动结构）
        ├─ Neuron 连续状态数组
        ├─ Path source/target/tendency/stability/trace
        └─ Signal 与调制波双缓冲
                    ↓ GPU 并行
GPU
        ├─ 所有活动 Path 传播与 target 聚合
        ├─ Neuron 连续积分
        ├─ Emotion 对活跃 Plastic Trace 的调制
        └─ Sleep 传播与可塑结构候选结算
                    ↓ 小量结果
CPU
        ├─ World、器官、身体与 Teaching
        ├─ Runtime 调度及活跃 Region 装入/退出
        ├─ Fixed Path 不变量检查
        └─ 可重建的观测增量日志
```

数组只是存储布局；每个 Neuron 和 Path 仍有稳定 ID 和完整多维状态。GPU 后端不能引入 Layer、Loss、Reward、最佳 Path 或反向传播。

## 为什么不能把全部数据每回环来回复制

PCIe 迁移会抵消 GPU 并行收益。正确方式是把活跃子图在显存中保持多个回环，只上传新的器官 Signal 和新唤醒的结构块；每回环只下载 Action、Prediction、Emotion 摘要和发生变化的观测增量。睡眠结束后再批量同步结构结算。

## Windows AMD 路径

AMD 当前官方支持 RX 7900 XTX 在 Windows 11 上使用 ROCm 7.2.1 和 PyTorch 2.9.1，但 Windows 只支持 PyTorch 子集，不是完整 ROCm 栈。项目需要的无自动微分数组传播、`index_add`聚合和状态更新已经在本机验证通过。

ROCm/PyTorch已经安装在项目独立虚拟环境中，没有改变原CPU基准环境。

## 分阶段验证

1. ✅ NumPy 连续数组与当前对象版逐传播结果对照；
2. ✅ ROCm PyTorch 在 RX 7900 XTX 上验证设备可见和关键聚合算子；
3. ✅ 3 万、100 万、1000 万 Path 吞吐和显存测试；
4. ✅ RAM/显存往返内容及带宽测试；
5. ✅ 同一结构下 CPU 与 GPU 活动误差验证；
6. 待完成：把Emotion Trace、Memory热窗口和Sleep完整状态机接入混合后端；
7. 完成后扩大人工脑规模。
