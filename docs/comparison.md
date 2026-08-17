# 与类似项目对比

> 相关工作按三类区分：**工程基础**（本项目直接使用）、**OS 思想来源**（启发性概念）与**相近 MoE 工作**（同类问题空间）。本项目不声称复刻任何一项，增量集中在“将模型资源条件与推理阶段语义接入 Linux mmap / Page Cache 决策”。

## 1. 对比范围

### 工程基础

| 项目 | 主要能力 | 本项目的关系 |
| --- | --- | --- |
| llama.cpp | 本地量化模型推理与跨平台执行 | 本项目保留完整上游工程，在推理路径旁增加默认关闭的 trace、OS hint 和实验可信度工具 |
| GGUF | 量化模型分发与 mmap 布局 | 模型权重以 GGUF 单文件 mmap 为映射基础 |

### OS 思想来源

| 思想 / 论文 | 核心概念 | 与本项目的关系 |
| --- | --- | --- |
| Denning Working Set（P. J. Denning，CACM 1968） | 按进程引用局部性定义工作集 | 思想来源：启发 Semantic Working Set 的“语义需求定义成员资格”；本项目将其对象化为 `(layer, expert, tensor)` |
| Application-Controlled File Caching（Cao / Felten / Li，OSDI 1994） | 应用参与文件缓存决策优于纯 LRU | 思想来源：启发“Runtime 比 OS 更了解后续访问”的用户态 hint 路线 |
| Informed Prefetching and Caching（Patterson et al.，SOSP 1995） | 用披露的访问信息指导预取与缓存 | 思想来源：启发 trace → hint → first-use 反馈闭环的实验形态 |

### 相近 MoE 工作

| 项目或方向 | 主要能力 | 本项目的关系与差异 |
| --- | --- | --- |
| MoE-Infinity（Xue et al.，arXiv 2024） | 序列级 expert 激活追踪、缓存与预取 | 相近工作：同处 expert 语义缓存问题空间；本项目把真实路由信息映射到 Linux 页面提示，并记录物理内存和缺页反馈 |
| ProMoE（Song et al.，arXiv 2024） | Expert offloading 场景中的预测、预取与主动缓存（proactive caching） | 相近工作：面向 PCIe host→GPU offloading 的预测式预取；本项目面向 CPU mmap / Page Cache，不搬运 tensor |
| SpecMD Least-Stale | 基于陈旧程度的替换决策 | 本项目仅在离线 trace 模拟器中实现和比较，不把模拟命中率等同于运行时收益 |
| PagedAttention、vAttention | KV cache 分页和虚拟内存管理 | 本项目已有 KV trace、预算模拟和 cgroup 压力矩阵，尚未声称完成运行时分页 KV allocator |
| Linux MGLRU、DAMON、PSI、cgroup v2 | 页面回收、访问监测、压力观测和资源限制 | 本项目不修改内核，利用模型语义补充内核不可见的信息，并通过官方接口施加提示和采集反馈 |
| FlexInfer、SP-MoE | 设备端卸载、异步预取、批量 I/O 和及时到达模型 | 本项目不做 CPU-GPU tensor 搬运，而是控制 Linux 文件映射页；异步 hint 是保留的实验路径 |
| OD-MoE 等跨层预测工作 | 提前预测后续层 expert 并及时加载 | 本项目曾做过无训练的相邻层预测探索，但当前已归档，不把预测准确率写成系统收益 |

## 2. 相对基础 llama.cpp 的新增能力

- tensor、KV、expert、memory 等 JSONL trace，以及统一的事件时间线。
- 以一次 `process_ubatch()` 为边界的 `STEP_BEGIN/STEP_END` 权威阶段计时。
- 基于 MoE routed expert 的预取、异步提示队列和 deadline/route score 调度。
- expert 与 KV 的离线替换策略模拟，以及 cgroup 受限内存实验矩阵。
- Memory Object、Working Set、Calibration Shadow 和 object-level residency attribution 观测。
- 每个 trace sink 的 `enqueued/written/dropped` 计数和零丢失检查。
- 冷缓存准备、GNU time 全进程指标、运行 Manifest、输出哈希和重复实验一致性验证。
- 延迟、缺页、RSS、swap、hint 开销联合比较的 Pareto 分析。

这些修改均为可选实验功能。关闭 trace 和 OS hints 时，不改变模型权重、计算图或生成算法。

## 3. 相对通用页面替换的特点

Linux 页面替换器能观察页访问、回收和系统压力，但通常不知道“哪个文件区间对应下一层即将使用的 expert”。本项目的用户态策略能够获得 layer、expert、router score 和推理阶段，从而做语义相关的提示。两者不是替代关系：内核负责真实页面生命周期，本项目负责提供模型侧先验并测量提示是否值得。

当前实现相对通用 LRU/LFU 的主要不同是：

1. 缓存项对应 `(layer, expert, tensor)` 的文件映射区间，而不是无语义的单页。
2. 预取顺序考虑到达使用点的层距离和 route score。
3. 策略评价同时计算延迟收益、major faults、常驻内存、swap 和系统调用成本。
4. 离线模拟只用于候选筛选，最终结论必须由受控真实运行确认。

## 4. 当前自主工作与创新边界

本项目的增量集中在三点：

1. 将模型资源条件（mmap 大小、Sparse MoE 结构、cgroup / MemAvailable）与推理阶段（prefill / decode）接入 Linux mmap / Page Cache 决策：`DEFAULT / POPULATE / SKIP` 准入与 `FADV_SEQUENTIAL → 首次 Decode 前 FADV_NORMAL` 阶段切换。
2. 将 Router 的 Expert 精确映射为 `(layer, expert, tensor)` 对应的 GGUF mmap **虚拟地址区间**与 Memory Object，使其具备身份、需求状态与可审计生命周期。
3. 结合页面驻留、缺页、内存压力、first-use、readmission 等运行信息验证页面行为与控制边界；Prefetch / COLD / DONTNEED / Rescue 的受控结果（包括负结果）共同界定语义控制的实际能力范围。

Controller 入口当前为 `off` 与 `expert_prefetch`；双反馈、slack、Stage priority 和跨层预测已经归档，不能再按当前能力宣传。

## 5. 与简单 top-k 预取的差异

历史探索中，直接减少 routed expert 数量虽然降低 hint 数，却可能损失预取覆盖率并增加 major faults。当前路线保留语义覆盖，通过异步执行、优先级和 TTL 降低关键路径干扰。COLD、Working Set 和 Residency Attribution 主要用于研究对象状态与驻留风险，不能简单等同于页回收或性能提升。

## 6. 可信度方面的改进

与只保存一份终端输出或单次性能数字的实验方式相比，本项目新增以下约束：

- 一次运行同时记录代码、模型、输入、参数、硬件和 cgroup 实际值。
- 冷缓存准备失败即判定无效，不静默变成热缓存实验。
- 阶段延迟来自 `STEP_END`，全进程 faults/RSS 来自 GNU time，各指标口径明确。
- 正式证据要求 trace 零丢失、输出哈希一致、仓库干净和运行条件一致。
- 四种候选按位置轮换并重复运行，缺失数据不参与“更优”判断。

## 7. 当前不足

- 主要实验已在冻结前完成；其余数据缺口（swap peak、跨模型/跨硬件泛化）保留为已知边界，不再补测。
- Router Prefetch 在 16–19 GiB transient sweep 与 12 GiB server anchor 中的行为已测定（16G Critical 全抑制、12G 真实 reclaim 下仍发 hint），但未证明端到端净收益。
- page-in 完成时间无法从 `madvise` 返回值精确观测；Residency Attribution 提供的是 demand 前的对象级驻留观察。
- Working Set / DONTNEED 当前策略性能为负（wall 约 +92.9% / +41.6%），逻辑 budget 不等同于物理 Page Cache residency。
- KV cache 目前以分析与模拟为主，尚未与 expert 页面共享统一运行时内存预算。
- 泛化验证仍需覆盖不同输入长度、内存上限、模型和硬件。

## 8. 决赛阶段口径

代码已冻结。答辩主线为三项贡献（模型映射准入、阶段感知文件预读、MoE 语义内存对象），性能表述以各自 A/B 数字与边界为准；Prefetch / COLD / DONTNEED / Rescue 作为机制探索与负结果呈现，不作为当前性能贡献。
