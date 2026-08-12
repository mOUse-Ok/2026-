# 与类似项目对比

## 1. 对比范围

| 项目或方向 | 主要能力 | 本项目的关系与差异 |
| --- | --- | --- |
| llama.cpp | 本地量化模型推理与跨平台执行 | 本项目保留完整上游工程，在推理路径旁增加默认关闭的 trace、OS hint 和实验可信度工具 |
| MoE-Infinity | 基于请求的 expert 追踪、缓存与预取 | 本项目借鉴 expert 语义，但把真实路由信息映射到 Linux 页面提示，并记录物理内存和缺页反馈 |
| SpecMD Least-Stale | 基于陈旧程度的替换决策 | 本项目仅在离线 trace 模拟器中实现和比较，不把模拟命中率等同于运行时收益 |
| PagedAttention、vAttention | KV cache 分页和虚拟内存管理 | 本项目已有 KV trace、预算模拟和 cgroup 压力矩阵，尚未声称完成运行时分页 KV allocator |
| Linux MGLRU、DAMON、PSI、cgroup v2 | 页面回收、访问监测、压力观测和资源限制 | 本项目不修改内核，利用模型语义补充内核不可见的信息，并通过官方接口施加提示和采集反馈 |
| FlexInfer、SP-MoE | 设备端卸载、异步预取、批量 I/O 和及时到达模型 | 本项目不做 CPU-GPU tensor 搬运，而是控制 Linux 文件映射页；异步 hint 和 deadline-score 是当前保留的轻量主线 |
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

本项目目前具有以下本队完成的系统机制：

- 将真实 MoE 路由语义映射到 Linux 页面提示，而不是仅做顺序文件预取。
- 将 `madvise` 从 decode 同步路径移入异步队列，在线估计 layer deadline，并在出队时取消过期任务。
- 使用 Memory Object 和 Working Set 记录 Expert Slice 的 semantic demand、保护、淘汰和重新进入状态。
- 使用 Calibration Shadow 和 Residency Attribution 补充环境尺度与对象级驻留观察。
- 通过同一套 trace 闭环联合筛选推理速度和物理内存指标，而不是只优化单一命中率。

当前主线是 `off` 与 `expert_prefetch` 两个 controller。后者使用异步 hint、deadline-score 和可选 gate，但仍属于实验性 profile；双反馈、slack、Stage priority 和跨层预测已经归档，不能再按当前能力宣传。

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

- 旧 N=3 数据只能作为探索记录，尚缺按新协议完成的 N=8 受控复测。
- 当前主线仍需要在固定冷缓存、cgroup 和重复次数下重新确认 Prefetch 的端到端表现。
- page-in 完成时间无法从 `madvise` 返回值精确观测；Residency Attribution 提供的是 demand 前的对象级驻留观察。
- Memory Object、Calibration Shadow 和 Residency Attribution 目前主要有代码/单测/观测链路证据，尚无独立性能收益结论。
- KV cache 目前以分析与模拟为主，尚未与 expert 页面共享统一运行时内存预算。
- 泛化验证仍需覆盖不同输入长度、内存上限、模型和硬件。

## 8. 决赛阶段建议主线

当前阶段不再继续堆叠新的调度器。优先完成 `off`/`expert_prefetch` 的冷缓存重复矩阵，再单独观察 Memory Object、Calibration Shadow 和 Residency Attribution 的开销与数据质量。任何新机制都应先证明输出、trace 完整性和指标口径一致，再讨论是否存在性能收益。
