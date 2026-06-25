# 与类似项目对比

## 对比对象

| 项目/方向 | 主要关注点 | 与本项目关系 |
| --- | --- | --- |
| llama.cpp | 本地 LLM 推理、量化、跨平台运行 | 本项目基于其完整工程扩展 trace 与 OS hint 实验 |
| MoE-Infinity | MoE expert tracing/cache/prefetch | 本项目借鉴 request-level expert 追踪和 expert cache 思路，但实现为用户态 trace/OS hint 实验 |
| SpecMD Least-Stale | 基于 stale 程度的替换策略 | 本项目实现 least-stale 离线模拟，并与 LRU/LFU/window-LFU 比较 |
| ST-MoE 相关工作 | MoE expert 跨 token/跨层可预测性 | 本项目使用 layer/expert 维度分析 routed expert 行为 |
| PagedAttention/vAttention | KV cache 的分页式管理 | 本项目当前侧重 KV cache 分析与投影，未改写 KV allocator |
| Linux MGLRU/DAMON/madvise | 内核页面回收和用户态 hint | 本项目不改内核，使用 `madvise`/`posix_fadvise` 做用户态实验 |

## 与基础 llama.cpp 的差异

基础 `llama.cpp` 提供高效本地推理能力，但默认不输出本项目所需的细粒度内存 trace，也不基于 MoE routing 做 expert slice OS hint。本项目新增：

- tensor/KV/expert/memory JSONL trace。
- OS hint 事件记录与分析。
- expert-aware prefetch。
- expert slice cache policy 离线模拟。
- 异步 expert hint queue。
- deadline-aware priority 调度。
- repeated-run 对比和 Pareto 分析。

## 与内核页面替换方案的差异

本项目不修改 Linux 内核，不直接替换 MGLRU 或 DAMON。原因：

- 初赛阶段更需要可复现、低侵入、便于演示的用户态方案。
- LLM 推理过程具有模型结构和 expert routing 先验，用户态能获得内核不可见的语义信息。
- 使用 `madvise` 和 `posix_fadvise` 可以在不改内核的前提下影响页面预取/回收行为。

## 与单纯 top-k 预取的差异

实验显示 top-k 截断虽然能减少 hint calls，但会显著损失 prefetch coverage，导致 major faults 回升。因此最终主线不是简单减少专家数，而是：

1. 保留完整 routed expert coverage。
2. 将 hint call 从 decode 关键路径迁出。
3. 用 deadline-aware priority 提升异步 hint 的时效性。

## 当前优势

- 工程侵入小：默认关闭，不改变 baseline。
- 可复现：提供 pipeline、matrix、analysis 和 summary 脚本。
- 指标完整：同时比较 latency、faults、RSS、swap、hint events。
- 结果诚实：没有强行宣称某个策略单方面最优，而是给出 Pareto 权衡。

## 当前不足

- hint calls 仍接近 10 万，未达到最初大幅减少 syscall 的理想目标。
- 只在当前本地模型和机器上完成 N=3 验证，泛化性还需更多模型和硬件验证。
- KV cache 目前主要是分析和投影，尚未实现分页式 KV 管理。
- `MADV_DONTNEED`/`MADV_PAGEOUT` 仍是显式实验项，未作为默认策略。
