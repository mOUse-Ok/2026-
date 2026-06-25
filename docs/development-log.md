# 开发过程记录

## 说明

本文档记录初赛阶段主要开发过程、关键问题、解决方法和阶段性结论。更细粒度的本地实验流水记录保存在 `llama.cpp/trace_output/contest_runs/progress_log.md`，该路径用于本地实验记录，不提交仓库。

## 阶段 1：建立 trace 与 baseline

### 目标

建立可运行的 LLM 推理访存分析链路，观察 tensor load、KV cache、expert routing、RSS、swap 和 page faults。

### 处理

- 在 `llama.cpp/trace/` 中扩展 trace sink。
- 使用 `run_trace_pipeline.sh` 自动运行推理、收集 JSONL trace、生成分析结果。
- 使用 `analyze_trace.py` 聚合关键指标。

### 结果

baseline 能稳定产出 tensor、KV、expert、memory trace，为后续 OS hint 实验提供对照。

## 阶段 2：安全 OS hint 原型

### 问题

LLM 推理首次访问权重和 MoE expert slice 时存在 major faults 集中爆发，decode latency 受到影响。

### 解决方法

实现默认关闭的 OS hint 实验路径：

- `madvise(MADV_WILLNEED)`
- `madvise(MADV_SEQUENTIAL)`
- `posix_fadvise(POSIX_FADV_WILLNEED)`
- expert-aware prefetch

所有策略通过环境变量开启，避免改变默认推理行为。

### 结论

expert-aware prefetch 能显著降低 major faults 和 decode latency，但会提高 RSS，并产生约 10 万次 hint calls。

## 阶段 3：Expert cache 替换策略模拟

### 问题

需要判断 LRU/LFU/window-LFU/least-stale 是否能在有限 cache budget 下替代 route prefetch。

### 解决方法

实现 trace-driven 离线模拟，比较：

- `route`
- `lru`
- `lfu`
- `window_lfu`
- `least_stale`

并测试 128/256/512/768/1024 MiB 预算。

### 遇到的问题

heap 化 `least_stale` eviction 时初版排序方向错误，可能淘汰更早复用的 item。修正后改为优先淘汰预计更晚复用的 item。

### 结论

在当前 trace 下，朴素 LRU/LFU 类策略在 <=1 GiB 预算下 miss 和 eviction 过高，不适合作为真实运行主候选。

## 阶段 4：Route top-k 和 coalescing

### 问题

完整 route prefetch 保留 coverage，但 hint calls 太高。需要减少 syscall 数量。

### 解决方法

- 测试 `LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK=1/2/4/6`。
- 实现 route slice coalescing。

### 结论

top-k 会显著破坏 prefetch coverage，major faults 回升明显。coalescing 受限于 expert slice 地址连续性，收益有限。因此不能通过简单截断解决问题。

## 阶段 5：异步 expert prefetch

### 问题

同步 hint call 会干扰 decode 关键路径。

### 解决方法

实现用户态异步 hint queue：

- `LLM_MEM_TRACE_OPT_EXPERT_ASYNC`
- `LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE`
- `LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS`

增加 `EXPERT_ASYNC_SUMMARY`，记录 enqueue、issued、fallback、queue depth 等指标。

### 结论

异步化能降低 syscall 对 decode 路径的直接影响，但 FIFO 不够，需要优先级调度。

## 阶段 6：Deadline-aware priority

### 问题

异步队列如果不区分任务紧迫度，可能先处理距离使用点较远的 expert slice。

### 解决方法

实现 priority mode：

- `score`
- `deadline`
- `deadline_score`

其中 `deadline_score` 先按 step/layer 接近程度排序，再按 route score 排序。

### 结论

N=3 重复实验中，`deadline_score` 是当前综合最优策略。

## 阶段 7：Route TTL 与重复实验

### 问题

需要减少重复 hint，并避免单次运行噪声影响结论。

### 解决方法

- 实现 route hint TTL。
- 将大量 skip 明细改成 `EXPERT_ROUTE_HINT_SUMMARY`。
- 新增 `summarize_repeat_runs.py` 聚合 N 次运行。
- 新增 `run_finalist_repeat_matrix.sh` 固化最终矩阵。

### 结论

`decode_ttl1` 能减少 hint calls，但延迟和 RSS 不如 `deadline_score`。最终报告采用 N=3 均值和标准差。

## 当前状态

- 主推荐策略：`route_all_async4_deadline_score`
- 对照策略：`expert_prefetch_route_all`
- Pareto 备选：`decode_ttl1`
- 已完成构建、Python 静态检查、脚本 dry-run 和 N=3 重复实验。

## 后续计划

- 继续探索在保持 coverage 的同时减少 syscall 数量，例如更精确的 range batching。
- 在更多模型、不同内存压力、不同输入长度下验证稳定性。
- 将当前本地实验结果整理进正式答辩材料。
