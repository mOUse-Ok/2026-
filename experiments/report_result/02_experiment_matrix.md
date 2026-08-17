# 冻结代码下的答辩实验体系

## 资源判定

当前机器观测到总内存约 7.6 GiB、可用约 6.3 GiB，而目标模型为 16,356,375,168 bytes（约 15.23 GiB）。即使不考虑 KV、运行时、文件缓存和系统余量，也无法可靠完成任何真实模型实验。

因此：本机只用于代码健康检查、分析器与归档校验；所有加载该模型、cgroup 压力、server、trace-on/off 或 mmap 实验必须迁移到可用内存至少 24 GiB、物理内存至少 32 GiB 的机器。

## 实验矩阵

| ID | 答辩问题 | 条件与重复 | 资源 | 现有产物 | 最终结论类型 |
|---|---|---|---|---|---|
| H0 | 当前代码是否健康 | 8 CTest + 3 Python tests，HEAD | 本机 | 本会话已通过 | 正确性门槛 |
| M1 | mmap 三开关的真实代价 | baseline/skip-populate N=5；skip-sequential/random N=3 | 32GB，无限内存 | `experiments/experiment_0815/` 已恢复：16 个 run、原始 JSONL、CSV、输出和 manifest | **可复用**为源码等价的当前负结果；模型 SHA 缺失 |
| M2 | memory pressure 下 admission 是否避免不合适 populate | `default/auto/skip`，MemoryMax=20G、15G、12G；每格 N=3 | 32GB + cgroup v2 | `experiments/experiment_M2_mmap_pressure/`：27/27 完成 | A−：当前源码等价的 admission decision 与压力取舍 |
| T1 | trace 最小开销 | trace-off vs trace-on/controller-off，参数完全一致，N=5 | 32GB，MemoryMax=20G | `experiment_T_current_head/T1_plain_vs_trace` 的两组均写出完整 trace | **INVALID**：trace-off 对照失效，必须重跑 |
| T2 | Router prefetch 是否有净收益 | controller off vs `expert_prefetch`，同一 cgroup、Latin 交错、N=5 | 32GB，先做 20G；仅在均可完成后做 12G，N=3 | `experiment_T_current_head/T2_*` 的 task created/admitted/issued 均为 0 | **INVALID**：未实际执行预取，必须重跑 |
| T3 | 生命周期是否闭合 | Memory Object off/on，各 1 个完整 trace；必要时 N=3 开销对照 | 32GB | `experiment_T_current_head/T3_memory_object`：N=3/组 | A−机制/负结果：working-set 生命周期真实触发；不覆盖 COLD/DONTNEED/rescue |
| R1 | Memory Object + COLD/DONTNEED + rescue 是否安全、可归因 | object-only、COLD、DONTNEED、COLD+rescue，各 N=3；同一受压 cgroup、交错顺序 | 32GB + cgroup v2 | 无合格当前 HEAD A/B | 当前 HEAD 完成率、回收/readmission/refault 与输出一致性 |
| A1 | 384MiB 场景是否可归因 | 同一 survival 参数：trace-off、trace-on/off、working-set、reclaim；每组 N=3 | 32GB + cgroup v2 | `experiment_A1_matched_completion`：12/12 完成、输出一致 | A−机制/负结果：DONTNEED 已实际 issued；不支持“rescue 提升完成率” |
| S1 | KV Slot Admission 是否有服务价值 | off/on，`--parallel 2`，固定请求序列，N=3 | 32GB + server/cgroup | `experiment_S1_kv_slot_admission`：on 组 2 checks/2 allowed/0 deferred | C：低压 pass-through 与输出等价；未覆盖 defer/rejection/公平性 |

## 执行顺序与停止规则

1. 先执行 H0；失败即停止全部性能实验。
2. M1 已恢复于 `experiments/experiment_0815/`，可复用为无 cgroup mmap 消融；不得拿它替代 M2。
3. 再运行 M2、T1、T2、T3、R1、A1；S1 只在核心矩阵完成后执行。
4. 每次启动前必须确认：HEAD 精确匹配、工作树干净、模型文件存在、模型 SHA、prompt SHA、二进制 SHA、可用内存 >=24 GiB、cgroup v2 可用。
5. 任一条件发生 OOM、输出 SHA 不一致、trace drop 非零、manifest 缺失、工作树变脏：该格标记 `INVALID`，不得补算或纳入均值。
6. 先报告每个指标的均值、标准差、所有样本和运行顺序；不因“首轮异常”删除样本。可单列敏感性分析，但不能替换主结果。

## 指标与比较规则

- 主指标：完成率、wall time、prefill、decode mean/p95、major/minor faults、RSS peak、cgroup peak/events、输出 SHA。
- 机制指标：hint created/admitted/issued/failed、first-use matched、Memory Object pending/active 终态、trace enqueued/written/dropped。
- 只把同一 host、同一 commit、同一 binary SHA、同一模型 SHA、同一 prompt SHA、同一参数和同一 cgroup 的样本做均值比较。
- `M1` 的 skip-populate 结论必须同时报告 load、prefill、decode、fault 与 RSS；不能只报总 wall 或 mmap 时间。
- `R1` 必须同时报告 cgroup `oom/oom_kill`、`memory.peak`、RSS、`workingset_refault_*`、Memory Object 的 evictions/readmissions、COLD/DONTNEED issued/failed、rescue state/trigger，以及成功组输出 SHA。`madvise` 返回成功不是“物理页已回收”的证据。

## 与图中 C 级缺口的对应

| 图中功能 | 需要迁移的实验 | 判定 |
|---|---|---|
| Router Expert prefetch | T2 | 是；补同提交、同 cgroup 的 off/on A/B。 |
| Memory Object + COLD/DONTNEED + rescue | R1（A1 只作单独完成性补充） | 是；这是图中最直接缺失的受压回收实验。 |
| trace 本身开销 | T1 | 是。 |
| KV slot admission | S1 | 是，但优先级低于核心矩阵。 |
| KV policy | 无 | 否；运行时 allocation/eviction/quantization 链路尚不存在，冻结代码下不能靠迁移实验补齐。 |
| auto mmap admission | M2 | 是；图中缺的是有限 `memory.max`，不是已丢失的 M1 无上限消融。 |

M1 已恢复为可复核的无上限消融；它仍不是图中 auto mmap 行所列“有限 cgroup”缺口的替代品。
