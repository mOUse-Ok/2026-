# 四套 32GB 实验结果的答辩准入审计

审计对象：`experiments/experiment_M2_mmap_pressure/`、`experiments/experiment_T_current_head/`、`experiments/experiment_A1_matched_completion/`、`experiments/experiment_S1_kv_slot_admission/`。它们均运行于 `88fc9e1`；当前 `a5d8005` 相对该提交只有归档文档差异。所有带 manifest 的模型 SHA 都是 `null`，故不得称为无条件 A 级。

## 可直接用于说明增量的结果

| 实验 | 能佐证的实现增量 | 可使用的答辩表述 | 不能使用的表述 |
|---|---|---|---|
| M2 | `llama_mmap_populate_admit()` 的 `default/auto/skip`、Sparse-MoE headroom 判定、cgroup 观测和 `MMAP_POPULATE_ADMISSION` 事件 | 27/27 完成、输出一致；auto 在 20G 选择 `DEFAULT/MODEL_FITS_HEADROOM`，在 15G/12G 稳定选择 `SKIP_POPULATE/SPARSE_MOE_MODEL_EXCEEDS_HEADROOM`。12G 下 default 的 mean wall/major faults 为 106.82s/563k，auto 为 70.45s/441k，证明该 workload 下 auto 避开了 default 的较差压力路径。 | “auto 在所有条件下性能最优”、“避免 OOM”或“20G 有加速”；所有格均完成，20G auto 与 default 的性能差异未作统计显著性证明。 |
| T3 | `ExpertMemoryObjectTracker` 的 demand→working-set→eviction→readmission 状态机与终态不变量 | 20G N=3/组：working-set=512MiB 时每 run 有 77,664 admissions、76,481 evictions、53,799 readmissions，`invariant_violations=0` 且终态 pending/active 为 0；输出一致。它还给出负结果：wall 72.58→139.98s、decode p50 114.6→853.8ms、RSS 基本不变。 | “Memory Object 提升性能/降低 RSS”，以及任何 COLD、DONTNEED 或 rescue 的效果结论。T3 本身的 COLD/DONTNEED issued 均为 0。 |
| A1（trace-on 组内） | 受 384MiB cgroup 时的 working-set 与 `MADV_DONTNEED` 真实执行、输出不变性 | working-set/reclaim 各 N=3 都完成、输出相同；reclaim 每 run `MADV_DONTNEED issued=2240, failed=0`，并完成 115,089 admissions、114,942 evictions、92,067 readmissions，终态不变量为 0。可表述为“回收通路已在极限 cgroup 下真实运行，未改变本 workload 输出”。 | “reclaim/rescue 提高完成率或降低 major fault”；四组均完成且 major faults 约 2.93M，reclaim 相对 trace-on observation 的 wall/prefill 明显变差。 |
| S1 | server `--parallel 2` 的 KV slot admission hook、开关可观测性 | 开关 off 的三次运行无 summary；on 的三次每次都是 `checks=2, allowed=2, deferred=0`，两请求输出哈希一致。可表述为“低压下 admission pass-through 已接入 server，且未改变输出”。 | “具备背压/拒绝/公平性能力已验证”、“改善 TTFT/TPS”或“统计上无开销”；没有 manifest/cgroup 快照，且没有任何 deferred/rejected slot。 |

## 不得纳入答辩性能结论的 T 子实验

### T1：无效的 trace 开销对照

所谓 plain/trace-off 与 trace-on 两组都把 `LLM_MEM_TRACE_EXPERT=1`、`LLM_MEM_TRACE_MEMORY=1` 和各自 trace 目录传给进程；两组都产生完整 `expert_trace.jsonl` 和 `memory_trace.jsonl`，都有 `TRACE_START/TRACE_END`、`EXPERT_ROUTE`、`EXPERT_TASK_SUMMARY` 和 Memory sink 事件。它们不是 trace-off vs trace-on；因此 5.8% wall 或 3.4% decode p50 不能作为 trace 开销。目录里没有 CMakeCache 或构建日志，无法在“所谓 trace-off binary 实际仍以 `LLM_MEM_TRACE` 构建”“启动器调用错二进制”“运行后复制了 trace”三者间取证；重做时 trace-off 组必须没有 JSONL/event sink，并归档 CMakeCache、二进制 SHA 和启动命令。原提示词没有这些构建和反证验收条件，故这是提示词的验收缺口，而非可由现有数据归因的算法问题。

### T2：无效的 Router prefetch 效果对照

prefetch 组 manifest 的 `LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=0`，虽把 `LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=expert_prefetch` 写成了名称。源码的实际 gate 只检查前者，因此 Router 回调直接返回，20G 与 12G 的全部 `EXPERT_TASK_SUMMARY` 都是 `created=0, admitted=0, issued=0`。`TOPK=0` 不是根因：开启 gate 后它会退化为选择全部 Router 专家。虽然存在 Router trace，但没有生成 `ExpertHintTask`，更没有可归因的 `MADV_WILLNEED`。因此该实验只能证明 Router 观测还在工作，不能证明预取无效，更不能证明预取有效。根因是原提示词直接建议使用仅设 controller 的矩阵 case，未覆盖/清理已存在的 `PREFETCH=0`，也未把 task/hint 计数设为准入门槛；修订提示词已修复这两点。

## 最终答辩定位

本批数据最适合展示“系统增量已运行、受压时可观测、并且负结果被如实保留”：mmap auto admission 具有可复核的分支行为；Memory Object 与 DONTNEED 已跑过真实状态机和系统调用；KV admission 已接入 server 的低压路径。性能正向主张应限于 M2 在 12G 下相对 default 的该 workload 取舍。Router prefetch、trace 开销、COLD/rescue、KV defer/rejection/fairness 仍是待补实验，不能提前宣传。
