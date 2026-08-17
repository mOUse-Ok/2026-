# 决赛 README 核心证据闭环报告

考察日期：2026-08-09  
当前源码锚点：`main@01817a0aab55791fbcb54d2021f78a15e02307ab`

## 结论先行

本轮没有得到“当前 HEAD 的 Expert Prefetch 稳定加速”证据。相反，在固定的 Qwen/CPU/7040 MiB cgroup 冷缓存实验中，`expert_prefetch` 相对 OFF 的五次均值 decode 平均延迟增加 **12.24%**，P95 增加 **17.65%**；major fault 仅少 **0.50%**，远小于性能波动和控制器实际发 hint 数的波动。因此 README 只能把它写成“可运行、可观测的 router-driven prefetch 主线”，不能写成当前配置下的性能加速承诺。

Memory Object Lifecycle、Shadow Working Set、probation 和 `MADV_COLD` 都已在真实 Qwen 推理中执行。它们的正确性/语义证据是强的；性能收益不是。特别是历史的 Shadow-only vs COLD 三次对照显示，COLD 真正执行且无 syscall 失败，但平均 decode 延迟增加 **26.48%**、major fault 增加 **45.67%**。这是一条应保留的负结论。

Runtime Rescue 已有“检测到坏状态后暂停/限速 COLD、恢复发 hint”的阶段性端到端证据；本轮的当前 HEAD 高-issue guard 也证明它在一个正常样本中保持未触发。但尚没有与坏状态严格配对、重复的 Rescue OFF/ON 性能 A/B，因此不能写成“已证明恢复端到端性能”或“零误触发”。

## 证据等级与可比性

| 标签 | 含义 | 本报告中的用途 |
|---|---|---|
| **HEAD-Qwen** | 本轮的当前 `01817a0` binary + 真实 Qwen 运行 | P0 五次 OFF/ON、当前 Lifecycle/COLD guard。 |
| **历史 Qwen** | `dc44a60` 提交前的 dirty 工作树运行；有 manifest 和 raw JSONL | Lifecycle 五次、Working Set scan、COLD A/B、Rescue/Calibration 阶段结果。用于机制和历史结论，不能冒充 HEAD 性能数字。 |
| **单元** | 当前 HEAD CTest | mmap phase admission、状态机、优先级、router 同步、memory object 与 calibration 公式。 |

HEAD-Qwen 的 manifest 均记录 `git_dirty=true`，原因是本工作区已有未跟踪的 `docs/technical-archaeology-report.md`；运行前 `git diff --name-only` 为空，且所有 manifest 指向同一 HEAD、同一 binary SHA-256、同一模型 size/mtime、同一 prompt hash 和同一 cgroup。原有 `summarize_repeat_runs.py` 正确地拒绝 dirty manifest，因此本报告直接从原始 `analysis/metrics.json`、`process_metrics.json` 与 JSONL 汇总，**没有改写 manifest 来绕过校验**。

P0 的 `CACHE_MODE=cold` 采用 `prepare_model_cache.py` 的 `posix_fadvise(POSIX_FADV_DONTNEED)`；它不是 root 级的全局 drop-caches。cgroup 实际 `memory.max=7,381,975,040` bytes（7040 MiB），`MemorySwapMax` 未限制。当前 trace/profile 没有记录 swap peak，故“swap”不是这组数据可安全报告的指标。

---

## 1. 最终机制成熟度表

“default mainline”指项目的默认**优化主线/profile**，不等于无环境变量的普通 llama.cpp 默认行为：`LLAMA_MEM_TRACE` 为编译期开关，`run_trace_pipeline.sh` 的控制器默认值仍为 `off`。

| 机制 | Implementation | Evidence | Product / Mainline status | 可安全理解 |
|---|---|---|---|---|
| Trace Infrastructure | implemented | repeated end-to-end | default mainline | 当前 P0 十次均有退出码 0、expert/memory sink 零丢失；编译关闭时为空 stub。 |
| Expert Prefetch | implemented | repeated controlled A/B | default mainline | 当前默认优化 profile 是 `expert_prefetch`；P0 已量化其本机效果，但结果为不稳定且均值退化。 |
| Memory Object Lifecycle | implemented | repeated end-to-end | optional mechanism | demand/activation/completion、slot acquire/release 与退出不变量在真实 Qwen 中闭合。 |
| Single-flight（inflight hint aggregation） | implemented | repeated end-to-end（路径） | optional mechanism | slot 生命周期反复执行；本工作负载的 `inflight_hint_aggregated=0`，故不能宣称已观察到实际合并收益。 |
| Stale Cancellation | implemented | repeated end-to-end | optional mechanism | 当前 guard 与历史运行都出现少量语义 stale cancel，且状态最终归零。 |
| Shadow Working Set | implemented | repeated end-to-end | optional mechanism | 有 budget、admission、eviction、readmission、protection 语义；不是硬物理内存上限，也未证明提速。 |
| Probation | implemented | repeated end-to-end | optional mechanism | eviction 后进入 probation，部分被 readmission 取消；当前 COLD trace 可复核。 |
| `MADV_COLD` | implemented | repeated controlled A/B | research controller | 在 HEAD-Qwen 真正调用且无失败；历史 A/B 否定了稳定净收益。默认关闭。 |
| Runtime Rescue | implemented | repeated end-to-end | research controller | 历史触发路径发生过 COLD suspend/gate bypass；当前 guard 一次保持静默。没有严格重复的坏状态 OFF/ON 性能证明。 |
| Calibration Shadow | implemented | repeated end-to-end | optional mechanism | 真实运行达到 16 个 healthy sample 并写出 environment-specific baseline；其本身只观测。 |
| Calibrated Controller | implemented | smoke / staged end-to-end | research controller | 多个阶段运行到 calibrated、probe、recovery 或 disabled 状态；没有统一、受控的性能 A/B。 |

实现定位：`trace/tensor_trace.cpp` 负责环境开关、任务、COLD、Rescue 与校准控制；`trace/expert_memory_object.{h,cpp}` 保存 lifecycle/working-set/probation；`trace/expert_calibration_shadow.{h,cpp}` 是 observation-only calibration；`trace/trace_writer.cpp` 负责有界 JSONL sink。当前构建中 `test-router-tensor-observation-sync`、`test-expert-hint-priority`、`test-expert-task-lifecycle`、`test-expert-memory-object`、`test-expert-calibration-shadow` 均通过。

## 2. Current HEAD Expert Prefetch A/B

### 2.1 固定条件与有效性

| 项目 | 值 |
|---|---|
| 运行组 | `readme_head_ab_20260809_{baseline,expert_prefetch}_r1..r5` |
| 次数 / 顺序 | OFF 与 `expert_prefetch` 各 5；Latin 交错顺序 |
| binary | `llama-cli` SHA-256 `9695cb81a7e34311…` |
| 模型 | `Qwen3.5-35B-A3B-Q3_K_M.gguf`，16,356,375,168 bytes；全部同一 mtime |
| prompt / output | 同一 prompt SHA-256 `59f51358b13d…`；十次输出 SHA-256 均为 `87bfd3e1e62672cb…` |
| 推理参数 | CPU-only、8 threads、80 predicted tokens、batch 512、ctx 2048、seed 1234、temp 0 |
| CPU / kernel | affinity 0–31；Linux 6.6.114.1-microsoft-standard-WSL2 |
| 内存与缓存 | 同一 7040 MiB transient user cgroup；`cold` / `posix_fadvise_dontneed` |
| trace 完整性 | 十次均 exit 0；expert 与 memory sink 都是 `dropped=0` 且 `enqueued=written` |

### 2.2 五次汇总（均值 ± 标准差；括号为 min–max）

| 指标 | OFF | expert_prefetch | 相对 OFF |
|---|---:|---:|---:|
| wall time (s) | 62.04 ± 2.68 (58.80–65.14) | 67.78 ± 6.60 (58.31–76.33) | +9.26% |
| decode avg (µs) | 286,065 ± 5,651 | 321,073 ± 52,533 | **+12.24%** |
| decode P95 (µs) | 415,008 ± 11,292 | 488,252 ± 115,673 | **+17.65%** |
| decode P99 (µs) | 483,592 ± 7,973 | 574,821 ± 141,116 | +18.87% |
| decode throughput (tok/s) | 3.497 ± 0.068 | 3.173 ± 0.445 | -9.27% |
| prefill avg (µs) | 9,231,239 ± 156,571 | 9,793,542 ± 782,906 | +6.09% |
| major faults | 802,297 ± 13,564 | 798,302 ± 6,795 | -0.50% |
| minor faults | 969,215 ± 4,683 | 972,263 ± 4,659 | +0.32% |
| peak RSS (GiB) | 6.282 ± 0.255 | 6.363 ± 0.035 | +1.30% |

在 ON 组中，真实 `MADV_WILLNEED` hint 数为 **4–2,420**（均值 505.4），建议数据量为 **1.84–1,055.65 MiB**（均值 220.62 MiB），无 OS hint error。first-use 任务匹配率为 75.0–93.3%，但所有逻辑 first-use 的覆盖率仅 0.003–2.13%：这是 value gate 仅放行少量任务的直接证据，不应将“匹配率高”解释为覆盖了所有专家需求。

### 2.3 P0-1 判定

**判定：当前条件下是“无明显 fault 收益、速度收益有限且均值退化”。**  
它仍然是一个真实运行的主线优化 profile：任务、queue、first-use 和 OS hint 都被记录且正确收尾；但本组数据不支持以性能提升作为 README 结论。控制器实际发 hint 的跨 run 波动也是应继续调查的事实，而不是应删除的异常点。

## 3. Memory Object Lifecycle 证据

### 3.1 历史 Phase 1：Lifecycle OFF vs ON（各 N=5）

Phase 1 的 ON 组开启 Memory Objects、single-flight 与 semantic stale cancellation，OFF 组关闭三者；两组均为历史 dirty-worktree Qwen 数据，故只用来估计开销量级。

| 指标 | OFF 均值 | Lifecycle ON 均值 | ON 相对 OFF |
|---|---:|---:|---:|
| wall time (s) | 58.54 | 61.76 | +5.50% |
| decode avg (µs) | 276,233 | 317,790 | +15.04% |
| major faults | 721,164 | 681,474 | -5.50% |
| minor faults | 986,461 | 994,105 | +0.78% |
| peak RSS (GiB) | 6.365 | 6.513 | +2.33% |

这不是“Lifecycle 提升性能”的实验，且方差明显；正确的表述是：该历史 workload 中 Lifecycle 的端到端开销不是零，也没有因此损坏正确性。

### 3.2 当前 HEAD 的语义闭合

本轮两次 HEAD-Qwen Rescue guard 都启用 Lifecycle；每次均有：

| 检查项 | OFF guard | Rescue-ON guard |
|---|---:|---:|
| semantic demands / activation / completion | 102,222 / 102,222 / 102,222 | 102,222 / 102,222 / 102,222 |
| hint slot acquire / release | 102,222 / 102,222 | 102,222 / 102,222 |
| pending / active at shutdown | 0 / 0 | 0 / 0 |
| invariant violations | 0 | 0 |
| semantic stale cancellations | 12 | 5 |
| output hash | 相同 | 相同 |

所以 README 可以说“Lifecycle 在真实 Qwen 推理中维护并闭合 demand/slot 状态”；不能说它已被证明改善时延。

## 4. Semantic Working Set 证据

### 4.1 Budget scan（历史 Qwen，16 decode tokens，每档一次）

| budget | current bytes | peak bytes | admissions | evictions | readmissions | protected skips | unresolved due to protection |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 MiB | 268,361,728 | 348,684,288 | 40,782 | 40,191 | 17,778 | 6,477,371 | 852 |
| 512 MiB | 536,502,272 | 537,427,968 | 35,169 | 33,986 | 12,165 | 6,658,418 | 0 |
| 1024 MiB | 1,073,496,064 | 1,074,298,880 | 32,471 | 30,103 | 9,467 | 6,508,915 | 0 |

机制行为符合预期：增大 budget 后 admission、eviction 与 readmission 都下降，且 current working-set size 回到各自预算附近。它同时暴露一个必须如实保留的边界：256 MiB 下因 active/protected 对象出现 852 次无法立即满足 budget，peak 可短暂超过 budget；因此它是**容量约束的语义工作集**，不是严格的物理 bytes cap。

### 4.2 当前 HEAD 佐证

当前 guard 的 1024 MiB Working Set 有 63,748 admissions、61,380 evictions、39,904 readmissions、6,933,415 protected skips，shutdown 时 current=1,073,496,064 bytes，`budget_unresolved_due_to_protection=0`。这确认 budget scan 中的语义仍在 HEAD 上实际运行。

## 5. MADV_COLD 的真实结论

### 5.1 它确实作用于 Linux 页面

当前 HEAD guard 每次均记录：

- `madv_cold_candidates = madv_cold_issued = 46,256`
- `madv_cold_failed = 0`
- `madv_cold_bytes = 20,965,908,480`
- `post_cold_readmissions = 26,192`
- `cold_protected_violation = 0`

因此可以安全写“系统实际执行了 `MADV_COLD` 并记录了 probation/readmission”；不能把成功 syscall 写成“已物理回收了同等字节数”。

### 5.2 Shadow-only vs COLD（历史配对 A/B，各 N=3）

两组均为 1024 MiB Shadow Working Set、相同 Qwen workload；差异是 COLD 组启用 `LLM_MEM_TRACE_OPT_EXPERT_MADV_COLD_RECLAIM=1`。每个 COLD run 均发出 46,256 个 COLD（20.97 GB）且失败为 0。

| 指标 | Shadow only | Shadow + COLD | COLD 相对 Shadow |
|---|---:|---:|---:|
| wall time (s) | 78.43 ± 2.07 | 84.98 ± 7.66 | +8.36% |
| decode avg (µs) | 472,062 ± 4,292 | 597,071 ± 85,365 | **+26.48%** |
| decode P95 (µs) | 644,849 ± 6,388 | 862,428 ± 154,076 | +33.74% |
| major faults | 471,983 ± 3,953 | 687,518 ± 177,210 | **+45.67%** |
| minor faults | 1,057,869 ± 9,020 | 1,017,652 ± 63,992 | -3.80% |
| peak RSS (GiB) | 6.425 ± 0.102 | 6.478 ± 0.112 | +0.83% |

**最终结论：主动 COLD 的净性能收益没有证明；在这条受控历史线中，它更像破坏 Prefetch/增加重大缺页的来源。** 这应作为 README 的负结果与设计边界，而非继续调参的理由。

## 6. Runtime Rescue 的真实结论

### 6.1 坏状态中的动作证据（历史 Qwen）

`phase2d_gr_highn1_r0` 的 `gate_recovery` 在 decode step 3 触发：`runtime_rescue_triggered=1`、`runtime_rescue_cold_suspended=1`、`runtime_rescue_gate_bypass_steps=5`。触发前 3 steps 的 issued=0、major faults=14,096；触发后的首 5 steps issued=4,789、major faults=3,798。另一个 `phase2d_gr_highn1_r25` 也在 step 3 触发，并出现相同的 suspend/bypass 行为。

`phase2e_highn1_on1` 的 COLD-rate recovery 也在 step 3 触发：先 suspend COLD，step 12/17/22 分别进入 25%/50%/100% probe；step 35 检测到再退化并将 COLD 禁用于本次运行。它的 early-3 issued=0、major faults=14,439，post-trigger first-5 issued=4,796、major faults=3,085。

这些记录直接证明“检测 → 控制动作 → 后续 hint 发出”的状态机执行；early/post 窗口不是独立随机 A/B，故不能单靠它们宣称全局性能因果恢复。

### 6.2 False-positive guard（当前 HEAD）

本轮在高-issue 正常路径中运行 Rescue OFF 与 ON；唯一配置差异是 ON 设置 `LLM_MEM_TRACE_OPT_EXPERT_RUNTIME_RESCUE=1` 和 `gate_recovery`。两次的 output SHA-256 相同。ON 的总结为：

| 字段 | 值 |
|---|---:|
| early 3 steps issued | 2,880 |
| early 3 steps major faults | 1,778 |
| runtime_rescue_triggered | **0** |
| runtime_rescue_cold_suspended | 0 |
| gate-bypass steps | 0 |
| COLD issued / failed | 46,256 / 0 |

OFF/ON 的 COLD candidates、COLD issued、COLD bytes、Working Set admission/eviction/readmission 均相同；两次的发 hint 数只相差 7（102,210 vs 102,217），属调度扰动量级。两次单 run 的 decode 分别为 557,624 与 633,210 µs，不能用来声称 ON 的速度影响。

**判定：基本 false-positive guard 通过一次——正常高-issue 样本未触发。** 仍不能写“不会误触发”或“Rescue 已在严格 A/B 中恢复性能”。

## 7. Calibration 与 Calibrated Controller

Calibration Shadow 的设计为 observation-only（`expert_calibration_shadow.h`），最少 16 个 healthy sample 后才标为 calibrated。历史 `phase2eb` 运行中已多次进入 `calibrated`，例如：

| 运行 | MemoryMax | calibration valid step | median issue ratio | median major faults | 最终 controller 状态 |
|---|---:|---:|---:|---:|---|
| `phase2eb_r1_v2` | 8192 MiB | 36 | 0.992 | 571.0 | disabled: RE_DEGRADATION |
| `phase2eb_r1_v3` | 8192 MiB | 35 | 1.000 | 385.0 | disabled: LOW_BENEFIT |
| `phase2eb_r2_v2` | 7168 MiB | 27 | 1.000 | 357.5 | recovery |
| `phase2eb_r2_v3` | 7168 MiB | 19 | 1.000 | 556.5 | disabled: LOW_BENEFIT |
| `phase2eb_r3_v3` | 8192 MiB | 19 | 1.000 | 425.0 | disabled: LOW_BENEFIT |

这是有价值的“环境尺度会变化、控制器能进入/退出状态”的端到端证据，但运行配置、修订版本与结果状态不构成统一 A/B。Calibration 可以表述为“为 research controller 提供本进程、当前环境的观测基线”；Calibrated Controller 只能表述为“实验性、已跑通多种状态”，不能表述为性能产品能力。

## 8. Correctness 总表

| 范围 | 证据 | 结果 |
|---|---|---|
| 当前 P0 OFF/ON | 10 个 manifest、output hash、process metrics、summary.json | 10/10 exit 0；同一输出 hash；所有 enabled sink 零 drop 且完全写入。 |
| 当前 Lifecycle | 两个 HEAD-Qwen guard 的 `EXPERT_MEMORY_OBJECT_SUMMARY` | 102,222 demand/activation/completion；slot acquire=release；pending=active=invariant violations=0。 |
| 当前 COLD | 两个 HEAD-Qwen guard | 每次 46,256 issued、0 failed、0 protected violation；存在 post-COLD readmission。 |
| 当前 Rescue guard | `EXPERT_RUNTIME_RESCUE_SUMMARY` | 触发计数 0；无 suspend/bypass；输出一致。 |
| 历史 Working Set | 256/512/1024 budget summaries | admission/eviction/readmission/protection 全部实际计数；256 MiB 的 protection 边界被显式记录。 |
| 当前单元 | 8 个 CTest | mmap phase admission、router tensor sync、priority、lifecycle、memory object、calibration shadow 全通过。 |

## 9. 最终 README Claim → Evidence 表

| Claim | Evidence | Runs | Result | README wording |
|---|---|---:|---|---|
| Router-driven Expert Prefetch 是真实可运行主线 | HEAD 源码 + P0 trace | 5 ON | task/queue/OS hint/first-use 均真实记录，零 drop | “提供 opt-in 的 router-driven expert prefetch profile。” |
| Expert Prefetch 对当前 HEAD 的性能影响 | HEAD P0 Latin A/B | 5 OFF + 5 ON | decode +12.24%，major faults -0.50%，hint 数高波动 | “在本机 7040 MiB 冷缓存条件下未观察到稳定加速。” |
| Memory Object Lifecycle 正确闭合 | HEAD guards + Phase 1 | 2 HEAD + 5 历史 ON | demand、slot、pending/active/invariant 闭合 | “维护专家需求与 hint 生命周期，并记录可审计状态。” |
| 容量约束 Semantic Working Set | budget scan + HEAD guard | 3 历史 budget + 2 HEAD | budget、admit/evict/readmit/protection 均有实测 | “维护容量约束的语义工作集；protected 对象可造成暂时超额。” |
| MADV_COLD 能真实作用于 Linux 页面 | HEAD guard memory/OS events | 2 HEAD | 46,256 syscall/run，0 failed | “可选地向 Linux 发出 MADV_COLD 建议，并记录其结果。” |
| MADV_COLD 的净性能收益 | 历史 Shadow/COLD A/B | 3 + 3 | decode 与 major faults 均变差 | “未证明主动 COLD 的稳定净性能收益。” |
| 过度回收会造成 Prefetch degradation | COLD A/B + Rescue windows | 6 A/B + stage runs | COLD 线性能反噬；trigger 前 issued collapse | “将 COLD 保留为受保护的研究控制器。” |
| Runtime Rescue 的恢复动作 | 历史 step summaries | 至少 2 trigger runs | detect、suspend、bypass、post-trigger issuance 都发生 | “实验性 guard 可在检测到定义的坏状态后暂停 COLD。” |
| 正常状态下 Rescue 是否误触发 | HEAD false-positive guard | 1 ON + 1 OFF | ON trigger=0，COLD/WS/output 保持一致 | “一个高-issue guard 样本保持静默。” |
| 整体 correctness | P0 + guard + unit tests | 12 HEAD Qwen + 8 CTest | 输出 hash 一致，trace 完整，状态不变量闭合 | “实验路径保留输出一致性与 trace 完整性检查。” |

## 10. 可视化所需的数据表

下面是可直接制图的聚合数据；本轮按要求不画图。

### 图 1：Current HEAD OFF vs Expert Prefetch

| condition | N | decode avg µs mean | decode avg range | major faults mean | major faults range |
|---|---:|---:|---:|---:|---:|
| OFF | 5 | 286,065 | 280,675–295,554 | 802,297 | 787,557–823,920 |
| expert_prefetch | 5 | 321,073 | 284,230–411,256 | 798,302 | 793,747–810,068 |

### 图 2：Semantic Working Set budget sweep

| budget MiB | admissions | evictions | readmissions | protected skips | current bytes | peak bytes | major faults |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 40,782 | 40,191 | 17,778 | 6,477,371 | 268,361,728 | 348,684,288 | 492,277 |
| 512 | 35,169 | 33,986 | 12,165 | 6,658,418 | 536,502,272 | 537,427,968 | 432,455 |
| 1024 | 32,471 | 30,103 | 9,467 | 6,508,915 | 1,073,496,064 | 1,074,298,880 | 431,597 |

### 图 3：COLD 的副作用

| condition | N | COLD issued | COLD bytes / run | post-COLD readmission | prefetch comparison proxy | major faults mean | decode avg µs mean |
|---|---:|---:|---:|---:|---|---:|---:|
| Shadow only | 3 | 0 | 0 | 0 | 无 COLD | 471,983 | 472,062 |
| Shadow + COLD | 3 | 46,256 | 20,965,908,480 | 26,192 | probation canceled by readmission=13,712 | 687,518 | 597,071 |

### 图 4：Runtime Rescue 时间线

窗口有重叠（first-5 是 first-10 的子集），制图时不要累计相加。

| run / window | decode steps | prefetch issued | major faults | latency total ms | control state |
|---|---:|---:|---:|---:|---|
| `phase2d_gr_highn1_r0`, early | 3 | 0 | 14,096 | 1,992.98 | trigger at step 3 |
| 同 run, post-trigger first 5 | 5 | 4,789 | 3,798 | 2,154.50 | COLD suspended; gate bypass |
| 同 run, post-trigger rest | 66 | 63,210 | 40,958 | 33,218.29 | recovery state |
| `phase2e_highn1_on1`, early | 3 | 0 | 14,439 | 2,171.45 | trigger at step 3 |
| 同 run, post-trigger first 5 | 5 | 4,796 | 3,085 | 2,438.22 | COLD-rate recovery |
| 同 run, post-trigger rest | 66 | 18,485 | 251,325 | 51,573.56 | re-degradation; COLD disabled at step 35 |

## 11. README 现在可以安全写出的核心结论

1. 项目将 llama.cpp 的真实 MoE router 选择连接到 expert slice、异步任务、logical first-use 和 Linux memory advice 的可审计 trace。
2. `expert_prefetch` 是项目的 opt-in 默认优化 profile；当前 HEAD 已在 Qwen 上完成 5×5 对照、保持输出一致和零 trace drop。
3. 该 5×5 对照在本机 7040 MiB 冷缓存环境中没有显示稳定性能提升：平均 decode 和尾延迟反而变慢，major fault 差异很小。
4. Memory Object Lifecycle 与 Semantic Working Set 已在真实运行中维护 demand、slot、admission/eviction/readmission/probation 语义，并提供不变量与计数。
5. `MADV_COLD` 是真实执行但默认关闭的研究控制器；当前证据显示它可能破坏 prefetch，未证明稳定净收益。
6. Runtime Rescue/Calibration 属于实验性 guard/controller：已有状态机和阶段性真实运行证据，但尚未完成统一的性能主张验证。

## 12. 仍然不能写的内容

- “Expert Prefetch 在当前 HEAD 上加速 X%”或“稳定降低 major faults”。
- “MADV_COLD 降低内存/缺页/延迟”或“成功 COLD 就等于物理内存已回收”。
- “Runtime Rescue 已证明恢复整体性能”、“不会误触发”或“Calibrated Controller 已产品化”。
- “Single-flight 已带来合并收益”；本 workload 的 aggregation counter 为 0。
- 任何 swap peak 数字；P0 runner 没有收集该指标。
- 基于 `grace_high/low/med` 目录名的压力结论；命名不能替代 manifest/cgroup/runtime counter。
- 把历史 dirty-worktree Phase 1/2 性能数据表述为 current HEAD 的严格复现结果。

---

## 停止条件核对

- [x] HEAD OFF vs expert_prefetch：各 5 次、同 binary/model/prompt/cgroup/cache/order 记录。
- [x] Lifecycle：历史重复性 + 当前 HEAD correctness 闭合。
- [x] Working Set：256/512/1024 budget 行为数据。
- [x] COLD：真实 syscall、反作用和 N=3 历史 A/B 结论。
- [x] Rescue：坏状态的阶段性重复触发/动作证据。
- [x] False-positive Guard：当前 HEAD 基本验证（一次未触发）。

实验到此停止。后续若要提升证据等级，唯一优先项应是**在干净工作树、统一版本与统一 cgroup 下重复 Rescue OFF/ON 的坏状态配对**；不要为 README 前再增加新的 predictor、priority、fairness、aging、worker controller 或 cache replacement。
