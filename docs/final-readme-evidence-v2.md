# 决赛最终证据报告 V2

> 范围：README 编写前的最终实验收口。本文只报告已经落盘的证据；不引入新算法、不改动控制逻辑，也不修改 README。
>
> 原始证据根目录：`llama.cpp/trace_output/final-readme-v2/`。每个样本目录保留 `run_manifest.json`、`cache_preparation.json`、`process_metrics.json`、stdout/stderr、输出 SHA-256；trace 样本另保留 JSONL、`summary.json` 与分析产物。

## 1. 环境与版本

| 项目 | 值 |
|---|---|
| 源码提交 | `01817a0aab55791fbcb54d2021f78a15e02307ab` |
| 源码状态 | 独立 worktree，`git_dirty=false`；两种二进制都从同一提交构建 |
| 主机 | WSL2 / Linux `6.6.114.1-microsoft-standard-WSL2` |
| CPU | 13th Gen Intel Core i9-13980HX，32 logical CPUs，affinity `0-31` |
| 模型 | `Qwen3.5-35B-A3B-Q3_K_M.gguf`，16,356,375,168 bytes（为避免扰动页缓存，未计算整文件 SHA-256） |
| 输入 | 固定 2,284-byte prompt，SHA-256 `59f51358…7897e4f` |
| 推理参数 | CPU、`-n 80 -t 8 -b 512 -c 2048 --gpu-layers 0 --temp 0.0 --seed 1234`；`--no-warmup` |
| 内存隔离 | 每个 transient service 设 `MemoryMax=7040M`；实际 `memory.max=7,381,975,040` bytes |
| 缓存策略 | 每样本运行 `prepare_model_cache.py --mode cold`，使用模型文件级 `POSIX_FADV_DONTNEED`；没有 root/global `drop_caches` |
| swap | `memory.swap.max=max`，但未记录可靠 peak；本文不把 swap 写成性能结论 |
| Plain binary | `LLAMA_MEM_TRACE=OFF` 编译，SHA-256 `3b272e25…f29dbe0` |
| Trace binary | `LLAMA_MEM_TRACE=ON` 编译，SHA-256 `6727d422…76ee0b` |

构建在隔离 worktree 中进行。上游源码把 `src/models/` 与 `tools/mtmd/models/` 忽略，但它们是构建输入；使用工作区已有的同版本生成源副本以完成干净 worktree 构建。该前提不改变 tracked HEAD，也不会作为性能结论的一部分。

所有 trace 实验使用 `TRACE_PROFILE=benchmark`：tensor/KV/residency/smaps 关闭，expert/memory sink 开启，队列上限 65,536，且不允许 drop。所有统计为 wall time；括号内为样本标准差。

## 2. Plain llama.cpp → instrumentation overhead

这是本轮唯一将**未编译追踪的普通 llama.cpp**与追踪构建直接比较的实验。两侧 controller 都关闭；Trace 侧仅保留 benchmark profile 所需的 expert/memory 事件。

交替顺序为 `A1 B1 B2 A2 A3 B3 B4 A4 A5 B5`，其中 A=Plain、B=Trace/controller-off，`N=5`/组。

| 指标 | Plain (`LLAMA_MEM_TRACE=OFF`) | Trace 最小观测、controller=off | Trace 相对 Plain |
|---|---:|---:|---:|
| wall time (s) | 53.394 ± 0.687 | 55.348 ± 1.052 | **+3.66%** |
| major faults | 785,218.2 | 783,389.0 | -0.23% |
| max RSS (KiB) | 6,746,882 | 6,799,906 | +0.79% |
| exit code | 5/5 为 0 | 5/5 为 0 | — |
| output SHA-256 | `3bd36df9…a210` | 同一值 | 一致 |

每个 Trace 样本写入 22,272 条 expert 与 7,950 条 memory 事件（共 30,222）；`enqueued == written` 且 `dropped=0`。终态 `EXPERT_TASK_SUMMARY` 为 `in_flight=0`、`invalid_transitions=0`。

结论：在本机、这个模型和冷缓存条件下，最小可用观测基础设施的 wall-time 成本约为 **3.66%**。它不是跨硬件的通用常数，也不是预取收益/损失。

## 3. 复用既有 Expert Prefetch 5×5 负结果

本轮不重跑该项。它已在先前的 current-HEAD 5×5 交替对照中完成，且控制语义未改；V2 只复用，不把历史数据伪装成本轮新数字。

| 既有 current-HEAD 结果 | Controller off | `expert_prefetch` | 相对变化 |
|---|---:|---:|---:|
| decode 平均 (µs) | 286,064.831 | 321,072.612 | **+12.24%** |
| major faults | 802,297.2 | 798,301.8 | -0.50% |
| 样本数 | 5 | 5 | 输出哈希均一致 |

这是一条应保留的负结论：当前机器的 7,040 MiB 冷缓存配置下，router-driven expert prefetch 是真实可运行的 opt-in 路径，但**未观察到稳定加速**。不得写成“默认提升推理速度”。

## 4. Memory Object Lifecycle overhead

当前 HEAD、冷缓存、`expert_prefetch` 相同。OFF/ON 唯一变更为 Memory Object lifecycle 家族：`MEMORY_OBJECTS`、`INFLIGHT_HINT_AGGREGATION`、`SEMANTIC_STALE_CANCEL` 均从 0 切到 1；Working Set、COLD、Rescue、Calibration 均显式为 0。因最初 N=3 的 ON 方差较大，按预设规则扩至 N=5。

| 指标 | Lifecycle OFF | Lifecycle ON | ON 相对 OFF |
|---|---:|---:|---:|
| wall time (s) | 55.488 ± 0.644 | 63.094 ± 7.884 | **+13.71%** |
| major faults | 751,611.2 | 661,964.0 | -11.93% |
| max RSS (KiB) | 6,862,934 | 6,897,098 | +0.50% |
| exit / output hash / trace drop | 5/5 正常 | 5/5 正常 | 全部一致 / 0 drop |

ON 的每个真实 Qwen run 都记录：102,222 demand 注册、102,222 activation、102,222 completion、102,222 slot acquire 与 102,222 release；最终 `pending=0`、`active=0`、`invariant_violations=0`。

结论：Lifecycle 的状态闭环是实证成立的；其在本 workload 的端到端成本不是零，且高方差下不能宣称它改善性能。

## 5. Semantic Working Set

未重跑历史 256/512/1024 MiB budget scan：当前 HEAD 的 1024 MiB 运行已经逐项复现该扫描的语义计数，且没有机制语义变更。历史扫描仍标为历史数据：

| budget（历史 Qwen，16 decode tokens） | current bytes | peak bytes | admission | eviction | readmission | protected skips | unresolved |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 MiB | 268,361,728 | 348,684,288 | 40,782 | 40,191 | 17,778 | 6,477,371 | 852 |
| 512 MiB | 536,502,272 | 537,427,968 | 35,169 | 33,986 | 12,165 | 6,658,418 | 0 |
| 1024 MiB | 1,073,496,064 | 1,074,298,880 | 32,471 | 30,103 | 9,467 | 6,508,915 | 0 |

本轮 1024 MiB、80-token current-HEAD 的 6 个 Shadow/COLD run 全部一致地记录：

| 字段 | 当前值 |
|---|---:|
| budget / shutdown current / peak bytes | 1,073,741,824 / 1,073,496,064 / 1,074,298,880 |
| admissions / evictions / readmissions | 63,748 / 61,380 / 39,904 |
| protected skips | 6,933,415 |
| `budget_unresolved_due_to_protection` | 0 |
| current `pending` / `active` / invariant violations | 0 / 0 / 0 |

结论：它是被真实执行的**容量约束语义工作集**，支持 admission、eviction、readmission、protection/probation 的可审计行为。它不是严格物理 bytes cap：历史 256 MiB 档的 protected 对象曾令预算暂时未解。

## 6. 当前 HEAD Shadow-only vs COLD

两组均为 1024 MiB Working Set、grace=3、相同 `expert_prefetch`、Lifecycle 与 stale handling 开启；唯一差异是 `LLM_MEM_TRACE_OPT_EXPERT_MADV_COLD_RECLAIM=0/1`。顺序为 `Shadow1 COLD1 COLD2 Shadow2 Shadow3 COLD3`，`N=3`/组。

| 指标 | Shadow-only | Shadow + COLD | COLD 相对 Shadow |
|---|---:|---:|---:|
| wall time (s) | 82.120 ± 5.887 | 87.430 ± 5.226 | **+6.47%** |
| major faults | 634,192.7 | 693,656.7 | **+9.38%** |
| max RSS (KiB) | 6,852,608 | 6,894,183 | +0.61% |
| exit / output hash / trace drop | 3/3 正常 | 3/3 正常 | 全部一致 / 0 drop |

每个 COLD run 都有 `madv_cold_candidates=issued=46,256`、`madv_cold_bytes=20,965,908,480`、`madv_cold_failed=0`、`post_cold_readmissions=26,192`、`cold_protected_violation=0`。

结论：`MADV_COLD` 确实被 Linux 接收并被追踪，但成功 syscall 不等于物理回收同量 bytes。本轮均值仍变慢、major faults 也增加；结合既有历史 N=3（decode +26.48%、major faults +45.67%），它没有稳定净收益，并可能损害 Prefetch。保持 research/默认关闭定位。

## 7. Rescue OFF/ON

### 固定坏状态 pilot

使用历史已触发配置的**不调参**复现：1024 MiB Working Set、COLD、grace=3、`gate_recovery`、re-entry rate=0。两个 current-HEAD pilot 结果不同，说明状态由实际运行指标而非目录名判定：

| run | early-3 issued | triggered | COLD suspended | gate bypass | post-trigger first-5 issued |
|---|---:|---:|---:|---:|---:|
| `rescue_pilot_on_1` | 2,812 | 0 | 0 | 0 | 0 |
| `rescue_pilot_on_2` | 0 | 1 | 1 | 5 | 4,800 |

### 同配置 OFF/ON 交替 N=5

顺序为 `OFF1 ON1 ON2 OFF2 OFF3 ON3 ON4 OFF4 OFF5 ON5`。所有十个样本 exit=0、输出哈希一致、trace 零 drop，且 Memory Object 最终 `pending=active=invariant_violations=0`。

| 指标 | Rescue OFF | Rescue ON | ON 相对 OFF |
|---|---:|---:|---:|
| wall time (s) | 86.850 ± 8.093 | 82.298 ± 3.566 | -5.24%（观测值） |
| major faults | 672,263.4 | 532,100.6 | -20.85%（观测值） |
| max RSS (KiB) | 6,874,879 | 6,837,241 | -0.55% |

ON 的 5 次中有 2 次进入真实坏状态（`ON4`、`ON5`）：均在 decode step 3 触发，early-3 issued=0，随后 `cold_suspended=1`、`gate_bypass_steps=5`，post-trigger first-5 issued=4,800。另 3 次为正常态并保持未触发。

这完成了同配置 N=5 的 OFF/ON 观察，但**不能把均值直接写成“Rescue 已证明恢复整体性能”**：ON 与 OFF 不能在同一次 OS 调度中拥有相同的坏状态 counterfactual，且 ON 的触发率为 2/5。可安全的结论是：在检测到该定义的坏状态时，当前 HEAD 的状态机确实暂停 COLD、绕过 gate，并恢复后续 hint 发出；全局性能因果收益仍需更强的状态匹配设计才能宣称。

## 8. False-positive guard

仅因第 7 节已实际重现触发路径，才统计 guard。`rescue_on_1`、`rescue_on_2`、`rescue_on_3` 都是同一固定 ON 配置下的正常态：early-3 issued 分别为 2,832、840、1,632，三个样本均 `runtime_rescue_triggered=0`、`cold_suspended=0`、`gate_bypass_steps=0`，输出哈希仍一致。

结论：在这三次**有实际 early issuance 的正常态**中，guard 保持静默。不能写“零误触发”或“永不触发”。

## 9. Correctness 与完整性

- 新鲜真实 Qwen run：38 个；Plain 5、Trace 33。
- 38/38 `process_metrics.json.exit_code=0`；输出 SHA-256 全为 `3bd36df99d6aabd977ce1d4927c71af75767689fb91c895316be37665912a210`。
- 33/33 Trace run 的每个 enabled sink 均 `enqueued == written` 且 `dropped == 0`；无完整性失败样本。
- P0 的 task summary 为 `in_flight=0`、`invalid_transitions=0`；Lifecycle/COLD/Rescue 的 Memory Object 终态均为 `pending=0`、`active=0`、`invariant_violations=0`，slot acquire/release 对称。
- 当前 trace 构建重新运行并通过 5 个单元测试：`test-router-tensor-observation-sync`、`test-expert-hint-priority`、`test-expert-task-lifecycle`、`test-expert-memory-object`、`test-expert-calibration-shadow`。

## 10. Evidence staircase

```text
Plain llama.cpp (trace compiled out)
  └─ P0 N=5 vs minimal Trace/controller-off N=5
       → 定量观测成本：+3.66%，输出一致、零 drop
          └─ Router-driven Expert Prefetch
               → 复用既有 current-HEAD 5×5：未观察到稳定加速
                  └─ Memory Object Lifecycle
                       → current N=5 bookkeeping 成本 +13.71%；demand/slot 闭合
                          └─ Semantic Working Set
                               → 当前 1024 MiB admit/evict/readmit/protection 真实执行
                                  └─ MADV_COLD
                                       → current N=3 无净收益，fault/wall 均变差
                                          └─ Runtime Rescue
                                               → current pilot + N=5：坏状态时 suspend/gate-bypass/hint 恢复
                                               → 正常态 N=3：保持静默（条件性 guard）
```

## 11. 最终 Claim → Evidence 表

| 问题 | 证据结论 | README 可用表述 |
|---|---|---|
| 1. 相比普通 llama.cpp，观测成本？ | 当前 clean-HEAD P0：+3.66% wall，N=5+5 | “在该 CPU/Qwen 冷缓存基准下，最小追踪成本约 3.66%。” |
| 2. Expert Prefetch 有收益？ | 既有 current-HEAD 5×5 为 decode +12.24% | “提供 opt-in router-driven prefetch；本机未观察到稳定加速。” |
| 3. Lifecycle 成本？ | 当前 N=5 为 +13.71% wall、高方差；状态闭合 | “可审计 lifecycle bookkeeping 有可测成本。” |
| 4. Working Set 是否真实维护？ | 当前 1024 MiB run 有 admit/evict/readmit/protection 计数与终态 | “维护容量约束的语义工作集。” |
| 5. COLD 有净收益？ | 当前 N=3 +6.47% wall、+9.38% major faults；历史同向负结果 | “未证明 COLD 的稳定净收益。” |
| 6. COLD 会破坏 Prefetch？ | COLD 真实执行且与更高 faults/wall 相关；历史线更显著 | “COLD 可能伤害预取，故保持实验性。” |
| 7. Rescue 能恢复坏状态？ | 2 pilots/ON4/ON5 有 trigger→suspend→bypass→4,800 post hints | “坏状态检测后可暂停 COLD 并恢复 hint 发出。” |
| 8. 正常时 Rescue 静默？ | 当前同配置正常态 N=3 都未触发 | “在三个有实际 early issuance 的正常样本中保持静默。” |
| 9. 整套系统正确？ | 38/38 hash、exit、trace 完整；5 CTest；所有终态不变量闭合 | “实验路径保持输出一致、trace 完整与状态收尾。” |

## 12. README 最终可以写的话

- 提供 CPU/MoE 推理的 trace、router observation、可选 expert prefetch、Memory Object lifecycle、Semantic Working Set、可选 `MADV_COLD` 与实验性 Rescue 状态机。
- 真实 Qwen 运行中，Memory Object 的 demand、activation/completion、slot acquire/release 与终态不变量可以审计。
- Working Set 具有容量预算、admission/eviction/readmission/protection 语义；保护对象可导致短暂超额。
- `MADV_COLD` 会真实执行并记录 syscall 结果，但默认应保持关闭；当前证据没有显示稳定性能收益。
- Rescue 能在观测到定义的坏状态时暂停 COLD、绕过 gate、恢复 hint 发出；其正常态 guard 有条件性 N=3 静默证据。
- 在本报告的固定硬件、模型、冷缓存条件下，最小 trace overhead 为约 3.66%。

## 13. README 最终不能写的话

- “Expert Prefetch 默认加速/稳定降低时延。”
- “Memory Object Lifecycle 提升性能。”
- “Working Set 是严格物理内存上限。”
- “`MADV_COLD` 已证明减少内存、缺页或时延”，或“成功 COLD 等于回收同量物理内存”。
- “Rescue 已证明恢复整体端到端性能”或“Rescue 永不误触发”。
- 未记录的 swap peak、跨硬件/模型的普适百分比，或任何未做状态匹配的因果性能主张。

实验在此停止。下一阶段应直接进入数据可视化、架构图、README 与答辩材料，而不是继续策略探索。
