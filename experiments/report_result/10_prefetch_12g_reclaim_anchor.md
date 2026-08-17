# 12 GiB Router Prefetch / 真实 reclaim anchor（server 模式）

> **Provenance**：本报告是正式版本化证据摘要，依据原始实验目录 `experiments/scenario_b_12g/`（r1/r2/r3 三次重复，每次含 baseline 与 performance 两组）整理。主要使用的原始文件：各 rep 的 `memory_trace.jsonl`（`EXPERT_TASK_SUMMARY`）、`cgroup_after_requests.json`（`memory_events` / `memory_stat`）、`request_metrics.json`（输出 SHA）、`startup.json`、`server_stdout.log`。原始载荷因 `.gitignore` 的 `experiments/*` 规则不入库，仅存在于原实验目录；本文件中的每个数字均由上述原始文件直接读取并复核（r1 为报告主行，r2/r3 用于一致性确认）。

## 1. 实验身份

| 项 | 值 |
| --- | --- |
| 场景 | **server 模式**（`llama-server` 持久服务 + HTTP 请求），非 transient `llama-cli` |
| cgroup | `MemoryMax = 12 GiB`（12884901888 字节），`MemorySwapMax = 0` |
| 请求 | 3 次请求 × 32 tokens 生成 |
| 重复 | N=3（scene_b_r1/r2/r3，各含 baseline / performance 两组） |
| 主机路径 | `/home/liziheng/`（user-1000 systemd scope）；该实验未记录 run_manifest（commit/binary 无法严格对齐，见第 4 节边界） |
| 输出一致性 | 6 个 run（3 baseline + 3 performance）的 combined output SHA-256 全部为 `6ce9175f8039ffbc0c713a6dc558a55f88f4d2c7dae6ef36276adc81ce76ba0a` |

## 2. 核实数据（r1 performance 组，r2/r3 复核）

### 2.1 Task / gate 计数（`EXPERT_TASK_SUMMARY`）

| 指标 | r1 | 说明 |
| --- | ---: | --- |
| created | **20,184** | r2/r3 同为 20,184 |
| admitted | 18,003 | |
| rejected_pressure | **2,181** | 全部拒绝均来自 pressure gate |
| rejected_value | **0** | value gate 未拒绝任何 task |
| issued hint | **18,003** | enqueued=dequeued=issued，全部真实发出 |
| cancelled（pressure/value/queue_full） | 0 / 0 / 0 | |
| invalid_transitions | 0 | 状态机闭合 |
| terminal / in_flight | 20,184 / 0 | 终态闭合 |

### 2.2 cgroup 回收证据（`cgroup_after_requests.json`，r1）

| 指标 | r1 | r2 | r3 |
| --- | ---: | ---: | ---: |
| memory.events.max | **81,142** | 80,577 | 81,450 |
| pgscan | **7,307,338** | 7,291,542 | 7,324,519 |
| pgsteal | **2,703,337** | 2,685,003 | 2,715,828 |
| pgmajfault | 68,864 | 62,499 | 63,232 |
| oom / oom_kill | **0 / 0** | 0 / 0 | 0 / 0 |
| memory.peak | 12,884,905,984（≈上限） | — | — |

三次重复均出现百万级 pgscan/pgsteal 与 8 万级 `events.max`，即**真实发生 cgroup reclaim**（限额持续被打满触发扫描与回收），且全程未触发 OOM kill。

## 3. 该实验证明什么

- 在真实 reclaim 压力下，controller 的 **pressure gate 会拒绝部分 task**（2,181/20,184 ≈ 10.8%），而 **value gate 未拒绝**（rejected_value=0），hint 仍在发出（18,003）。
- gate / queue / worker 路径行为闭合：admitted=enqueued=dequeued=issued=18,003，cancelled=0，invalid_transitions=0，终态 in_flight=0。
- 12 GiB 限额下 server 完成全部请求且输出与 baseline 一致、未 OOM。

**“仍发出 18,003 个 Hint”只表示 gate/queue/worker 的行为学事实，不等于这些 Hint 改善了性能。本报告不含也不支持任何 Prefetch 性能收益结论。**

## 4. 实验边界（不可比性声明）

- 本实验是**独立的 12 GiB server tight-memory anchor**。
- 与 `08_prefetch_pressure_window_result.md` 的 **16–19 GiB transient `llama-cli` sweep** 存在系统性差异：运行模式（持久 server vs 一次性进程）、token 数（3×32 vs 1×80）、trace profile（control_only=true、expert sink 关闭 vs benchmark）、task 数量级（20,184 vs 102,186）、provenance（无 run_manifest、无法对齐 commit/binary vs 有完整 manifest）。
- **两组不能放在同一张表内直接数值横向比较**；12 GiB 档不构成 16–19 GiB sweep 的延伸点。
- 本实验的用途是单一问题：**真实 cgroup reclaim 发生时 controller 实际做什么**。它不作为 Prefetch 性能收益的证据。

## 5. 原始文件清单（原目录，未入库）

`experiments/scenario_b_12g/scene_b_r{1,2,3}_{baseline,performance}/`：`memory_trace.jsonl`、`cgroup_before_server.json`、`cgroup_after_requests.json`、`request_metrics.json`、`startup.json`、`server_stdout.log`、`health_ready.json`；及 `scene_b_r{1,2,3}_report/scenario_b_summary.json`。
