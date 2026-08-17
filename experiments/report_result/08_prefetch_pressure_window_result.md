> **Provenance**：本文件是由原始实验目录 `experiments/experiment_06_prefetch_pressure_window/RESULT.md` 冻结版 RESULT.md 复制入库的正式版本化证据；底层数据为该目录冻结的 `group_stats.csv` / `all_runs_metrics.csv` / 各 run `analysis/metrics.json`（原始载荷按仓库约定不入库）。

# T2 Prefetch Pressure Window 实验结果（T2_prefetch_pressure_window）

## 0. 实验环境与前置

| 项目 | 值 |
|------|-----|
| 仓库 HEAD commit | `a5d80057701a759edd40f477e9375e0daffbe757` |
| 工作树状态 | 首次创建输出目录前 clean；只有未跟踪辅助文件（build/、experiments/、run/、trace 脚本等），无源码改动。运行后产生的 `raw/` 文件不属源码改动。 |
| 主机 | LAPTOP-8V5RQPEC（WSL2） |
| 内核 | `Linux 6.6.87.2-microsoft-standard-WSL2 x86_64` |
| CPU | AMD Ryzen 9 7940H（16 logical） |
| 内存 | total 27 GiB / available 25 GiB（≥24 GiB ✓）/ swap 4 GiB |
| cgroup v2 | 可用（cgroup2fs），`systemd-run --user --scope` 可用 |
| 模型 | `/home/prince/project3136859-386203/models/Qwen3.5-35B-A3B-Q3_K_M.gguf`，size 16,356,375,168 B（15.24 GiB），SHA256 `5607c8fcc8b04ada7d1a1152b9a5b6c1e67e6768232c16f6b03d9719d5ab1b2d` |
| trace-on binary | `build-trace-on/bin/llama-cli`，SHA256 `97321119b6e29cb54f13e01310be62f1c6f31c291a100fac273371fdb6b2f545`，`LLAMA_MEM_TRACE:BOOL=ON` |
| 输出根目录 | `experiments/report_result/raw/T2_prefetch_pressure_window/` |

### 0.1 固定参数（spec §0.3 / §1）

CPU-only、`-n 80 -t 8 -b 512 -ub 512 -c 2048 --cache-type-k f16 --cache-type-v f16 --temp 0 --seed 1234`、相同 prompt、`CACHE_MODE=cold`（posix_fadvise(DONTNEED)）、`TRACE_PROFILE=benchmark`、`LLM_MEM_TRACE_EXPERT=1`、`LLM_MEM_TRACE_MEMORY=1`、`LLM_MEM_TRACE_EXPERT_TASK_MODE=summary`、`LLM_MEM_TRACE_ALLOW_DROP=0`。

prefetch 配置（spec §1）：`LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=expert_prefetch`、`LLM_MEM_TRACE_OS_HINTS=1`、`LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1`、`LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_BUDGET_MB=512`、`LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK=0`、async=1、async_priority=1、feedback=1、value_gate=1。`TOPK=0` 保持冻结语义，未修改为非零 top-k。

---

## 1. §2 窗口筛选结果（19/18/17/16 GiB 各 1 次 prefetch screening run）

### 1.1 5 项候选条件核验

按 spec §2 line 29-35，候选窗口必须同时满足：
1. exit=0、输出 SHA 完整、无 OOM/oom_kill、两类 trace 有 TRACE_END、expert/memory drop=0
2. manifest 明示 controller=expert_prefetch、PREFETCH=1、OS_HINTS=1
3. `EXPERT_TASK_SUMMARY.created/admitted/issued` 均大于 0，且 `OS_HINT` 数大于 0
4. 存在实际回收证据：`memory.events.max>0` 或 `memory.stat.pgscan>0` 或 `memory.stat.pgsteal>0`
5. `EXPERT_PRESSURE` 至少出现 High，且整个 run 不得出现 Critical

| MemoryMax | 条件1 | 条件2 | 条件3 | 条件4 | 条件5 | 状态 |
|----------|-------|-------|-------|-------|-------|------|
| 19 GiB | ✓ | ✓ | ✓（5072/5067, OS_HINT=5067） | ✗（events.max=0, pgscan=0, pgsteal=0） | ✗（最高 moderate，未达 High） | **NO_PRESSURE** |
| 18 GiB | ✓ | ✓ | ✓（102173/102172, OS_HINT=102172） | ✗（events.max=0, pgscan=0, pgsteal=0） | ✓（high=237, 无 Critical） | **NO_PRESSURE** |
| 17 GiB | ✓ | ✓ | ✓（102185/102177, OS_HINT=102177） | ✗（events.max=0, pgscan=0, pgsteal=0） | ✓（high=567, 无 Critical） | **NO_PRESSURE** |
| 16 GiB | ✓ | ✓ | ✗（admitted=0, issued=0, OS_HINT=0） | ✗（events.max=0, pgscan=0, pgsteal=0） | ✗（出现 critical=303） | **GATED_CRITICAL** |

**没有任何档位同时满足全部 5 项候选条件 → 没有候选窗口。**

### 1.2 详细指标

| MemoryMax (GiB) | bytes | exit_code | output_sha | cgroup memory.current (GiB) | cgroup memory.peak (GiB) | cgroup memory.max (GiB) | events.max | pgscan | pgsteal | pgmajfault | pressure_max | task created/admitted/issued | OS_HINT | first_use matched/unmatched | screening_status |
|------------------|-------|-----------|------------|------------------------------|---------------------------|---------------------------|------------|--------|---------|------------|--------------|------------------------------|---------|----------------------------|-------------------|
| 19 | 20401094656 | 0 | 693f2012f73db26e | 15.484 | 15.868 | 19.000 | 0 | 0 | 0 | 323 | moderate (325) | 102186 / 5072 / 5067 | 5067 | 5050 / 97136 | NO_PRESSURE |
| 18 | 19327352832 | 0 | 693f2012f73db26e | 15.509 | 15.885 | 18.000 | 0 | 0 | 0 | 0 | high (237), moderate (361) | 102186 / 102173 / 102172 | 102172 | 102161 / 25 | NO_PRESSURE |
| 17 | 18253611008 | 0 | 693f2012f73db26e | 15.512 | 15.885 | 17.000 | 0 | 0 | 0 | 0 | high (567) | 102186 / 102185 / 102177 | 102177 | 102167 / 19 | NO_PRESSURE |
| 16 | 17179869184 | 0 | 693f2012f73db26e | 15.510 | 15.811 | 16.000 | 0 | 0 | 0 | 0 | critical (303) | 102186 / 0 / 0 | 0 | 0 / 102186 | GATED_CRITICAL |

- 4 个 run 的 `expert_dropped=0`、`memory_dropped=0`（trace 不丢数据）
- 4 个 run 的 output SHA 全部一致 `693f2012f73db26ed093f5747a5e0b88456db4a282d155c446629eb03e3b3b52`
- 4 个 run 的 `exit_code=0`、`events.oom=0`、`events.oom_kill=0`
- 17 GiB 的 `EXPERT_FIRST_USE_SUMMARY`：`matched_tasks=102167`、`unmatched_first_uses=19`、`unmatched_tasks=10`（详见原始 jsonl）

### 1.3 根因分析

#### 1.3.1 cgroup v2 watermark 与 prefetch controller 压力感知阈值不同步

四个档位的 `cgroup memory.peak` 都在 **15.81–15.89 GiB** 之间。这是模型 MMAP 工作集的稳定上限。cgroup v2 在 memory.current 接近 memory.max 但仍低于 watermark（远低于 MemoryMax）时不会触发回收，所以即使 MemoryMax=17/18/19 GiB（远大于 peak），cgroup 也不会回收；而 MemoryMax=16 GiB 时 peak=15.81 GiB < MemoryMax，仍未达回收阈值。

prefetch controller 的压力感知基于 `memory.current / memory.max` 的 ratio：
- 19 GiB：ratio ≈ 83% → moderate，value-gate 拒绝 97,114 task（保守）
- 18 GiB：ratio ≈ 88% → high，几乎全部 admitted
- 17 GiB：ratio ≈ 91% → high，几乎全部 admitted
- 16 GiB：ratio ≈ 98.8% → critical，value-gate 全数拒绝

**两者之间存在一个"压力感知但未回收"的真空区**：prefetch controller 已经感知 High/Critical 并调整其 task admission 策略，但 cgroup 还没到 watermark 没有真正回收。这个真空区使得"实际 hint + 实际回收"无法在 17–19 GiB 同时发生，而 16 GiB 又因 value-gate Critical 关闭而无法发 hint。

#### 1.3.2 cgroup v2 不主动回收 file-backed 模型页

cgroup v2 的回收优先 anonymous 页；file-backed 模型页（来自 MMAP）在 cgroup 视角下属于可重读的页，cgroup 通常不主动 swap 或 steal 它们，除非 anonymous 内存也被压到极限。本实验中模型 size=15.24 GiB，cgroup memory.current 主要由模型页构成，但 cgroup 没有触发 file-backed 回收（pgsteal=0）。

#### 1.3.3 prefetch value-gate 在 Critical 时正确关闭

16 GiB 的 `EXPERT_TASK_SUMMARY` 显示 `rejected_value=102186`（完全等于 created），证明 value-gate 在 Critical 压力下正确拒绝了所有 task。这是 controller 的安全设计，避免加剧回收；同时也意味着 16 GiB 不是一个"安全可测"的预取窗口。

---

## 2. §3 配对性能实验：未执行

按 spec §3 line 41："只对第 2 步选出的一个新 cgroup 档位运行 baseline 与 prefetch，各 N=5。"

因第 2 步未选出任何候选窗口，跳过 §3 配对实验。不报告 wall/TPS 性能对比；仅保留 §2 的 4 个 screening run 作为机制观察证据。

按 spec §4 line 49 仍生成 `RESULT.md`、`all_runs_metrics.csv`、`group_stats.json/csv`、`screening_summary.csv`。

---

## 3. 综合结论

### 3.1 验收门槛总结

| 验收项 | 结果 |
|--------|------|
| 前置门槛（HEAD、模型 SHA、可用内存、cgroup v2） | ✓ 全部满足 |
| §2 4 个档位 screening run 各 1 次 | ✓ 全部完成（exit=0、无 OOM、trace drop=0、SHA 一致） |
| §2 找到候选窗口 | ✗ **0 个候选窗口** |
| §3 配对实验 N=5 | 跳过（未找到窗口） |
| 性能结论 | 不报告（spec §3 line 45：未找到窗口只能保留机制观察，禁止比较 wall/TPS） |

### 3.2 核心结论

1. **当前冻结 prefetch profile 在 19/18/17/16 GiB 四个档位下未找到同时满足"Router prefetch 实际发出 hint"与"cgroup 已发生真实内存回收"的安全窗口。**
   - 19/18/17 GiB：prefetch 能发出 hint（admitted/issued > 0），但 cgroup 不回收（pgscan=0, pgsteal=0, events.max=0）→ NO_PRESSURE
   - 16 GiB：cgroup 仍不回收（peak=15.81 GiB < 16 GiB），但 prefetch value-gate 在 Critical 压力下关闭全部 task（admitted=0, issued=0, rejected_value=102186）→ GATED_CRITICAL

2. **根因：cgroup v2 watermark 与 prefetch controller 压力感知阈值不同步。** cgroup 在 memory.current 接近 memory.max 但仍低于 watermark 时不会触发回收，而 prefetch controller 此时已感知 High/Critical 并调整 admission。两者之间存在"压力感知但未回收"的真空区，使得"实际 hint + 实际回收"无法在 17–19 GiB 同时发生，16 GiB 又因 value-gate Critical 关闭而无法发 hint。

3. **机制本身正确：** prefetch controller 在 18/17 GiB 正确 admit/issue ~102,172 个 task（与 created 数量一致），在 16 GiB 正确触发 value-gate 全数拒绝（rejected_value=102186），`invalid_transitions=0`、`expert_dropped=0`、`memory_dropped=0`。trace instrumentation 完整保留 TRACE_END，4 个 run 的 SHA 一致。

4. **如实报告：** 该结果如实报告为"当前冻结 profile 未找到同时发 hint 且发生回收的安全窗口"，不能通过关闭 value-gate、提高 budget 或改 top-k 强行制造结果（spec §2 line 39 明令禁止）。

### 3.3 建议

- **如未来需要找到候选窗口：** 可考虑 1) 低于 16 GiB 但仍能完成推理的档位（如 15.5 GiB，模型 15.24 GiB + ~256 MiB KV）；2) 改用 swap 友好的模型或更短 prompt；3) 调整 cgroup watermark（需要 root，超出 user-scope）；4) 在更高负载场景（多 client 并发推理）下复现回收。但当前 32GB 机器的 user-scope 限制下，本冻结 profile 无法找到窗口。
- **不建议：** 关闭 value-gate、提高 budget、改 top-k 以强行制造结果（spec §2 line 39 明令禁止，且会改变 profile）。

---

## 4. 输出物清单

| 文件 | 路径 |
|------|------|
| 本报告 | `RESULT.md` |
| 未找到窗口声明 | `NO_VALID_HINT_UNDER_PRESSURE.md` |
| 前置信息 | `prelude.json` |
| 全 run 指标 CSV | `all_runs_metrics.csv` |
| 筛查汇总 CSV | `screening_summary.csv` |
| 分组统计 CSV | `group_stats.csv` |
| 分组统计 JSON | `group_stats.json` |
| 筛查脚本 | `run_screening.sh` |
| 验证脚本 | `verify_screening.py` |
| 聚合脚本 | `aggregate_screening.py` |
| 4 个 screening run | `screening_runs/screen_{19,18,17,16}gib/latest/` |
| 各 run 完整产物 | `run_manifest.json`、`process_metrics.json`、`summary.json`、`cgroup_after_inference.json`、`cache_preparation.json`、`screening_status.json`、`inference_output.txt`、`inference_stderr.txt`、`output.sha256`、`test_prompt.txt`、`expert_trace.jsonl`、`memory_trace.jsonl`、`analysis/` |
| 各档位 pipeline 日志 | `screening_runs/screen_<N>gib/run.log` |

---

## 5. 不混入旧数据声明

按 spec §2 line 39 与 §3 line 45 要求：本报告所有数值均来自 HEAD `a5d80057701a759edd40f477e9375e0daffbe757` 在 32GB WSL2 机器上 2026-08-17 当日产生的 4 个 screening run。**未混入** 12 GiB / 20 GiB 历史 T2 数据，亦未混入 03/04 实验的 384MiB/12G survival 数据。所有 4 个 run 的 `run_manifest.git_commit` 均为 `a5d80057701a759edd40f477e9375e0daffbe757`，输出 SHA 全部一致 `693f2012f73db26ed093f5747a5e0b88456db4a282d155c446629eb03e3b3b52`。
