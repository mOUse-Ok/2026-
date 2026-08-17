# 给 32GB 机器 Agent 的提示词：T2 未测压力窗口（实际 hint + 实际回收）

在 32GB Linux 机器上，对冻结代码的当前 HEAD 搜索一个**此前未测**、同时满足“Router prefetch 实际发出 hint”与“cgroup 已发生真实内存回收”的运行窗口；只在找到该窗口后进行一次配对性能实验。不得改源码、策略、脚本、Git 历史、模型或 value-gate 阈值；不得下载模型；不得运行或重报已完成的 12GiB、20GiB T2 条件。所有新产物只能写入 `<repo>/experiments/report_result/raw/T2_prefetch_pressure_window/`。

## 0. 前置门槛与固定条件

1. 首次创建输出目录前，HEAD 必须是 `a5d80057701a759edd40f477e9375e0daffbe757`，记录 `git rev-parse HEAD`、`git status --porcelain`、主机/内核/CPU/可用内存、cgroup v2 状态。此时工作树必须干净；运行产生的 `raw/` 文件导致后续工作树变脏时，须在报告中单独说明，不能把它误写为源码改动。
2. 模型必须为 `models/Qwen3.5-35B-A3B-Q3_K_M.gguf`；记录绝对路径、字节数和完整 SHA256。可用内存须至少 24GiB，cgroup v2 必须可用；任一失败则写 `STATUS_BLOCKED.md` 并停止。
3. 仅用 trace-on 二进制。保留本次使用的 `CMakeCache.txt`，其中 `LLAMA_MEM_TRACE:BOOL=ON`，并记录 `llama-cli` SHA256。运行参数固定为 CPU-only、`-n 80 -t 8 -b 512 -ub 512 -c 2048 --cache-type-k f16 --cache-type-v f16 --temp 0 --seed 1234`，prompt 与旧 T2 相同；每个 run 均记录 prompt SHA、输出 SHA、缓存准备记录、manifest、process metrics、cgroup 前后快照、stdout/stderr、`expert_trace.jsonl` 和 `memory_trace.jsonl`。
4. 每个 run 使用既有的 `posix_fadvise(DONTNEED)` 冷缓存准备步骤并保存返回结果；这只能称为“相同的冷缓存准备方法”，不得声称能证明物理页驻留状态完全相同。

## 1. 固定的两种配置

两组均开启 `LLM_MEM_TRACE_EXPERT=1`、`LLM_MEM_TRACE_MEMORY=1`、`LLM_MEM_TRACE_EXPERT_TASK_MODE=summary`、`LLM_MEM_TRACE_ALLOW_DROP=0`，且使用相同模型、二进制、prompt、seed、CPU 参数、trace profile、缓存准备方法和 cgroup。启动前清理或逐项覆盖所有同名遗留环境变量。

- **baseline：** `LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=off`、`LLM_MEM_TRACE_OS_HINTS=0`、`LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=0`、async/priority/feedback/value-gate=0。
- **prefetch：** `LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=expert_prefetch`、`LLM_MEM_TRACE_OS_HINTS=1`、`LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1`、`LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_BUDGET_MB=512`、`LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK=0`、async=1、async priority=1、feedback=1、value-gate=1；其余参数与 baseline 相同。`TOPK=0` 保持当前冻结语义（选择全部 Router 专家），不得为了得到更好结果修改为非零 top-k。

## 2. 只做未测档位的窗口筛选

依次测试 **19GiB、18GiB、17GiB、16GiB** 的 cgroup `MemoryMax`，每个档位只运行一次 prefetch screening run；不运行 12GiB 或 20GiB。每个筛选 run 都保存完整原始产物，但不报告性能均值。

筛选结果必须从 schema v3 原始文件读取，不能依赖旧 `run_log.txt` 的字符串解析：

- 配置从 `run_manifest.json.environment` 读取；
- `EXPERT_TASK_SUMMARY`、`EXPERT_FIRST_USE_SUMMARY`、`EXPERT_PRESSURE`、`OS_HINT` 从 **`memory_trace.jsonl`** 读取；
- trace drop 从 `summary.json.sinks.{expert,memory}.dropped` 读取。

某一档位仅在同时满足以下条件时才是候选窗口：

1. exit=0、输出 SHA 完整且无 OOM/oom_kill、两类 trace 都有 `TRACE_END`、expert/memory drop 均为 0；
2. manifest 明示 controller=expert_prefetch、PREFETCH=1、OS_HINTS=1；
3. `EXPERT_TASK_SUMMARY.created/admitted/issued` 均大于 0，且 `OS_HINT` 数大于 0；
4. 存在实际回收证据：`memory.events.max>0` 或 `memory.stat.pgscan>0` 或 `memory.stat.pgsteal>0`；
5. `EXPERT_PRESSURE` 至少出现 High，且整个 run **不得出现 Critical**。这避免把“已被安全 gate 完全关闭”的状态误当成可测预取窗口。

若某档位只满足 hint 条件而没有实际回收，标为 `NO_PRESSURE` 后继续降低上限；若有 Critical 且 admitted/issued=0，标为 `GATED_CRITICAL` 后继续降低上限；若 exit 非零或 OOM，标为 `UNSAFE` 并停止继续降低上限。筛选到第一个候选窗口后停止筛选，不再运行更低档位。

如果 19/18/17/16GiB 没有任何候选窗口，写 `NO_VALID_HINT_UNDER_PRESSURE.md`，列出每个档位的 `memory.max`、memory.current/peak、events.max、pgscan/pgsteal、最高 pressure level、created/admitted/issued、OS_HINT、退出码与拒绝原因；随后停止。该结果应如实报告为“当前冻结 profile 未找到同时发 hint 且发生回收的安全窗口”，不能通过关闭 value-gate、提高 budget 或改 top-k 强行制造结果。

## 3. 命中窗口后才做一次配对性能实验

只对第 2 步选出的**一个**新 cgroup 档位运行 baseline 与 prefetch，各 N=5。顺序按 `B1,P1,P2,B2,B3,P3,P4,B4,B5,P5` 交错，保存顺序、开始/结束时间和每次冷缓存准备记录。

每一条 prefetch run 都必须再次满足第 2 步的五项候选条件；任一条失败则整个 prefetch 性能组为 `INVALID`，只能保留机制观察，禁止比较 wall/TPS。baseline 必须证明 controller=off、PREFETCH=0、OS_HINTS=0 且 `OS_HINT=0`。全部 run 的输出 SHA 必须一致；否则停止并标记 `INVALID_OUTPUT_MISMATCH`。

## 4. 交付与结论边界

写 `RESULT.md`、`all_runs_metrics.csv`、`group_stats.json/csv` 和 `screening_summary.csv`。逐 run 与分组报告 wall、prefill/decode、TPS、RSS、major/minor faults、cgroup memory.current/peak/events.max/pgscan/pgsteal/pgmajfault、PSI、task created/admitted/issued/rejected_value/rejected_pressure、OS_HINT、first-use、trace drop、输出 SHA。保留原始值，不能删除离群点；同时给出 N=5 的均值、median、p95、CV 和双侧置换检验。

性能结论只能表述为“在 `<selected MemoryMax>`、本模型、本 prompt、TOPK=0 和当前完整 prefetch profile 下的端到端结果”。不得将差异单独归因于 madvise、Router、trace I/O 或任一子组件；不得外推到全部模型、全部内存上限或生产环境。
