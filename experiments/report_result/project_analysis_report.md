# 项目工程代码级审计报告

## 结论摘要

本工程不是从零实现 LLM 推理引擎，而是在 vendored `llama.cpp` / `ggml` 基线之上，新增了一套面向 **CPU-host 稀疏 MoE 推理** 的内存观测、Linux advisory、实验可追溯和受压服务控制系统。以项目首次完整源码提交 `720c62f` 为基线，到当前 HEAD `a5d8005`，`llama.cpp/` 净增 **22,039 行**、删除 25 行；其中 `llama.cpp/trace/` 净增 **19,859 行**。HEAD 相对代码锚点 `88fc9e1` 只增加实验归档和 `.gitignore`，没有再改动 C/C++/Python 实现，因此以下代码结论对应 `88fc9e1` 的实现、也适用于当前 HEAD。

可成立的核心结论是：项目完成了从 `ggml` Router 输出到 Expert slice 地址、再到任务生命周期/OS hint/first-use 归因的真实运行时通路；并将 mmap 加载策略、cgroup/PSI/refault 信号、KV slot admission 和 JSONL/manifest 证据链接入了推理系统。当前可量化且同版本、干净工作树的性能证据只覆盖 mmap 消融：跳过 `MAP_POPULATE` 显著减少加载时间和 RSS，但将代价转移至 prefill/decode，吞吐反而下降。因此不能把它包装为普适加速。

专家预取、Memory Object 回收、Runtime Rescue、KV admission 都已有可编译实现与单元/分析器验证，但当前仓库内没有同一 commit、同一模型指纹、同一受限 cgroup 下的端到端 A/B 结果来证明它们提升吞吐、尾延迟或 OOM 存活率。它们应表述为“机制实现完成、效果待实测”，而不是既有性能结论。

## 1. 审计范围、方法与证据等级

### 1.1 审计对象

- 当前仓库 HEAD：`a5d8005`；工作树在审计开始时干净。
- 代码增量基线：`720c62f`（首次纳入完整 `llama.cpp` 源码）。使用 `git diff 720c62f..HEAD -- llama.cpp` 确定增量，而非根据 README 或设计文档推断。
- 代码锚点：`88fc9e1`；`88fc9e1..HEAD` 的差异只包含 `experiments/report_result/` 与 `.gitignore`。
- 实测验证：以 `llama.cpp/build/CMakeCache.txt` 中 `LLAMA_MEM_TRACE:BOOL=ON` 的构建目录，重新构建 8 个项目新增 CTest target，并运行全部对应 CTest；再运行 `python3 -m unittest discover -s llama.cpp/trace/tests -v`。
- 性能原始证据：`experiments/results/experiment_0815/` 中的 16 个 run 的 `run_manifest.json`、`memory_trace.jsonl`、`analysis/metrics.json`、`all_runs_metrics.csv` 与 `group_stats.csv`。这些文件记录的源码为 `88fc9e1`、`git_dirty=false`。

### 1.2 证据判定规则

| 等级 | 判定标准 | 本报告中的用法 |
|---|---|---|
| A：代码+重新执行 | 本次重新构建并通过测试，且可定位到生产代码 | 并发 Router 观测、mmap admission、状态机、分析器契约 |
| B：代码+可追溯实测 | 原始 manifest、trace、指标均存在，commit/二进制/prompt 可核验 | 当前 HEAD 的 mmap 四组消融 |
| C：代码存在、未做端到端量化 | 已接入生产调用点且有单元测试，但没有合格 A/B 性能数据 | Expert prefetch/reclaim、KV admission、trace 开销 |
| D：离线模拟或文献参考 | 脚本模拟，或只有参考关系而无运行时实现 | Paged KV、H2O/KIVI、若干论文映射 |

这一区分很关键：`madvise()` 返回 0 只说明 syscall 被内核接受，不等价于页已物理驻留、页已回收或必然降低延迟。

## 2. 代码级架构与创新/增量工作

### 2.1 端到端数据通路

```text
llama_context::decode / encode
  └─ 设置 (phase, step, llama_ubatch)，记录 STEP_BEGIN/END
       └─ ggml_cpu::ggml_graph_compute_thread
            ├─ thread 0: tensor begin/end、KV SET_ROWS 观测
            └─ Router GET_ROWS: 生产者 barrier → thread 0 安全读取 Router IDs/scores
                 └─ ExpertTensorRegistry: (layer, tensor, addr, stride, n_expert)
                      └─ Expert slice = addr + expert * stride
                           ├─ ExpertMemoryObjectTracker / FirstUseMatcher
                           ├─ 可选 ExpertHintTask（同步或异步、优先级、TTL 去重）
                           │    └─ MADV_WILLNEED / posix_fadvise
                           └─ JSONL：expert/memory/tensor/KV 四类 sink

模型加载：llama_model_loader → mmap admission → MAP_POPULATE 决策/审计
服务入口：server_context 空闲 slot → cgroup/PSI/refault pressure → KV admission/defer
```

上图是从实际调用关系归纳的：关键跨层调用可在 `llama.cpp/src/llama-context.cpp`、`llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c`、`llama.cpp/trace/tensor_trace.cpp`、`llama.cpp/trace/expert_trace.cpp` 和 `llama.cpp/tools/server/server-context.cpp` 中逐一追踪，而非由文档示意图得出。

### 2.2 增量 A：可关闭、分 sink 的推理期内存观测框架

| 代码锚点 | 实现与创新点 | 可验证边界 |
|---|---|---|
| `llama.cpp/CMakeLists.txt:118-128`、`ggml/src/ggml-cpu/CMakeLists.txt:55-84` | `LLAMA_MEM_TRACE` 是编译期开关；仅开启时把 `trace_writer.cpp`、`tensor_trace.cpp`、Expert/KV/OS 模块编入 CPU backend。 | 未开启时 `trace_event.h:113-190` 提供 inline no-op，不需要 trace 的链接符号；这是避免改变常规构建链接关系的适配。 |
| `trace/trace_event.h:16-110` | 定义稳定的 C ABI 事件面：phase、step、ubatch、tensor、KV、Router、mmap、slot admission。 | 所有后续 hook 通过此头文件解耦，避免 `src/`/`ggml` 直接依赖大型控制器实现。 |
| `src/llama-context.cpp:1308-1323,1714-1744` | 在 encode/decode 的真实 `llama_ubatch` 生命周期设置 phase/step、记录 step/token 边界；`llm_mem_trace_ubatch_guard` 防止异常路径留下悬垂 ubatch 指针。 | STEP 指标覆盖 prefill 与 decode；`ubatch.n_tokens > 1` 判作 PREFILL，单 token 判作 DECODE。 |
| `trace/trace_writer.cpp:19-115,249-337,372-463` | 四个独立 JSONL sink（tensor/KV/expert/memory）各有异步 writer、队列上限、drop 或阻塞策略，并在 `summary.json` 输出 enqueued/written/dropped。 | `LLM_MEM_TRACE_ALLOW_DROP=0` 默认保数据但可能反压推理；`=1` 可降低侵入性但必须检查 `dropped`。 |
| `trace/os_trace.cpp:41-309` | 采集 `/proc/self/stat`、`smaps_rollup`（可选）、`/proc/self/maps`、cgroup v2 `workingset_refault_*`，写出 RSS/VMS/fault/refault。 | Linux 外仅退化输出 minimal event；不是跨平台物理驻留测量器。 |

工程意义：这将“总进程时间”拆为模型映射、prefill、逐 decode step、KV append、Router 选择、first-use 和 syscall 事件，能够定位负优化究竟发生在加载还是运行期。学术上，它为页级/对象级策略提供可复查的因果时间轴，而不是事后仅报一个平均 TPS。

### 2.3 增量 B：Router 输出到 Expert slice 的同步观测与语义预取通路

| 代码锚点 | 核心数据结构/函数 | 实现细节 |
|---|---|---|
| `ggml/src/ggml-cpu/ggml-cpu.c:3044-3073` | 图执行 hook；`llm_mem_trace_moe_weights_requires_sync()` | 对 Router `GGML_OP_GET_ROWS` 先让所有 CPU worker 完成 producer shard，再由 `ith==0` 读取；随后复用已有 barrier 阻止其他线程跨入下一个 node。避免 thread 0 读到半完成 Router tensor。 |
| `trace/expert_trace.cpp:107-211` | `llm_mem_trace_moe_weights()` | 只接受 host tensor、I32/I64 expert ids 与 F32/F16/BF16 权重；解析 layer、token、top-k 后传递真实 expert id/score，而非训练外 predictor。即使 EXPERT sink 关闭，只要控制器需要 Router，仍要求同步。 |
| `trace/expert_tensor_registry.h`、`trace/expert_tensor_registry.cpp:21-107` | `ExpertTensorInfo`、`ExpertTensorRegistry`、`expert_slice_range()` | 从 `_exps.weight` 的 `ne[2]` / `nb[2]` 记录 expert 数和 stride；slice 地址精确计算为 `addr + expert * expert_stride`，长度截断到 tensor 边界。 |
| `trace/tensor_trace.cpp:6724-6885` | `llm_mem_trace_prefetch_expert_layer()` | 先把原始 Router demand 登记给 Memory Object；随后可选按 Router 或可复现 random 对照选择 top-k，按 phase TTL 去重，构造 `ExpertHintTask`。随机选择只影响预取 target，不污染真实 Router demand 观测。 |
| `trace/tensor_trace.cpp:5200-5725`、`trace/expert_hint_priority.cpp:5-58` | `ExpertHintQueue`、`ExpertHintTask`、`ExpertHintPriorityKey` | 支持同步或异步 worker、容量回退、score/deadline/deadline_score 排序、出队时 pressure/value 重判，并实际调用 `MADV_WILLNEED`（可选 `posix_fadvise`）。 |
| `trace/expert_task_lifecycle.{h,cpp}`、`trace/expert_first_use_matcher.cpp:61-167` | `ExpertTaskState`、`ExpertFirstUseMatcher` | 强制 `New→Created→Admitted→Enqueued→Dequeued→Issued` 等合法转移；first-use 按 `(step,layer,expert,tensor)`、stage、因果时间与地址区间 overlap 匹配，能区分 `issue_after_first_use`、`stage_mismatch` 与 `address_mismatch`。 |

这是本项目最具辨识度的增量：不少系统只在请求或 layer 粗粒度做 expert cache 统计；这里的控制对象是 `(layer, expert, tensor slice)`，并通过 Router 完成后的真实数据绑定其语义身份，再用 first-use 验证“hint 是否在实际消费前发出”。它没有复制 GPU/SSD expert offload，而是把问题落在 Linux file-backed mmap 页的 advisory 层。

### 2.4 增量 C：Memory Object、shadow working set 与安全回收

| 代码锚点 | 核心数据结构/函数 | 实现与保护条件 |
|---|---|---|
| `trace/expert_memory_object.h:12-136` | `ExpertMemoryObject`、`ExpertMemoryObjectCounters` | 对象键为 `layer:expert:tensor`；保存 `pending_users`、`active_users`、inflight hint、shadow working-set、probation 与 COLD/DONTNEED episode 状态。统计量明确区分语义 demand、候选、issued、失败和受保护跳过。 |
| `trace/expert_memory_object.cpp:146-246` | `register_demand()` / `observe_first_use()` | 同一步重复 demand 合并；first-use 必须同一步存在 pending demand 才转入 active，否则记录 unmatched；计数下溢会转化为 `invariant_violations`，而不是静默回绕。 |
| `trace/expert_memory_object.cpp:89-144` | `evict_to_working_set_budget_unlocked()` | 超预算时在无 pending/active 的对象中按 `last_touch_seq` 选 deterministic LRU victim；受保护对象不可驱逐，无法满足预算时记录 `budget_unresolved_due_to_protection`。这里的 working set 是逻辑 membership，并不声称等同物理 residency。 |
| `trace/expert_memory_object.cpp:353-530` | `end_layer_and_collect_madv_cold_candidates()` / `...dontneed...` | 只有对象已 shadow eviction、无 pending/active（DONTNEED 还要求不 inflight）、达到 grace step 且通过完整 step byte budget 才产生候选；COLD 与 DONTNEED 分别记录一次 episode。 |
| `trace/tensor_trace.cpp:6074-6297` | `LayerTracker::on_end()` | 在 layer 结束时把候选接到 `MADV_COLD`/`MADV_DONTNEED`，并受 TTL、pressure、Runtime Rescue 或 calibrated controller 限制。`MADV_COLD` 不可用的平台会记录失败，不伪造成功。 |
| `trace/expert_calibration_shadow.cpp:18-362` | `CalibrationProfile` | 只在 decode、rescue 安全、无新 invariant/保护违例、pending/active/inflight 均为 0、且 issue ratio ≥0.70 的样本入库；16 个健康样本后才计算 median/p25/p75。Shadow 计数不修改 hint/reclaim 状态。 |

工程意义：该模块将“请求已路由”“hint 处理中”“tensor 首次被用”“layer 已结束”“可进入 reclaim probation”显式化，能避免把尚有语义需求的对象作为回收对象。学术意义在于把 prefetch 评估从“发了多少 hint”提升到 demand–first-use–reclaim 的可检验状态机；但其 LRU/阈值仍是工程启发式，不是已证明最优的学习策略。

### 2.5 增量 D：mmap phase advice 与受限内存 admission

| 代码锚点 | 实现 | 可验证含义 |
|---|---|---|
| `src/llama-mmap.h:16-74` | `llama_mmap_populate_policy`（default/populate/skip/auto）和包含 expert 数、模型映射大小、headroom 的 admission contract。 | 将决策输入固定为显式结构，而不是散落的环境变量判断。 |
| `src/llama-mmap.cpp:230-313` | `llama_mmap_populate_admit()` | Linux 上优先读取当前 cgroup v2 `memory.current/memory.max`，失败时退到 `/proc/meminfo:MemAvailable`；仅当模型为 sparse MoE、headroom 可知且 `model_bytes/headroom > threshold` 时，`auto` 选择 skip。forced populate/skip 优先级高于 legacy switch。 |
| `src/llama-model-loader.cpp:1352`、`src/llama-model.cpp:1405` | 映射初始化把 `n_expert/n_expert_used` 和所有 mapping 字节传给同一次 admission。 | split GGUF 的多个 mapping 共享一个决定，避免每片文件独立猜测。 |
| `src/llama-mmap.cpp:784-846,1021-1056` | mmap flags、`POSIX_FADV_SEQUENTIAL`、`llama_mmap_decode_normal_once()` | 非 skip 时实际设置 `MAP_POPULATE`；可保留 `dup(fd)` 指向同一 open file description，在首 decode 一次性恢复 `POSIX_FADV_NORMAL`。 |
| `src/llama-context.cpp:1717-1729`、`trace/tensor_trace.cpp:6454-6548` | T0/T1/T2 mmap audit、`MMAP_POPULATE_ADMISSION`、`MODEL_MMAP`、`MODEL_DECODE_BEGIN`。 | 记录 admission 与真实 mmap flag 是否一致；不以日志代替 syscall 结果。 |

价值在于把“大模型加载是否抢占受限内存”从固定开关提升为针对 sparse MoE 和当前 cgroup 的 admission 问题，同时把加载阶段与 decode 阶段分开审计。它不是通用内存预测器：模型全映射字节、headroom 和阈值的比值没有建模共享 page cache、KV、GPU offload 或并发进程。

### 2.6 增量 E：KV 观测、离线策略筛选和服务 admission

| 代码锚点 | 实现 | 审计判断 |
|---|---|---|
| `trace/kv_trace.cpp:104-195` 与 `src/llama-kv-cache.cpp:1022-1093` | 识别 `GGML_OP_SET_ROWS` 的 K/V cache append，记录 layer、bytes、token IDs、ctx length 和 backend；另记录 cache cell reuse。 | 是真实运行时观测，不改 KV 分配/淘汰语义。 |
| `trace/simulate_kv_cache_policy.py:164-533` | 从 `KV_APPEND` 离线构造 full、paged_blocks、量化比例、sliding window、sink_recent、budget LRU 估算。 | 属于 D 级模拟；不重跑模型，不在 C++ runtime 中分块分配或量化 KV。 |
| `trace/simulate_kv_cache_policy.py:323-352,696` | Paged block 注释明确标注为 PagedAttention/vAttention 风格；H2O 被明确拒绝，因 trace 无 attention score。 | 防止把脚本结果误称为 PagedAttention/H2O runtime 实现。 |
| `trace/tensor_trace.cpp:3283-3550` 与 `tools/server/server-context.cpp:1789-1913` | `ExpertPressureController` 读取 current-cgroup 的 memory/PSI/refault；`kv_slot_admission_allows()` 决定 active slot 上限。server 在已找到 idle slot、尚未 attach state 时 defer 非 parent task。 | 这是保守的入场控制，不会回收或修改飞行中的 KV；High/Critical 至少保留一个 slot，避免仅因压力产生永久零服务。 |

工程价值是把模型内核、进程内存、cgroup 和 server slot 连成可观测闭环；学术边界是 KV 的“策略”目前主要是 trace-driven 容量估算，尚不能宣布实现了 vLLM PagedAttention、heavy-hitter KV 或 KIVI。

### 2.7 增量 F：实验可复查性与分析器约束

| 代码锚点 | 实现 | 作用 |
|---|---|---|
| `trace/write_run_manifest.py:31-243` | 捕获 git commit/dirty、模型/提示/binary hash、host、CPU affinity、cgroup、环境变量。 | 把“实验条件一致”变为机器可读证据。 |
| `trace/summarize_repeat_runs.py:179-221` | 拒绝 code/model/binary/host/cache/cgroup/workload fingerprint 不一致的 run。 | 避免把不等价样本混入均值。 |
| `trace/compare_trace_runs.py:42-60` | official mode 要求 whole-process GNU time fault、`STEP_END` latency、存在 decode step 和完整 Pareto metric。 | 防止 prefill-only smoke、缺字段或局部 fault 指标支配比较。 |
| `trace/analyze_trace.py`、`trace/trace_metrics.py`、`trace/tests/` | 从 JSONL 计算 step latency、fault window、Router/KV/任务生命周期指标。 | 保留 legacy token latency 作为显式 fallback，而非默默混用。 |

## 3. 量化效果评估

### 3.1 本次重新执行的正确性证据（A 级）

在 `LLAMA_MEM_TRACE=ON` 构建中重新构建并运行，结果如下：

| 测试组 | 数量 | 结果 | 覆盖的实现 |
|---|---:|---|---|
| CTest | 8/8 PASS | `test-mmap-phase-advice`、Router double-barrier/control decoupling、control-only trace、hint priority、task lifecycle、Memory Object、calibration shadow | policy 优先级/FD 生命周期、并发观测顺序、状态转移、first-use 因果/overlap、working-set 保护、16 样本校准门槛 |
| Python `unittest` | 14/14 PASS | `test_analysis_metrics.py` 7、`test_compare_metrics.py` 3、`test_repeat_validation.py` 4 | 以 STEP_END/whole-process fault 为准、缺字段拒绝、输出/commit/fingerprint 不匹配拒绝 |

测试并不证明真实大模型性能：Router 同步测试是合成 4 线程 shard 场景，Memory Object/校准测试是确定性 unit test，Python 测试主要校验 JSON 解析和统计规则。它们证明的是状态机和分析契约未回归。

### 3.2 当前 HEAD mmap 消融（B 级，唯一可安全量化的性能结论）

原始输入来自 `experiments/results/experiment_0815/`：

- commit 固定为 `88fc9e13808739640ed1d2305c76358cc14d98d4`，所有 run `git_dirty=false`；二进制 SHA-256 一致为 `2caadda1...4480f9e4e`，prompt SHA-256 一致为 `59f51358...3217897e4f`。
- 模型映射大小为 16,356,375,168 bytes，运行环境为 WSL2 Linux 6.6.87.2、AMD Ryzen 9 7940H、8 CPU threads、`ctx=2048`、`batch=ubatch=512`、128 个生成 token、cold cache。
- baseline 与 skip-populate 均 N=5、交错顺序；skip-sequential 与 expert `MADV_RANDOM` 均 N=3。16 个 run 的 `exit_code` 均为 0。
- 重要限制：manifest 的 `model.sha256` 为 `null`，且没有有限 `memory.max`。因此模型路径/大小/mtime 已记录，但模型内容不是密码学固定；结果是“32GB 无限制内存下 mmap 消融”，不是受压 cgroup 结论。

| 指标（组均值） | Baseline N=5 | Skip `MAP_POPULATE` N=5 | 相对 Baseline | 解释 |
|---|---:|---:|---:|---|
| model load-to-ready | 39,414.54 ms | 4.18 ms | **-99.99%** | `MAP_POPULATE` 成本从加载路径移走。 |
| RSS peak | 15.5814 GiB | 12.7365 GiB | **-18.26%** | 进程峰值 RSS 较低。 |
| major faults | 3.4 | 386,551.4 | 大幅增加 | 按需页错误被转移到运行期。 |
| prefill total | 20,848.61 ms | 49,096.03 ms | **+135.49%** | 首次实际访问显著变慢。 |
| decode p95 | 139,599.64 µs | 164,237.10 µs | **+17.65%** | 尾延迟变差。 |
| decode throughput | 8.083 tok/s | 7.294 tok/s | **-9.76%** | 无净吞吐收益。 |
| whole-process wall time | 84.82 s | 73.11 s | -13.81% | 只看 wall 会错误掩盖 prefill/decode 退化，不能单独作为“加速”。 |

`memory_trace.jsonl` 可直接复核这一因果链：baseline 的 `MODEL_MMAP.map_populate=true`、样例耗时 61,173 ms；skip 样例为 `map_populate=false`、耗时约 0.022 ms，但随后 prefill/major-fault 指标上升。故正确结论是 **加载—运行期的代价迁移**，不是“skip-populate 优化了推理”。

补充负结果同样应保留：

| 条件 | N | 主要结果（相对 baseline 均值） | 审计结论 |
|---|---:|---|---|
| 跳过 `POSIX_FADV_SEQUENTIAL` | 3 | prefill -3.87%，但 decode p95 +11.15%，TPS -1.86% | 没有稳定净改善。 |
| Expert `MADV_RANDOM` | 3 | decode p95 +34.65%，TPS -12.91% | syscall 实验不带来收益；默认不应开启。 |

### 3.3 尚不能量化的已实现功能（C 级）

| 功能 | 已有代码/测试证据 | 缺失的实验，因而不能声称的效果 |
|---|---|---|
| Router Expert prefetch | 真实 Router→slice→`MADV_WILLNEED` 通路、task/first-use 测试通过 | 同 commit 的 controller off/on、相同 cgroup/输出 hash 下 TPS、p95、fault、first-use match rate 与 hint failure 对照。 |
| Memory Object + COLD/DONTNEED + rescue | 状态机、保护条件、校准测试通过 | 受压 cgroup 下 OOM 完成率、RSS/cgroup peak、refault、回收后 readmission、输出一致性；当前无合格 A/B。 |
| trace 本身开销 | queue/drop accounting 与分析器测试通过 | trace-off vs trace-on/controller-off 的 N≥5 同条件开销测试。 |
| KV slot admission | server defer 调用点真实存在 | `--parallel` 压力下 off/on 的完成率、排队时间、TTFT、吞吐和公平性对照。 |
| KV policy | 真实 KV append trace + 离线模拟 | runtime block allocation/eviction/quantization 及质量/正确性验证。 |
| auto mmap admission | unit test 覆盖 forced/default/auto 的分支 | 在有限 `memory.max` 下 default/auto/skip 的完成率和多次重复；当前 M1 是无 cgroup 上限。 |

## 4. 工程与学术意义（含边界）

1. **工程可落地性**：项目没有把策略埋在一个脚本中，而是同时提供 compile-time no-op、runtime env gate、JSONL sink 统计、manifest、trace-drop 检查与 server integration。对生产系统而言，先观测再渐进开启 policy 的可回退结构比“默认启用回收”更安全。
2. **可证伪的 MoE 内存研究对象**：`ExpertMemoryObjectTracker` 和 `ExpertFirstUseMatcher` 将 expert 策略拆成可测状态与因果关系；这比只统计 Router 热度更接近能被实验反驳的系统假设。
3. **负结果有价值**：M1 证实单纯取消 `MAP_POPULATE` 和注入 `MADV_RANDOM` 都不是此工作负载的有效优化。保留失败结果可约束后续 controller，避免选择性报告。
4. **学术突破边界**：当前代码是“OS mmap 页的可观测/可控制原型”，不是新的通用 MoE cache replacement 理论，也没有实现 GPU/SSD expert offload、学习型跨层 expert predictor、PagedAttention runtime 或 KV heavy-hitter/量化算法。论文级性能主张须先补齐 C 级功能的受控实测。

## 5. 借鉴/继承工作与一一映射

### 5.1 直接继承或链接的开源代码/库

下表只列出了在仓库中可由源码、许可证、CMake 或 `#include` 直接证实的来源；“本项目改进”是本项目对其使用位置的改动，不把上游原有功能算作自研。

| 借鉴的源项目/库 | 本项目对应模块/代码路径 | 本项目的改进/重构/适配说明 |
|---|---|---|
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)（MIT） | 全部 `llama.cpp/src/`、`llama.cpp/ggml/`、`common/`、`tools/`；许可证 `llama.cpp/LICENSE` | 这是推理、GGUF loader、KV cache、server 与 graph runtime 基座。本项目只在 `src/llama-context.cpp`、`llama-model-loader.cpp`、`llama-mmap.*`、`llama-kv-cache.cpp`、`tools/server/server-context.cpp` 和 `ggml-cpu.c` 做 hook/admission 修改，并新建 `trace/`。 |
| [ggml CPU backend](https://github.com/ggml-org/llama.cpp/tree/master/ggml)（随 llama.cpp） | `ggml/src/ggml-cpu/ggml-cpu.c` | 在 CPU graph compute loop 中加入仅 thread 0 的 tensor/KV/Router hooks；针对 Router GET_ROWS 加生产者完成 barrier，未重写算子实现。GPU backend 不含等价 hook。 |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib)（MIT） | `vendor/cpp-httplib/`；`common/http.h`、`common/download.cpp`、`tools/server/server-http.cpp`、`server-models.cpp`；由根 `CMakeLists.txt:206` 加入 | 继承 HTTP client/server/TLS 能力；本项目的 KV admission 修改在 server 的任务调度层，未修改该库。 |
| [nlohmann/json](https://github.com/nlohmann/json)（MIT） | `vendor/nlohmann/json.hpp`、`licenses/LICENSE-jsonhpp`；`common/*` 与 `tools/server/*` 中大量 `#include <nlohmann/json...>` | 用于上游的 chat/HTTP/JSON；项目新增 trace 文件采用手写 JSONL 拼接，未向 nlohmann/json 增加实现。 |
| [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) | `vendor/sheredom/subprocess.h`；`tools/server/server-models.cpp`、`server-tools.cpp` | 上游 server 子进程封装；KV admission 未触及此依赖。 |
| [stb_image](https://github.com/nothings/stb)（public domain/MIT 双许可） | `vendor/stb/stb_image.h`；`tools/mtmd/mtmd-helper.cpp` | 上游多模态图像解码；与本项目内存策略无直接关系。 |
| [miniaudio](https://github.com/mackron/miniaudio)（public domain/MIT-0） | `vendor/miniaudio/miniaudio.h`；`tools/mtmd/mtmd-helper.cpp` | 上游多模态音频解码；与 trace 增量无直接关系。 |
| [LLGuidance](https://github.com/guidance-ai/llguidance)（可选下载依赖） | `common/CMakeLists.txt:142-166` 在 `LLAMA_LLGUIDANCE` 时 `ExternalProject_Add` | 上游 structured output 依赖；本项目未改其代码，也不属于默认 trace build。 |
| [KleidiAI](https://github.com/ARM-software/kleidiai)、llamafile SGEMM | `ggml/src/ggml-cpu/CMakeLists.txt:600-652`、`ggml-cpu/kleidiai/`、`ggml-cpu/llamafile/` | 都是随上游 CPU backend 的可选优化路径；当前 trace 实验为 CPU host 路径，新增 hook 没有复制或改写这些 kernel。 |
| Linux `mmap(2)` / `madvise(2)` / `posix_fadvise(2)` | `src/llama-mmap.cpp:784-846`、`trace/tensor_trace.cpp:620-700,6074-6297` | 本项目把上游 mmap 使用扩展为 MAP_POPULATE admission、SEQUENTIAL→NORMAL phase transition、WILLNEED/COLD/DONTNEED hint，并输出 syscall event；仍完全依赖内核最终页面策略。 |
| Linux cgroup v2 / PSI / `memory.stat` | `src/llama-mmap.cpp:78-313`、`trace/os_trace.cpp:41-309`、`trace/tensor_trace.cpp:3283-3450` | 适配当前进程所属 cgroup，读取 memory.current/high/max、memory.pressure、workingset_refault；用作 admission、预算和 slot defer 信号。不是直接移植内核回收器。 |

### 5.2 论文/系统思想到代码的映射

项目的 `docs/source-attribution.md` 列出了若干研究方向。为避免“引用即实现”的误导，下表按代码审计结果区分为 **有功能语义映射**、**仅离线模拟** 和 **未发现直接实现**。后两类不应写成代码移植或已复现论文结果。

| 借鉴的源论文/系统 | 本项目对应模块/代码路径 | 本项目的改进/重构/适配说明 |
|---|---|---|
| [MoE-Infinity](https://arxiv.org/abs/2401.14361) | `trace/expert_trace.cpp`、`expert_tensor_registry.*`、`tensor_trace.cpp:6724-6885`、`expert_memory_object.*` | 有语义映射：均关注 Router 驱动的 expert 访问与预取。本项目改为 llama.cpp CPU file-backed mmap 的 slice-level `MADV_WILLNEED`，并加 first-use/lifecycle 验证；**未实现**其 GPU/CPU/SSD 分层 offload/cache manager。 |
| SpecMD（Least-Stale 思路） | `expert_memory_object.cpp:89-144,353-530`、`expert_task_lifecycle.*` | 仅启发式映射：本项目以 `last_touch_seq` 做逻辑 LRU、以 pending/active/step 做 semantic stale 判断；未发现 SpecMD 代码、公式或可识别的 Least-Stale 原算法移植。 |
| ST-MoE | `expert_trace.cpp` 的真实 Router route，`expert_first_use_matcher.cpp` 的 step/layer 统计 | 只保留“跨 token/layer 可预测性值得观测”的研究动机；当前代码没有 learned cross-layer predictor，也没有论文名称或模型参数的直接实现。 |
| [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)、[vAttention](https://arxiv.org/abs/2405.04437) | `trace/simulate_kv_cache_policy.py:323-352` 的 `simulate_paged()` | 仅离线容量估算：按固定 token block 估算保持完整上下文的占用；C++ KV cache 没有 block table、virtual mapping 或 paged allocation。 |
| [StreamingLLM](https://arxiv.org/abs/2309.17453) | `simulate_kv_cache_policy.py:375-427` 的 `sink_recent` | 仅离线 sink+recent 保留模拟；不改变运行时 attention/KV。 |
| [H2O](https://arxiv.org/abs/2306.14048) | `simulate_kv_cache_policy.py:696` | 明确未实现：脚本指出只有 append/reuse/bytes，缺 attention score/重要度埋点，不能判断 heavy hitter。 |
| [KIVI](https://arxiv.org/abs/2402.02750) | `simulate_kv_cache_policy.py:352-374` 的比例估算 | 仅估算 int8/int4 占用比例；没有 C++ KV quant/dequant、质量或速度测试。 |
| FlexInfer、SP-MoE、OD-MoE | 未在 `trace/`、`src/`、`ggml` 生产源码中发现项目名、导入或对应训练 predictor/offload 实现 | 只能标注为文献背景：异步预取、受压保留、timely/multi-layer prediction 的问题定义被借鉴；不能称为移植或复现。 |
| Linux MGLRU、DAMON/DAMON_RECLAIM | `expert_memory_object.*`、`tensor_trace.cpp` 的 MADV_COLD/DONTNEED 分支 | 仅借鉴“冷热/回收”方向；代码没有调用 DAMON API、没有实现 MGLRU，实际动作是用户态对确定 slice 调用 `madvise`。 |

## 6. 审计发现的工程风险与改进优先级

| 优先级 | 发现 | 代码证据 | 风险与建议 |
|---|---|---|---|
| P0（证据完整性） | 现有 M1 模型 SHA 为 `null`，没有有限 cgroup。 | `experiments/results/experiment_0815/runs/*/latest/run_manifest.json` | 结果不能抵抗“同路径模型被替换”，也不能验证 auto admission 的目标场景。后续所有正式 run 必须强制 model SHA、`memory.max` 与 cgroup snapshot；缺失即 INVALID。 |
| P0（效果结论） | Expert prefetch/reclaim/KV admission 没有当前 HEAD 的受控 A/B。 | 功能已在 `tensor_trace.cpp`/`server-context.cpp` 调用，但 M1 manifest 中 `LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=0`，没有对应实验矩阵结果。 | 禁止宣称降低 p95、提高 TPS 或改善 OOM；先做 controller off/on、same host/binary/model/prompt/cgroup、输出 hash 一致、N≥3/5 的测量。 |
| P1（扩展性） | `ExpertMemoryObjectTracker` 在 LRU eviction 与每层 end 时遍历整个 `unordered_map`；`ExpertTensorRegistry::hinted` 在 TTL=0 时以 `step:slice` 永久累积。 | `expert_memory_object.cpp:89-144,327-350`；`expert_tensor_registry.cpp:50-83` | 长请求/常驻 server、large top-k 可能出现 O(objects × layers) 扫描与无界去重集合增长。应以 layer 索引/heap 替代全表扫描，并为 zero-TTL 使用每-step 可清理集合或 bounded generation 标记。 |
| P1（平台范围） | Router hook 仅编入 `ggml-cpu`，且 `expert_trace.cpp` 要求 host-readable Router tensor。 | `ggml-cpu/CMakeLists.txt:55-84`；`expert_trace.cpp:129-151` | GPU/offloaded execution 不等价覆盖；报告和运行时应明确标出 CPU-host 限制，避免把 CPU 结果外推至 CUDA/Metal。 |
| P1（启发式泛化） | mmap auto 使用全模型 mapping bytes/headroom 的一次比值；校准 summary 仍报告硬编码 Phase-2D/2E 常量。 | `llama-mmap.cpp:230-313`；`expert_calibration_shadow.cpp:252-326` | 它未计入 page cache 共享、KV、并发请求或 GPU buffer；硬编码常量只适合 scale audit。应将决策阈值与收敛样本分离，并在每机型/模型重新校准。 |
| P2（可观测性干扰） | writer 默认不 drop，队列满时阻塞生产线程。 | `trace_writer.cpp:49-70` | 这是“数据完整性优先”的合理选择，但会改变延迟。trace-on 性能报告必须同时给出 queue drop/queue limit，并补 trace-off 对照。 |
| P2（进程生命周期） | mmap NORMAL transition 是 process-wide one-shot；trace init 使用 `std::once_flag`。 | `llama-mmap.cpp:137-172,1021-1056`；`trace_writer.cpp:249-293` | 对一个进程内热重载模型或多轮独立 trace 不可自然重置。若 server 支持此类生命周期，需要显式 reset/re-init 设计与测试。 |

## 7. 最终审计判断

1. **创新真实性**：成立。新增实现集中在 trace、Router slice 语义、Memory Object 状态机、mmap admission、cgroup pressure、KV admission 与实验可追溯链，不是只改 README 或套壳调用。
2. **性能优化真实性**：对 mmap 的“加载快/RSS 低，但 prefill/decode 更差”这一权衡成立；“总体推理更快”不成立。`MADV_RANDOM` 当前是明确负结果。
3. **复杂策略有效性**：代码完成度和单元测试覆盖较好，但端到端收益尚未证实；应作为下一阶段实验对象，不应提前当成论文结论。
4. **借鉴关系**：推理基础与多种依赖直接来自 llama.cpp 生态；论文系统多为问题/方法启发。特别是 PagedAttention、H2O、KIVI、FlexInfer/SP-MoE/OD-MoE 不应被表述为本工程的 runtime 移植。

---

### 附：本次实际执行的验证命令

```bash
cmake --build llama.cpp/build --target \
  test-mmap-phase-advice test-router-tensor-observation-sync \
  test-router-control-decoupling test-trace-control-profile \
  test-expert-hint-priority test-expert-task-lifecycle \
  test-expert-memory-object test-expert-calibration-shadow -j 4

ctest --test-dir llama.cpp/build --output-on-failure \
  -R 'test-(mmap-phase-advice|router-tensor-observation-sync|router-control-decoupling|trace-control-profile|expert-hint-priority|expert-task-lifecycle|expert-memory-object|expert-calibration-shadow)'

python3 -m unittest discover -s llama.cpp/trace/tests -v
```

执行结果：8/8 CTest PASS；14/14 Python test PASS。