# 测试报告

## 1. 测试目标

本项目的测试同时覆盖功能正确性、实验数据完整性和性能结论可信度。测试不只检查程序能否运行，还检查一次实验是否具备可比较条件：代码版本、模型、输入、推理参数、缓存状态、内存限制和 trace 完整性必须可追溯。

## 2. 测试环境

- 操作系统：Linux，使用 cgroup v2 的实验需启用统一层级。
- 构建系统：CMake。
- 编译目标：`llama-cli`。
- Trace 构建选项：`LLAMA_MEM_TRACE=ON`。
- 默认测试模型：`models/Qwen3.5-35B-A3B-Q3_K_M.gguf`。
- Python 分析依赖：见 `llama.cpp/trace/requirements-analysis.txt`。

模型和 trace 输出体积较大，不放入仓库。复现时通过 `MODEL_FILE` 指向本地 GGUF 文件。

## 3. 自动检查

### 3.1 构建与默认关闭回归

```bash
cmake --build llama.cpp/build --target llama-cli

cmake -S llama.cpp -B llama.cpp/build-no-trace \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_MEM_TRACE=OFF
cmake --build llama.cpp/build-no-trace --target llama-cli
```

第二组命令用于确认关闭 trace 后仍能构建，新增接口在该配置下由空实现承接，不改变默认推理路径。

### 3.2 静态检查与单元测试

```bash
python3 -m py_compile llama.cpp/trace/*.py
python3 -m unittest discover -s llama.cpp/trace/tests -v

for script in llama.cpp/trace/*.sh; do
  bash -n "$script"
done

git diff --check
```

单元测试重点覆盖：

- `STEP_END` 优先于兼容字段 `TOKEN_END`。
- 全进程 major/minor faults 和峰值 RSS 使用 GNU time 数据。
- 缺失指标不会在 Pareto 计算中被当作 0。
- 聚合器拒绝 trace 丢失、脏仓库、参数不一致和输出哈希不一致的运行。

## 4. 功能测试

### 4.1 单次可信证据运行

```bash
MODEL_FILE=/path/to/model.gguf \
RUN_NAME=evidence_smoke \
TRACE_PROFILE=evidence \
CACHE_MODE=as-is \
NUM_TOKENS_PREDICT=1 \
bash llama.cpp/trace/run_trace_pipeline.sh
```

正式证据模式要求每个已启用 trace sink 满足：

```text
enqueued == written
dropped == 0
```

单次运行至少生成以下关键产物：

- `run_manifest.json`：版本、模型、输入、参数、硬件、cgroup 和关键环境变量。
- `cache_preparation.json`：缓存准备方式及系统调用结果。
- `process_metrics.json`：GNU time 提供的进程 wall time、峰值 RSS、缺页和文件 I/O。
- `summary.json`：各 trace sink 的入队、写入和丢失计数。
- `output.sha256`：模型标准输出哈希。
- `analysis/metrics.json`：阶段时间线和综合分析指标。

### 4.2 轻量性能运行

```bash
MODEL_FILE=/path/to/model.gguf \
RUN_NAME=benchmark_smoke \
TRACE_PROFILE=benchmark \
CACHE_MODE=as-is \
NUM_TOKENS_PREDICT=1 \
bash llama.cpp/trace/run_trace_pipeline.sh
```

`benchmark` profile 关闭高开销的全量 tensor、KV、residency 和 smaps 采样，保留策略判断所需的 expert/memory 事件和全进程指标，避免插桩开销主导被测结果。

### 4.3 重复矩阵 dry-run

```bash
REPEAT_COUNT=2 \
RUN_PREFIX=dryrun_credible \
MEMORY_MAX=8G \
MEMORY_SWAP_MAX=1G \
bash llama.cpp/trace/run_finalist_repeat_matrix.sh
```

dry-run 应显示四种方案在相邻重复轮次中的位置轮换，并打印缓存准备、内存限制、运行和聚合命令，不启动长时间推理。

## 5. 正式性能测试协议

正式对比推荐采用以下条件：

- `REPEAT_COUNT=8`，四方案按拉丁方顺序轮换。
- `CACHE_MODE=cold`；文件级 `POSIX_FADV_DONTNEED` 失败时本次运行无效。
- 使用同一 clean commit、模型文件、prompt、CLI、推理参数和 cgroup 限制。
- trace 不允许丢失，进程退出码必须为 0。
- 同组运行的 `output.sha256` 必须一致。
- decode 延迟必须来自 `STEP_END`，总 faults 和峰值 RSS 必须来自 GNU time。
- 聚合前由 `summarize_repeat_runs.py` 拒绝不完整或不可比的样本。

主要比较指标为：decode 均值与 p95、吞吐、全进程 major/minor faults、峰值 RSS、swap 峰值、hint 数和进程 wall time。Pareto 排序对缺失值保持缺失，不将其视为 0。

## 6. 历史探索结果

早期 N=3 结果曾显示 `deadline_score` 在当时采集口径下具有较好的延迟和 swap 表现，`expert_prefetch` 的 major faults 较低，`decode_ttl1` 的 hint 数较少。该批数据存在以下限制：

- 运行顺序未做充分位置轮换。
- 冷/热文件缓存状态没有形成强制证据。
- decode 计时与全进程 faults 的权威来源尚未分离。
- trace 队列完整性、输入输出一致性和脏仓库状态未被聚合器强制检查。

因此该批数据仅作为研发过程中的候选筛选依据，不再用于宣称某策略已经获胜。正式性能结论须以本报告第 5 节的新协议复测结果为准。

## 7. 已知限制与后续测试

- 当前 KV 工作以 trace 分析和离线策略模拟为主，尚未完成运行时分页式 KV 管理。
- `MADV_DONTNEED` 和 `MADV_PAGEOUT` 仍只作为显式实验项，不作为默认策略。
- 正式 N=8 矩阵耗时较长，需要在 clean commit、稳定供电和可写 cgroup 环境中单独执行。
- 最终报告需补充至少两类内存限制和一组不同输入长度；条件允许时再增加第二种模型或硬件平台。

## 8. 回归标准

- `LLM_MEM_TRACE_OS_HINTS=0` 时不执行 expert 页面提示策略。
- `LLM_MEM_TRACE=0` 或 `LLAMA_MEM_TRACE=OFF` 时新增 trace 接口不改变模型计算。
- 所有优化策略默认关闭，必须通过环境变量显式启用。
- 功能测试失败、产物缺失、trace 丢失或实验条件不一致时，不生成正式获胜结论。

## 9. 本轮验证记录（2026-07-11）

- 启用 trace 的增量构建通过。
- 关闭 trace 的独立 Release 构建通过；验证过程中发现并修正了 no-trace 配置缺少 `trace_event.h` 包含路径的问题。
- Python 语法检查、全部 shell 语法检查和 `git diff --check` 通过。
- 9 个分析与聚合单元测试全部通过，其中新增压力/slack/预测指标聚合回归。
- 四方案 N=2 拉丁方矩阵 dry-run 通过，第二轮顺序相对第一轮轮换一个位置。
- baseline/deadline_score 的 4096 MiB cgroup 矩阵 dry-run 通过。
- 使用现有 GGUF、`TRACE_PROFILE=benchmark`、`CACHE_MODE=as-is`、`NUM_TOKENS_PREDICT=1` 完成短 smoke。
- smoke 中 expert sink 为 `19112/19112/0`，memory sink 为 `1233/1233/0`，格式依次为 `enqueued/written/dropped`。
- 六类可信证据产物和输出哈希均生成，分析指标来源为 `STEP_END` 与 GNU time。

本次 smoke 只出现 3 个 prefill step，没有形成 decode 样本，因此仅证明流水线和证据产物可用，不作为性能结论。正式 N=8 矩阵未在本轮自动执行。

## 10. 双反馈与预测功能验证（2026-07-11）

使用 `feedback_slack_predict`、`TRACE_PROFILE=benchmark`、`CACHE_MODE=as-is`、`NUM_TOKENS_PREDICT=1` 完成功能 smoke：

- expert sink `19112/19112/0`，memory sink `28495/28495/0`，无 trace 丢失。
- 产生 91 个 `EXPERT_PRESSURE` 样本，并记录 memory ratio、PSI、refault 和动态预算。
- 异步队列入队 663 项、执行 636 项，fallback 为 0；micro-batch 合并节省 3 次系统调用，按 slack 取消 24 项。
- 在线预测器观察 19112 条 route，形成 301 个可评估候选，命中 181 个；precision 60.13%、recall 14.98%、set hit rate 80.13%。
- 当次 WSL 根 cgroup 的 PSI/refault 主要被判定为高或临界压力；852 条预测相关决策中，72 条实际发出 predicted hint，其余由 value/pressure gate 拒绝。

以上数据证明预测、压力反馈、成本门控和指标分析链路可运行。由于本次使用 `as-is` 缓存、脏工作区、无限制根 cgroup 且没有 decode 样本，不能据此判断控制器优于 baseline，也不能把预测命中率直接写成端到端收益。下一步必须在 delegated cgroup 中执行四方案 N=8 冷缓存矩阵。

## 11. Expert Prefetch 任务生命周期验证（2026-07-13）

- Trace-On Release 构建、Trace-Off Release 构建和 `test-expert-task-lifecycle` 均通过；Python 分析回归共 6 项通过。
- detail smoke 产生 26,382 个唯一任务、26,183 个 `issue_id`，其中 199 个 coalesced group 为一对二；字段缺失、非法状态迁移、重复 task ID、issue 计数不一致和 syscall 链接错误均为 0。
- 逻辑 first-use 匹配 19,806/26,382（75.07%）；未匹配主要来自异步 hint 晚于实际使用。事件明确写入 `semantics=logical_first_use` 和 `physical_load_observed=false`，不把 `madvise` 返回解释为 page-in 完成。
- benchmark 默认 summary 只比 off 多任务汇总记录，不写逐任务事件，也不为每条 syscall 分配 `issue_id`。

使用 `as-is` 缓存、7 个 Decode token，按 summary/off 交替顺序各运行 N=3；中位数结果如下：

| 指标 | off | summary | 相对变化 |
| --- | ---: | ---: | ---: |
| 全进程 wall time | 34.44 s | 34.62 s | +0.52% |
| Decode throughput | 5.8447 tok/s | 5.8376 tok/s | -0.12% |
| Decode p95 | 189.47 ms | 189.90 ms | +0.22% |
| memory trace 字节 | 12,318,443 | 12,320,006 | +0.013% |

6 次运行均为零丢事件，输出 hash 均为 `5ee568424c71ea7436b6c4c3b899ae9dde3c6bc1c48bb3da71ca254c22073b68`。一轮 summary 因 major faults 升至 159,120，wall time 达 137.72 s；该异常值保留，说明当前 16 GB 模型/8 GB 内存的 `as-is` 环境仍有明显换页噪声，正式结论需要 delegated cgroup、固定缓存条件和更多重复。

同配置的 detail N=3 相对 off 中位数使 Decode 吞吐下降 11.60%、p95 增加 18.18%，memory trace 事件约为 6.48 倍。因此 detail 明确限定为 Evidence Profile，不进入性能排名。所有模式输出 hash 一致，任务 trace 未改变调度策略或模型输出。

## 12. Expert Tensor Stage 与 logical first-use 时序观测（2026-07-14）

实现只增加唯一 Stage 分类、Task/first-use 关联和聚合；stage 没有进入 Admission、priority、coalescing、Slack、Pressure Control、worker 数量或预取范围的任何决策表达式。Trace-On/Trace-Off Release 构建、C++ 状态机与 Stage 分类测试、Python 分析回归、shell 语法和 diff 检查均通过。

`stage_timing_detail_evidence_tasks` 使用 35B MoE 模型、Evidence Profile、3 个生成 token、既有 route prefetch、async 单 worker 和 as-is 缓存。四个 sink 均 `enqueued == written` 且 `dropped == 0`。28,302 个 Task 中 EARLY=18,868、LATE=9,434、UNKNOWN=0；eligible=28,302、matched=21,444、unmatched=6,858、ambiguous=0、duplicate-first-use=0、matcher peak live=522、expired=0。28,091 个 issue group 全部为 same-stage，cross-stage=0。真实运行未出现同键重复 Task；C++ 回归通过两 Task/一次 logical first-use 的一对多语义验证，并验证 ambiguous 和 duplicate-first-use 计数。

该次 Evidence 按 `(run_id, step, layer, expert)` 得到 9,434 组配对，未匹配组为 0；其中 PREFILL=8,794、DECODE=640。本次观测的 `late_after_early_count=9,434`、before=0、equal=0；总体有符号 delta p25/p50/p75/p95 为 148.62/161.96/176.57/250.45 ms，Decode 为 0.73/1.50/1.94/3.11 ms。模型同时使用 gate 和 up 两个 EARLY tensor，因此每个配对键有多条 EARLY logical first-use；分析固定采用该 stage 的最早时间戳并保留 `multiple_early_observation_groups` 诊断。以上只是一次运行中的时序结果，不构成 EARLY 必然早于 LATE 的保证。

summary/off 使用相同 8-token benchmark、as-is 缓存和既有 prefetch 配置，按 off/summary 交替各 N=3。6 次运行全部零丢失，输出 hash 均为 `5ee568424c71ea7436b6c4c3b899ae9dde3c6bc1c48bb3da71ca254c22073b68`；两次 3-token detail Evidence 的输出 hash 也一致。中位数如下：

| 指标 | off | summary | summary 相对 off |
| --- | ---: | ---: | ---: |
| 全进程 wall time | 34.88 s | 34.16 s | -2.06% |
| Decode throughput | 6.2797 tok/s | 6.1593 tok/s | -1.92% |
| Decode p95 | 169.13 ms | 176.11 ms | +4.13% |

off 第 3 次 wall time 为 48.44 s，显示 as-is 运行仍有明显换页长尾；因此 wall time 的负增量不解释为 summary 收益。Decode 吞吐和 p95 均未达到文档中的 1%/2%建议门槛，当前 summary 只能视为功能已验证，开销仍需在固定缓存和 delegated cgroup 条件下继续测量或优化。
