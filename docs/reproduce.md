# 复现说明

## 1. 推荐环境

- Linux x86_64，推荐 Ubuntu 22.04 或更新版本
- cgroup v2；受限内存真实矩阵需要可写 delegated parent 或可用的用户级 systemd
- CMake、支持 C++17 的编译器、Python 3.10+
- 足够存放源码、构建目录、GGUF 模型和本地 trace 的磁盘空间

Ubuntu/Debian 安装示例：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3 python3-pip time
python3 -m pip install -r llama.cpp/trace/requirements-analysis.txt
```

为避免 Matplotlib 写入不可用的用户配置目录，运行分析时建议设置：

```bash
export MPLCONFIGDIR=/tmp/llm_mem_trace_matplotlib
```

## 2. 获取源码

```bash
git clone <repo-url> llmop
cd llmop
```

确认关键入口：

```bash
test -f README.md
test -f llama.cpp/CMakeLists.txt
test -f llama.cpp/trace/run_trace_pipeline.sh
```

## 3. 准备模型

模型文件不提交仓库。默认脚本查找：

```text
models/Qwen3.5-35B-A3B-Q3_K_M.gguf
```

也可显式设置绝对路径：

```bash
export MODEL_FILE=/absolute/path/to/model.gguf
```

正式重复实验建议预先计算一次模型 hash，并在矩阵中复用，避免每次运行前读取整个模型而污染冷缓存：

```bash
export MODEL_SHA256="$(sha256sum "$MODEL_FILE" | awk '{print $1}')"
```

## 4. 构建

```bash
cmake -S llama.cpp -B llama.cpp/build \
  -DLLAMA_MEM_TRACE=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --target llama-cli -j"$(nproc)"
test -x llama.cpp/build/bin/llama-cli
```

关闭 trace 的编译回归可使用独立目录：

```bash
cmake -S llama.cpp -B llama.cpp/build-no-trace -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build-no-trace --target llama-cli -j"$(nproc)"
```

## 5. 静态与单元测试

```bash
python3 -m py_compile \
  llama.cpp/trace/analyze_trace.py \
  llama.cpp/trace/trace_metrics.py \
  llama.cpp/trace/compare_trace_runs.py \
  llama.cpp/trace/simulate_expert_cache.py \
  llama.cpp/trace/simulate_kv_cache_policy.py \
  llama.cpp/trace/stage_scheduling_analysis.py \
  llama.cpp/trace/summarize_repeat_runs.py \
  llama.cpp/trace/prepare_model_cache.py \
  llama.cpp/trace/write_run_manifest.py \
  llama.cpp/trace/validate_trace_summary.py

python3 -m unittest discover -s llama.cpp/trace/tests -p 'test_*.py' -v

bash -n llama.cpp/trace/run_trace_pipeline.sh
bash -n llama.cpp/trace/run_finalist_repeat_matrix.sh
bash -n llama.cpp/trace/run_cgroup_memory_matrix.sh
git diff --check
```

## 6. Trace profile

- `TRACE_PROFILE=evidence`：完整 tensor/KV/expert/memory trace，适合解释行为。
- `TRACE_PROFILE=benchmark`：关闭高流量 tensor/KV 和驻留采样，适合正式性能对比。
- `TRACE_PROFILE=custom`：允许用 `LLM_MEM_TRACE_TENSOR` 等变量逐项指定。

正式结果应分别保留 evidence run 和 benchmark run，不能用高开销全量 trace 的绝对耗时冒充无插桩性能。

Expert Prefetch 任务 trace 可独立切换：

| `LLM_MEM_TRACE_EXPERT_TASK_MODE` | 输出 | 用途 |
| --- | --- | --- |
| `off` | 不记录任务事件或任务汇总 | Trace 开销基线 |
| `summary` | 仅任务/stage/matcher 聚合，不写逐任务事件，不分配输出用 task/issue ID | benchmark 默认值 |
| `detail` | 完整任务事件、`issue_id`、stage 和逻辑 first-use 时序 | evidence 默认值 |

benchmark profile 仍保留真实 `madvise/posix_fadvise` 调用及错误，但把高流量 skip/reject 明细聚合到 summary。进入正式性能矩阵前，应以 `off` 为基线轮换运行三种模式；建议门槛为 Decode 吞吐下降不超过 1%、Decode p95 增加不超过 2%、trace 零丢失且输出 hash 相同。`detail` 未通过门槛时只能用于 evidence，不能用于性能排名。

`EXPERT_FIRST_USE_SUMMARY` 的 M1 语义检查包括 `eligible_tasks/matched_tasks/unmatched_tasks/ambiguous_matches/duplicate_first_use_ignored/matcher_peak_live_tasks/matcher_expired_tasks`。多 Token ubatch 的同键重复 Task 采用一次 logical first-use 对多 Task 的关联语义。`EXPERT_TASK_SUMMARY` 另记录 `same_stage_issue_groups/cross_stage_issue_groups`。分析结果 `metrics.json` 中的 `expert_stage_pairing` 按 `(run_id, step, layer, expert)` 输出总体、PREFILL、DECODE、逐 Layer 和未匹配原因；分析脚本不得从 tensor 名重新分类 stage。

M2.5 离线调度分析要求 detail Task 中存在真实 `sequence` 和 `deadline_ts_ns`。`analyze_trace.py` 会额外生成 `analysis/stage_scheduling_opportunity.json`；其中 `data_quality` 必须确认 `sequence_semantics=trace_sequence`、没有 deadline 代理，并报告观测到的 priority mode 和 worker 数量。若字段缺失，兼容分析会显式标记代理来源，该结果不能作为正式 M2.5 Evidence。

## 7. 单次 smoke test

开发中的脏工作区只能用于功能冒烟，不得写入正式报告：

```bash
ALLOW_DIRTY_REPO=1 \
CACHE_MODE=as-is \
TRACE_PROFILE=benchmark \
NUM_TOKENS_PREDICT=1 \
RUN_NAME=smoke_baseline \
MODEL_FILE="$MODEL_FILE" \
LLM_MEM_TRACE_OS_HINTS=0 \
bash llama.cpp/trace/run_trace_pipeline.sh
```

## 8. 单次双反馈控制器运行

```bash
RUN_NAME=feedback_slack \
TRACE_PROFILE=benchmark \
CACHE_MODE=cold \
MODEL_FILE="$MODEL_FILE" \
LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=feedback_slack \
bash llama.cpp/trace/run_trace_pipeline.sh
```

增加相邻层预测实验时，将控制器改为：

```bash
LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=feedback_slack_predict
```

高层 profile 会启用异步 deadline heap、4 workers、batch=8、100 us batch wait、压力反馈、slack 和 value gate。关键调参接口如下：

| 变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `LLM_MEM_TRACE_OPT_EXPERT_CACHE_MB` | 512 | low 压力下的基础预取预算 |
| `LLM_MEM_TRACE_OPT_EXPERT_PRESSURE_SAMPLE_MS` | 50 | cgroup/PSI/refault 采样周期 |
| `LLM_MEM_TRACE_OPT_EXPERT_PAGEIN_MBPS` | 512 | page-in 一阶带宽估计 |
| `LLM_MEM_TRACE_OPT_EXPERT_VALUE_MIN_RATIO` | 1.0 | 最小预计收益/成本比 |
| `LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH` | 8 | worker 一次最多取出的任务数 |
| `LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_WAIT_US` | 100 | 等待相邻任务形成 micro-batch 的上限 |
| `LLM_MEM_TRACE_OPT_EXPERT_PREDICT_TOPK` | 2 | 下一层预测候选数 |
| `LLM_MEM_TRACE_OPT_EXPERT_PREDICT_MIN_SAMPLES` | 8 | 启用一条转移统计前的最小样本数 |
| `LLM_MEM_TRACE_OPT_EXPERT_PREDICT_MIN_CONFIDENCE` | 0.10 | 预测候选最低置信度 |

不要仅根据 prediction precision 选择策略；必须同时检查实际 predicted prefetches、取消原因、hint 数、major faults、RSS/swap 和 decode p95。

## 9. 正式重复矩阵

默认只打印命令：

```bash
REPEAT_COUNT=1 RUN_PREFIX=dryrun_check \
bash llama.cpp/trace/run_finalist_repeat_matrix.sh
```

正式推荐 N=8，使四种方案在四个运行位置各出现两次：

```bash
RUN_REPEAT_MATRIX_EXECUTE=1 \
RUN_PREFIX=final_cold_8g \
REPEAT_COUNT=8 \
TRACE_PROFILE=benchmark \
CACHE_MODE=cold \
ORDER_MODE=latin \
ORDER_SEED=0 \
MEMORY_MAX=8G \
MEMORY_SWAP_MAX=1G \
MODEL_FILE="$MODEL_FILE" \
bash llama.cpp/trace/run_finalist_repeat_matrix.sh
```

设置 `MEMORY_MAX` 后脚本使用 `systemd-run --user --scope` 创建独立限制；无法建立限制时直接失败，不会退化为无限制实验。

## 10. Delegated cgroup 压力矩阵

先 dry-run：

```bash
MEMORY_LIMITS_MB=4096,5120,6144 \
RUN_GROUPS=baseline,deadline_score,feedback_slack,feedback_slack_predict \
REPEAT_COUNT=1 \
bash llama.cpp/trace/run_cgroup_memory_matrix.sh
```

真实执行需要可写的 delegated parent：

```bash
RUN_MEMORY_PRESSURE_EXECUTE=1 \
CGROUP_PARENT=/sys/fs/cgroup/<delegated-parent> \
MEMORY_LIMITS_MB=4096,5120,6144 \
RUN_GROUPS=baseline,deadline_score,feedback_slack,feedback_slack_predict \
REPEAT_COUNT=8 \
MODEL_FILE="$MODEL_FILE" \
bash llama.cpp/trace/run_cgroup_memory_matrix.sh
```

## 11. 运行产物与有效性

每个有效运行至少包含：

```text
run_manifest.json
cache_preparation.json
process_metrics.json
summary.json
output.sha256
analysis/metrics.json
```

以下任一情况会使正式聚合失败：仓库脏、进程非零退出、缺少产物、启用的 sink 丢事件或写入不完整、没有 `STEP_END`、没有 GNU time 全进程 faults、模型/二进制/prompt/参数/cgroup 不一致、确定性输出 hash 不一致。

## 12. 离线模拟

Expert cache：

```bash
python3 llama.cpp/trace/simulate_expert_cache.py \
  --trace-dir llama.cpp/trace_output/<run> \
  --output-dir llama.cpp/trace_output/contest_runs/expert_cache_simulation
```

KV policy：

```bash
python3 llama.cpp/trace/simulate_kv_cache_policy.py \
  --trace-dir llama.cpp/trace_output/<run>
```

M2.5 Stage Scheduling Opportunity Analysis 使用正常 pipeline 的 detail 输出，不单独启动模型：

```bash
ALLOW_DIRTY_REPO=1 \
RUN_NAME=stage_scheduling_opportunity_evidence \
TRACE_PROFILE=evidence \
CACHE_MODE=as-is \
NUM_TOKENS_PREDICT=3 \
CTX_SIZE=1024 \
LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=feedback_slack \
bash llama.cpp/trace/run_trace_pipeline.sh
```

`feedback_slack` 在此只用于采集仓库现有 `deadline_score` 配置的真实非零 deadline、sequence 和 4-worker Task。分析阶段离线重放同一批已 ISSUE Task，并输出 1/2/4-worker A/B 模拟；分析脚本本身不会修改 C++ queue。复用已有 trace 时可执行：

```bash
python3 llama.cpp/trace/analyze_trace.py \
  --trace-dir llama.cpp/trace_output/<detail-run> \
  --output-dir llama.cpp/trace_output/<detail-run>/analysis
```

运行时 Stage priority 使用独立矩阵，不改 `run_finalist_repeat_matrix.sh`：

```bash
# 先检查两组命令；默认不执行模型。
REPEAT_COUNT=1 \
bash llama.cpp/trace/run_stage_deadline_score_repeat_matrix.sh

# 正式 A/B 建议 delegated cgroup、cold cache 和 N=8。
RUN_STAGE_PRIORITY_EXECUTE=1 \
REPEAT_COUNT=8 \
MEMORY_MAX=6G \
MEMORY_SWAP_MAX=2G \
bash llama.cpp/trace/run_stage_deadline_score_repeat_matrix.sh
```

该矩阵只比较 `deadline_score` 与 `stage_deadline_score`，两组均使用全部 routed experts（`PREFETCH_TOPK=0`）、4 workers、单任务 batch、无 coalescing，并关闭 Slack/feedback/value admission 和跨层预测；不配置 Chunk 切分。两组设置 `LLM_MEM_TRACE_OPT_EXPERT_DEADLINE_OBSERVE=1` 以生成同口径 deadline，且新 mode 默认也会估计 deadline 供 UNKNOWN legacy 仲裁和 late telemetry 使用；Slack 关闭时不会据此拒绝或取消任务。正式结果须核对输出 hash、零 trace loss、每阶段 late count 及 `queue_wait_ns_by_stage` 的 `max_ns`。

## 13. 常见问题

### 仓库存在未提交修改

正式运行必须先提交或清理预期修改。`ALLOW_DIRTY_REPO=1` 仅用于开发 smoke test。

### 冷缓存准备失败

`CACHE_MODE=cold` 需要 Linux Python 的 `os.posix_fadvise`。不要静默改用热缓存，也不要默认使用系统级 `drop_caches`；应修复环境或将不同缓存方法拆成独立实验组。

### systemd-run 不可用

检查：

```bash
systemctl --user status
systemd-run --user --scope -p MemoryMax=1G -- true
```

若比赛环境不提供用户级 systemd，使用管理员预先创建的 delegated cgroup 和 `run_cgroup_memory_matrix.sh`。

### Trace 输出较大

`llama.cpp/trace_output/` 已排除在版本控制之外。正式提交不包含模型、trace、构建目录或 Python 缓存。
