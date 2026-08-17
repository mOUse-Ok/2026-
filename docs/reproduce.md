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
  llama.cpp/trace/analyze_residency_attribution.py \
  llama.cpp/trace/trace_metrics.py \
  llama.cpp/trace/compare_trace_runs.py \
  llama.cpp/trace/simulate_kv_cache_policy.py \
  llama.cpp/trace/summarize_repeat_runs.py \
  llama.cpp/trace/summarize_experiment_4b.py \
  llama.cpp/trace/prepare_model_cache.py \
  llama.cpp/trace/write_run_manifest.py \
  llama.cpp/trace/validate_trace_summary.py

python3 -m unittest discover -s llama.cpp/trace/tests -p 'test_*.py' -v

bash -n llama.cpp/trace/run_trace_pipeline.sh
bash -n llama.cpp/trace/run_finalist_repeat_matrix.sh
bash -n llama.cpp/trace/run_cgroup_memory_matrix.sh
bash -n llama.cpp/trace/run_experiment_4b.sh
git diff --check
```

## 6. Trace profile

- `TRACE_PROFILE=evidence`：完整 tensor/KV/expert/memory trace，适合解释行为。
- `TRACE_PROFILE=benchmark`：关闭高流量 tensor/KV 和驻留采样，适合正式性能对比。
- `TRACE_PROFILE=attribution`：开启 tensor/memory residency attribution，适合对象级驻留分析。
- `TRACE_PROFILE=custom`：允许用 `LLM_MEM_TRACE_TENSOR` 等变量逐项指定。

正式结果应分别保留 evidence run 和 benchmark run，不能用高开销全量 trace 的绝对耗时冒充无插桩性能。

Expert Prefetch 任务 trace 可独立切换：

| `LLM_MEM_TRACE_EXPERT_TASK_MODE` | 输出 | 用途 |
| --- | --- | --- |
| `off` | 不记录任务事件或任务汇总 | Trace 开销基线 |
| `summary` | 仅任务/stage/matcher 聚合，不写逐任务事件，不分配输出用 task/issue ID | benchmark 默认值 |
| `detail` | 完整任务事件、`issue_id`、stage 和逻辑 first-use 时序 | evidence 默认值 |

benchmark profile 仍保留真实 `madvise/posix_fadvise` 调用及错误，但把高流量 skip/reject 明细聚合到 summary。进入正式性能矩阵前，应以 `off` 为基线轮换运行三种模式；建议门槛为 Decode 吞吐下降不超过 1%、Decode p95 增加不超过 2%、trace 零丢失且输出 hash 相同。`detail` 未通过门槛时只能用于 evidence，不能用于性能排名。

`EXPERT_FIRST_USE_SUMMARY` 的 M1 语义检查包括 `eligible_tasks/matched_tasks/unmatched_tasks/ambiguous_matches/duplicate_first_use_ignored/matcher_peak_live_tasks/matcher_expired_tasks`。多 Token ubatch 的同键重复 Task 采用一次 logical first-use 对多 Task 的关联语义。`EXPERT_TASK_SUMMARY` 另记录 `same_stage_issue_groups/cross_stage_issue_groups`。分析结果 `metrics.json` 中的 `expert_stage_pairing` 按 `(run_id, step, layer, expert)` 输出总体、PREFILL、DECODE、逐 Layer 和未匹配原因；分析脚本不得从 tensor 名重新分类 stage。Stage scheduling 的旧分析字段只用于历史报告，不再作为当前复现必需项。

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

## 8. 单次 Expert Prefetch 运行

当前 pipeline 只接受 `off` 和 `expert_prefetch` 两个 controller。`expert_prefetch`
会启用 routed-expert hint、异步 worker、deadline-score priority、feedback/value
gate 和 OS hint；它是可选实验 profile，不代表已经证明端到端加速。

```bash
RUN_NAME=expert_prefetch \
TRACE_PROFILE=benchmark \
CACHE_MODE=cold \
MODEL_FILE="$MODEL_FILE" \
LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=expert_prefetch \
bash llama.cpp/trace/run_trace_pipeline.sh
```

若只想观察当前代码的基础行为，将 controller 改为 `off`。不要再使用旧文档中的
`feedback_slack`、`feedback_slack_predict` 或 Stage priority 入口；这些属于历史探索。

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
RUN_GROUPS=baseline,expert_prefetch \
REPEAT_COUNT=1 \
bash llama.cpp/trace/run_cgroup_memory_matrix.sh
```

真实执行需要可写的 delegated parent：

```bash
RUN_MEMORY_PRESSURE_EXECUTE=1 \
CGROUP_PARENT=/sys/fs/cgroup/<delegated-parent> \
MEMORY_LIMITS_MB=4096,5120,6144 \
RUN_GROUPS=baseline,expert_prefetch \
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

Expert cache 离线模拟脚本 `simulate_expert_cache.py` 属于历史探索（已随 Expert Cache 路线归档删除，结论为负结果：通用 cache replacement 与 Router 语义不匹配），当前仓库不再提供该入口。

KV policy：

```bash
python3 llama.cpp/trace/simulate_kv_cache_policy.py \
  --trace-dir llama.cpp/trace_output/<run>
```

复用已有 trace 重新分析时可执行：

```bash
python3 llama.cpp/trace/analyze_trace.py \
  --trace-dir llama.cpp/trace_output/<run> \
  --output-dir llama.cpp/trace_output/<run>/analysis
```

### 12.1 Experiment 4B：对象级 Residency Attribution

该实验只做观察，不打开 Expert Prefetch、COLD 或 Runtime Rescue。默认 dry-run；
需要真实运行时再设置 `RUN_EXPERIMENT_4B_EXECUTE=1`。脚本会准备 Unlimited 与
`MemoryMax=7G` 两组 N=1 运行，并把结果汇总到 `trace_output/experiment_4b/report/`。

```bash
MODEL_FILE="$MODEL_FILE" \
RUN_EXPERIMENT_4B_EXECUTE=1 \
bash llama.cpp/trace/run_experiment_4b.sh
```

单次 attribution profile 也可以直接运行：

```bash
RUN_NAME=residency_attribution_smoke \
TRACE_PROFILE=attribution \
CACHE_MODE=as-is \
NUM_TOKENS_PREDICT=1 \
MODEL_FILE="$MODEL_FILE" \
bash llama.cpp/trace/run_trace_pipeline.sh
```

报告包含 `residency_attribution.json`、`.csv` 和 `.md`。其中
`nonresident_before_use_bytes` 是对象级驻留观察指标，不是 Major Fault 的因果归因，
也不能理解为 `madvise` 或 `mincore` 已证明页面完成换入。

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
