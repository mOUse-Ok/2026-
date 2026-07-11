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
- 7 个分析与聚合单元测试全部通过。
- 四方案 N=2 拉丁方矩阵 dry-run 通过，第二轮顺序相对第一轮轮换一个位置。
- baseline/deadline_score 的 4096 MiB cgroup 矩阵 dry-run 通过。
- 使用现有 GGUF、`TRACE_PROFILE=benchmark`、`CACHE_MODE=as-is`、`NUM_TOKENS_PREDICT=1` 完成短 smoke。
- smoke 中 expert sink 为 `19112/19112/0`，memory sink 为 `1233/1233/0`，格式依次为 `enqueued/written/dropped`。
- 六类可信证据产物和输出哈希均生成，分析指标来源为 `STEP_END` 与 GNU time。

本次 smoke 只出现 3 个 prefill step，没有形成 decode 样本，因此仅证明流水线和证据产物可用，不作为性能结论。正式 N=8 矩阵未在本轮自动执行。
