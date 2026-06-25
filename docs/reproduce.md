# 复现说明

## 1. 环境准备

推荐环境：

- Linux x86_64
- CMake
- C/C++ 编译器
- Python 3
- 足够内存和磁盘空间

安装依赖示例：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake python3 python3-pip git
```

## 2. 获取源码

```bash
git clone <repo-url> llmop
cd llmop
```

确认目录：

```bash
ls
ls llama.cpp
ls llama.cpp/trace
```

## 3. 准备模型

模型文件不提交仓库。默认脚本查找：

```text
models/Qwen3.5-35B-A3B-Q3_K_M.gguf
```

可改用任意 GGUF 模型：

```bash
export MODEL_FILE=/path/to/model.gguf
```

## 4. 构建

```bash
cmake -S llama.cpp -B llama.cpp/build -DLLAMA_MEM_TRACE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --target llama-cli -j"$(nproc)"
```

验证：

```bash
test -x llama.cpp/build/bin/llama-cli
```

## 5. 单次 baseline 运行

```bash
MODEL_FILE=/path/to/model.gguf \
RUN_NAME=baseline \
NUM_TOKENS_PREDICT=80 \
LLM_MEM_TRACE_OS_HINTS=0 \
bash llama.cpp/trace/run_trace_pipeline.sh
```

输出：

```text
llama.cpp/trace_output/baseline/
```

## 6. 单次 deadline_score 运行

```bash
MODEL_FILE=/path/to/model.gguf \
RUN_NAME=deadline_score \
NUM_TOKENS_PREDICT=80 \
LLM_MEM_TRACE_OS_HINTS=1 \
LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1 \
LLM_MEM_TRACE_OPT_EXPERT_POLICY=route \
LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK=0 \
LLM_MEM_TRACE_OPT_EXPERT_ASYNC=1 \
LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE=131072 \
LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS=4 \
LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY=1 \
LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE=deadline_score \
bash llama.cpp/trace/run_trace_pipeline.sh
```

## 7. 最终重复实验矩阵

默认 dry-run：

```bash
REPEAT_COUNT=1 RUN_PREFIX=dryrun_check bash llama.cpp/trace/run_finalist_repeat_matrix.sh
```

执行 N=3：

```bash
RUN_REPEAT_MATRIX_EXECUTE=1 \
RUN_PREFIX=contest_finalist \
REPEAT_COUNT=3 \
MODEL_FILE=/path/to/model.gguf \
bash llama.cpp/trace/run_finalist_repeat_matrix.sh
```

矩阵包含：

- `baseline`
- `expert_prefetch`
- `deadline_score`
- `decode_ttl1`

## 8. 重复实验结果聚合

如果需要手动聚合已有 run：

```bash
python3 llama.cpp/trace/summarize_repeat_runs.py \
  --base-dir llama.cpp/trace_output \
  --baseline-group baseline \
  --group baseline=contest_finalist_baseline_r1,contest_finalist_baseline_r2,contest_finalist_baseline_r3 \
  --group expert_prefetch=contest_finalist_expert_prefetch_r1,contest_finalist_expert_prefetch_r2,contest_finalist_expert_prefetch_r3 \
  --group deadline_score=contest_finalist_deadline_score_r1,contest_finalist_deadline_score_r2,contest_finalist_deadline_score_r3 \
  --group decode_ttl1=contest_finalist_decode_ttl1_r1,contest_finalist_decode_ttl1_r2,contest_finalist_decode_ttl1_r3 \
  --output-dir llama.cpp/trace_output/contest_runs/repeat_summary
```

## 9. 离线 expert cache 模拟

```bash
python3 llama.cpp/trace/simulate_expert_cache.py \
  --trace-dir llama.cpp/trace_output/os_hint_compare/expert_prefetch \
  --output-dir llama.cpp/trace_output/contest_runs/full_expert_cache_matrix
```

## 10. 静态检查

```bash
python3 -m py_compile \
  llama.cpp/trace/analyze_trace.py \
  llama.cpp/trace/compare_trace_runs.py \
  llama.cpp/trace/simulate_expert_cache.py \
  llama.cpp/trace/summarize_repeat_runs.py

bash -n llama.cpp/trace/run_trace_pipeline.sh
bash -n llama.cpp/trace/run_finalist_repeat_matrix.sh
```

## 11. 常见问题

### 找不到模型

设置 `MODEL_FILE`：

```bash
MODEL_FILE=/absolute/path/to/model.gguf bash llama.cpp/trace/run_trace_pipeline.sh
```

### 找不到 llama-cli

重新构建：

```bash
cmake --build llama.cpp/build --target llama-cli
```

### trace 输出太大

`llama.cpp/trace_output/` 已被 `.gitignore` 排除。正式提交不要包含该目录。
