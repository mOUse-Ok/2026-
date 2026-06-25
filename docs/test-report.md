# 测试报告

## 测试环境

- 操作系统：Linux
- 构建系统：CMake
- 编译目标：`llama-cli`
- Trace 功能：`LLAMA_MEM_TRACE=ON`
- 模型：本地 GGUF 模型，默认脚本路径为 `models/Qwen3.5-35B-A3B-Q3_K_M.gguf`
- 说明：模型文件体积较大，不提交仓库；复现时需自行准备或通过 `MODEL_FILE` 指定。

## 构建测试

命令：

```bash
cmake --build llama.cpp/build --target llama-cli
```

结果：

```text
[100%] Built target llama-cli
```

结论：构建通过。

## 静态检查

命令：

```bash
python3 -m py_compile \
  llama.cpp/trace/analyze_trace.py \
  llama.cpp/trace/compare_trace_runs.py \
  llama.cpp/trace/simulate_expert_cache.py \
  llama.cpp/trace/summarize_repeat_runs.py

bash -n llama.cpp/trace/run_trace_pipeline.sh
bash -n llama.cpp/trace/run_finalist_repeat_matrix.sh
```

结论：Python 脚本和 shell 脚本语法检查通过。

## 功能测试

### 单次 trace pipeline

命令模板：

```bash
MODEL_FILE=/path/to/model.gguf \
RUN_NAME=baseline \
NUM_TOKENS_PREDICT=80 \
bash llama.cpp/trace/run_trace_pipeline.sh
```

期望输出：

- `tensor_trace.jsonl`
- `kv_trace.jsonl`
- `expert_trace.jsonl`
- `memory_trace.jsonl`
- `analysis/metrics.json`
- 分析图表和报告文件

结论：已在本地实验中验证可生成完整 trace 与分析结果。

### 最终矩阵 dry-run

命令：

```bash
REPEAT_COUNT=1 RUN_PREFIX=dryrun_check bash llama.cpp/trace/run_finalist_repeat_matrix.sh
```

结论：脚本会打印 baseline、expert_prefetch、deadline_score、decode_ttl1 和 summary 命令，不会默认执行长时间实验。

## 性能与内存测试结果

N=3 重复实验结果：

| 方案 | Decode 均值 | Decode std | Major faults 均值 | RSS 均值 | Swap 均值 | Hint calls 均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 376,069 us | 92,579 | 757,052 | 6.228 GiB | 217.86 MiB | 0 |
| expert_prefetch | 214,724 us | 17,077 | 49,613 | 6.486 GiB | 157.44 MiB | 99,907 |
| deadline_score | 194,971 us | 12,390 | 71,060 | 6.433 GiB | 123.67 MiB | 98,893 |
| decode_ttl1 | 216,819 us | 15,777 | 69,406 | 6.501 GiB | 174.25 MiB | 82,711 |

分析：

- `expert_prefetch` major faults 最低。
- `deadline_score` decode latency、swap 和综合表现最好。
- `decode_ttl1` hint calls 最低，但 RSS/swap/latency 不占优。
- baseline decode latency 波动较大，因此采用 repeated-run 均值作为主要证据。

## 失败项与限制

- LRU/LFU/window-LFU/least-stale 在 <=1 GiB expert slice cache budget 下离线表现差，未作为真实运行最终候选。
- route top-k 能减少 hint calls，但会破坏 prefetch coverage，导致 major faults 和 decode latency 变差。
- 当前最佳策略没有显著降低 hint calls，主要收益来自降低同步 syscall 对 decode 路径的影响。
- 真实性能受模型、内存压力、输入长度和机器状态影响，正式复现应至少重复 3 次。

## 回归检查

- `LLM_MEM_TRACE_OS_HINTS=0` 时，优化策略默认关闭。
- 新增 async、priority、TTL、coalescing、phase top-k 均为 opt-in。
- 本地 trace 输出、模型文件、构建产物和缓存目录不提交。
