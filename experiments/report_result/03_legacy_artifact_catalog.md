# 既有实验产物：复用与排除目录

| 产物 | 版本/完整性 | 结论 | 答辩处理 |
|---|---|---|---|
| `experiments/results/experiment_0815/` | `88fc9e1`、`git_dirty=false`、16 个 manifest、binary/prompt/output SHA、模型 SHA 缺失 | mmap baseline/skip-populate N=5；另两开关 N=3。 | **直接复用为 M1**；补记模型 SHA 后可升为 A 级。 |
| `llama.cpp/trace_output/final-readme-v2/` | 干净 `01817a0`、N=3/5，完整但非当前 HEAD | trace 开销、prefetch/COLD 负结果、lifecycle。 | B 级历史背景；不可写“当前”。 |
| `llama.cpp/trace_output/admission_v1/`、`map_populate_perf/`、`phase_advice_matrix/` | `git_dirty=true`、模型 SHA 缺失 | mmap 机制线索与早期性能信号。 | D 级；不用于答辩数字。 |
| `llama.cpp/trace_output/scenario_a_final/` | `5505465`、dirty、组间二进制/KV/context/ubatch 不等价 | plain OOM 与 survival 完成。 | D 级；不能归因 Working Set 或 reclaim。 |
| `experiments/results/scenario_b/` | 汇总有输出一致性缺失/不通过，缺标准 manifest | 充足内存组 hint=0。 | D 级；不得报性能。 |
| `experiments/results/scenario_b_12g/` | 有请求指标与 hint 计数，但缺标准 manifest | Router hint 可发出，性能无稳定结论。 | C 级机制线索。 |
| `experiments/expert_prefetch/negative_results/` | 历史归档 | cache、slack、pressure、max-wait、aging 等负结果。 | B/C 级“已否定方向”，不作当前性能。 |
| `experiments/results/experiment_0/1/2/5a/` | 旧机器/旧实现或链路不足 | 早期探索。 | D 级，不混合数据。 |

## M1 当前 HEAD 复用结论

`experiment_0815` 运行于 32GB 级 AMD 主机，当前 HEAD、工作树干净、`TRACE_PROFILE=benchmark`、CPU-only、固定 prompt，所有 16 个输出 SHA 一致且 trace 无 dropped event。

- `skip-populate` 显著缩短加载并降低 RSS，但把代价转移到 prefill：major faults 大幅增加，decode p95/TPS 退化。
- `skip-sequential` 无明显收益。
- `expert-madv-random` 已实际下发 advice，但 decode 指标变差。

这是一组“当前 HEAD 的机制取舍/负结果”证据，不是默认策略的加速宣传；完整数值和逐 run 数据仍以 [`../../results/experiment_0815/REPORT.md`](../results/experiment_0815/REPORT.md) 及其 `runs/` 为准。
