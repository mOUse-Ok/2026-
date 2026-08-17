# 给 32GB 机器 Agent 的提示词：T1/T2/T3 当前 HEAD

在 32GB Linux 机器上只运行冻结代码的当前 HEAD 验证。不得改源码、策略、脚本、Git 历史或模型；不得下载模型。输出唯一写入 `<repo>/experiments/report_result/raw/T_current_head/`。

前置门槛：HEAD 必须为 `a5d80057701a759edd40f477e9375e0daffbe757`、首次写入结果前工作树干净、模型存在且记录 SHA、可用内存至少 24GiB、cgroup v2 可用。任一失败则写 `STATUS_BLOCKED.md` 并停止。

1. 先运行 8 个自定义 CTest 与 3 个 Python 分析测试；保存命令、版本和完整输出。失败或跳过均不得写成“通过”；缺失依赖须写入 `STATUS_BLOCKED.md` 并说明。
2. T1：在两个**全新、互不复用**的 build 目录构建 trace-off（`-DLLM_MEM_TRACE=OFF`）和 trace-on（`-DLLM_MEM_TRACE=ON`）二进制。保存两份 `CMakeCache.txt`、配置命令、二进制 SHA 和 `LLM_MEM_TRACE:BOOL` 行。用同一模型、prompt、seed、CPU 参数、KV 类型、ctx、ubatch、20G cgroup、冷缓存规则，按 Latin 交错运行 N=5/组。
   - plain 组必须使用 trace-off binary，并显式设 `LLM_MEM_TRACE_EXPERT=0`、`LLM_MEM_TRACE_MEMORY=0`、`LLM_MEM_TRACE_KV=0`、`LLM_MEM_TRACE_TENSOR=0`、`LLM_MEM_TRACE_CONTROL_ONLY=1`、controller=off；trace-on 组才设 Expert/Memory sink=1、controller=off。
   - **逐 run 硬性验收：**plain 组不得存在非空 `expert_trace.jsonl` 或 `memory_trace.jsonl`，不得出现 `TRACE_START`/`TRACE_END`；trace-on 组必须有两类 JSONL、`TRACE_END` 和 zero drop。任一反向出现即整组 `INVALID`，停止汇总，不得把它称为 trace-off 对照。两组均须有 manifest、process metrics、cgroup 前后快照与输出 SHA。
3. T2：只能比较“无 hint baseline”与“实际产生 Router-driven hint 的 prefetch”。不要未经显式环境清理就调用 `run_finalist_repeat_matrix.sh` 的 `expert_prefetch` case：该 case 只设 controller，已存在的 `LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=0` 会被 `run_trace_pipeline.sh` 保留。
   - baseline 必须显式设 controller=off、`LLM_MEM_TRACE_OS_HINTS=0`、`LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=0`、async/feedback/value-gate=0。
   - prefetch 必须显式设 controller=expert_prefetch、`LLM_MEM_TRACE_OS_HINTS=1`、`LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1`、固定 budget、async/priority/feedback/value-gate 的值，并在 launch 前 `unset` 或覆盖同名遗留环境变量。`TOPK=0` 在当前源码会选择全部 Router 专家；若要控制开销，必须显式记录选用的非零 top-k。除此以外的模型、prompt、seed、二进制、trace profile、缓存、cgroup 与运行顺序必须一致。
   - 先做 20G N=5；两组均完成、输出一致且 prefetch 组每一个 run 都通过下列门槛后，才做 12G N=3：manifest 显示 `PREFETCH=1`；`EXPERT_TASK_SUMMARY.created/admitted/issued` 全部大于 0；有可归因的 OS hint/first-use 汇总；trace drop=0。任何一项不满足即标记该条件 `INVALID`，只保留 Router 观测，禁止报告“预取有效/无效”的性能结论。
4. T3：在 trace-on/controller-off 条件下，Memory Object off/on 各运行一次完整 trace；检查 demand/activation/completion/slot acquire/release 与终态 pending/active/invariant。若做开销比较，N=3/组且与 T1 不混合。
5. 汇总时分别给出 wall、prefill、decode mean/p95、fault、RSS、cgroup、hint task、first-use、trace drop 与输出 SHA。若无稳定收益，如实写为负结果；不能删除离群点替换主均值。

禁止将旧 `01817a0`、AMD/Intel 或不同 cgroup 的数据混入当前 HEAD 平均值。
