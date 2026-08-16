# 给 32GB 机器 Agent 的提示词：T1/T2/T3 当前 HEAD

在 32GB Linux 机器上只运行冻结代码的当前 HEAD 验证。不得改源码、策略、脚本、Git 历史或模型；不得下载模型。输出唯一写入 `<repo>/experiments/report_result/raw/T_current_head/`。

前置门槛：HEAD 必须为 `88fc9e1`、工作树干净、模型存在且记录 SHA、可用内存至少 24GiB、cgroup v2 可用。任一失败则写 `STATUS_BLOCKED.md` 并停止。

1. 先运行 8 个自定义 CTest 与 3 个 Python 分析测试；保存命令、版本和完整输出。失败则停止性能实验。
2. T1：从同一 HEAD 构建 trace-off 与 trace-on 二进制。用完全相同模型、prompt、seed、CPU 参数、KV 类型、ctx、ubatch、20G cgroup、冷缓存规则运行 plain-vs-trace/controller-off，N=5/组、Latin 交错。plain 组不要求 JSONL，但必须有相同的 manifest/process/output SHA/cgroup 记录。
3. T2：trace-on 下仅比较 `EXPERT_CONTROLLER=off` 与 `expert_prefetch`，其他参数完全一致。先 20G N=5；两组均完成且输出一致后，再做 12G N=3。优先使用现有 `run_finalist_repeat_matrix.sh` 的 `CASES_CSV=baseline,expert_prefetch`，输出根目录改为本任务 `raw/`。
4. T3：在 trace-on/controller-off 条件下，Memory Object off/on 各运行一次完整 trace；检查 demand/activation/completion/slot acquire/release 与终态 pending/active/invariant。若做开销比较，N=3/组且与 T1 不混合。
5. 汇总时分别给出 wall、prefill、decode mean/p95、fault、RSS、cgroup、hint task、first-use、trace drop 与输出 SHA。若无稳定收益，如实写为负结果；不能删除离群点替换主均值。

禁止将旧 `01817a0`、AMD/Intel 或不同 cgroup 的数据混入当前 HEAD 平均值。
