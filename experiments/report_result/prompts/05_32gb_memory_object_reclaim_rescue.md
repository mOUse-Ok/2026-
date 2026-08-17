# 给 32GB 机器 Agent 的提示词：R1 Memory Object / COLD / DONTNEED / rescue

你只能在 32GB Linux 机器上运行冻结实验，不能修改源码、算法、策略、脚本、Git 历史或模型，也不能下载模型。输出只写入 `<repo>/experiments/report_result/raw/R1_memory_object_reclaim_rescue/`；不得覆盖已有目录。

1. 开始前确认 `HEAD=a5d80057701a759edd40f477e9375e0daffbe757`，首次写入前工作树干净；模型存在，记录完整模型 SHA；可用内存至少 24GiB，cgroup v2 和 `systemd-run --user --scope` 可用。任一条件失败，写 `STATUS_BLOCKED.md` 后停止。
2. 用现有源码构建 trace-on `llama-cli`，记录二进制 SHA。固定 CPU-only、相同模型/prompt/seed、`-n 80 -t 8 -b 512 -ub 512 -c 2048`、`TRACE_PROFILE=benchmark`、`CACHE_MODE=cold`、controller=off，关闭 prefetch/async/feedback/value gate。每个 run 在独立 scope 内运行并记录 `memory.max`、`memory.swap.max=0`、前后 cgroup snapshot、manifest、process metrics、stdout/stderr、JSONL 和 output SHA。
3. 先做不计入正式统计的单次 feasibility sweep，寻找一个固定 `memory.max`：必须使 trace 可启动、可产生 cgroup 压力/回收信号，且不会因环境配置失败而失真。把选择规则和所有试探结果原样写入 `CALIBRATION.md`；随后全部正式组使用同一 cap，不得按组调 cap。
4. 正式组均设 `LLM_MEM_TRACE_OPT_EXPERT_MEMORY_OBJECTS=1`、`LLM_MEM_TRACE_OPT_EXPERT_WORKING_SET_MB=128`、`LLM_MEM_TRACE_OPT_EXPERT_RECLAIM_GRACE_STEPS=3`、`LLM_MEM_TRACE_OPT_EXPERT_RECLAIM_MAX_MB_PER_STEP=64`，并按 Latin 交错运行 N=3/组：
   - `object_only`：COLD=0、DONTNEED=0、RUNTIME_RESCUE=0；
   - `cold`：COLD=1、DONTNEED=0、RUNTIME_RESCUE=0；
   - `dontneed`：COLD=0、DONTNEED=1、RUNTIME_RESCUE=0；
   - `cold_rescue`：COLD=1、DONTNEED=0、RUNTIME_RESCUE=1、`LLM_MEM_TRACE_OPT_EXPERT_RUNTIME_RESCUE_MODE=reclaim_backoff`。
   其他环境变量必须完全相同。不得同时开启 COLD 与 DONTNEED。
5. 对每个 run 汇总完成/退出/OOM、RSS 与 `memory.peak`、`memory.events`、`workingset_refault_*`、major/minor faults、Memory Object 的 evictions/readmissions/protected skips、COLD/DONTNEED issued/failed bytes、rescue summary/state transitions、trace dropped 和 output SHA。成功组 output SHA 不一致、trace drop 非零、manifest 缺失或 dirty 均为 INVALID；OOM 必须保留并计入完成率，不能静默重跑替换。
6. 输出 `RESULT.md`、逐 run CSV、机器可读汇总 JSON 和全部原始文件。只写“在该 workload/cgroup 下观察到的完成率与指标”；不得由 `madvise` 成功返回推断物理页必然已经被回收，也不得将 rescue 未触发解释为有效。
