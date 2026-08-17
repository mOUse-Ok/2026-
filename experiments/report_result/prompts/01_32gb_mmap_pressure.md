# 给 32GB 机器 Agent 的提示词：M2 mmap admission

你在一台仅与云端仓库一致的 32GB Linux 机器上工作。只执行实验，不修改任何 tracked/untracked 源码、算法、策略、脚本、Git 历史或模型；不得下载模型。实验输出唯一写入 `<repo>/experiments/report_result/raw/M2_mmap_pressure/`。

1. `git rev-parse HEAD` 必须为 `a5d80057701a759edd40f477e9375e0daffbe757`，首次写入结果前 `git status --porcelain` 必须为空；否则停止并写 `STATUS_BLOCKED.md`。该提交相对 `88fc9e1` 只有归档文档差异。
2. 确认模型 `models/Qwen3.5-35B-A3B-Q3_K_M.gguf` 存在；记录完整 SHA-256。确认可用内存至少 24GiB、cgroup v2 与 `systemd-run --user --scope` 可用；否则停止。
3. 使用当前仓库构建 trace-on `llama-cli`；记录二进制 SHA。不得修改 CMake 或源码。
4. 使用既有 `trace/run_trace_pipeline.sh`，固定：CPU-only、`-n 80 -t 8 -b 512 -ub 512 -c 2048`、固定 prompt/seed、`TRACE_PROFILE=benchmark`、`CACHE_MODE=cold`、controller=off。每次 run 的 `TRACE_BASE_DIR` 设为本实验 `raw/`，不可写 `trace_output/`。
5. 对 `MemoryMax=20G,15G,12G` 分别运行 `LLAMA_MMAP_POPULATE_POLICY=default,auto,skip`；每格 N=3，按 Latin 交错顺序。`MemorySwapMax=0`。不得使用 root `drop_caches`。
6. 每个 run 必须保留 manifest、模型 SHA 文本、binary SHA 文本、cgroup 前后快照、process metrics、summary、JSONL、output SHA 与 stderr。若 OOM、输出不一致、trace drop 非零或 manifest 显示 dirty，标记 INVALID，不重试替换。
7. 输出 `RESULT.md`、逐 run CSV 和汇总 JSON。分别报告完成率、load-to-ready、prefill、decode p95、major faults、RSS/cgroup peak；不得只报 wall time。明确 `auto` 的实际 decision/reason。

不要把任何 `madvise` 返回码写成“物理页已加载/释放”。
