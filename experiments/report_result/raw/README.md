# 合格新实验的唯一落盘位置

目录命名：`raw/<experiment-id>/<run-id>/`。

每个 run 必须保留：

- `run_manifest.json`：当前 commit、clean 状态、完整 model/prompt/binary SHA；
- `process_metrics.json`、`summary.json`、cgroup 前后快照；
- `inference_output.txt`、`output.sha256`、stderr/stdout；
- 启用 trace 时的全部 JSONL 与分析目录；
- `STATUS.md`：`VALID`、`INVALID` 或 `BLOCKED` 及原因。

禁止用同名目录覆盖既有 run。只允许追加新的、带唯一 run ID 的结果。
