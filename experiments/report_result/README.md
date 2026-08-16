# 答辩实验归档

本目录是冻结源码 `main@88fc9e13808739640ed1d2305c76358cc14d98d4` 的答辩证据入口。

它保存实验体系、证据分级、原始产物索引、结果解读和 32GB 迁移提示词。算法、策略、控制器和推理代码均不在本轮修改范围内；后续运行只能改变实验环境变量和启动参数。

## 目录约定

```text
report_result/
├── README.md                         # 本文件：归档规则
├── 01_architecture_and_claims.md     # 当前系统与答辩主张边界
├── 02_experiment_matrix.md           # 完整实验体系、资源判定与停止规则
├── 03_legacy_artifact_catalog.md     # 旧产物的复用/排除判定
├── 04_current_head_mmap_analysis.md  # 已复用的 88fc9e1 / 32GB 结果
├── prompts/                          # 32GB 机器 Agent 专属提示词
└── raw/                              # 未来合格 run 的唯一写入位置（按实验 ID）
```

## 证据等级

| 等级 | 准入条件 | 答辩用途 |
|---|---|---|
| A | 当前 HEAD、`git_dirty=false`、参数/二进制/prompt/输出完整、N 满足矩阵要求 | 当前结果 |
| B | 固定旧提交且完整可复核 | 历史背景或负结果 |
| C | 缺模型 SHA、缺 manifest 或只做一次机制观察 | 机制线索，不能报性能结论 |
| D | dirty worktree、跨机器混拼、对照不等价 | 仅存档，不用于答辩结论 |

## 零拷贝归档规则

已有 `llama.cpp/trace_output/` 约 16 GiB，直接复制会产生第二份不可控数据。因此本目录用 [`03_legacy_artifact_catalog.md`](03_legacy_artifact_catalog.md) 记录每份既有原始产物的相对路径、版本、可用范围和校验状态；源目录保持只读。

从本目录建立之后产生的合格实验，必须直接写入 `raw/<experiment-id>/<run-id>/`，不可再写入 `llama.cpp/trace_output/` 或临时目录。每个 run 必须包含：`run_manifest.json`、`process_metrics.json`、`summary.json`、cgroup 前后快照、stdout/stderr、输出 SHA、JSONL（如启用 trace）和分析结果。

禁止把不同机器、不同模型哈希、不同 prompt、不同 cgroup、不同 trace profile 或不同 commit 的数字合并为一个均值。
