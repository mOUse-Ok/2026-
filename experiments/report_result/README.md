# 答辩实验归档

本目录是冻结源码的答辩证据入口。当前可运行 HEAD 为 `a5d80057701a759edd40f477e9375e0daffbe757`；其父提交 `88fc9e13808739640ed1d2305c76358cc14d98d4` 是最后一个源码提交，二者的差异仅为本目录和 `.gitignore`，不含算法、策略或推理源码变更。

它保存实验体系、证据分级、原始产物索引、结果解读和 32GB 迁移提示词。算法、策略、控制器和推理代码均不在本轮修改范围内；后续运行只能改变实验环境变量和启动参数。

## 目录约定

```text
report_result/
├── README.md                         # 本文件：归档规则
├── 01_architecture_and_claims.md     # 当前系统与答辩主张边界
├── 02_experiment_matrix.md           # 完整实验体系、资源判定与停止规则
├── 03_legacy_artifact_catalog.md     # 旧产物的复用/排除判定
├── 04_current_head_mmap_analysis.md  # 已恢复 M1 的可复用边界与答辩解读
├── 05_recovery_audit.md              # 删除产物的可恢复性、价值与处置
├── 06_32gb_result_audit.md            # 四套 32GB 实验的逐项准入判定
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

## 归档规则

历史产物只要仍在原目录、Git 备份分支或 dangling object 中，均先通过 [`03_legacy_artifact_catalog.md`](03_legacy_artifact_catalog.md) 和 [`05_recovery_audit.md`](05_recovery_audit.md) 建立来源、版本与可用范围；不得把旧汇总数字伪装成当前 HEAD 的原始数据。

从本目录建立之后产生的合格实验，必须直接写入 `raw/<experiment-id>/<run-id>/`，不可再写入 `llama.cpp/trace_output/` 或临时目录。每个 run 必须包含：`run_manifest.json`、`process_metrics.json`、`summary.json`、cgroup 前后快照、stdout/stderr、输出 SHA、JSONL（如启用 trace）和分析结果。

禁止把不同机器、不同模型哈希、不同 prompt、不同 cgroup、不同 trace profile 或不同 commit 的数字合并为一个均值。
