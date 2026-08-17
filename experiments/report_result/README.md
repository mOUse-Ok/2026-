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
├── 07_m2_mmap_pressure_result.md      # 正式证据：M2 mmap admission 压力实验（冻结汇总）
├── 08_prefetch_pressure_window_result.md  # 正式证据：16–19 GiB Router Prefetch 压力 sweep
├── 09_decode_normal_ab_result.md      # 正式证据：Decode 前 FADV_NORMAL 严格 A/B
├── 10_prefetch_12g_reclaim_anchor.md  # 正式证据：12 GiB server 真实 reclaim anchor
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

## 正式证据索引（07–10）

以下前 07–09 三份是冻结前完成的三大主实验 RESULT 的**正式版本化版本**：由原始实验目录中冻结的 `group_stats.csv`、`all_runs_metrics.csv` 与各 run `analysis/metrics.json` 汇总得到，从原目录 RESULT.md 复制入库；10 为依据原始 trace/cgroup/请求日志整理核实的正式证据摘要（原始大体积载荷按仓库约定不入库，仅存在于原实验目录）。

| 报告 | 对应机制 | 核心验证内容 |
|---|---|---|
| [`07_m2_mmap_pressure_result.md`](07_m2_mmap_pressure_result.md) | 贡献 1：资源约束感知的模型映射准入（`DEFAULT / POPULATE / SKIP`） | 27 runs（3 MemoryMax × 3 policy × N=3，Latin square）全部输出 SHA 一致、无 OOM；auto 的 decision/reason（`MODEL_FITS_HEADROOM` vs `SPARSE_MOE_MODEL_EXCEEDS_HEADROOM`）；12 GiB 下 auto 相对强制 populate：wall −34.1%、major faults −21.7%、decode p95 −11.1%、TPS +13.95% |
| [`08_prefetch_pressure_window_result.md`](08_prefetch_pressure_window_result.md) | Router Prefetch 的压力档位行为（16–19 GiB transient sweep） | 各档 hint 抑制/放行：19G Moderate 5,067 / 18G High 102,172 / 17G High 102,177 / 16G Critical 0；四档均无 reclaim；16G 为 value-gate 全抑制档 |
| [`09_decode_normal_ab_result.md`](09_decode_normal_ab_result.md) | 贡献 2：推理阶段感知的文件预读控制（FADV_SEQUENTIAL → 首次 Decode 前 FADV_NORMAL） | 严格 A/B（20 GiB、skip-populate cold-page、N=3 interleaved）：未路由相邻 Expert 新增驻留页 2246.7 → 965.0（−57.0%）；Router 选中 Expert 5265 → 5265 保持一致；端到端基本持平 |
| [`10_prefetch_12g_reclaim_anchor.md`](10_prefetch_12g_reclaim_anchor.md) | **真实 reclaim 下的 Prefetch controller 行为边界**（12 GiB server tight-memory anchor） | N=3：created 20,184 / rejected_pressure 2,181 / rejected_value 0 / issued 18,003；pgscan 7,307,338 / pgsteal 2,703,337 / events.max 81,142（r1）；无 OOM，6 个 run 输出 SHA 一致。与 16–19 GiB transient sweep 不可横向比较，不作为 Prefetch 性能收益证据 |

## 归档规则

历史产物只要仍在原目录、Git 备份分支或 dangling object 中，均先通过 [`03_legacy_artifact_catalog.md`](03_legacy_artifact_catalog.md) 和 [`05_recovery_audit.md`](05_recovery_audit.md) 建立来源、版本与可用范围；不得把旧汇总数字伪装成当前 HEAD 的原始数据。

从本目录建立之后产生的合格实验，必须直接写入 `raw/<experiment-id>/<run-id>/`，不可再写入 `llama.cpp/trace_output/` 或临时目录。每个 run 必须包含：`run_manifest.json`、`process_metrics.json`、`summary.json`、cgroup 前后快照、stdout/stderr、输出 SHA、JSONL（如启用 trace）和分析结果。

禁止把不同机器、不同模型哈希、不同 prompt、不同 cgroup、不同 trace profile 或不同 commit 的数字合并为一个均值。
