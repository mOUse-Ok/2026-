# 历史产物目录与复用边界

本表按当前工作区、Git 可达对象和原始 run 是否仍可复核分类。历史文字、截图或派生汇总均不等同于逐 run 原始证据。

| 资产 | 当前状态 | 可恢复性 | 答辩用途 |
|---|---|---|---|
| `experiments/experiment_0815/`（M1，16 runs） | 已恢复：16 个 manifest、JSONL、输出、分析和 CSV | 完整逐 run 数据存在；`88fc9e1`、clean、binary/prompt/输出一致；模型 SHA 为 null | A−：可复用为无 cgroup mmap 负结果。 |
| `experiments/scenario_b/`、`experiments/scenario_b_12g/` | 已恢复：server 请求、cgroup、输出和 3 次报告 | 缺标准 run manifest；12G 组能证明 Router hint 实际 issued 且成功组输出一致 | C：机制与历史负结果线索；T2/S1 仍须当前标准矩阵。 |
| `llama.cpp/trace_output/m3b_formal_summary`、`shadow_slack`、`m4a1_*` | `docs/test-report.md` 有文字引用，原始目录缺失 | 未纳入 Git | 旧实现阶段的叙事不可作答辩指标；不恢复旧策略代码，冻结期内不重做。 |
| `experiments/expert_prefetch/`（31 文件） | 当前工作区缺失 | **可从本地分支 `backup-before-remove-experiments-20260807202154^` 精确恢复** | 历史 B/C 级背景、负结果和取证样本；不可升级为当前 HEAD 性能。 |
| dangling blob `3bfa5cc7…60dba3` | 无路径、尚在对象库 | 可按 object id 导出；执行 `git gc` 前须先固定 | 历史 4C Router semantic stability 记录，模型 SHA 缺失、binary/output 不全等价；仅机制背景。 |
| `llama.cpp/trace_output/final-readme-v2`、`scenario_a_final`、`admission_v1` 等仍存目录 | 仍在本机 | 可读，不复制 | 按 `evidence_catalog.json` 的 B/D 级边界使用，不与当前结果混算。 |

## `expert_prefetch` 备份的价值

该备份包含 83,197,117-byte 的 async memory JSONL gzip、baseline trace、fault/pressure/lifecycle 汇总、图表、负结果和 provenance。其 provenance 明确是旧提交且工作树 dirty；其中 async trace 的控制器为 off、与 baseline 并非合格效率 A/B。因此适合保存“实现曾被观测过”和负结果脉络，不适合报当前 TPS、p95 或 OOM 改善。

恢复动作未在本次审计执行。若确认恢复，应只导出到 `report_result/raw/recovered_legacy_expert_prefetch/`，保留原始 commit、对象 SHA 和 `historical-only` 标签，绝不能覆盖新实验的 ID 目录。
