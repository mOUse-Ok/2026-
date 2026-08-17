# 误删实验产物恢复审计（只读）

审计时的源码 HEAD 为 `a5d80057701a759edd40f477e9375e0daffbe757`；其父 `88fc9e1` 是最后源码提交。本审计没有恢复、移动、覆盖或删除任何用户文件；未跟踪的 `project_analysis_report.md` 保持不动。

## 已确认可恢复且有价值

1. `backup-before-remove-experiments-20260807202154` 是删除提交；其父提交 `backup-before-remove-experiments-20260807202154^` 保留 `experiments/expert_prefetch/` 的 31 个文件。可以逐字节从 Git 导出，其中包含约 83 MiB 压缩 async trace、baseline trace、fault/pressure/lifecycle 汇总、图表、负结果和 provenance。
2. Git object `3bfa5cc7edf7b8955084e20044015857fb60dba3`（487,080 bytes）是一个无路径的 4C Router semantic-stability JSON 记录，含两个 binary SHA、多个 7G 组的 N=3 Router trace hash 与输出信息。它尚可导出，但没有 model SHA，二进制/输出也并非全组一致，因此仅可作历史机制记录。

以上两项都不能当作当前 HEAD 的性能 A/B：前者明确记录旧提交、dirty worktree 或非等价对照；后者无完整路径和完整不可变输入。它们的价值是避免丢失负结果、行为追溯和实现历史。

**保全提示：** 在决定是否导出 dangling blob 前，不要运行 `git gc`、`git prune` 或会清理 unreachable object 的维护命令。

## 已恢复与仍缺失的原始证据

| 资产 | 核验结果 | 正确处置 |
|---|---|---|
| `experiments/experiment_0815/` | 已恢复于实际目录；16 个 manifest、JSONL、输出、分析和 CSV 齐全 | M1 可复用；限制是模型 SHA 缺失、无 cgroup 上限。 |
| `experiments/scenario_b/`、`scenario_b_12g/` | 已恢复于实际目录；包含服务请求、cgroup 与三次报告 | 作为 C 级机制/历史负结果线索；T2/S1 仍须标准 manifest 的当前矩阵。 |
| `trace_output/m3b_formal_summary`、`shadow_slack`、`m4a1_shadow_slack_20260716_report`、`contest_runs/expert_cache_simulation` | 只有文档引用，没有版本控制原始目录 | 不作为当前答辩证据；其对应旧策略不在冻结期恢复或重做。 |

本机回收站目录不存在；没有发现同名压缩包或工作区副本。文件系统层面的删除恢复需要停止向原磁盘写入并由用户提供块设备/备份权限；这超出 Git 可恢复范围，且成功率无法保证。

## 结论

- 图中五项可由 32GB 运行补齐：M2、T1、T2、R1、S1；KV policy 不能补，因为缺的是 runtime 实现而非机器资源。
- M1 已恢复，不需要重做；它仍不替代图中 auto mmap 的 M2。
- `m3b_formal_summary`、`shadow_slack`、`m4a1_shadow_slack_20260716_report` 与 `contest_runs/expert_cache_simulation` 对应的策略源码均不在当前 HEAD，故不恢复、不重做，也不作答辩证据。
- 若需要保留历史材料，优先恢复 `expert_prefetch` 快照和固定 dangling 4C blob；若目的是答辩性能结论，优先完成 M2/T1/T2/R1，而不是花时间修复已退役策略的旧汇总。
