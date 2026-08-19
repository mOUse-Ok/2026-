## AI使用概况

- 是否使用AI辅助：是
- 使用的AI模型名称：deepseek-v4-pro；Codex（chatgpt 5.5、chatgpt 5.6 Terra）
- 使用时间范围：大致覆盖比赛时间，不定期使用
- 使用成员：李子恒

## 使用场景

- [√] 实验脚本编写辅助
- [√] 实验数据整理、汇总和报告结构整合
- [√] 文档表述整理
- [√] 代码审计与Bug原因定位
- [√] 代码优化建议与质量评估

## AI 产出内容

请列出 AI 工具辅助产出的内容，并说明最终是否经过人工修改：

| 内容 | 文件/位置 | AI 参与方式 | 人工审核与修改情况 |
| --- | --- | --- | --- |
| 实验脚本编写 | `llama.cpp/trace/run_trace_pipeline.sh`、`run_finalist_repeat_matrix.sh`、`run_experiment_4b.sh` 及相关 Python 脚本 | 协助组织脚本结构、参数传递、输出目录和结果汇总代码 | 队员根据当前代码、实际运行环境和 trace 格式手工修改并检查；脚本能否运行、实验是否执行由队员确认 |
| 实验数据整理与整合 | `llama.cpp/trace/analyze_*.py`、`summarize_*.py`、`docs/test-report.md` 等 | 协助整理 JSON/CSV/Markdown 输出、统一字段和报告结构 | 队员人工核对原始 trace、指标口径、样本条件和异常情况；AI 未独立决定实验指标或结论 |
| 文档表述整理 | `docs/design.md`、`docs/reproduce.md`、`docs/development-log.md`、`docs/comparison.md` | 协助将已有代码和实验记录整理为较易阅读的说明 | 队员根据当前仓库状态重新筛选、删改，历史结果和最终表述由队员确认 |
| 技术审计与数据核验 | `docs/technical-archaeology-report.md`（素材核实）、`experiments/experiment_M2_mmap_pressure/`（指标复核）、`experiments/scenario_b_12g/`（12 GiB anchor 核验） | AI 回到原始 trace / cgroup 快照 / 冻结 CSV 逐项重算比对，定位 RESULT.md 中的陈旧数字与 P95 口径冲突，核实 12 GiB server 场景的 task 计数与回收计数 | 采用哪组数字、如何表述边界由队员决定；AI 仅报告证据与冲突，不自行选择结论 |
| 文档一致性清理与过期内容修正 | `README.md`、`docs/*.md`（comparison、design、reproduce、source-attribution、test-report 等） | 统一测试数字口径（8/8 CTest、14/14 Python）、清理已退出主线机制的引用与旧性能表述 | 队员逐项核对修改点；实验数字与结论、越界表述经队员确认边界后才提交 |
| 正式证据报告整理 | `experiments/report_result/07–10_*.md` | 依据原始实验目录冻结的 CSV/JSON/trace 汇总整理为版本化证据报告（10 号报告由 AI 依据原始文件逐项核实后撰写） | 队员核对报告数字与原始数据一致，确认不可比性边界与"不作性能收益"的口径 |
| 相关工作出处补充 | `docs/comparison.md`、`docs/source-attribution.md` | 协助整理 Denning Working Set、Cao/Felten/Li、Patterson、ProMoE、MoE-Infinity 的标题、作者、年份与 venue 分类 | 队员确认最终引用信息；|


## 人工审核说明

- 脚本审核方式：队员人工检查参数、路径、环境变量、默认关闭行为和输出字段，并通过语法检查、dry-run 或实际运行验证。
- 数据审核方式：实验设计、采样条件、指标选择、数据分析和最终结论均由队员完成；AI 只协助整理和整合已有结果，不替代人工判断。
- 文档审核方式：队员对照当前代码、原始 trace、实验报告和仓库路径逐项核对，删除无法由仓库或实验记录支持的表述。
- 是否存在未采纳的 AI 建议：存在。凡是缺少本地验证依据、改变实验设计或可能夸大性能结论的建议，均不直接采用。

## 责任声明

```text
本队确认：提交作品中的代码、文档和实验结论已经由队员审核。AI 工具仅作为辅助工具使用，不替代队伍对作品正确性、原创性、合规性和可复现性的责任。
```

## 风险与限制

- 可能风险：AI 生成的脚本或整理结果可能与当前 `llama.cpp` 版本、trace 字段或实验目录不一致，也可能把相关性写成因果结论。
- 处理方式：所有 AI 输出均作为参考草稿；脚本由队员人工修改和验证，实验设计、数据分析、结果解释和最终结论由队员完成；不提交模型权重、trace 输出、缓存、Token、密码或个人隐私数据。
- 人工复核结果：队员已对脚本、数据整理、文档路径和最终结论进行复核，并对提交作品承担完整责任。
