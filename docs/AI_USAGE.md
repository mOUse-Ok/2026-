## AI使用概况

- 是否使用AI辅助：是
- 使用的AI模型名称：deepseek-v4-pro；Codex
- 使用时间范围：大致覆盖比赛时间，不定期使用
- 使用成员：李子恒

## 使用场景

请按实际情况勾选或填写：

- [√] 需求梳理
- [√] 文献检索
- [√] 代码实现辅助
- [√] Bug 定位
- [√] 文档初稿整理
- [√] 测试命令整理

## AI 产出内容

请列出 AI 工具辅助产出的内容，并说明最终是否经过人工修改：

| 内容 | 文件/位置 | AI 参与方式 | 人工审核与修改情况 |
| --- | --- | --- | --- |
| 需求与方案梳理 | `README.md`、`docs/design.md`、`docs/comparison.md` | 辅助整理赛题需求、LLM 推理内存瓶颈、MoE expert 预取和页面替换策略的表达方式；提供方案讨论中的备选角度 | 队员结合赛题目标和实际工程实现重新筛选、删改，最终设计以本仓库代码和实验结果为准 |
| Trace、OS hint 和 expert prefetch 相关代码 | `llama.cpp/trace/tensor_trace.cpp`、`llama.cpp/trace/expert_trace.cpp`、`llama.cpp/trace/trace_event.h` | 辅助讨论局部实现思路、接口组织、边界条件和调试方向 | 队员手工集成到 `llama.cpp`，检查默认关闭、环境变量开关、fallback 行为和 trace 事件字段，未直接采纳未经验证的整段方案 |
| 分析、模拟和对比脚本 | `llama.cpp/trace/analyze_trace.py`、`llama.cpp/trace/simulate_expert_cache.py`、`llama.cpp/trace/compare_trace_runs.py`、`llama.cpp/trace/summarize_repeat_runs.py` | 辅助生成部分脚本结构、指标命名、图表/报告整理逻辑和异常处理建议 | 队员根据实际 trace JSONL 格式修改并运行语法检查、dry-run 和本地 trace 分析，保证输出字段与实验目录一致 |
| 运行和测试命令整理 | `llama.cpp/trace/run_trace_pipeline.sh`、`llama.cpp/trace/run_finalist_repeat_matrix.sh`、`docs/reproduce.md`、`docs/test-report.md` | 辅助整理构建命令、运行参数、重复实验矩阵和复现步骤 | 队员在本地环境执行构建、脚本语法检查、dry-run 和重复实验；长时间实验结果以本地输出为准 |
| 文档初稿与语言润色 | `docs/development-log.md`、`docs/test-report.md`、`docs/source-attribution.md`、`docs/reproduce.md`、`docs/AI_USAGE.md` | 辅助生成文档初稿、补全说明结构、统一术语和中文表述 | 队员逐项核对项目路径、命令、许可证、实验数据和结论，删除或改写无法由本仓库验证的内容 |

## 人工审核说明

- 代码审核方式：队员人工阅读 AI 建议涉及的代码段，按 `llama.cpp/trace/` 结构手工集成；重点检查默认行为不改变 baseline、环境变量开关是否明确、异常路径是否有 fallback、trace 输出字段是否可被分析脚本消费。
- 文档审核方式：队员对照 `README.md`、`docs/design.md`、`docs/test-report.md`、`docs/source-attribution.md` 和实际脚本逐项核对，确保路径、命令、实验指标、许可证来源和结论表述一致，不保留无法验证的 AI 推断。
- 测试验证方式：使用 CMake 构建 `llama-cli`，对 Python 脚本执行 `python3 -m py_compile`，对 shell 脚本执行 `bash -n`，运行 trace pipeline、finalist matrix dry-run 和本地 N=3 重复实验；性能与内存结论以本地 trace 输出和聚合结果为准。
- 是否存在未采纳的 AI 建议：存在。未采纳缺少本地验证依据、侵入性过强、需要修改内核或评测环境、可能破坏 baseline 可复现性、或与许可证/来源说明不清晰的建议。

## 责任声明

```text
本队确认：提交作品中的代码、文档和实验结论已经由队员审核。AI 工具仅作为辅助工具使用，不替代队伍对作品正确性、原创性、合规性和可复现性的责任。
```

## 风险与限制

- 可能风险：AI 可能给出不准确的比赛理解、不可编译的代码片段、与 `llama.cpp` 当前版本不兼容的接口、未经验证的性能判断、遗漏第三方来源说明，或把文档表述写得超过实验数据能够支持的范围。
- 处理方式：所有 AI 输出均作为参考草稿处理；核心代码由队员手工审查和修改，实验结论必须来自本地构建、运行和 trace 分析；第三方项目、模型和参考方向在 `docs/source-attribution.md` 中单独说明；不提交模型权重、trace 输出、缓存、Token、密码或个人隐私数据。
- 人工复核结果：已对提交文档、主要修改代码、复现命令和实验结论进行人工复核。AI 未替代队员完成最终设计决策、实验执行、结果解释或合规确认，队伍对提交作品承担完整责任。
