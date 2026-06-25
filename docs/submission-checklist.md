# 初赛提交检查清单

## 工程完整性

- [x] 根目录存在清晰入口：`README.md`
- [x] 完整源码位于 `llama.cpp/`
- [x] 本项目新增 trace、分析和复现实验代码位于 `llama.cpp/trace/`
- [x] 不依赖截图、压缩包或局部补丁复原工程
- [x] 存在开源协议文件：`LICENSE`

## 构建与运行

- [x] 构建系统：CMake
- [x] 构建命令已写入 `README.md` 和 `docs/reproduce.md`
- [x] 运行命令已写入 `README.md` 和 `docs/reproduce.md`
- [x] 测试命令已写入 `README.md` 和 `docs/test-report.md`
- [x] 最终重复实验入口：`llama.cpp/trace/run_finalist_repeat_matrix.sh`

## 文档

- [x] `README.md`
- [x] `docs/design.md`
- [x] `docs/development-log.md`
- [x] `docs/test-report.md`
- [x] `docs/comparison.md`
- [x] `docs/source-attribution.md`
- [x] `docs/AI_USAGE.md`
- [x] `docs/reproduce.md`

## 需要队伍人工填写或确认

- [ ] 赛题编号
- [ ] 队伍名称
- [ ] 队员信息
- [ ] 模型来源、许可证、下载方式和使用许可
- [ ] AI 使用说明中的具体工具、使用场景、人工审核过程
- [ ] 最终提交前的真实复现实验运行记录

## 不应提交内容

- [x] 模型文件和大权重文件已通过 `.gitignore` 排除
- [x] `llama.cpp/trace_output/` 已通过 `.gitignore` 排除
- [x] `llm_mem_trace_report.md` 已通过 `.gitignore` 排除
- [x] `.claude/`、`.codex/`、`.agents/` 已通过 `.gitignore` 排除
- [x] 构建产物和 Python 缓存已通过 `.gitignore` 排除

## 已执行检查

- [x] Python trace 脚本语法检查
- [x] shell 脚本语法检查
- [x] 文档和忽略规则 diff 空白检查
- [x] 本次提交范围精确 secret 模式扫描
- [x] 大文件扫描，排除 `.git`、`build`、`trace_output`、`models`

## 注意事项

- 上游 `llama.cpp` 自带示例脚本和 WebUI bundle 中可能包含 `api key`、`password` 等占位字符串或变量名，这类内容属于第三方项目原始示例，不应误判为本队泄露的真实密钥。
- 正式提交前如需清理构建产物，只能清理确认可重新生成的目录，例如 `llama.cpp/build/` 和 `llama.cpp/trace_output/`，不得删除源码、测试数据和文档。
