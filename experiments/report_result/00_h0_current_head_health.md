# H0：当前 HEAD 本机健康门槛

执行日期：2026-08-16

源码锚点：`88fc9e13808739640ed1d2305c76358cc14d98d4`；执行前工作树干净。仅重建了既有忽略目录 `llama.cpp/build` 中的测试目标，未改变算法或源码。

## 结果

| 检查 | 结果 |
|---|---|
| CTest：mmap phase advice | PASS |
| CTest：Router tensor observation sync | PASS |
| CTest：Router control decoupling | PASS |
| CTest：trace control profile | PASS |
| CTest：hint priority | PASS |
| CTest：task lifecycle | PASS |
| CTest：Memory Object | PASS |
| CTest：calibration shadow | PASS |
| `test_analysis_metrics.py` | 7/7 PASS |
| `test_compare_metrics.py` | 3/3 PASS |
| `test_repeat_validation.py` | 4/4 PASS |

结论：H0 通过。它只证明当前代码中的逻辑、状态机、分析器与已覆盖的集成钩子没有回归；不替代真实模型性能或 OS syscall 效果实验。
