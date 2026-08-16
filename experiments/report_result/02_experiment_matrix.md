# 冻结代码下的答辩实验体系

## 资源判定

当前机器观测到总内存约 7.6 GiB、可用约 6.3 GiB，而目标模型为 16,356,375,168 bytes（约 15.23 GiB）。即使不考虑 KV、运行时、文件缓存和系统余量，也无法可靠完成任何真实模型实验。

因此：本机只用于代码健康检查、分析器与归档校验；所有加载该模型、cgroup 压力、server、trace-on/off 或 mmap 实验必须迁移到可用内存至少 24 GiB、物理内存至少 32 GiB 的机器。

## 实验矩阵

| ID | 答辩问题 | 条件与重复 | 资源 | 现有产物 | 最终结论类型 |
|---|---|---|---|---|---|
| H0 | 当前代码是否健康 | 8 CTest + 3 Python tests，HEAD | 本机 | 本会话已通过 | 正确性门槛 |
| M1 | mmap 三开关的真实代价 | baseline/skip-populate N=5；skip-sequential/random N=3 | 32GB，无限内存 | 可直接复用 `experiment_0815` | 当前 HEAD 性能/负结果 |
| M2 | memory pressure 下 admission 是否避免不合适 populate | `default/auto/skip`，MemoryMax=20G、15G、12G；每格 N=3 | 32GB + cgroup v2 | 旧产物 dirty，仅作线索 | 当前 HEAD 完成性与取舍 |
| T1 | trace 最小开销 | trace-off vs trace-on/controller-off，参数完全一致，N=5 | 32GB，MemoryMax=20G | 01817a0 只能历史复用 | 当前 HEAD 开销 |
| T2 | Router prefetch 是否有净收益 | controller off vs `expert_prefetch`，同一 cgroup、Latin 交错、N=5 | 32GB，先做 20G；仅在均可完成后做 12G，N=3 | 01817a0 为历史负结果 | 当前 HEAD 性能或负结果 |
| T3 | 生命周期是否闭合 | Memory Object off/on，各 1 个完整 trace；必要时 N=3 开销对照 | 32GB | 01817a0 可作历史背景 | 当前 HEAD 机制闭环 |
| A1 | 384MiB 场景是否可归因 | 同一 survival 参数：trace-off、trace-on/off、working-set、reclaim；每组 N=3 | 32GB + cgroup v2 | 旧 Scenario A 不可复用 | 完成性；禁止因果跳跃 |
| S1 | KV Slot Admission 是否有服务价值 | off/on，`--parallel 2`，固定请求序列，N=3 | 32GB + server/cgroup | 无合格旧数据 | 补充机制证据；非主答辩必需 |

## 执行顺序与停止规则

1. 先执行 H0；失败即停止全部性能实验。
2. M1 已满足当前 HEAD、干净工作树和 N 要求，可直接复用，不重跑。
3. 再运行 M2、T1、T2、T3、A1；S1 只在核心矩阵完成后执行。
4. 每次启动前必须确认：HEAD 精确匹配、工作树干净、模型文件存在、模型 SHA、prompt SHA、二进制 SHA、可用内存 >=24 GiB、cgroup v2 可用。
5. 任一条件发生 OOM、输出 SHA 不一致、trace drop 非零、manifest 缺失、工作树变脏：该格标记 `INVALID`，不得补算或纳入均值。
6. 先报告每个指标的均值、标准差、所有样本和运行顺序；不因“首轮异常”删除样本。可单列敏感性分析，但不能替换主结果。

## 指标与比较规则

- 主指标：完成率、wall time、prefill、decode mean/p95、major/minor faults、RSS peak、cgroup peak/events、输出 SHA。
- 机制指标：hint created/admitted/issued/failed、first-use matched、Memory Object pending/active 终态、trace enqueued/written/dropped。
- 只把同一 host、同一 commit、同一 binary SHA、同一模型 SHA、同一 prompt SHA、同一参数和同一 cgroup 的样本做均值比较。
- `M1` 的 skip-populate 结论必须同时报告 load、prefill、decode、fault 与 RSS；不能只报总 wall 或 mmap 时间。
