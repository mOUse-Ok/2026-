# 项目技术考古报告：从 Qwen + llama.cpp 到当前系统

考古日期：2026-08-12
考察对象：`main@af1e286438235ab88ab827aaade3a698602c087d`，以及其可达 Git 历史、保留的本地实验归档和当前构建产物。

## 证据口径与边界

本文不把 README 当作事实来源。结论按下列强度给出：

- **事实**：可由当前源代码、Git 对象、脚本、构建配置、保留的原始/汇总数据直接复核。
- **高可信推断**：由多条独立证据共同支持，但缺少一次直接记录。
- **不确定**：缺少关键一手证据；不会把它写成项目已证明的能力。

已做的可复核检查包括：全量可达提交的 first-parent 时间线、当前代码和构建配置、当前流水线脚本、`experiments/expert_prefetch` 的归档数据与 provenance、以及构建目录中的五个自定义 CTest。后者覆盖 router 同步、priority、task lifecycle、Memory Object 和 Calibration Shadow。新增的 Residency Attribution 还需要通过实际 trace 进一步检查数据质量；这些检查都不自动证明真实模型上的时延或内存收益。

相反，以下事项**不能**由仓库证明：最初的 Qwen 命令行、硬件/内核环境、初始 baseline 的完整原始日志，以及 HEAD 上一组干净、受控、可复现的端到端性能结果。根提交只有 `.gitignore` 和导入的 `llama.cpp`，因此“最初已经在 Qwen 上跑通并得到某数字”是合理背景，却不是 Git 可验证的事实。

---

## A. 一句话定义当前项目

**它目前是一个编译期开关保护的 llama.cpp/Linux CPU 实验性运行时：把 MoE router 的实际选择映射为 mmap 专家权重切片的 `madvise(MADV_WILLNEED)` 工作，并用任务—首次使用—OS 内存遥测链条评估该提示；它不是一个已经验证完成的通用动态内存管理器。**

默认关闭 `LLAMA_MEM_TRACE` 时，接口是空实现；打开后才有追踪和可选的专家预取（`llama.cpp/trace/trace_event.h`、`llama.cpp/CMakeLists.txt`、`ggml/src/ggml-cpu/CMakeLists.txt`）。当前 HEAD 还保留默认关闭的 Memory Object、Calibration Shadow 和 Residency Attribution 观察层，但没有由这些开关直接支撑的 HEAD 端到端性能结论。

## B. 起点：从仓库能确认与不能确认的初始状态

**事实。** 2026-05-20 的 `720c62f` 是“add project files and llama.cpp source”：树中除 `.gitignore` 外是完整的 `llama.cpp` 导入，未含项目自有 runner、追踪代码、prompt、模型或实验结果。它更像“在 upstream 推理引擎上开始研究”的快照，而不是可复现的首个 Qwen 实验。

**高可信推断。** 随后团队很快观察到的不是通用 token 生成质量问题，而是本地大模型、文件映射权重和 Linux 内存行为的耦合：2026-06-06 的 `7b32d48` 一次性引入 tensor/KV/expert/OS trace、异步 writer、分析脚本和 trace pipeline。这样的首批投入说明最初遇到的瓶颈很可能是“推理为何慢、哪些张量/页面造成慢”，而不是模型算法本身。

**不确定。** 项目名称中“Qwen + llama.cpp”的“Qwen”在现存可执行流水线中有明确实物（默认 `Qwen3.5-35B-A3B-Q3_K_M.gguf`），但 Git 没有保存第一次使用的模型、GGUF hash 或起始命令。因此不能把当前默认模型倒灌为 5 月时的精确模型。

## C. 演化时间线：每一阶段新增了什么、验证了什么、放弃了什么

| 时间 / 提交 | 新能力或决策 | 直接证据 | 对研究方向的含义 |
|---|---|---|---|
| 05-20, `720c62f` | 导入 llama.cpp | 根树无项目自定义运行逻辑 | 起点是推理引擎，不是独立系统。 |
| 06-06～06-10, `7b32d48`、`bf0a46b` | tensor、KV、MoE expert、OS trace；JSONL 写入和分析 | `trace_writer.cpp`、`tensor_trace.cpp`、`expert_trace.cpp`、`kv_trace.cpp`、`os_trace.cpp` | 先建立“看得见”的测量面；这是一个观察系统。 |
| 06-10, `375ec99` | 增加页面驻留观察 | `os_trace.cpp`、`analyze_trace.py` 改动 | 问题被进一步收敛为权重映射/缺页，而非单纯算力。 |
| 06-14～06-22, `27c3c5c`、`d8a2715`、`fc82acf`、`a2c6c39` | OS hint 原型、专家缓存模拟、trace 工程化、异步预取/队列 | `tensor_trace.cpp` 历史改动，`simulate_expert_cache.py`（后删除），现存任务队列基础 | 从“观测”跨到“利用 router 预测权重需求”。 |
| 07-11～07-13, `7839774`、`a07c53f`、`6018114` | manifest、冷缓存准备、输出 hash、零丢失校验、重复汇总；张量注册表/策略类型分离 | `write_run_manifest.py`、`prepare_model_cache.py`、`validate_trace_summary.py`、`expert_tensor_registry.*` | 研究标准从一次好看的 run 转为可审计、可比的 run。 |
| 07-13～07-16, 合入 `fd71f7a`、`06a3915`、`014397e` | expert task lifecycle、first-use matcher、阶段信息与 deadline-score 调度 | `expert_task_lifecycle.*`、`expert_first_use_matcher.*`、`expert_tensor_stage.*`、`expert_hint_priority.*` | 认识到“发 hint”不等于“对关键路径有帮助”；开始测量任务何时真正被消费。 |
| 07-28～08-06, `b18c3fb`、`d641099`、`3300e12`、`60814cc` | shadow slack、压力观测、最大等待保护、连续老化/保留服务等大批策略 | 提交增删规模；`d641099` 留下 router 同步修复；连续老化随后回滚 | 策略空间迅速膨胀，但多条支线没有通过受控实验。 |
| 08-07, `7650056`、`dc44a60` | 大规模删除实验路径并忽略 `experiments/` | 约 36,160 行删除；备份引用 `backup-before-remove-experiments-20260807202154` 指向 `67dcf29` | 这是明确的收缩/归档，不是“所有策略都成为产品能力”。文档未同步。 |
| 08-08, `01817a0` | memory object、working-set shadow、calibration shadow、可选 `MADV_COLD` 候选 | `expert_memory_object.*`、`expert_calibration_shadow.*` 及单测 | 删除后又开始新的、默认关闭的假设验证；目前仍是实验种子。 |
| 08-10, `a4079e7` | 更新 README、证据图表和技术考古材料 | `README.md`、`docs/final-readme-evidence-*`、`docs/assets/` | 开始把当前 HEAD 证据、历史结果和负结果分开叙述。 |
| 08-12, `af1e286` | object-level residency attribution 与 Experiment 4B | `residency_attribution.*`、`analyze_residency_attribution.py`、`run_experiment_4b.sh` | 进一步回答不同语义对象在 demand 前的驻留/缺页风险；当前仍是观察实验，不是因果性能结论。 |

时间线里最重要的转折是 07-11：项目不再把“有一个快一点的样本”当作足够结论，而开始要求 manifest、输入/输出 hash、缓存准备、trace 完整性和重复汇总。第二个转折是 08-07：大量控制器被撤出当前主线。08-10～08-12 的工作则把叙事和观测面补回当前 HEAD，说明仓库现在应由“精简的可执行主线 + 明确标注的实验种子”共同代表。

## D. 机制谱系：从症状到仍存主线

```text
文件映射权重在内存压力下缺页
        │
        ├─ 06 月：tensor / KV / router / OS trace
        │       └─ 页面驻留、major/minor fault、RSS、swap 的关联观察
        │
        ├─ router 已经给出“将用哪个 expert”
        │       └─ expert tensor registry：expert id → mmap tensor slice
        │               └─ MADV_WILLNEED / 可选 fadvise
        │                       ├─ 同步提示（早期有效样本，但会干扰关键路径）
        │                       └─ 异步 task queue（当前保留）
        │                               └─ deadline + route-score priority（当前保留）
        │
        ├─ task issue 不等于需求满足
        │       └─ first-use matcher / tensor stage（当前主要作为遥测）
        │               ├─ stage priority → 后期反噬，关闭
        │               ├─ slack predictor/shadow → 精度和泛化不足，关闭
        │               └─ max-wait/fairness → 尾延迟和锁竞争变差，关闭
        │
        └─ “内存对象”语义化
                ├─ working-set / calibration shadow / COLD candidate（HEAD 新增，默认关闭）
                └─ residency attribution（HEAD 新增，观察 demand 前的对象级驻留情况）
```

仍在主线的关键路径可以逐段复核：

1. 模型加载时，`expert_tensor_registry.add` 识别 `blk.*_(gate/up/down/gate_up)_exps.weight`，记录 tensor 的基址、字节数、`ne[2]` 专家数和切片 stride（`expert_tensor_registry.cpp`）。
2. CPU 图执行到 MoE 权重操作时，`ggml-cpu.c` 的 `moe_weights` hook 进入 `llm_mem_trace_moe_weights`（`expert_trace.cpp`）。它读取 router 的 expert id/score，按层和 expert 映射为实际切片。
3. `llm_mem_trace_prefetch_expert_layer` 建立任务；TTL、admission、feedback/value gate 都是选项；异步 worker 以 `deadline_score` 的词典序选择任务，然后发 `madvise(MADV_WILLNEED)`。POSIX fadvise 仅在能由 `/proc/self/maps` 反查映射 offset 时可用（`tensor_trace.cpp`、`expert_hint_priority.cpp`）。
4. 首次真实消费发生在 `GGML_OP_MUL_MAT_ID` 的逻辑 first-use 记录；`expert_first_use_matcher` 用 step/layer/expert/tensor、stage、range overlap 与因果顺序匹配已发任务（`expert_first_use_matcher.cpp`）。它测的是“这次计划与这次张量使用的关系”，不是内核保证的页已在 RAM。
5. 四个有界异步 JSONL sink 分别写 tensor、KV、expert、memory 事件，并暴露 enqueued/written/dropped；pipeline 强制检查 trace 是否零丢失（`trace_writer.cpp`、`validate_trace_summary.py`）。

这里有一个不能省略的正确性节点：`d641099` 后，`ggml-cpu.c` 在需要读取 router tensor 前通过 `llm_mem_trace_moe_weights_requires_sync` 建 barrier；对应 `test-router-tensor-observation-sync` 通过。它解决的是“观察到的 router 值会因多线程时序不可靠”的风险，不是调度收益证明。

## E. 研究者的认知如何变化

最初的思路近似“观察到 mmap/缺页，给模型文件一点 OS 提示即可”。随后有三次成熟化：

1. **从全局内存到语义对象。** router 选择把一个抽象的“模型权重”缩小为某层、某 expert、某个 gate/up/down 切片；这也是 registry、切片范围和 task key 出现的原因。
2. **从发出动作到验证因果。** `madvise` 返回成功只表示内核接受建议；它没有证明数据在第一次矩阵乘前完成 page-in。first-use matcher、issue/first-use 时序、deadline 遥测把“建议”变成可被证伪的假设。
3. **从均值优化到干扰管理。** 异步队列、deadline priority、反馈门控、压力 shadow、最大等待和 reserved service 都在问同一个更难的问题：怎样减少缺页，同时不把 syscall、队列和锁竞争移到 decode 关键路径。

因此项目真正的演进不是“不断叠加更聪明的预取算法”，而是逐渐承认预取是一个有副作用、受时序和内核调度支配的控制问题。

## F. 失败、退化和被撤销的路线

| 路线 | 结果 | 证据及为何重要 |
|---|---|---|
| 专家缓存替换（LRU/LFU 等） | 关闭 | 归档 `negative_results/expert_cache*.json` 的离线 trace 模拟中，1 GiB 下 LRU 命中约 15.07%、LFU 约 14.29%，而 router 路径的非传统 proxy 约 80.28%。这是模拟而非物理 run，却足以否定“先做通用 expert cache”作为当前优先级。 |
| stage priority / stage deadline | 关闭 | 32 个正确的受控 run 显示 early wait 降约 45.62%/51.07%，但 late wait 增约 77.31%/76.22%，major fault 也增约 8.74%/3.41%。局部公平性改善换来尾部和 I/O 反噬。 |
| shadow slack / decode template | 关闭 | 观测样本的最佳 first-use MAE 约 26.077 ms，issue-late precision 0.3564、recall 0.1956；模板在 worker 2/4 下 MAE 退化约 2792%/1835.5%。预测不足以安全驱动控制。 |
| pressure shadow → 主动压力控制 | 停在观测 | `M5A_pressure_shadow_report_20260726.md` 记录 off/off 也有 router-score/task 等价性不一致，压力模块没有资格进入正式 A/B。cgroup/PSI 的可测性是成功，控制器的可比较性不是。 |
| max-wait/fairness | 关闭 | `negative_results/max_wait*.json` 的 30 个有效 directed run 未给出稳定优势，锁 acquire p95 增约 21.7%～34.9%，扫描 p95 也增约 27.9%～33.9%。 |
| continuous aging / reserved service | 回滚或关闭 | `3300e12` 后很快有 `60814cc` 回滚；归档显示没有稳定 p95/p99/max 改善，且至少一组 deadline miss 变差；reserved service 的 12 个 A/B 和 18 个压力测试未改变赢家，并带来高压锁尾延迟回归。 |

这些失败不是噪声：它们共同说明，MoE 专家需求虽可预测，但预测窗口短、系统调用与锁的成本真实存在，过度调度会将小的 page-fault 收益转化为更大的 decode 尾延迟。

## G. 当前架构与实际运行路径

当前运行路径不是 README 所述的一组历史控制器。直接执行的入口是 `trace/run_trace_pipeline.sh`：

- 默认 CPU-only（`--gpu-layers 0`）、8 threads、80 tokens，并指向 Qwen3.5 35B-A3B GGUF。
- 该脚本只接受 `LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=off|expert_prefetch`；其它值会退出。选择 `expert_prefetch` 时同时启用 feedback、value gate、async、priority、deadline-score、4 workers 和 OS hint/prefetch。
- 它保存 manifest、模型缓存准备信息、GNU time 指标、输入/输出 hash 和 JSONL，并拒绝有丢失事件的 trace。

| 层 | 当前职责 | 边界 |
|---|---|---|
| 编译/侵入层 | `LLAMA_MEM_TRACE` 把 hook 接到 llama context、model loader、KV 与 ggml CPU op | 关闭时不应改变正常推理路径。 |
| 观测层 | step/phase、加载、KV reuse、router、tensor、OS-memory 事件 | 观测会有开销；writer 采用 bounded queue，故必须检查 drop。 |
| 映射层 | 注册 expert tensor，计算 expert slice/range | 只处理识别到的 MoE 命名/布局；不是所有 GGUF 都自动适配。 |
| 控制层 | task 生命周期、dedup/TTL、queue、deadline-score 与 `MADV_WILLNEED` | `madvise` 是 hint，不是强制预读或页锁定。 |
| 解释层 | first-use 匹配、trace metrics、对比/重复统计 | first-use 为逻辑消费证据，不是 residency 真值。 |
| 新实验层 | memory-object 状态、working-set shadow、calibration shadow、可选 cold reclaim、residency attribution | 需显式环境变量或 `attribution` profile；主要是语义/观测证据，没有新的 HEAD 性能报告。 |

当前代码还保留 KV trace 和 `simulate_kv_cache_policy.py`，但没有证据显示它实现了在线 KV 替换策略；不要把“KV 被观测/模拟”说成“项目解决了 KV 内存管理”。Residency Attribution 也只解释 demand 前的对象级驻留状态，不替代 `/proc`/cgroup 的全进程 fault 统计。

## H. 它真正想解决的问题，以及它刻意没有解决什么

它要解决的是：**在 CPU、文件映射的稀疏 MoE 推理中，router 给出专家选择之后、对应专家矩阵首次被执行之前的很短窗口内，能否以目标明确的 Linux hint 降低代价高的缺页，而不恶化 decode 关键路径。**

它不直接解决：

- 内核页回收或真正的物理内存分配；`MADV_WILLNEED`/fadvise 没有提供 page-in 完成保证。
- 通用 MoE cache replacement；该路线被离线证据否决。
- GPU-offload、分布式推理、模型质量、训练或 router 改造。
- 已证明的主动冷页回收；HEAD 的 `MADV_COLD` 只是一条受环境变量保护的候选路径。
- 通用 KV cache 策略。

这一定义也解释了为什么 router 同步、任务等价性、输出 hash、冷缓存和 cgroup 可信度会成为核心工程问题：如果需求信号或 A/B 对比本身不可信，任何“预取加速”数字都没有解释力。

## I. 最能改变结论的实验

1. **早期 core-effect 对照：方向成立，但证据等级低。** 归档 `core_effect/summary.json` 中单次样本的 baseline/optimized decode 平均为 274,479/195,171 µs（-28.894%），major fault 772,703/50,293（-93.492%）；N=3 描述性汇总为 decode -42.903%、major fault -93.447%。输出 hash 相同。但 provenance 明示归档工作树 dirty、原始 manifest/binary/model hash 不可得；优化配置还是同步 route-all、无 TTL 的早期原型。因此它证明“值得继续研究”，不能证明当前异步控制器的稳定收益。
2. **缓存模拟：改变了技术路线。** 即使它是离线 proxy，其结果使团队没有把资源继续投向 LRU/LFU，而转向 router 驱动的需求提示。
3. **stage 调度矩阵：否定“更细优先级一定更好”。** early 指标提升却牺牲 late/major fault，是从局部 scheduling 指标回到端到端系统指标的关键反例。
4. **pressure shadow 的 off/off 不等价：提升了证据门槛。** 在准备进入主动控制前发现 router score 不能严格复现，迫使项目先处理观察同步；这比仓促发布压力收益更有价值。
5. **max-wait、aging、reserved-service 的负结果：确立简化原则。** 多个“看起来合理”的公平/保护机制没有跨指标稳定获益，随后大规模 cleanup 是对实验结果的工程响应，而非单纯删代码。

可以把实验强度概括为：早期效果有“信号”，阶段/公平性支线有较强的否定证据，HEAD 上 memory-object/calibration 只有单测与代码语义证据，尚无端到端结论。

## J. 五个最意外、最值得保留的发现

1. **最早看似巨大的收益并不自动属于当前架构。** 早期优化是同步 route-all，而现在是异步、门控、优先级队列；两者不能用同一张收益表宣传。
2. **router 是最有价值的预测器，也曾是实验可信度的薄弱点。** `d641099` 的同步 barrier 与测试表明，连“读到正确 router 值”都需要专门工程。
3. **任务 lifecycle 比预取策略本身更可迁移。** issue、匹配、first-use、deadline、drop accounting 让未来方案可被比较；它们是项目最通用的资产。
4. **改善 early 等待会伤害 late 阶段。** stage priority 的反例说明 MoE 的“先服务谁”并不是免费选择。
5. **代码规模与证据强度呈反比的时段真实存在。** 07-28 到 08-06 曾加入数万行 shadow/pressure/fairness 代码，08-07 又删除约 36k 行；最终留下的是较小的可观测核心和一组明确为实验的种子。

## K. 容易被低估的价值

项目最可贵的产物不是一个声称万能的 prefetch controller，而是一套能让 Linux/llama.cpp MoE 内存问题变得可测、可审计、可证伪的基础设施：

- 将 GGML 操作、router 路由、expert 切片、KV、OS counter 和输出一致性放进同一 trace 时钟；
- 对异步观测承认并显式处理 drop；
- 用 manifest、hash、缓存准备和重复汇总限制“偶然跑得快”的叙事；
- 用 first-use matcher 区分“建议已发出”和“真实需求到达”；
- 保留负结果归档，而不是只保留正向图表。

如果未来迁移到别的模型、不同 Linux I/O 策略或真正的 cache controller，这套因果测量面仍能使用。

## L. 容易被高估、或需要降级表述的部分

1. **README / design / reproduce / comparison 的“当前 mainline”措辞曾过期。** 8 月清理后，本轮已将 `feedback_slack`、`feedback_slack_predict`、`stage_deadline_score`、`stage_scheduling_analysis.py` 等内容标为历史阶段材料，并以 `off`/`expert_prefetch` 和当前 attribution 入口为准。
2. **`MADV_WILLNEED` 成功返回不是预取完成。** 它只能说明 hint 被接受；页是否在 first-use 前到位必须由 OS/时序证据间接评估。
3. **一次或 N=3 的百分比不是通用性能承诺。** 早期归档缺少严格版本/环境 provenance；其数字最多是探索信号。
4. **单测通过不是系统收益。** 8 月 8 日新增 memory-object/calibration 的五个相关单测可证明状态机/公式基本语义，不可替代真实 Qwen + 内存压力下的 A/B。
5. **“动态内存管理”这个名字会过宽。** 当前真正在线动作主要是专家切片的 advice；working set、cold reclaim、calibration 和 residency attribution 仍应按默认关闭或观察性实验表述。

## M. 三种叙事视角

### 1. 性能优化叙事

从 major fault 和 decode 慢的观察出发，利用 router 信号对专家权重发出提前 hint。早期有很强的正向信号，随后发现排队、锁、阶段公平和压力会吞噬收益，于是收缩为更小、更可控的 async deadline prefetch。

### 2. 系统架构叙事

这是一次在 upstream llama.cpp 中建立“语义—内存—内核”桥梁的改造：GGML MoE op 提供语义锚点，registry 将其翻译为文件映射范围，trace writer 记录系统现象，task/first-use 构成控制闭环。它的核心成果是跨层可观测性。

### 3. 研究方法叙事

项目从“观察到一个效果”走到“先证明比较有效”：补 manifest、hash、冷缓存、repeat、zero-drop；发现 pressure 下 off/off 不等价后停止主张；把无稳定收益的策略明确关停。08-07 的删除是研究纪律的组成部分。

## N. 最终回答

**最终做成了什么？**  
做成了一个可在 llama.cpp CPU/MoE 路径上编译启用的专家权重切片 trace + 异步 hint 实验平台，以及一个较小的 router 驱动 prefetch 主线；它还留下了默认关闭的 memory-object/calibration 原型。

**放弃了什么？**  
放弃或归档了通用 expert cache、stage priority、slack predictor、压力主动控制、max-wait/fairness、continuous aging 和 reserved service 等控制器；不是因为这些想法不优雅，而是因为它们缺少跨指标、受控、可复现的净收益。

**真正核心创新是什么？**  
不是 `madvise` 本身，而是把 router 选择、专家 tensor slice、异步任务、first-use 和 OS memory/fault 事件建立成可复核的因果链，并把 trace 完整性和 A/B 等价性纳入实验设计。

**最大失败或教训是什么？**  
把局部指标（早期 wait、hint 数、某次 decode 平均）当成系统收益会误导设计；尤其是 asynchronous 观察和 router 读取若不先证明一致性，后续性能结论没有地基。另一个教训是文档必须随 08-07 的架构收缩同步，否则历史功能会被误读为当前能力。

**如果现在接手，第一步该做什么？**  
先不增加控制策略。应建立一个与 HEAD 完全匹配的最小可复现实验包：固定 GGUF/input/binary/kernel/cgroup/CPU affinity，使用当前 `off` 与 `expert_prefetch` 两案，保存完整 manifest、raw JSONL、输出 hash、major/minor fault、decode 分位数和 trace-drop；至少做足以报告方差的重复。只有这个基线通过等价性和统计检查后，才单独打开 memory-object 或 calibration-shadow 的一个开关，验证它是否改善“首次使用前准备”而不伤害尾延迟。

---

## 供维护者立即处理的文档债

此前工作树与项目叙事的主要不一致是：`docs/design.md`、`docs/reproduce.md`、
`docs/development-log.md` 和 `docs/comparison.md` 把已删除的 controller / 脚本写成
当前能力。本轮已将这些内容标为历史，并补充当前的 Memory Object、Calibration
Shadow 和 Residency Attribution 入口；后续仍应在有新实验数据后再更新具体性能结论。

根目录的初赛 PDF/PPT 尚未在本轮修改，若重新导出它们，还需要沿用同一条“历史实验线 /
HEAD 可执行路径”边界，避免二进制交付物再次落后于 Markdown 文档。
