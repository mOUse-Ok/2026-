# proj5 决赛机制审计与答辩收敛报告（Phase 2）

> 审计对象：/home/liziheng/llmop 当前工作树的可执行代码、Git 历史、单元测试和 experiments 下原始实验产物。  
> 审计时间：2026-08-17。  
> 源码基线：88fc9e1（扩展 mmap phase advice 与 trace 链路）。当前 HEAD a5d8005 相对该提交只改实验归档/文档，源码无差异，故源码审计以 88fc9e1 为准。  
> 证据纪律：README、设计稿和历史报告只用于找线索；事实结论必须回到当前代码和 JSONL/CSV/manifest。无法回溯的数据明确降级。

## 0. 结论摘要

本项目的可信主线不是“增加若干 madvise 开关”，而是把 LLM 语义需求接到 Linux 的 mmap/Page Cache 控制面：

~~~text
Router / Expert / KV / decode 阶段
             ↓
语义事件、专家 Memory Object、真实 mmap 地址区间
             ↓
工作集、压力、价值、时限等策略
             ↓
MAP_POPULATE / posix_fadvise / madvise / Page Cache / cgroup
~~~

必须分开讲两类结论。

1. 已有端到端正向性能证据的机制是**mmap populate 准入**。16.36 GB 稀疏 MoE 模型在 12 GB cgroup 下，auto 选择 SKIP；相对强制默认 MAP_POPULATE，平均 wall -34.1%、major fault -21.7%、decode p95 -11.1%、TPS +13.95%。每组 N=3，全部 27 次运行完成且输出一致。
2. 已实现并有真实状态证据、但不能宣称性能收益的机制是 Router 驱动的语义专家对象、工作集、异步 hint 队列、COLD/DONTNEED 回收和压力反馈。对象生命周期已经跑通；但工作集和 DONTNEED 的现有对照都是显著负收益。

最终答辩建议收敛为三项：

| 优先级 | 可讲贡献 | 证据等级 | 边界 |
|---|---|---:|---|
| P0 | 基于模型大小、稀疏 MoE 拓扑和 cgroup 内存的 mmap populate 准入 | E3 | 证明避开错误 eager 路径；未证明避免 OOM。 |
| P0 | Router 语义到专家 Memory Object、再到真实 mmap 区间的映射 | E1–E2 | 已证明代码和对象状态；无有效 prefetch ON A/B。 |
| P1 | 语义工作集/回收控制面 | E2，且 E3 为负 | 真实执行过，当前策略不宜默认启用。 |

KV 准入、COLD/Rescue/Calibration、phase advice 和历史调度策略只应作为备份页。

## 1. 审计方法、范围与证据等级

### 1.1 范围

- 已定位源码：llama.cpp/src、llama.cpp/ggml/src/ggml-cpu、llama.cpp/trace、相应 CMake 与单测。
- 已读取原始实验：M1/M2、T3、A1、S1 的 run_manifest.json、memory_trace.jsonl、summary.json、group_stats.csv 和 RESULT.md。
- 当前工作树存在用户/其他流程产生的未提交实验归档；本报告不改动它们。
- 远程 32 GB 机器正在运行的结果不在本地，未将其当作已审计结果。

### 1.2 证据等级

| 等级 | 含义 | 可说什么 |
|---|---|---|
| E0 | 只有想法、注释或文档 | 不能当作结果。 |
| E1 | 源码存在完整可达调用链，且相关单测通过 | 已实现、可调用。 |
| E2 | 一次真实运行的原始 trace/计数证明状态或系统调用发生 | 运行时执行过，不代表更快。 |
| E3 | 干净、匹配配置的 A/B 或重复实验，包含端到端指标 | 在该工作负载下有/无收益。 |
| E4 | 跨模型、内存档位、机器且元数据可复现的重复验证 | 才可声称稳健泛化。 |

本次复核构建并执行的单测：

~~~text
test-mmap-phase-advice
test-router-tensor-observation-sync
test-router-control-decoupling
test-trace-control-profile
test-expert-hint-priority
test-expert-task-lifecycle
test-expert-memory-object
test-expert-calibration-shadow
=> 8 passed / 0 failed

python3 -m unittest discover -s llama.cpp/trace/tests -v
=> 14 passed / 0 failed
~~~

这些测试只说明接口/状态机基本正确，不能替代端到端性能证明。

### 1.3 实验元数据缺口

M2 的 12G auto 原始 manifest 固定了 Git、二进制、提示词哈希和模型字节数，但 model_sha256 为 null；RESULT.md 却写有模型哈希前缀。冲突时以原始 manifest 为准：M2 的相对复现性较好，但模型文件内容无法独立证明。T3 等当前实验 manifest 也有模型 SHA 缺失。

S1 聚合器统计 kv_admission_events=1，但每次 ON 原始 trace 的 KV_SLOT_ADMISSION_SUMMARY 是 checks=2、allowed=2、deferred=0。本报告以原始事件为准。

后续每次运行须记录模型 SHA-256、二进制 SHA、机器/内核、cgroup、全部环境变量、warm/cold cache 状态、提示词/输出哈希和退出码。

## 2. 真实系统现象与问题定义

| 现象 | 原始证据 | 代码/系统解释 | 设计约束 |
|---|---|---|---|
| 跳过 populate 能显著缩短 ready/load，却可能拖慢首次推理 | M1：skip ready/load 约 4 ms，对照约 39.4 s；skip prefill 约 49.1 s，对照约 20.8 s，decode p95 +17.7%、TPS -9.8%。 | mmap 建映射、Page Cache 填充、首次缺页不是同一成本。 | 不能把启动快当作总吞吐更好。 |
| 受限内存下 eager populate 会走坏路径 | M2 12G：default wall 106.82 s、major faults 563,255、p95 327.1 ms、TPS 4.284；auto 选择 SKIP。 | 全模型页填充与解码执行在有限 cgroup 中竞争内存。 | 加载时要看资源条件和模型形态。 |
| readahead 邻页不总是 Router 所选专家页 | trace_output/expert_boundary_matrix2_baseline/expert_boundary_summary.json：有效 pair=2，Router 新选择页=270，邻/跨界页=200，weighted overfetch=0.4255。 | OS 只看文件序列/页访问；稀疏 MoE 的后续需求由 Router 语义决定。 | 需要语义输入，而不只是调 readahead 参数。 |
| 逻辑工作集可闭环，却不保证物理页或性能改善 | T3 ON 创建 23,865 个对象、102,186 次 demand 全完成、unmatched=0，但性能恶化。 | 对象 budget 不等于 RSS 或 Page Cache residency；维护本身有成本。 | 状态正确、系统调用发生、性能收益必须分别验证。 |
| 低压力 KV admission 应透传 | S1 ON 三次都有 pressure=low、checks=2、allowed=2、deferred=0。 | 低压力无需无端限制服务请求。 | 还必须测 moderate/high/critical。 |

项目要解决的精确不对称是：

- mmap/Page Cache 按地址、缺页和访问序列工作；
- Sparse MoE 的真实后续需求由 Router 决定，且仅消费少数专家；
- load、prefill、decode 的访问模式不同，统一 MAP_POPULATE 或顺序预读会把成本放错阶段；
- 一个 expert 分散在多类权重张量和字节区间，裸地址和全模型大小都不足以表达需求；
- cgroup 下的逻辑预算与内核的实际 residency 有偏差，策略需要可观测、准入、失效及反证。

## 3. 四层架构和代码落点

| 层 | 关键代码 | 输入/输出 | 审计判断 |
|---|---|---|---|
| L1 语义观测 | src/llama-context.cpp:1308-1323、1714-1744；ggml/src/ggml-cpu/ggml-cpu.c:3044-3073 | prefill/decode、Router GET_ROWS、MoE 权重、KV SET_ROWS | CPU 图执行边界可提取真实语义；无 GPU 对等接入。 |
| L2 语义存储对象 | trace/expert_tensor_registry.cpp:23-31、100-107；trace/expert_memory_object.h:10-195 | (layer,expert,tensor) 到 addr,nbytes | 把 Router 选择变为真实 mmap 区间和可核验对象。 |
| L3 决策/调度 | trace/tensor_trace.cpp:5390-5929、6193-6300、6707-6880；trace/expert_memory_object.cpp:32-236 | pressure/value/TTL/预算到任务或回收 | 多数控制开关默认关闭。 |
| L4 OS 执行/反馈 | src/llama-mmap.cpp:230-315、784-846、1021-1056；trace/os_trace.cpp:234-306 | mmap/madvise/fadvise/cgroup/fault | OS hint 不是物理页已加载的保证。 |

L1/L2 是项目区别于文件级 readahead 调参的核心；L3/L4 决定它能否落地。

## 4. 候选机制总表

| 候选 | 关键实现 | 默认/开关 | 当前证据 | 缺失证据 | 处置 |
|---|---|---|---|---|---|
| mmap populate 准入 | src/llama-mmap.cpp:230-315 | policy 默认 default；显式 auto 才自适应 | E1 + M2 E3 | 多模型 E4 | 主讲、保留。 |
| phase advice / decode NORMAL | src/llama-mmap.cpp:1021-1056；src/llama-context.cpp:1714-1744 | LLAMA_MMAP_DECODE_NORMAL=1，默认关 | E1，边界 probe 有 E2 线索 | 干净 A/B E3 | 诊断/备份。 |
| Router 到 expert prefetch | trace/expert_trace.cpp:107-229；trace/tensor_trace.cpp:6724-6880 | LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1，默认关 | E1 | 有效 ON task/OS hint/first-use/A-B | 主线组成，不单列成功预取。 |
| 异步 hint 队列/优先级 | trace/tensor_trace.cpp:5390-5752 | 可选 worker/priority | E1 | 实际队列端到端收益 | 附属执行器。 |
| Memory Object/工作集 | trace/expert_memory_object.h/.cpp | 默认关 | E1 + T3 E2 | 低开销稳定参数 | 主讲抽象，必须报负结果。 |
| COLD、DONTNEED、rescue | trace/tensor_trace.cpp:6073-6300、2346-2659 | 可选开关 | A1 DONTNEED E2 | COLD/rescue 正向 E3 | 备选。 |
| cgroup pressure/budget | trace/tensor_trace.cpp:3283-3425 | controller 后采样 | E1，M2 是受限场景 | 状态切换闭环 A/B | 支撑机制。 |
| KV trace/slot admission | trace/kv_trace.cpp:104-189；examples/server/server-context.cpp:1903-1913 | slot admission 默认关 | E1 + S1 E2 low-pass | 高压 defer/服务 A-B | 一页扩展。 |
| calibration shadow | trace/expert_calibration_shadow.cpp:18-320 | 默认只观察 | E1 | 16+ 样本后的收益 | 测量基础设施。 |

## 5. 机制、策略与结果的边界

| 分类 | 内容 | 结论纪律 |
|---|---|---|
| 机制 | Router 同步读取、张量注册/切片、Memory Object 生命周期、mmap 准入、OS trace、真实 madvise/fadvise 调用 | 回答“能感知什么、能做什么”，应有代码和状态证据。 |
| 策略 | budget、LRU 式逐出、TTL、任务优先级、pressure 阈值、COLD grace、DONTNEED 条件、fit ratio | 回答“何时做、做多少”，可替换且可能失败。 |
| 结果 | wall、p95、TPS、fault、RSS、输出一致性 | 不能从代码存在或单次 trace 推导。 |

T3/A1 否定的是当前策略组合及开销，而不是“语义对象能否表达专家工作集”这一机制。

## 6. Router 驱动专家预取：逐段调用链

### 6.1 Router 何时被读取

ggml/src/ggml-cpu/ggml-cpu.c:3044-3073 对 MoE GET_ROWS 设置元数据判定和同步 barrier；线程 0 在 Router GET_ROWS 结束处调用 trace hook。这样读取发生在 Router ids 已写完之后。

trace/expert_trace.cpp:107-229 依次：

1. 判断 Expert sink 或控制器是否需要 Router；
2. 判断 Router weights/ids 是否为 host tensor；
3. 校验 GET_ROWS 语义；
4. 获取 layer、top-k、token 数；
5. 为实际路由 expert 调用 llm_mem_trace_prefetch_expert_layer，并可写 EXPERT_ROUTE。

结论：这不是“Router 计算前预取”。它是 Router 计算后、expert 权重首次使用前的短窗口；安全性已由同步点保证，隐藏 I/O 的能力尚待测量。

### 6.2 语义 expert 如何映射为 OS 地址

trace/expert_tensor_registry.cpp:23-31 只登记四类 expert 权重，保存：

~~~text
{ name, layer, addr, nbytes, n_expert = ne[2], expert_stride = nb[2] }
~~~

expert_slice_range（100-107）计算 addr + expert × expert_stride，长度取 stride 与张量剩余长度的安全最小值。故 Router 的 layer/expert 必须先补上张量身份，才成为一个 mmap 子区间。

Memory Object 的键为 layer:expert:tensor（trace/expert_memory_object.cpp:11-14），不使用裸地址，避免丢失层号/张量角色，也使 demand、first-use、回收能在同一语义对象上核验。

### 6.3 task、队列、worker 与 OS 调用

trace/tensor_trace.cpp:6724-6880 先登记 Memory Object demand，随后过 feature、pressure、价值和边界 gate，构造 ExpertHintTask（5881-5929），由 submit_expert_hint_task（5818-5879）投递。

异步路径在 5390-5752：

- 队列容量和 worker 数可配，worker 上限 16；
- 支持 score、deadline、deadline_score 优先级；
- worker 出队后重采样 pressure/value；
- TTL=0 时用 has_live_demand 取消语义过期任务；
- 取消有终态记录，不能把 created 当 issued。

真正发起 Linux hint 的唯一位置是 issue_expert_hint_task（5337-5387）：调用 madvise(addr,length,MADV_WILLNEED)，可选 posix_fadvise(POSIX_FADV_WILLNEED)，记录返回码和耗时。返回成功只是 OS 接受 hint，不是已证明物理页驻留。

### 6.4 first-use、stale 与 single-flight 的真实边界

observe_expert_logical_first_use（6356-6445）在 MUL_MAT_ID 且 ids 在 host 可读时读取实际 expert。ExpertFirstUseMatcher（trace/expert_first_use_matcher.cpp:61-167）要求：

~~~text
(step, layer, expert, tensor) 相同
且 stage 相同
且 issue 早于 first use
且地址区间重叠
~~~

事件明确带 physical_load_observed:false：它证明 hint 与逻辑首次消费关联，不证明页已物理装入。

single-flight 需准确表述：

- registry 的 was_hinted/mark_hinted（expert_tensor_registry.cpp:45-74）按 step/layer/expert/addr 去重；
- try_acquire_hint_slot 仅在 expert_inflight_hint_aggregation_enabled 且对象已登记时避免同一对象并发；
- 不会合并不同对象的字节范围或 syscall，不是通用批合并；
- zero-TTL 的 hinted 集合没有清理路径，存在长运行增长风险。

### 6.5 现有 prefetch 实验为什么无效

T2 类运行被标记为 prefetch，但 LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=0。expert_prefetch_control_enabled（trace/tensor_trace.cpp:372-374）只看这个环境变量，controller 名称中有 expert_prefetch 不会开启功能。因此 task=0 的 T2 是禁用路径，不能用于支持或反驳预取。

**结论**：Router 是语义 OS action 的必要动作生成器，应融入主线；在拿到 feature 真开、task>0 的运行前，不应单列为“已成功预取创新”。

## 7. Memory Object、工作集、COLD 与救援

### 7.1 对象状态和工作集实现

trace/expert_memory_object.h:10-195 的 ExpertMemoryObject 包含：

- 身份：layer、expert、tensor、地址、字节数；
- 活跃度：pending_users、active_users、最后 demand/use/touch；
- 控制态：hint_inflight、in_working_set、eviction/probation；
- 回收/重入态：COLD、DONTNEED、readmission 及计数。

admit_to_working_set（trace/expert_memory_object.cpp:32-93）负责入集/重入/probation 取消；evict_to_working_set_budget（95-144）超预算时扫描整个 unordered_map，逐出最久未 touch 且未保护的对象；register_demand（146-192）创建/合并 demand 并入集；observe_first_use（195-236）完成 pending 到 active。

工作集 eviction 在上述函数中只是**逻辑状态迁移**，本身不调用 OS。512 MiB 逻辑预算不等于 RSS=512 MiB，也不等于 Page Cache 仅占 512 MiB。

### 7.2 T3：状态闭环为真，性能为负

目录：experiments/experiment_T_current_head/T3_memory_object。

ON 组原始 EXPERT_MEMORY_OBJECT_SUMMARY：

| 指标 | 值 |
|---|---:|
| created | 23,865 |
| semantic demands registered | 102,186 |
| merged demands | 432,342 |
| activations/completions | 102,186 / 102,186 |
| unmatched / invariant violation | 0 / 0 |
| working-set budget/current | 512 MiB / 536,502,272 B |
| admissions/evictions/readmissions | 77,664 / 76,481 / 53,799 |

这给出 E2：真实运行的对象状态机闭合。

但 N=3 汇总：

| 组 | wall | decode p50 | decode p95 | major faults |
|---|---:|---:|---:|---:|
| OFF | 72.58 s | 114.6 ms | 129.3 ms | 1.3 |
| ON | 139.98 s | 853.8 ms | 1,108.1 ms | 8.3 |

ON 的 wall +92.9%、p50 +645%、p95 +757%。这明确否定“当前工作集是加速器”。可疑来源包括高频对象/哈希/trace、全表扫描、频繁入出集和 demand 合并，但未做 profile 前不能断言唯一根因。

### 7.3 COLD、DONTNEED、rescue 是否统一

LayerTracker::on_end（trace/tensor_trace.cpp:6193-6300）在启用 DONTNEED 时进入 end_expert_layer_with_madv_dontneed_reclaim；否则才进入 COLD/rescue/calibrated-control 分支。因此它们共享对象和工作集语义底座，但当前为**互斥实验分支**，不是已经验证的 COLD→DONTNEED→rescue 统一闭环。

end_expert_layer_with_madv_dontneed_reclaim（6073-6147）只在 decode，经过 pressure、grace、每步字节预算后页对齐，并调用 apply_madvise_hint(..., MADV_DONTNEED)。

runtime rescue（2346-2659）观察 major fault 和 issued 数；在早期 decode 步数中触发 COLD suspend、value-gate bypass 或 re-entry probe。latency 被记录但不参与当前触发阈值。结构上已有反馈控制，阈值硬编码且无恢复曲线 E3，不能称作已验证自适应器。

### 7.4 A1：DONTNEED 真发出，但效果为负

目录：experiments/experiment_A1_matched_completion。12/12 run 在 384 MiB cgroup 内完成，输出哈希一致。

每个 reclaim run 的原始事件：

- candidates=2,240、issued=2,240、failed=0；
- 提示约 1,004,068,864 B；
- 115,089 次 demand 均完成；
- 362 次 post-DONTNEED readmission。

这是 E2 的“已提交且未报错”，并验证输出一致性；不是“物理内存优化成功”。组均值：

| 组 | wall |
|---|---:|
| plain | 293.16 s |
| trace observation | 292.34 s |
| working set | 407.80 s |
| reclaim | 415.05 s |

reclaim 比 plain 慢 41.6%，major faults 没有改善（均约 2.93M）。正确结论是回收粒度/时机导致重入或额外开销，当前不可默认启用。

## 8. mmap、Page Cache 与 phase advice

### 8.1 mmap populate 准入

llama_mmap_populate_admit（src/llama-mmap.cpp:230-315）在 cgroup 受限时读取 memory.current/memory.max，否则读取 MemAvailable；结合模型映射字节、headroom、fit ratio 和 expert_used < expert_count 的 Sparse MoE 条件，返回 DEFAULT、POPULATE 或 SKIP。

src/llama-model-loader.cpp:1352-1359 构造真实模型输入并调用准入。映射位于 src/llama-mmap.cpp:784-846：默认对 retained fd 建议 POSIX_FADV_SEQUENTIAL；仅准入允许时使用 MAP_POPULATE，可选 POSIX_MADV_WILLNEED。

这不是一刀切地“不预加载”，而是将 eager 映射限制在模型放得下且策略允许的条件下。

### 8.2 M2 的端到端结果

目录：experiments/experiment_M2_mmap_pressure。三种 cgroup（12/15/20 GB）× default/skip/auto × N=3，**27/27 exit=0**，输出哈希一致。

| 12 GB 策略 | auto 决策 | wall 均值 | major fault 均值 | decode p95 | TPS |
|---|---|---:|---:|---:|---:|
| default | DEFAULT / MAP_POPULATE | 106.82 s | 563,255 | 327.1 ms | 4.284 |
| auto | SKIP / Sparse MoE 模型超限 | 70.45 s | 440,827 | 290.9 ms | 4.882 |
| skip | SKIP | 68.57 s | 442,453 | 275.5 ms | 5.122 |

auto 相对 default：wall -34.1%、major fault -21.7%、p95 -11.1%、TPS +13.95%，是本项目唯一稳妥的 E3 正向主证据。

边界：

- 固定 skip 在 12 GB 略快，是因为 auto 此处也选择 skip；结果证明 auto 避开 default 坏路径，**不**证明它优于已知最佳固定 skip。
- 15 GB 中 default 虽少 major fault，但加载时间更长，不能仅用一个 fault 指标判断。
- 20 GB auto/default 均为 DEFAULT，差异受方差影响，不能说 auto 在宽裕内存普遍更快。
- 全部运行完成，不能说“避免 OOM”。

### 8.3 decode NORMAL 的角色

llama_mmap_decode_normal_once（src/llama-mmap.cpp:1021-1056）由 src/llama-context.cpp:1714-1744 在首次 decode 前、LLAMA_MMAP_DECODE_NORMAL=1 时执行，对 retained fd 发 POSIX_FADV_NORMAL。它只改变文件描述符 readahead advice，不是按 expert 精确控制页面。

现有 phase matrix 来自 dirty 提交 cc6b3a6 且模型 SHA 为空，不能报告其性能。boundary probe 只说明“存在跨专家/邻页过取的可观测现象”；phase NORMAL 是 B 类待验证机制。

## 9. KV：接入了服务端准入，不等于 KV 内存管理

trace/kv_trace.cpp:104-171 在 GGML_OP_SET_ROWS 上识别 cache_k_l/cache_v_l，记录 append 的层、token、context、bytes 和地址；llama-kv-cache.cpp:1018-1093 记录 overwrite/reuse。

实际 slot admission：

- trace/tensor_trace.cpp:3441-3469：由 pressure 计算最大 active slot；low 放行全部，moderate 默认半数，high/critical 至少保留一个；
- examples/server/server-context.cpp:1903-1913：请求绑定 slot 前调用 llm_mem_trace_kv_slot_admission_allows，失败则 defer。

trace/tensor_trace.cpp:394-400 明确该机制只控制“请求到 slot”的入场，**不会**改变 active KV 的大小、类型或内容。项目没有运行时 KV eviction、量化、offload 或 PagedAttention；trace/simulate_kv_cache_policy.py 是离线模拟，不能当作 C++ 实现。

S1（20 GB、parallel=2、N=3）只证明低压力时不阻塞，ON/OFF 最大 wall 约 9.45/9.43 s，未发生 defer。当前不能称 KV 降低尾延迟或实现服务隔离。

## 10. Git 历史和策略收敛

| 历史线索 | 可核验事实 | 得出的工程教训 | 答辩处置 |
|---|---|---|---|
| 60814cc | 删除 continuous aging 约 2,220 行 | 长期老化维护未形成保留价值。 | 不讲主创新。 |
| 7650056 | 删除 89 文件、约 36,160 行，涉及 queue/pressure shadow/cache simulation/stage scheduling 等 | 复杂调度和模拟不能替代真实 Page Cache 端到端验证。 | 仅回应“为何收敛”。 |
| 3300e12 及更早 | 策略实验曾快速膨胀，当前源码已删除大部分 | 应保留最小的可观测、可证伪机制。 | 不用旧图表承诺性能。 |

旧文档中 M3B 等“早任务更快、晚任务饥饿”的图表找不到完整原始产物和对应 manifest，只能作为设计动机，不能算定量证据。

## 11. 最多四项核心 Claim

| Claim | 直接证据 | 强度 | 不能越过的边界 |
|---|---|---:|---|
| C1：压力感知 mmap populate 准入可避开 Sparse MoE 的错误 eager 路径 | mmap 准入代码 + M2 12 GB N=3 | 强，E3 | 仅该模型/压力档；非 OOM 证明、非全局最优。 |
| C2：专家语义可稳定映射为 mmap Memory Object，形成 demand 到 first-use 到工作集生命周期 | registry/对象代码 + T3 计数 | 中强，E2 | 不等于物理页 residency；当前性能为负。 |
| C3：Router 语义可生成 OS hint 并匹配后续逻辑 first-use | CPU hook、task/matcher、单测 | 中，E1 | 尚无有效 ON 的 E2/E3，不谈预取加速。 |
| C4：pressure/value/budget/COLD-rescue 具备反馈骨架 | controller/gate 代码 | 弱至中，E1 | 阈值、恢复曲线和 DONTNEED 均未有正向验证。 |

8 分钟时讲 C1/C2；C3 讲“可验证接口和待验动作”；C4 只作为可扩展控制面。

## 12. 快速实验收敛计划

本机仅约 7 GB 可用内存，无法对 16.36 GB 模型做可信端到端实验。以下须在 32 GB 机器执行；时间是单次墙钟预算，未含排队及 cache 重置。

| 类别 | 实验 | 最小设计/验收条件 | 时间 | 结论用途 |
|---|---|---|---:|---|
| A：已闭环 | M2 mmap auto | 保留 12/15/20 GB × N=3；补模型 SHA、机器/内核、cache 状态 | 0，只补元数据 | 主性能证据。 |
| B：高价值缺口 | Router prefetch 真 ON | 设 LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1；先 1 次 control：created>0、issued>0、OS result/first-use 可见；再 OFF/ON N=3 | 5–15 min；后续 30–60 min | 未过 control 不做性能解释。 |
| B：高价值缺口 | phase NORMAL | 同提交/模型 SHA、固定 cache 条件，default vs NORMAL，至少 N=3，报告 boundary 与 p95 | 30–60 min | 需两个指标同向改善。 |
| C：机制正确性 | KV high-pressure defer | 安全地复现 moderate/high，parallel=2；证明 defer 后 slot 释放能恢复 | 5–15 min | 证明状态机，不谈性能。 |
| C：机制正确性 | COLD/rescue 状态 | 输出 fault、issued、suspend、re-entry/probe 事件 | 15–30 min | 无状态事件即降为 E1。 |
| D：已有反证 | 工作集/DONTNEED | T3/A1 已显著负结果 | 0 | 不 profile 不盲调。 |
| D：无效对照修复 | trace overhead T1 | 旧 T1 两组 trace 都开；只有需声称低开销才 binary-off vs trace-on | 15–30 min | 不作为本次前置。 |

没有提交 SHA、clean/dirty 状态、模型/二进制 SHA、环境变量、cgroup、cache 状态、输入/输出哈希的远程结果，最高只能计 E2。

## 13. 两套 8 分钟答辩叙事

### A. 有新增性能结果

| 时间 | 内容 | 证据 |
|---:|---|---|
| 0:00–0:50 | Sparse MoE 语义稀疏性与 OS 文件级 readahead 的错配 | M1 + boundary。 |
| 0:50–1:35 | eager populate 的阶段/资源错配 | M1/M2。 |
| 1:35–2:35 | Router 到对象到真实地址的核心抽象 | registry、CPU hook。 |
| 2:35–3:35 | 压力/预算控制动作，而非盲目预读 | mmap admission/gate。 |
| 3:35–4:45 | M2 12 GB 四项指标，27/27 输出一致 | M2。 |
| 4:45–5:50 | 若有效 Router ON 实验完成，展示 issued 到 first-use 和 A/B；否则只讲机制 | B 类新结果。 |
| 5:50–6:45 | 工作集/DONTNEED 的负结果与可证伪价值 | T3/A1。 |
| 6:45–8:00 | 贡献、边界与 E4 计划 | C1–C3。 |

### B. 没有新增性能结果

| 时间 | 内容 | 重点 |
|---:|---|---|
| 0:00–1:00 | M1/M2：问题不是“少预读必快”，而是阶段与约束错配 | 反直觉现象。 |
| 1:00–2:10 | 语义到 OS 的可验证桥梁 | Router hook、地址映射、对象。 |
| 2:10–3:20 | 对象生命周期真实闭环 | T3 unmatched=0。 |
| 3:20–4:35 | 唯一已确认性能主结果 M2 | 只谈 mmap admission。 |
| 4:35–5:45 | 主动展示 DONTNEED/工作集负结果 | 逻辑预算不等于物理页缓存。 |
| 5:45–6:50 | 从 Git 历史说明删去未经证实的复杂策略 | 收敛而非堆功能。 |
| 6:50–8:00 | Router ON 的最小可证伪实验 | 不盲目调参。 |

没有新增性能提升时，贡献依然成立：建立了可观测、可执行、可否证的语义—OS 控制面，并明确 mmap 准入的性能边界。

## 14. 材料取舍

### 必讲

- **mmap populate 准入及 M2**：唯一 E3 正结果；报告 12 GB 数值，也报告 15/20 GB 的边界。
- **Router 到 Memory Object 到真实地址**：项目的系统抽象创新；展示切片、first-use 和生命周期。
- **负结果**：T3/A1 说明粗粒度对象工作集/DONTNEED 目前放大了开销，可信度高于只选成功样本。

### 一句话带过

- OS trace、cgroup pressure、队列、value gate：主线的支撑设施。
- M1 skip 的成本迁移：为什么需要准入。
- KV slot admission：低压力透传的服务端扩展。

### 备份页

- COLD/Rescue/Calibration 状态机。
- boundary probe 和 decode NORMAL。
- 历史连续老化、阶段调度、shadow 的删除原因。
- 单测、manifest 字段、raw JSONL 查询。

### 不建议讲

- 在没有有效 ON 的情况下把 Router prefetch 说成性能成功。
- 把 DONTNEED、工作集或 KV 说成已成功的内存优化。
- 旧 stage priority、模拟 KIVI/Paged KV、随机 advice 或无原始数据的历史曲线。
- “逻辑 working-set budget 等于 RSS”“madvise 返回 0 等于物理页装入”“完成即避免 OOM”。

## 15. 十个关键问题的直接回答

1. **Router 真被读到吗？** 是，CPU GET_ROWS 完成后的同步安全点读取 host Router ids；不是 Router 计算前，更非 GPU 通用路径。  
2. **专家怎样定位到 OS 地址？** 通过四类 expert 张量登记的 layer/addr/nbytes/expert_stride，按 addr+expert×stride 取 mmap 子区间。  
3. **Router prefetch 已证实有效吗？** 代码/单测为 E1；T2 实际把开关设为 0，尚无有效 ON 的 E2/E3。  
4. **有队列、取消和 single-flight 吗？** 有异步队列、优先级、pressure/value/TTL 取消；只做单对象去重/可选 in-flight slot，不做跨对象 syscall 合并。  
5. **Memory Object、COLD、rescue、DONTNEED 是统一机制吗？** 共享语义底座；当前 COLD/rescue 与 DONTNEED 为互斥实验分支。  
6. **工作集真的执行了吗？** 是。T3 demand/activation/completion 闭合、不变量为零；但 wall/p95 大幅变差。  
7. **DONTNEED 真发出了吗？** 是。A1 每 run 2,240 次、约 1.004 GB、failed=0；reclaim 仍慢 41.6%。  
8. **mmap auto 证明了什么？** 12 GB Sparse MoE 下正确选 skip 并显著胜过强制 populate；不证明防 OOM 或全局最优。  
9. **KV 完成内存优化了吗？** 没有，只在请求绑定 slot 前准入/延后；没有 active KV 驱逐、压缩、卸载或分页。  
10. **Router prefetch 要单列创新吗？** 不建议。它是语义 OS action 的必要生成器，取得有效 ON 结果前单列会暴露证据不足。  

## 16. 最终工程判断

最有价值的增量，是将模型语义、物理 mmap 区间和 Linux 内存动作置于同一条可记录、可核验的路径，并已在 mmap 准入上得到受限内存下的端到端收益。

Memory Object/回收说明“可控”不等于“更快”：对象抽象与 trace 不变量有研究/工程价值，但当前策略执行频率、数据结构复杂度、回收时机和真实 Page Cache residency 间的鸿沟尚未解决。

最终建议：

1. 优先保留/产品化 mmap admission；默认策略继续保守，auto 仅在平台验证后启用。
2. 保留 Router、Memory Object、OS trace 作为研究与调优基础设施；先完成有效 prefetch ON 的最小 E2，再决定是否优化性能。
3. 暂不默认启用工作集逐出、COLD、DONTNEED、rescue、KV admission；当前证据不足或已为负。
4. 先补齐模型 SHA、cache 状态、feature 实际值和 trace-overhead 对照，再扩展到多模型、多内存档位的 E4。

这不是缩小创新，而是将创新从“策略数量”收敛为可验证的语义—OS 控制接口，并清晰给出可部署部分的性能边界。
