# 当前系统、展示主线与主张边界

源码锚点：`main@88fc9e13808739640ed1d2305c76358cc14d98d4`。

## 答辩架构图

```text
GGUF loader / mmap
  └─ ExpertTensorRegistry：记录 Expert tensor 地址与 stride
       ↓
ggml-cpu Router GET_ROWS（同步后读取真实 Router 输出）
       ↓
Router expert id → (layer, expert, tensor slice 地址范围)
       ├─ 默认：trace / first-use / residency 观测
       └─ opt-in expert_prefetch：ExpertHintTask → worker → WILLNEED advisory
                                                     ↓
                              Memory Object / lifecycle / working-set 语义状态
                                                     ↓
                  JSONL sink + process/cgroup/mincore 观测 + manifest
```

## 必须讲清的运行档位

| 档位 | 条件 | 实际行为 |
|---|---|---|
| 普通 llama.cpp | `LLAMA_MEM_TRACE=OFF` | 不编入项目 trace 实现。 |
| 观测基线 | trace build + `LLM_MEM_TRACE=1` + `EXPERT_CONTROLLER=off` | 记录运行证据；不发 Expert prefetch hint。 |
| 可选预取 | `EXPERT_CONTROLLER=expert_prefetch` | Router slice → 异步任务 → `MADV_WILLNEED` / 可选 fadvise。 |
| 可选回收 | Memory Object/working set/reclaim 环境变量 | 语义候选与 advisory；不是内核物理页管理器。 |
| mmap phase advice | `LLAMA_MMAP_POPULATE_POLICY=auto|skip|populate`、`LLAMA_MMAP_DECODE_NORMAL=1` | 加载期/首 decode 的可选 policy。 |

## 可答辩主张

1. 已实现真实 Router → Expert slice 地址范围 → Linux advisory 的语义映射通路。
2. 已实现并测试任务生命周期、first-use 匹配、Memory Object 语义状态、mmap admission 与控制/记录解耦。
3. 当前 HEAD 已有一组干净的 32GB mmap 消融证据，显示加载、prefill、decode 和 RSS 的明确权衡；详见 [`04_current_head_mmap_analysis.md`](04_current_head_mmap_analysis.md)。

## 不得主张

- `madvise` 成功等于物理页面已载入、释放或必然加速。
- 默认 pipeline 自动预取、回收或管理 KV。
- 旧 `01817a0` 结果是当前 HEAD 结果。
- Scenario A 证明 Working Set 单独导致 OOM 存活；其对照配置不等价。
- 任一单机结果可直接外推至另一机器、模型或 memory budget。
