# KV Slot 准入与空闲 Slot 清理

这是对 `llama-server` 多请求场景的轻量扩展。它复用已有的
`ExpertPressureController` 读取 cgroup v2 压力，但只在**新请求将要占用
空闲 Slot**时决定是否延后启动。它不会修改正在推理的 Context、KV 类型、
ctx-size 或 ubatch。

## 启用方式

需要使用 `-DLLAMA_MEM_TRACE=ON` 构建的 `llama-server`，并在有限 cgroup
内运行：

```bash
export LLM_MEM_TRACE=1
export LLM_MEM_TRACE_DIR=trace_output/kv_admission
export LLM_MEM_TRACE_MEMORY=1
export LLM_MEM_TRACE_CONTROL_ONLY=1
export LLM_MEM_TRACE_OPT_KV_SLOT_ADMISSION=1

./build/bin/llama-server -m /path/to/model.gguf --parallel 4 \
  --kv-unified --cache-ram 128 --cache-idle-slots
```

`--parallel` 必须大于 1，准入策略才有降低并发 KV 使用量的空间。单 Slot
服务仍可使用原生空闲 Slot 选项，但准入策略通常不会改变行为。

## 策略

| cgroup 压力 | 新请求最多可并发的 Slot 数 |
| --- | --- |
| Low | 全部 Slot |
| Moderate | 默认总 Slot 数的一半，向上取整 |
| High | 1 |
| Critical | 1 |

已经在运行的请求不会被中断。被限制的请求留在 llama-server 原有队列中，等
正在运行的请求释放 Slot 后重试。没有发现有限 cgroup 上限时，策略放行请求，
避免把不可验证的主机全局内存信息作为控制依据。

可按需覆盖三个上限：

```bash
export LLM_MEM_TRACE_OPT_KV_SLOT_ADMISSION_MODERATE_MAX_ACTIVE=2
export LLM_MEM_TRACE_OPT_KV_SLOT_ADMISSION_HIGH_MAX_ACTIVE=1
export LLM_MEM_TRACE_OPT_KV_SLOT_ADMISSION_CRITICAL_MAX_ACTIVE=1
```

三项均是“最多活跃 Slot 数”。Moderate 设置为 `0` 时使用默认的一半策略。
High 和 Critical 最小为 `1`，保证单个请求不会只因当前压力而永久滞留。

## 原生空闲 Slot 清理

`--cache-idle-slots` 是 llama.cpp 原有机制，需要同时启用
`--kv-unified --cache-ram N`。新请求开始时，空闲 Slot 的状态会先保存到受
`--cache-ram` 限制的提示词缓存，再清空该 Slot 的可用 KV 位置。这改善的是
KV 的可复用容量；由于提示词缓存本身也占内存，不能把它表述为保证降低
`memory.current`。

场景 B 的 helper 现可选择这些开关，默认保持关闭，不改变已有单 Slot 对比：

```bash
KV_UNIFIED=1 KV_CACHE_RAM_MB=128 KV_CACHE_IDLE_SLOTS=1 \
  bash trace/run_scenario_b_server_scope.sh
```

## 可审计结果

开启 `LLM_MEM_TRACE_MEMORY=1` 后，退出时 `memory_trace.jsonl` 会写入一条
`KV_SLOT_ADMISSION_SUMMARY`，包含检查次数、放行次数、延后次数以及各压力
等级的计数。`TRACE_PROFILE=control` 下只保留该汇总，不写逐请求 Trace。
