# M1：当前 HEAD mmap 消融的答辩解读

来源：`experiments/results/experiment_0815/`。该目录有 16 个 run，每个均记录 `88fc9e1`、`git_dirty=false`、同一 binary SHA、同一 prompt SHA、输出 SHA 和 trace summary。唯一缺口是 model SHA 为 `null`。

## 可展示结果

| 条件 | N | 可安全结论 |
|---|---:|---|
| baseline vs skip-populate | 5 vs 5 | skip-populate 将全模型 `MAP_POPULATE` 的加载成本转移到 prefill；RSS 更低，但 major fault、prefill、decode p95 与 TPS 更差。 |
| skip-sequential | 3 | 未观察到稳定改善。 |
| expert `MADV_RANDOM` | 3 | syscall 全部成功发出，但 decode p95/TPS 变差；不建议启用。 |

## 推荐展示方式

使用一个四维图或表同时呈现 load-to-ready、prefill、decode p95、RSS/major-fault。只呈现总 wall 或 mmap 时间会掩盖“延迟转移”的本质。

## 结论措辞

> 当前 HEAD 的干净 32GB 实验表明：跳过 MAP_POPULATE 能压低加载时间和 RSS，但会增加按需分页并拖慢 prefill/decode；`MADV_RANDOM` 虽被成功下发，仍不带来收益。因此默认配置维持关闭，只有在受限 cgroup 下才值得用 M2 重新评估 admission。

在 M2 完成前，不得宣称 `auto` 或 skip-populate 在内存受限场景必然提高吞吐或延迟。
