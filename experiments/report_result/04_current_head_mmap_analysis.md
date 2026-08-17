# M1：已恢复的 mmap 消融

来源为 `experiments/experiment_0815/`。该目录保存 16 个逐 run manifest、原始 JSONL、输出、分析结果和聚合 CSV：源码提交均为 `88fc9e1`、`git_dirty=false`，二进制与 prompt SHA 一致，16/16 输出 SHA 已逐文件复核一致。当前 HEAD `a5d8005` 相对该提交仅有归档文档差异，故可作为源码等价的当前 mmap 无上限消融。

唯一证据缺口是所有 manifest 的 `model.sha256=null`，并且没有有限 `memory.max`。因此等级为 A−：可以展示该主机/模型路径下的 mmap 取舍与负结果，不能把它外推为受压 cgroup 的 admission 结论。

| 条件 | N | 可安全结论 |
|---|---:|---|
| baseline vs skip-populate | 5 vs 5 | skip-populate 将全模型 `MAP_POPULATE` 的加载成本转移到 prefill；RSS 更低，但 major fault、prefill、decode p95 与 TPS 更差。 |
| skip-sequential | 3 | 未观察到稳定净改善。 |
| expert `MADV_RANDOM` | 3 | advice 实际 issued 且无失败，但 decode p95/TPS 变差；不建议启用。 |

推荐展示时并列 load-to-ready、prefill、decode p95、RSS 和 major fault；不得只展示总 wall 或 mmap 时间。M2 仍需在有限 `memory.max` 下单独验证 auto/default/skip admission。
