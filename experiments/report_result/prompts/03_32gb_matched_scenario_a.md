# 给 32GB 机器 Agent 的提示词：A1 等价对照完成性实验

在 32GB Linux 机器上重新验证 384MiB 完成性，但只修复实验对照，不修改算法、策略、源码或现有脚本。输出唯一写入 `<repo>/experiments/report_result/raw/A1_matched_completion/`。

前置：HEAD=`88fc9e1`、工作树 clean、模型与两种二进制均存在、记录 model/binary SHA、可用内存至少 24GiB、`systemd-run --user --scope` 可用。否则写 `STATUS_BLOCKED.md`。

固定所有推理参数为 survival 配置：CPU-only、`-n 16 -t 8 -b 512 -ub 64 -c 1024 --cache-type-k q8_0 --cache-type-v f16 --temp 0 --seed 1234`、同一 prompt、cold cache、MemoryMax=384MiB、MemorySwapMax=0。

运行四组、每组 N=3、轮换顺序：

- `plain_survival`: trace-off binary，以上固定参数；
- `trace_observation`: trace-on binary，controller off、Memory Object/reclaim 全关；
- `working_set`: trace-on，Memory Object/working-set 开，COLD/DONTNEED 关；
- `reclaim`: 与 working_set 相同，仅开启 DONTNEED reclaim。

每组输出必须包含 cgroup events、exit/oom、wall、fault、输出 SHA、manifest、完整环境和 trace summary。若条件本身未完成或输出不一致，不得以“存活”宣称策略收益。

最终只能回答三个问题：四组是否完成、同参数下 trace 是否改变完成性、working-set/reclaim 是否相对等价对照改变完成率。不得把 survival 参数变化归因到 Working Set。
