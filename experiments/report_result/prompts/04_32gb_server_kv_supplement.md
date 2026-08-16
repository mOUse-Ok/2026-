# 给 32GB 机器 Agent 的提示词：S1 KV Slot Admission 补充验证

这是补充实验，不得阻塞 M2/T1/T2/T3/A1。只在 32GB Linux 主机、当前 clean `88fc9e1` 上执行，不修改源码、算法、策略或现有脚本。结果写入 `<repo>/experiments/report_result/raw/S1_kv_slot_admission/`。

目标是验证 server 中 KV Slot Admission 的实际启用与多请求行为，不预设性能收益。

1. 记录模型 SHA、server binary SHA、完整命令、cgroup、请求负载和输出 hash。
2. 使用相同 server 参数与 `--parallel 2`，分别运行 admission off/on，各 N=3；固定 12G 或 20G cgroup，先做能够稳定完成的一档。
3. 使用相同的两并发请求序列，记录请求完成率、TTFT、decode/token、server metrics、cgroup、KV admission trace 和输出一致性。
4. 若 `LLM_MEM_TRACE_OPT_KV_SLOT_ADMISSION=1` 仍是默认 allow，结论仅能写“hook 被执行/未拒绝”；不得把它写成 KV 回收、吞吐优化或内存节省。
