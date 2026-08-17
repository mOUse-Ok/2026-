> **Provenance**：本文件是由原始实验目录 `experiments/experiment_M2_mmap_pressure/RESULT.md` 冻结版 RESULT.md 复制入库的正式版本化证据；底层数据为该目录冻结的 `group_stats.csv` / `all_runs_metrics.csv` / 各 run `analysis/metrics.json`（原始载荷按仓库约定不入库）。

# M2: mmap admission pressure experiment

## 1. 实验身份

- **仓库**: `https://gitlab.eduxiji.net/T2026100559910276/project3136859-386203`
- **Commit**: `88fc9e13808739640ed1d2305c76358cc14d98d4`
- **工作树**: clean（`git status --porcelain` 为空）
- **内核**: `Linux 6.6.87.2-microsoft-standard-WSL2 x86_64`
- **内存**: `MemAvailable: 26724704 kB`（~25.5 GiB），cgroup v2，systemd-run --user --scope 可用
- **模型**: `Qwen3.5-35B-A3B-Q3_K_M.gguf`，16356375168 字节（~15.23 GiB），SHA-256 `5607c8fcc8b04ada...`
- **二进制**: trace-on `llama-cli`，SHA-256 `06fdc66b71b954b8...`
- **推理参数**: CPU-only，`-n 80 -t 8 -b 512 -ub 512 -c 2048`，KV f16/f16，`--temp 0 --seed 1234`，`TRACE_PROFILE=benchmark`，`CACHE_MODE=cold`，controller=off
- **cgroup**: `systemd-run --user --scope -p MemoryMax=<limit> -p MemorySwapMax=0`
- **冷缓存**: `prepare_model_cache.py --mode cold`（POSIX_FADV_DONTNEED），未使用 root `drop_caches`

## 2. 实验矩阵

3 MemoryMax × 3 LLAMA_MMAP_POPULATE_POLICY × N=3 = 27 runs，Latin square 交错顺序。

| MemoryMax | Policy | N | Latin 交错 |
|:-:|:-:|:-:|------|
| 20G | default | 3 | 每rep轮换，无连续2个相同条件 |
| 20G | auto | 3 | |
| 20G | skip | 3 | |
| 15G | default | 3 | |
| 15G | auto | 3 | |
| 15G | skip | 3 | |
| 12G | default | 3 | |
| 12G | auto | 3 | |
| 12G | skip | 3 | |

## 3. 校验结果

- **完成率**: 27/27 = 100%（全部 exit_code=0）
- **TRACE_END**: 全部命中
- **metrics.json**: 全部生成
- **输出 SHA-256**: 全部一致 `b629427bdbcaf5c6...`（唯一值）
- **cgroup OOM**: 全部为 0（无 OOM 事件）
- **manifest dirty**: 全部 false

## 4. auto 策略的实际 decision/reason

| MemoryMax | Policy | mmap admission decision | reason | 含义 |
|:-:|:-:|:-:|:-:|------|
| 20G | auto | **DEFAULT** | **MODEL_FITS_HEADROOM** | 模型(15.2G)在 20G 内有足够余量，执行 MAP_POPULATE |
| 20G | default | DEFAULT | DEFAULT_POLICY | 默认策略强制 populate |
| 20G | skip | SKIP_POPULATE | FORCED_SKIP | 强制跳过 populate |
| 15G | auto | **SKIP_POPULATE** | **SPARSE_MOE_MODEL_EXCEEDS_HEADROOM** | Sparse-MoE 模型超过可用余量，自动跳过 populate |
| 15G | default | DEFAULT | DEFAULT_POLICY | 默认策略强制 populate（尽管模型>15G） |
| 15G | skip | SKIP_POPULATE | FORCED_SKIP | 强制跳过 populate |
| 12G | auto | **SKIP_POPULATE** | **SPARSE_MOE_MODEL_EXCEEDS_HEADROOM** | 同 15G，自动跳过 |
| 12G | default | DEFAULT | DEFAULT_POLICY | 默认策略强制 populate（尽管模型>12G） |
| 12G | skip | SKIP_POPULATE | FORCED_SKIP | 强制跳过 populate |

**关键发现**: `auto` 策略正确检测了 Sparse-MoE 模型与可用内存的关系——在 20G（模型可容纳）时执行 populate，在 15G/12G（模型超出）时自动跳过。`default` 策略不检查内存，始终执行 populate。

## 5. 分组统计（均值）

| 条件 | wall(s) | RSS peak(GB) | major faults | load→ready(ms) | prefill(ms) | decode p95(µs) | decode TPS | cgroup peak(GB) |
|------|---:|---:|---:|---:|---:|---:|---:|---:|
| 20G_default | 79.58 | 15.58 | **0** | 34876 | 25002 | 157354 | 7.63 | 15.65 |
| 20G_auto | 66.27 | 15.58 | 4 | 31868 | 22592 | 209266 | 6.80 | 15.65 |
| 20G_skip | 64.93 | 12.60 | 380809 | 3 | 49254 | 217181 | 6.12 | 14.37 |
| 15G_default | 79.08 | 14.95 | 104202 | 38693 | 28001 | 161198 | 7.70 | 15.00 |
| 15G_auto | 62.77 | 12.60 | 389822 | 4 | 48550 | 187096 | 6.56 | 14.37 |
| 15G_skip | 67.07 | 12.60 | 378479 | 4 | 50558 | 180267 | 6.75 | 14.37 |
| 12G_default | 106.82 | 11.96 | **563255** | 35674 | 53380 | 327087 | 4.28 | 12.00 |
| 12G_auto | 70.45 | 11.66 | 440827 | 4 | 50889 | 290915 | 4.88 | 12.00 |
| 12G_skip | 68.57 | 11.66 | 442453 | 5 | 49561 | 275509 | 5.12 | 12.00 |

注：本表数值以冻结数据 `group_stats.csv`（各组 N=3 均值）与 `all_runs_metrics.csv` 为准；`decode p95` 取各 run `analysis/metrics.json` 的 `decode_p95_latency_us` 算术平均。12 GiB 下 `auto` 相对 `default`：wall −34.1%、major faults −21.7%、decode p95 −11.1%、TPS +13.95%。

## 6. 关键分析

### 6.1 auto 策略的价值
- **20G**: auto 与 default 行为一致（都 populate），但 wall time 更短（66s vs 80s，可能是方差）
- **15G/12G**: auto 自动跳过 populate，与 skip 行为一致；wall time、major faults、decode TPS 均接近 skip
- **结论**: auto 策略在模型超过可用内存时自动降级为 skip，避免了 default 在 12G 下的严重退化

### 6.2 default 在压力下的退化
- **20G_default**: 完美（0 major faults，最佳 TPS）
- **15G_default**: 104K major faults（populate 部分成功，但超过 15G 的部分被 reclaim）
- **12G_default**: 563K major faults（最差！populate 尝试 fault 全部模型页 ~15.2 GiB，但 12G 限制导致大量 reclaim/refault 循环）
- **wall time**: 12G_default 106.8s，是所有条件中最慢的（比 12G_skip 慢 56%）

### 6.3 skip 的代价
- **所有 MemoryMax**: skip 产生 ~380K major faults（包括 20G，完全没必要）
- **20G_skip**: 不必要的 380K faults（模型完全可容纳，populate 不会出问题）
- **load→ready**: skip 几乎为 0（4ms），但 prefill 承担了全部代价

### 6.4 cgroup peak
- 20G: default/auto peak ~15.65 GB（模型完整驻留），skip ~14.37 GB（page cache 部分驻留）
- 15G: default peak ~15.00 GB（触及 15G 限制），auto/skip peak ~14.37 GB（只有访问的页驻留）
- 12G: 所有策略 peak ~12.00 GB（均触及 12G 限制）

## 7. 完成率汇总

| MemoryMax | default | auto | skip |
|:-:|:-:|:-:|:-:|
| 20G | 3/3 ✓ | 3/3 ✓ | 3/3 ✓ |
| 15G | 3/3 ✓ | 3/3 ✓ | 3/3 ✓ |
| 12G | 3/3 ✓ | 3/3 ✓ | 3/3 ✓ |

全部 27/27 完成，无 OOM，输出一致。

## 8. 结论

1. **auto 策略是 default 和 skip 的智能折中**: 在内存充足时（20G）执行 populate 获得最佳性能，在内存不足时（15G/12G）自动跳过避免严重退化。
2. **default 在 12G 下严重退化**: 563K major faults、wall time 106.8s（比 auto/skip 慢 50%+），因为 MAP_POPULATE 盲目 fault 全部页面导致 reclaim/refault 循环。
3. **skip 在 20G 下不必要地产生 380K faults**: 模型完全可容纳时不应跳过 populate。
4. **auto 的 decision/reason 明确**: `MODEL_FITS_HEADROOM`（20G）vs `SPARSE_MOE_MODEL_EXCEEDS_HEADROOM`（15G/12G），基于 Sparse-MoE 模型特性和可用内存余量。

---

附: 原始数据 `runs/`，汇总 `all_runs_metrics.csv`、`group_stats.csv`、`summary.json`。
