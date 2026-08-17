> **Provenance**：本文件是由原始实验目录 `experiments/experiment_0817_decode_normal_ab/experiment_0817_decode_normal_ab/RESULT.md` 冻结版 RESULT.md 复制入库的正式版本化证据；底层数据为该目录冻结的 `group_stats.csv` / `all_runs_metrics.csv` / 各 run `analysis/metrics.json`（原始载荷按仓库约定不入库）。

# Decode NORMAL A/B Experiment — RESULT

Spec: `run/实验0817.md` (strict paired A/B test for `LLAMA_MMAP_DECODE_NORMAL`).
Output root: `test/experiment_0817_decode_normal_ab/`.

## Provenance

| Field | Value |
| --- | --- |
| Git commit | `a5d80057701a759edd40f477e9375e0daffbe757` |
| Git clean/dirty | dirty (only untracked auxiliary files: `.trae/`, `build-trace-*/`, `experiments/`, `llama.cpp/trace_output`, `test/`, `run/`); no source modifications |
| Binary SHA-256 | `97321119b6e29cb54f13e01310be62f1c6f31c291a100fac273371fdb6b2f545` |
| Binary path | `build-trace-on/bin/llama-cli` (LLAMA_MEM_TRACE:BOOL=ON, single build shared by both groups) |
| Model SHA-256 | `5607c8fcc8b04ada7d1a1152b9a5b6c1e67e6768232c16f6b03d9719d5ab1b2d` |
| Model | `models/Qwen3.5-35B-A3B-Q3_K_M.gguf` (16,356,375,168 bytes ≈ 15.24 GiB) |
| Prompt SHA-256 | `59f51358b13d0600feaf78e0cccfb71c9f25bdce3259ddae301e6c3217897e4f` |
| Output SHA-256 (all 6 P runs) | `693f2012f73db26ed093f5747a5e0b88456db4a282d155c446629eb03e3b3b52` (identical across all runs) |
| Cgroup | systemd user scope, `MemoryMax=21474836480` (20 GiB), `MemorySwapMax=0` |
| Cold-cache prep | `prepare_model_cache.py --mode cold` (posix_fadvise DONTNEED) per run, after SHA computation |
| Inference config | `-n 80 -t 8 -b 512 -ub 512 -c 2048 -ngl 0 -temp 0 -seed 1234`, `--no-perf --no-show-timings` |
| Trace profile | `evidence` (sinks: tensor + kv + expert + memory; allow_drop=0) |
| Single independent variable | `LLAMA_MMAP_DECODE_NORMAL` (S=0, N=1) |
| Interleave order | S1 → N1 → N2 → S2 → S3 → N3 |
| Boundary probe (D) | `LLAMA_EXPERT_BOUNDARY_PROBE=1`, `MAX_PAIRS=1024` |
| Boundary probe (P) | disabled (`LLAMA_EXPERT_BOUNDARY_PROBE=0`) |
| Other mmap advice | `LLAMA_MMAP_SKIP_POPULATE=1` (constant in both groups — required for the boundary probe to find cold expert pages; otherwise MAP_POPULATE makes every expert resident before the probe runs, giving 0 valid pairs). `POSIX_FADV_SEQUENTIAL` is still applied at mmap time in both groups; only the in-flight transition to `POSIX_FADV_NORMAL` before first decode is gated by the single variable. Router prefetch / OS hints / controller / working set / MADV_COLD / runtime reclaim are all off. |
| Run interleave | S and N share binary/model/prompt/cgroup/cache/probe params; interleaved to absorb host-state drift. |

## Boundary probe — Experiment D (N=3 paired)

All 6 runs completed with `valid_pairs=39` each, `invalid_pairs=0`, `cold_rejected_candidates≈145,399/run` (prefill warms most experts; only the probe-sampled pairs that are still cold before decode are eligible — this is the expected behavior of the probe and is identical in both groups).

Per-run probes confirm the invariants required by spec §5:
- `selected_first_use=true`, `neighbor_selected_by_router=false`, `neighbor_logical_first_use=false`
- `selected_before.resident_pages=0` and `neighbor_before.resident_pages=0` (cold)
- `selected_after.resident_pages=135` (full expert slice loaded by router)
- `A_new_pages=135` for every pair in every run (deterministic, since A is fully loaded once)

Aggregate per-run metrics (full per-pair breakdown in `D_boundary_analysis.json`):

| Run | Variant | valid | sum_A_new_pages | sum_B_new_pages | B/A | weighted B/(A+B) |
| --- | --- | --- | --- | --- | --- | --- |
| D_S_r1 | Sequential | 39 | 5265 | 2214 | 0.42051 | 0.29603 |
| D_S_r2 | Sequential | 39 | 5265 | 2370 | 0.45014 | 0.31041 |
| D_S_r3 | Sequential | 39 | 5265 | 2156 | 0.40950 | 0.29053 |
| D_N_r1 | Decode NORMAL | 39 | 5265 | 1027 | 0.19506 | 0.16322 |
| D_N_r2 | Decode NORMAL | 39 | 5265 |  950 | 0.18044 | 0.15286 |
| D_N_r3 | Decode NORMAL | 39 | 5265 |  918 | 0.17436 | 0.14847 |

Group statistics (N=3, mean / min / max / std):

| Metric | Sequential (S) | Decode NORMAL (N) | Δ (N − S) | Δ % |
| --- | --- | --- | --- | --- |
| sum_B_new_pages | 2246.7 / 2156 / 2370 / 92.3 | 965.0 / 918 / 1027 / 48.4 | **−1281.7** | **−57.0 %** |
| B/A ratio | 0.4267 / 0.4095 / 0.4501 / 0.0168 | 0.1833 / 0.1744 / 0.1951 / 0.0086 | **−0.2434** | **−57.0 %** |
| weighted B/(A+B) | 0.2990 / 0.2905 / 0.3104 / 0.0081 | 0.1549 / 0.1485 / 0.1632 / 0.0061 | **−0.1441** | **−48.2 %** |
| sum_A_new_pages | 5265 / 5265 / 5265 / 0 | 5265 / 5265 / 5265 / 0 | 0 | 0 % |

The N group's three runs are completely non-overlapping with the S group on `sum_B_new_pages`, `B/A`, and `weighted B/(A+B)` — i.e. the reduction is reproducible across all three interleaved runs.

### Claim A — Reduce cross-Expert overfetch

**Supported.** In the strict paired boundary probe, all three N runs show a stable, large reduction versus all three S runs on every required metric (B_new pages, B/A ratio, weighted B/(A+B) ratio). The reduction is reproducible (no overlap between the two groups on any metric across N=3) and is mechanistically consistent with switching the file advice away from `POSIX_FADV_SEQUENTIAL` before decode. `A_new` stays identical (5265 pages) in both groups, confirming the comparison only affects cross-Expert overfetch, not the useful Expert load.

## Performance — Experiment P (N=3 paired, boundary probe off)

Per-run metrics (extracted from `memory_trace.jsonl` STEP_END latency_ns + `process_metrics.json` + `cgroup_after_inference.json`):

| Run | Variant | wall_s | prefill_s | decode_s | decode_p50_ms | decode_p95_ms | decode_tps | majfault_proc | max_rss_kb | cgroup_peak_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P_S_r1 | S | 90.32 | 55.152 | 17.377 | 217.78 | 262.10 | 4.5463 | 392603 | 13199664 | 15504834560 |
| P_S_r2 | S | 90.28 | 55.471 | 18.338 | 228.88 | 300.48 | 4.3079 | 383285 | 13200324 | 15502106624 |
| P_S_r3 | S | 86.26 | 52.450 | 17.692 | 216.60 | 297.30 | 4.4654 | 388624 | 13199532 | 15503360000 |
| P_N_r1 | N | 86.00 | 49.646 | 18.448 | 227.97 | 296.63 | 4.2824 | 383951 | 13196720 | 15435714560 |
| P_N_r2 | N | 88.98 | 55.182 | 17.593 | 213.35 | 289.92 | 4.4904 | 375706 | 13196912 | 15440044032 |
| P_N_r3 | N | 92.69 | 58.049 | 17.347 | 213.32 | 270.90 | 4.5542 | 374298 | 13198468 | 15450259456 |

Group statistics (N=3, mean / min / max / std):

| Metric | Sequential (S) | Decode NORMAL (N) | Δ mean (N − S) | Δ % |
| --- | --- | --- | --- | --- |
| wall_time_s | 88.953 / 86.260 / 90.320 / 1.905 | 89.223 / 86.000 / 92.690 / 2.737 | +0.270 | +0.30 % |
| prefill_total_s | 54.358 / 52.450 / 55.471 / 1.355 | 54.292 / 49.646 / 58.049 / 3.488 | −0.066 | −0.12 % |
| decode_total_s | 17.802 / 17.377 / 18.338 / 0.400 | 17.796 / 17.347 / 18.448 / 0.472 | −0.006 | −0.04 % |
| decode_p50_ms | 221.088 / 216.601 / 228.879 / 5.530 | 218.214 / 213.320 / 227.971 / 6.899 | −2.874 | −1.30 % |
| decode_p95_ms | 286.629 / 262.104 / 300.485 / 17.390 | 285.816 / 270.899 / 296.626 / 10.897 | −0.813 | −0.28 % |
| decode_tps | 4.4399 / 4.3079 / 4.5463 / 0.0990 | 4.4423 / 4.2824 / 4.5542 / 0.1160 | +0.0025 | +0.06 % |
| major_faults_proc | 388171 / 383285 / 392603 / 3818 | 377985 / 374298 / 383951 / 4258 | **−10186** | **−2.62 %** |
| minor_faults_proc | 530912 / 527833 / 532962 / 2217 | 524012 / 514073 / 533617 / 7982 | −6900 | −1.30 % |
| max_rss_kb | 13199840 / 13199532 / 13200324 / 346 | 13197367 / 13196720 / 13198468 / 783 | −2473 | −0.02 % |
| cgroup_memory_peak_bytes | 15503433728 / 15502106624 / 15504834560 / 1114895 | 15442006016 / 15435714560 / 15450259456 / 609784 | −61427712 | −0.40 % |
| cgroup_file_bytes | 15067103232 / 15065669632 / 15068725248 / 1254545 | 15005816149 / 14999576576 / 15014051840 / 6075727 | −61287083 | −0.41 % |
| cgroup_pgmajfault (memory.stat) | 85216 / 84465 / 86530 / 933 | 83602 / 82027 / 85029 / 1230 | −1614 | −1.89 % |
| cgroup_pgscan | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 | 0 % |

Output consistency: all 6 runs produce identical `inference_output.txt` SHA-256 → the LLM output is bit-identical between Sequential and Decode NORMAL (determinism preserved).

### Claim B — Improve end-to-end inference performance

**Not supported.** Across N=3 paired runs, Decode NORMAL does not produce a stable improvement on `wall_time`, `prefill`, `decode_total`, `decode_p50`, `decode_p95`, or `decode_tps`. All deltas are below 1.5 % in magnitude and within the per-group standard deviation (the N group is actually slightly more variable than S on `wall_time`, `prefill`, `decode_tps` and `decode_p50`). The only statistically-directional change is `major_faults_proc` (−2.62 %), which is mechanistically consistent with the Claim A overfetch reduction (fewer neighbor pages brought in ⇒ fewer major faults), but it does not translate into measurable wall-time savings at this prompt length / token budget / cgroup.

## Conclusions

- **Claim A (overfetch reduction): Supported.** Strict paired boundary probe with N=3 interleaved runs shows Decode NORMAL cuts cross-Expert overfetch by ~57 % on `B_new_pages` and `B/A`, ~48 % on weighted `B/(A+B)`, with the two groups completely non-overlapping across runs. `A_new` is unchanged (5265 pages in every run), confirming the mechanism specifically targets unselected-neighbor overfetch, not useful Expert loads.
- **Claim B (performance gain): Not supported.** End-to-end metrics (`wall_time`, `decode_total`, `decode p50/p95`, `decode_tps`) change by less than ~1.5 % and are within run-to-run noise; only `major_faults` shows a small directional decrease (−2.6 %) consistent with reduced overfetch but insufficient to move wall-time at this scale. This is the "overfetch down, performance flat" outcome explicitly allowed by spec §7 — it is reported as-is, no parameter tuning was performed.

## Sufficient for finalist main slides?

- **Claim A** is finalist-grade: the mechanism has a clean, reproducible, mechanistically-explained effect on cross-Expert overfetch, with strict paired-probe evidence.
- **Claim B** is **not** finalist-grade on its own: there is no end-to-end performance gain at this scale. If the slide intends to claim a performance benefit, it would need a different memory-pressure regime (e.g. tighter MemoryMax / longer decode / larger Expert tensor / no host page cache headroom) where the saved major faults actually translate into wall-time savings. The current 20 GiB cgroup + 80-token decode is too comfortable to expose the benefit.

## Deliverables

All deliverables are under `test/experiment_0817_decode_normal_ab/` only:

- `RESULT.md` (this file)
- `prelude.json` — provenance snapshot taken before any run
- `run_one.sh` — single A/B run orchestrator (sets the single independent variable + skip populate)
- `run_D.sh` — Experiment D interleave driver (S1→N1→N2→S2→S3→N3)
- `run_P.sh` — Experiment P interleave driver (same order, probe off)
- `D_run_driver.log` — Experiment D driver log
- `P_run_driver.log` — Experiment P driver log
- `D_boundary_analysis.json` — per-run + matched-pair probe summary from `analyze_expert_boundary_probe.py`
- `P_performance_analysis.json` — per-run + grouped stats from `analyze_performance.py`
- `analyze_performance.py` — performance stats generator
- `runs/D_S_r{1,2,3}/`, `runs/D_N_r{1,2,3}/` — Experiment D raw run outputs
- `runs/P_S_r{1,2,3}/`, `runs/P_N_r{1,2,3}/` — Experiment P raw run outputs

Each run directory under `runs/<RUN>/latest/` contains: `run_manifest.json`, `cache_preparation.json`, `pre_run_state.json`, `memory_trace.jsonl`, `tensor_trace.jsonl`, `expert_trace.jsonl`, `kv_trace.jsonl`, `inference_output.txt`, `inference_stderr.txt`, `output.sha256`, `process_metrics.json`, `cgroup_after_inference.json`, `summary.json`, `analysis/`, and the per-run `run.log` is at `runs/<RUN>/run.log`.

