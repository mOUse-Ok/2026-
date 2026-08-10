# Final README Visualization Delivery Report

## Result

Nine evidence-faithful figure sets were generated in `docs/assets/`. Every set
has a high-resolution PNG and editable SVG; all use a white, flat,
systems-paper visual language without gradients, shadows, or 3D effects. The
figures are derived only from `docs/final-readme-evidence-v2.md` and the
already-closed Rescue JSONL needed for its per-step trace.

| Figure set | Files | Primary message | README suitability | PPT suitability |
| --- | --- | --- | --- | --- |
| System architecture | `system-architecture.svg`, `.png` | LLM semantic signal crosses into memory objects, hints, and Linux VM; observation is a side path | Architecture section | Opening/system overview |
| Project evolution | `project-evolution.svg`, `.png` | The project learned two important negative boundaries before adding runtime feedback protection | Design rationale | Narrative transition slide |
| Evidence staircase | `evidence-staircase.svg`, `.png` | Cost, negative results, mechanism closure, and state-machine evidence are distinct | Evidence/methodology section | Evidence-quality slide |
| Trace overhead | `trace-overhead.svg`, `.png` | Minimal trace cost is +3.66% under the closed setup | Trace section | Measurement slide |
| Expert prefetch ablation | `expert-prefetch-ablation.svg`, `.png` | Current HEAD shows no stable speedup in the tested 5×5 setup | Negative-results section | Ablation slide |
| Working Set budget | `working-set-budget.svg`, `.png` | Budget changes show semantic admission/eviction/readmission behavior | Mechanism section, labelled historical | Mechanism behavior slide |
| COLD ablation | `cold-ablation.svg`, `.png` | Current controlled A/B associates COLD with worse wall time and faults | Negative-results section | Falsification/lesson slide |
| Runtime Rescue timeline | `runtime-rescue-timeline.svg`, `.png` | A selected trace visibly performs trigger, gate bypass, and COLD suspension | Runtime-protection section | State-machine walkthrough |
| Correctness summary | `correctness-summary.svg`, `.png` | Closed runs, output identity, trace integrity, MO closure, and tests all pass | Reliability section | Closing verification slide |

The data ledger is [`data/readme-figures-data.md`](data/readme-figures-data.md).
The architecture has a GitHub-friendly fallback in
[`data/system-architecture-mermaid.md`](data/system-architecture-mermaid.md).
The selected Rescue trace data is in
[`data/runtime-rescue-on4-step-data.csv`](data/runtime-rescue-on4-step-data.csv).

## Required caption caveats

- Every current-head number is specific to V2's recorded environment; it is not
  a universal performance constant.
- Label `working-set-budget` as **historical**. It is mechanism behavior, not a
  current-HEAD benchmark claim.
- The prefetch chart must keep “No stable speedup observed” and retain its
  7040 MiB cold-cache / N=5×5 context.
- The COLD chart may say “associated with higher wall time and major faults in
  this controlled A/B.” It must not say COLD physically reclaimed pages merely
  because `madvise` was issued or succeeded.
- The Rescue timeline is **mechanism evidence, not causal end-to-end speedup
  proof**. Do not turn Rescue’s historical aggregate deltas into a causal
  speedup claim.
- A first-use event is not a residency measurement. Calibration shadow remains
  observation-only and is not drawn as a default mechanism.

## Reproduction and validation

Run the presentation-only generator from the repository root:

```bash
MPLCONFIGDIR=/tmp/llmop-readme-mpl python3 docs/scripts/generate_readme_figures.py
```

It writes only the listed assets/data files. It runs no model inference and no
new performance experiment. The generator registers the local Noto CJK font so
Chinese text is retained in both PNG and SVG outputs.

No README file and no runtime/core-code file was modified by this delivery.
