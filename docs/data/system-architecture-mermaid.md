# System Architecture Mermaid Backup

```mermaid
flowchart LR
    Q[Qwen MoE
LLM Runtime Semantic] --> L[llama.cpp / GGML]
    L --> R[MoE Router]
    R --> I[Expert ID / Score]
    I --> T[Expert Tensor Registry]
    T --> S[Expert Slice]
    S --> M[Memory Object
Demand Lifecycle]
    M --> A[Async Hint Task]
    A --> D[Linux madvise]
    D --> V[Page Cache / VM]

    M --- W[Semantic Working Set
Stable mechanism]
    C[MADV_COLD
Research control] --> D
    X[Runtime Rescue
Research control] -. suspend COLD .-> C
    X -. gate bypass .-> A

    R -. Router metrics .-> TW[Trace Writer]
    T -. Expert metrics .-> TW
    A -. Task / first-use metrics .-> TW
    V -. OS metrics .-> TW
    TW --> J[JSONL]
    J --> AN[Analysis / validation]
```

`MADV_COLD` is an advisory Linux call, not a claim of physical page reclamation. Calibration shadow is observation-only and intentionally omitted from the default main path.
