# Auro RAH — Recursive Agent Harnesses

Same doctrine as POCKET RAH. The recursive unit is a **full Auro sub-agent harness**
(role + parent capacity + disk receipt), not a bare model call.

```python
from auro_native_llm.rah import run_rah, plan_fanout

run_rah(
    "split independent work",
    leaves=["embed the corpus", "match two PSDs", "plan the next train"],
    max_parallel=3,
)
```

- Receipts: `~/.auro/rah/<run_id>/`
- Caps: 8 parallel · 3 depth · 12 leaves
- Pocket: `mode=auro` RAH leaves call this module when `AURO14B_ROOT` is set
