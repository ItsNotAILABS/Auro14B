"""Report score-operation estimates for dense versus delta long memory."""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auro_native_llm.model.delta_attention import DeltaAttentionEngine

def main():
    p=argparse.ArgumentParser();p.add_argument("--tokens",type=int,default=4096);p.add_argument("--hidden",type=int,default=256);p.add_argument("--slots",type=int,default=256);a=p.parse_args()
    rng=np.random.default_rng(14); base=rng.normal(size=(1,1,a.hidden)); changes=rng.normal(scale=.01,size=(1,a.tokens,a.hidden)); hidden=base+np.cumsum(changes,axis=1)
    engine=DeltaAttentionEngine(a.hidden,max_slots=a.slots,novelty_threshold=.08);t=time.perf_counter();engine.observe(hidden);elapsed=(time.perf_counter()-t)*1000
    report={"schema":"auro.delta_attention.benchmark.v1","tokens":a.tokens,"hidden":a.hidden,"max_slots":a.slots,"observe_ms":elapsed,**engine.receipt.to_dict(),"claim_boundary":"Operation counts estimate the delta sidecar only; this is not an end-to-end decoder speedup claim."}
    print(json.dumps(report,indent=2));return 0
if __name__=="__main__": raise SystemExit(main())
