"""Train Auro with real physics AI formulas and save checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from auro_native_llm.organism.checkpoint import load_mind, save_mind
from auro_native_llm.physics import get_physics_engine


def main() -> int:
    mind = load_mind("checkpoints/auro_minds/Auro-2B_specialized", chrome_mock=True)
    eng = get_physics_engine()
    mind.language.physics = eng
    tok = mind.language.tokenizer
    texts = [
        "MESIE spectral action coherence Kuramoto Landau phi Schrodinger",
        "Nonlinear dispersion omega squared c k alpha k4 golden angle",
        "Fisher natural gradient embedding physics regularized CE loss",
        "Resonance matched filter Fourier spectral force on hidden state",
        "GHOST grounded hardened open scalable traceable hybrid MESIE",
    ]
    steps0 = mind.language.train_steps
    hist = []
    for step in range(40):
        t = texts[step % len(texts)]
        ids = tok.encode(t, add_bos=True, add_eos=True, max_length=96)
        while len(ids) < 64:
            ids.append(tok.pad_id)
        arr = np.array([ids[:64]], dtype=np.int64)
        m = mind.language.train_step(arr, arr, lr=2.5e-3, text_for_meaning=t)
        hist.append(m)
        if step % 10 == 0 or step == 39:
            print(
                f"step {step+1}/40 ce={m['ce']:.4f} L={m['loss']:.4f} "
                f"R={m.get('phys_resonance', 0):.3f}",
                flush=True,
            )
    out = Path("checkpoints/auro_minds/Auro-2B_physics")
    save_mind(mind, out)
    rep = {
        "schema": "auro.physics.train.v1",
        "scaffold": False,
        "fake": False,
        "steps_delta": mind.language.train_steps - steps0,
        "train_steps": mind.language.train_steps,
        "ce_first": hist[0]["ce"],
        "ce_last": hist[-1]["ce"],
        "loss_first": hist[0]["loss"],
        "loss_last": hist[-1]["loss"],
        "num_params_live": mind.language.num_params,
        "checkpoint": str(out),
        "equations": eng.report().equations,
        "last_physics": eng.last.to_dict(),
    }
    (out / "PHYSICS_TRAIN_REPORT.json").write_text(
        json.dumps(rep, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: rep[k] for k in rep if k != "equations"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
