"""Grant Chrome access, train with symbolic compression + multi-site + brains."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT))

    from auro_native_llm.organism.family import build_mind
    from auro_native_llm.organism.checkpoint import save_mind, load_mind
    from auro_native_llm.symbolic.compress import (
        SymbolicCompressor,
        future_2b_spec,
        future_4b_spec,
    )
    from auro_native_llm.chrome.tools import ChromeToolbelt

    t0 = time.time()
    out = ROOT / "checkpoints" / "auro_minds" / "Auro-2B_chrome_symbolic"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("CHROME ACCESS + SYMBOLIC COMPRESSION TRAIN", flush=True)
    print("=" * 60, flush=True)

    # Chrome grant
    belt = ChromeToolbelt(mock=False, auto_start=True, headless=True)
    grant = belt.grant_access(prefer_real=True)
    print(f"[chrome] grant={grant}", flush=True)
    chrome_mock = grant.get("mode") != "real_cdp"

    # Mind
    resume = ROOT / "checkpoints" / "auro_minds" / "Auro-2B_continual"
    if resume.exists():
        print(f"[train] resume {resume}", flush=True)
        mind = load_mind(resume, chrome_mock=chrome_mock)
    else:
        mind = build_mind("Auro-2B", lite=True, chrome_mock=chrome_mock)

    # Portal + chrome
    spun = mind.portal_open(chrome_mock=chrome_mock)
    print(f"[portal] spun={spun.get('ok')} url={spun.get('url')} tools={len(spun.get('tools') or [])}", flush=True)

    # Multi-site with chrome access
    urls = [
        "https://example.com",
        "https://example.org",
        "https://www.wikipedia.org/wiki/Golden_ratio",
    ]
    multi = mind.multi_site(
        "Survey titles and key text; note math or product cues",
        urls,
        chrome_mock=chrome_mock,
    )
    print(
        f"[multi_site] ok={multi.get('ok')} n={len(urls)} "
        f"latency_ms={multi.get('latency_ms')} findings_len={len(multi.get('mind_findings') or '')}",
        flush=True,
    )

    # Symbolic compressor
    sym = SymbolicCompressor(budget=50_000)
    ctx = sym.expand_context("build REST API multi-site research spectral math", top_k=6)
    print(f"[symbolic] programs={len(sym.programs)}", flush=True)
    print(ctx[:400], flush=True)

    # Future specs
    s2 = future_2b_spec()
    s4 = future_4b_spec()
    eff = sym.effective_intelligence(
        mind.language.num_params,
        retrieval_docs=1025,
        tools=len(spun.get("tools") or []) or 20,
    )
    print(
        f"[future] 2B target={s2.parameter_target:,} 4B target={s4.parameter_target:,}",
        flush=True,
    )
    print(f"[effective] {json.dumps(eff, indent=2)}", flush=True)

    # Teach domains (code/research/math brains)
    teach = mind.teach_domains(steps_per_lesson=1)
    print(
        f"[brains] lessons={teach.get('lessons')} steps={teach.get('train_steps')} "
        f"cuda={ (teach.get('cuda') or {}).get('backend') }",
        flush=True,
    )

    # Entangled train on symbolic + multi-site digests
    texts = [
        ctx,
        multi.get("summary_for_llm") or "",
        s2.thesis,
        s4.thesis,
        "Chrome multi-site agents control internet UI via interior MCP portal.",
        "Symbolic compression: programs not params. 2-4B + symbols beat stale 70B.",
    ]
    history = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        r = mind.train_entangled(text[:2000], steps=2)
        history.append(
            {
                "i": i,
                "ce": r["last"]["student"].get("ce"),
                "accel": r["last"]["student"].get("accel_backend"),
                "council": [c["source"] for c in r["last"].get("council") or []],
            }
        )
        print(f"  [entangled {i+1}/{len(texts)}] ce={history[-1]['ce']} accel={history[-1]['accel']}", flush=True)

    # More structured CE on symbolic expansions
    tok = mind.language.tokenizer
    ces = []
    for step in range(1, 13):
        block = sym.expand_context(
            ["code app", "research MESIE", "math DFT", "multi site chrome"][step % 4],
            top_k=4,
        )
        ids = tok.encode(block, max_length=96)
        arr = np.array([ids], dtype=np.int64)
        m = mind.language.train_step(arr, arr, lr=2e-3, text_for_meaning=block[:200])
        ces.append(float(m.get("ce", 0)))
        if step % 4 == 0 or step == 1:
            print(
                f"  [ce {step}/12] ce={m.get('ce'):.4f} accel={m.get('accel_backend')}",
                flush=True,
            )

    sym.save(out / "symbolic_store.json")
    meta = save_mind(mind, out)

    report = {
        "schema": "auro.chrome_symbolic.train.v1",
        "ok": True,
        "chrome_grant": grant,
        "chrome_mock": chrome_mock,
        "num_params_live": mind.language.num_params,
        "train_steps": mind.language.train_steps,
        "portal": spun.get("url"),
        "multi_site_ok": multi.get("ok"),
        "symbolic": sym.stats(),
        "effective": eff,
        "future_2b": s2.to_dict(),
        "future_4b": s4.to_dict(),
        "entangled_history": history,
        "ce_min": min(ces) if ces else None,
        "ce_last": ces[-1] if ces else None,
        "checkpoint": str(out),
        "checkpoint_meta": meta,
        "elapsed_s": time.time() - t0,
        "thesis": (
            "A 2–4B live core + symbolic programs + multi-site tools + polyglot "
            "teachers advances capability without stale 100B vanity."
        ),
    }
    (out / "CHROME_SYMBOLIC_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    md = (
        f"# Chrome + Symbolic Train\n\n"
        f"- Live params: **{mind.language.num_params:,}**\n"
        f"- Chrome mode: **{grant.get('mode')}**\n"
        f"- Portal: `{spun.get('url')}` · UI `/portal`\n"
        f"- Multi-site: **{multi.get('ok')}** ({len(urls)} urls)\n"
        f"- Symbolic programs: **{len(sym.programs)}**\n"
        f"- CE min/last: {report['ce_min']} / {report['ce_last']}\n"
        f"- Train steps: **{mind.language.train_steps}**\n"
        f"- ChaosCUDA: {(teach.get('cuda') or {}).get('backend')}\n"
        f"- Thesis: {report['thesis']}\n"
    )
    (out / "CHROME_SYMBOLIC_REPORT.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "ok",
                    "num_params_live",
                    "train_steps",
                    "chrome_mock",
                    "ce_min",
                    "ce_last",
                    "elapsed_s",
                    "checkpoint",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
