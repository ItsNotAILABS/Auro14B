"""Train AURO language models on MESIE compute (local NumPy engine)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np

from auro_native_llm.model.auro_lm import AuroLanguageModel
from auro_native_llm.model.checkpoint import save_checkpoint
from auro_native_llm.model.corpus import (
    collect_corpus_texts,
    collect_corpus_texts_single_repo,
    iter_training_sequences,
)
from auro_native_llm.model.tokenizer import AuroTokenizer


@dataclass
class TrainConfig:
    model_id: str = "Auro-2B"
    mode: str = "dev"
    steps: int = 50
    batch_size: int = 2
    seq_len: int = 96
    learning_rate: float = 3e-3
    lr_decay: float = 0.995
    vocab_size: int = 4096
    seed: int = 42
    output_dir: str = "checkpoints/auro"
    corpus_root: Optional[str] = None
    report_every: int = 5
    multi_repo: bool = False
    max_corpus_files: int = 80
    max_corpus_chars: int = 400_000
    neuromorphic_lr_control: bool = True
    neuromorphic_max_lr_reduction: float = 0.25
    neuromorphic_penalty_gain: float = 5.0
    extra: Dict[str, Any] = field(default_factory=dict)


def _neuromorphic_training_control(model: AuroLanguageModel, cfg: TrainConfig) -> tuple[dict[str, float], float]:
    """Read latest residual-gate receipt and derive next-step LR multiplier.

    The current NumPy trainer has a manual CE gradient path, so we do not pretend
    that adding the neuromorphic scalar to reported loss creates a differentiable
    auxiliary objective. The residual gate already changes forward logits; this
    control additionally makes activity/energy pressure affect the next update's
    learning-rate envelope.
    """
    bridge = getattr(model, "_neuro", None)
    gate = getattr(bridge, "spiking_gate", None)
    receipt = getattr(gate, "last_receipt", None)
    if receipt is None:
        return {}, 1.0
    regularizer = max(0.0, float(receipt.regularizer))
    reduction = min(max(0.0, cfg.neuromorphic_max_lr_reduction), regularizer * max(0.0, cfg.neuromorphic_penalty_gain))
    multiplier = 1.0 - reduction if cfg.neuromorphic_lr_control else 1.0
    metrics = {
        "neuro_activity_rate": float(receipt.activity_rate),
        "neuro_sparsity": float(receipt.sparsity),
        "neuro_inhibitory_tone": float(receipt.inhibitory_tone),
        "neuro_energy_proxy": float(receipt.energy_proxy),
        "neuro_control_regularizer": regularizer,
        "neuro_lr_multiplier": float(multiplier),
    }
    return metrics, float(multiplier)


def train_language_model(cfg: Optional[TrainConfig] = None) -> Dict[str, Any]:
    cfg = cfg or TrainConfig()
    np.random.seed(cfg.seed)
    t0 = time.time()

    if cfg.multi_repo:
        texts = collect_corpus_texts(
            cfg.corpus_root,
            max_files=cfg.max_corpus_files,
            max_chars=cfg.max_corpus_chars,
            multi_repo=True,
            include_github=False,
        )
    else:
        texts = collect_corpus_texts_single_repo(
            cfg.corpus_root,
            max_files=cfg.max_corpus_files,
            max_chars=cfg.max_corpus_chars,
        )
    tokenizer = AuroTokenizer(vocab_size=cfg.vocab_size)
    sample = texts[: min(40, len(texts))]
    tokenizer.train(sample, vocab_size=cfg.vocab_size)

    model = AuroLanguageModel.build(
        cfg.model_id,
        mode=cfg.mode,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        seed=cfg.seed,
        learning_rate=cfg.learning_rate,
        vocab_size=max(cfg.vocab_size, tokenizer.vocab_size),
    )

    sequences = iter_training_sequences(texts, tokenizer, max_len=cfg.seq_len)
    if not sequences:
        raise RuntimeError("no training sequences built from corpus")

    history: List[Dict[str, float]] = []
    lr = cfg.learning_rate
    for step in range(1, cfg.steps + 1):
        batch_ids = []
        for _ in range(cfg.batch_size):
            seq = sequences[np.random.randint(0, len(sequences))]
            if len(seq) < cfg.seq_len:
                seq = seq + [tokenizer.pad_id] * (cfg.seq_len - len(seq))
            else:
                seq = seq[: cfg.seq_len]
            batch_ids.append(seq)
        arr = np.array(batch_ids, dtype=np.int64)
        meaning_text = tokenizer.decode(batch_ids[0])[:400]
        metrics = model.train_step(arr, arr, lr=lr, text_for_meaning=meaning_text)
        neuro_metrics, neuro_lr_multiplier = _neuromorphic_training_control(model, cfg)
        metrics.update(neuro_metrics)
        lr *= cfg.lr_decay * neuro_lr_multiplier
        if step % cfg.report_every == 0 or step == 1 or step == cfg.steps:
            history.append({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, bool))})
            neuro_text = ""
            if neuro_metrics:
                neuro_text = (
                    f" neuro_act={neuro_metrics['neuro_activity_rate']:.3f}"
                    f" neuro_E={neuro_metrics['neuro_energy_proxy']:.3f}"
                    f" neuro_lr={neuro_metrics['neuro_lr_multiplier']:.3f}"
                )
            print(
                f"[{cfg.model_id} step {step}/{cfg.steps}] "
                f"loss={metrics['loss']:.4f} ce={metrics['ce']:.4f} "
                f"ppl={metrics['ppl']:.2f} moe={metrics['moe']:.4f}{neuro_text}"
            )

    out_dir = Path(cfg.output_dir) / cfg.model_id.replace("/", "_")
    meta = save_checkpoint(model, out_dir)

    gen = model.generate(
        "Auro MESIE spectral meaning ratio rta teotl",
        max_new_tokens=32,
        temperature=0.8,
    )

    neuro_info = None
    if getattr(model, "_neuro", None) is not None:
        try:
            neuro_info = model._neuro.info()
        except Exception:
            neuro_info = None

    report = {
        "schema": "auro.lm.train_report.v2",
        "ok": True,
        "model_id": cfg.model_id,
        "mode": cfg.mode,
        "compute_plane": "MESIE",
        "native": True,
        "steps": cfg.steps,
        "num_params": model.num_params,
        "parameter_target": model.config.parameter_target,
        "tokenizer_vocab": tokenizer.vocab_size,
        "corpus_docs": len(texts),
        "sequences": len(sequences),
        "history": history,
        "checkpoint": str(out_dir),
        "checkpoint_meta": meta,
        "sample_generation": gen.to_dict(),
        "neuromorphic_training": {
            "enabled": bool(neuro_info and neuro_info.get("neuromorphic_residual_enabled")),
            "lr_control_enabled": bool(cfg.neuromorphic_lr_control),
            "gradient_auxiliary_loss_implemented": False,
            "forward_residual_gate_affects_logits": bool(neuro_info and neuro_info.get("neuromorphic_residual_enabled")),
            "control_regularizer_affects_next_step_lr": bool(cfg.neuromorphic_lr_control),
            "checkpoint_quality_verified": False,
            "physical_energy_efficiency_verified": False,
            "bridge": neuro_info,
        },
        "elapsed_s": time.time() - t0,
        "train_config": asdict(cfg),
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Train Auro LM on MESIE compute")
    p.add_argument("--model", default="Auro-2B")
    p.add_argument("--mode", default="dev", choices=["dev", "full"])
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--output-dir", default="checkpoints/auro")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disable-neuromorphic-lr-control", action="store_true")
    args = p.parse_args()
    cfg = TrainConfig(
        model_id=args.model,
        mode=args.mode,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        seed=args.seed,
        neuromorphic_lr_control=not args.disable_neuromorphic_lr_control,
    )
    report = train_language_model(cfg)
    print(json.dumps({k: report[k] for k in report if k not in {"history", "sample_generation"}}, indent=2))
    print("--- sample ---")
    print(report["sample_generation"]["text"][:500])


if __name__ == "__main__":
    main()
