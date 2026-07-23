import numpy as np

from auro_native_llm.model.delta_attention import DeltaAttentionEngine, MultiSenseAdapter
from auro_native_llm.model.auro_lm import AuroLanguageModel


def test_delta_attention_skips_redundant_state_and_bounds_memory() -> None:
    engine=DeltaAttentionEngine(8,max_slots=4,novelty_threshold=0.2)
    hidden=np.ones((1,12,8)); hidden[:,6:,:]*=2
    fused,receipt=engine.fuse(hidden)
    assert fused.shape==hidden.shape
    assert receipt["deltas_skipped"]>0
    assert receipt["memory_slots"]<=4
    assert receipt["estimated_score_op_ratio"]<1


def test_multi_sense_is_deterministic_and_native() -> None:
    adapter=MultiSenseAdapter(16)
    first,receipt=adapter.fuse({"text":"hello","sensor":[1,2,3,5,8]})
    second,_=adapter.fuse({"text":"hello","sensor":[1,2,3,5,8]})
    assert np.allclose(first,second)
    assert receipt["modalities"]==["sensor","text"]
    assert np.isclose(np.linalg.norm(first),1.0)


def test_delta_attention_runs_inside_tiny_auro_core() -> None:
    model=AuroLanguageModel.build("Auro-14B",mode="dev",hidden_dim=32,num_layers=1,num_heads=2,head_dim=16,ffn_dim=64,vocab_size=320,max_seq_len=32,num_experts=2,top_k_experts=1,continuous_dim=8,spectral_input_dim=32,num_kv_heads=1)
    ids=np.asarray([[model.tokenizer.bos_id,10,11,12,13,14,15,16]])
    output=model.forward_ids(ids)
    assert output["delta_attention"]["tokens_seen"]==8
    assert output["logits"].shape==(1,8,320)
