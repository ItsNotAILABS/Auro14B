import torch
from torch import nn

from auro_native_llm.experiments.structured_transformer import (
    FastfoodLinear,
    OrthogonalMemory,
    StructuredArchitectureReceipt,
    StructuredMultiHeadAttention,
    StructuredTransformerBlock,
    fwht_torch,
)


def test_fwht_round_trip_and_energy():
    torch.manual_seed(3)
    x = torch.randn(4, 16)
    y = fwht_torch(x)
    assert torch.allclose(fwht_torch(y), x, atol=1e-5)
    assert torch.allclose(x.square().sum(), y.square().sum(), atol=1e-5)


def test_fastfood_linear_shape_gradient_and_no_dense_weight():
    layer = FastfoodLinear(13, 29, seed=7)
    x = torch.randn(2, 5, 13, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 5, 29)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.dense_weight_elements == 0
    assert not any(tuple(parameter.shape) == (29, 13) for parameter in layer.parameters())


def test_fastfood_seed_is_deterministic():
    first = FastfoodLinear(8, 12, seed=5)
    second = FastfoodLinear(8, 12, seed=5)
    x = torch.randn(3, 8)
    assert torch.allclose(first(x), second(x))


def test_structured_attention_causal_and_padding_mask():
    module = StructuredMultiHeadAttention(16, 4, seed=2)
    x = torch.randn(2, 6, 16)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    y = module(x, attention_mask=mask)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_transformer_block_trains_one_step():
    torch.manual_seed(11)
    block = StructuredTransformerBlock(16, 4, d_ff=32, seed=4)
    head = nn.Linear(16, 7)
    optimizer = torch.optim.AdamW(list(block.parameters()) + list(head.parameters()), lr=1e-2)
    x = torch.randn(4, 5, 16)
    target = torch.randint(0, 7, (4, 5))
    before = nn.functional.cross_entropy(head(block(x)).reshape(-1, 7), target.reshape(-1))
    for _ in range(4):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(head(block(x)).reshape(-1, 7), target.reshape(-1))
        loss.backward()
        optimizer.step()
    after = nn.functional.cross_entropy(head(block(x)).reshape(-1, 7), target.reshape(-1))
    assert after < before


def test_orthogonal_memory_retrieves_matching_slot():
    memory = OrthogonalMemory(8, 8, temperature=0.05)
    with torch.no_grad():
        memory.values.copy_(torch.eye(8))
    value, weights = memory.retrieve(memory.keys[3:4])
    assert weights.argmax(dim=-1).item() == 3
    assert value.argmax(dim=-1).item() == 3


def test_receipt_preserves_claim_boundaries():
    receipt = StructuredArchitectureReceipt(128, 4, 512, 1000, 700, 0).to_dict()
    assert receipt["benchmark_superiority_claimed"] is False
    assert receipt["hallucination_reduction_claimed"] is False
    assert receipt["promoted_to_production"] is False
