# Auro14B HuggingFace Agentic Protocol

The **Auro14B Agentic Protocol** establishes standard interoperability between Auro's native dual-VM reasoning engines and the Hugging Face Hub (SmolAgents, LangChain, Transformers Agents).

## Architecture

```
                                  ┌───────────────────────────┐
                                  │   HuggingFace Hub Repo    │
                                  │ auro-ai/Auro14B-Monad-v2  │
                                  └─────────────┬─────────────┘
                                                │
                                  ┌─────────────▼─────────────┐
                                  │   hf_agentic_bridge.py    │
                                  └─────────────┬─────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 │                                                             │
   ┌─────────────▼─────────────┐                                 ┌─────────────▼─────────────┐
   │    Monad EVM Execution    │                                 │    Solana SVM Execution   │
   │  - Giga-Match Engine      │                                 │  - Turbo-Auro Engine      │
   │  - Flash Loan Composer    │                                 │  - Anchor Program Auditor │
   └───────────────────────────┘                                 └───────────────────────────┘
```

## Quick Start via SmolAgents / Transformers

```python
from smolagents import CodeAgent, HfApiModel
from auro_native_llm.hf_agentic_bridge import AuroAgenticModel

# Initialize Hub model bridge
bridge = AuroAgenticModel(model_id="auro-ai/Auro14B-Monad-v2")

# Generate structured tool call
tool_call = bridge.generate_agentic_tool_call("Simulate cross-vm flash loan borrow 100000 USDC")
print(tool_call)
```

## Supported Agentic Tools
1. `monad_simulate_transaction`: EVM transaction simulation & state diff.
2. `solana_inspect_program`: SVM program static analysis & compute budget optimization.
3. `cross_vm_flash_loan`: Cross-VM atomic flash loan execution payload.
4. `mev_bundle_analyzer`: Block sandwiching & arbitrage opportunity detection.
5. `spectral_psd_match`: High-frequency signal matching engine.
6. `huggingface_hub_agentic_call`: Direct Hugging Face pipeline router tool.

## Model Card & Protocol Artifacts
- Hugging Face Model Card: [`artifacts/auro-hf/AGENTIC_PROTOCOL.json`](file:///c:/Users/Medin/Documents/GitHub/Auro14B/artifacts/auro-hf/AGENTIC_PROTOCOL.json)
- Python Integration Module: [`auro_native_llm/hf_agentic_bridge.py`](file:///c:/Users/Medin/Documents/GitHub/Auro14B/auro_native_llm/hf_agentic_bridge.py)
