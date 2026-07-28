# HIM Multi-Model Orchestrator

HIM is the sovereign model identity and NOVA is its governed deliberation runtime. A model is counted only when a real generator/checkpoint or explicit endpoint is configured. Agent roles, tokens, tools, and brain regions never inflate model or parameter counts.

## Fleet configuration

The primary lane continues to use AURO_NATIVE_CHECKPOINT when present. Additional real endpoints are supplied through AURO_MODEL_FLEET_JSON:

    [
      {
        "id":"him-coder",
        "base_url":"http://127.0.0.1:8091/v1",
        "model":"HIM-Coder-v1",
        "role":"coding",
        "provider":"repository-native-open-weights",
        "parameter_count":14000000000,
        "capabilities":["code","tool"],
        "priority":10,
        "local":true,
        "checkpoint_hash":"<sha256>"
      }
    ]

Every lane is returned by GET /v1/models. Responses include model_fleet, models_used, and routing_traces. Each trace records task classification, candidates, actual attempts, latency, provider identity, failure type, and a SHA-256 receipt.

## Routing

The current deterministic router classifies general, code, math, research, and tool tasks. It prefers a matching capability, then local execution, then operator priority. This provides a falsifiable baseline that can later be replaced by a trained router only when evaluation shows improvement.

AURO_ALLOW_HOSTED_FALLBACK defaults to 0. A failed repository-native lane cannot invoke a hosted provider unless the operator sets it to 1. Even then, the response reports the actual hosted provider and model; it is never relabeled as HIM weights.

## Honest maturity boundary

This upgrades inference orchestration, provenance, reliability, and API transparency. It does not train new weights or turn the existing small HIM-native-v0 checkpoint into a 14B checkpoint. Each new checkpoint must separately pass tokenizer, corpus, training, benchmark, portability, safety, and readiness promotion gates.
