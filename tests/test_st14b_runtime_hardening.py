from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from auro_native_llm.model.st14b import ST14BArchitecture
from auro_native_llm.model.st14b_runtime import AuroST14BForCausalLM, ST14BRuntimeConfig
from auro_native_llm.serve.st14b_serving import OpenAICompatibleAdapter, render_command, serving_manifest, write_serving_manifest


class ST14BHardeningTests(unittest.TestCase):
    def test_parameter_target_matches_implemented_geometry(self) -> None:
        spec = ST14BArchitecture()
        self.assertEqual(spec.total_parameter_target, spec.dense_parameter_estimate())
        self.assertEqual(spec.total_parameter_target, 14_339_691_520)
        self.assertEqual(spec.research_parameter_label, 14_200_000_000)

    def test_reduced_runtime_uses_native_gqa_when_supported(self) -> None:
        cfg = ST14BRuntimeConfig(
            vocab_size=256,
            hidden_size=64,
            num_layers=2,
            num_heads=8,
            num_kv_heads=2,
            intermediate_size=128,
            max_seq_len=64,
        )
        model = AuroST14BForCausalLM(cfg)
        cache = model.new_cache()
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        logits = model.prefill(ids, cache)
        self.assertEqual(tuple(logits.shape), (1, 8, cfg.vocab_size))
        backends = model.attention_backends()
        self.assertEqual(len(backends), cfg.num_layers)
        self.assertTrue(all(name in {"torch-sdpa-native-gqa", "compat-expanded-kv"} for name in backends))
        if "torch-sdpa-native-gqa" in backends:
            self.assertTrue(all(name == "torch-sdpa-native-gqa" for name in backends))
        self.assertEqual(cache.layers[0].key.size(1), cfg.num_kv_heads)

    def test_serving_profiles_are_concrete(self) -> None:
        manifest = serving_manifest()
        self.assertEqual(set(manifest["profiles"]), {"transformers", "vllm", "tensorrt-llm", "llama.cpp"})
        self.assertEqual(render_command("vllm", model_path="/models/auro", tp=4, dtype="bfloat16")[:3], ["vllm", "serve", "/models/auro"])
        self.assertIn("llama-server", render_command("llama.cpp", gguf_path="/models/auro.gguf"))

    def test_remote_plaintext_adapter_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OpenAICompatibleAdapter("vllm", "http://example.com:8000", "AURO-ST-14B")

    def test_serving_manifest_can_be_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_serving_manifest(Path(tmp) / "serving.json")
            payload = json.loads(path.read_text())
            self.assertEqual(payload["status"], "PROTOTYPE")


if __name__ == "__main__":
    unittest.main()
