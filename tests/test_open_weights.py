from auro_native_llm.open_weights import ByteTokenizer, OpenHIM, OpenHIMConfig, corpus_digest
from auro_native_llm.production_fleet.runtime import NativeOpenWeightGenerator


def _save(model, directory, key='test-runner-key'):
    model.save(directory, {"open_weights": True}, corpus_sha256=corpus_digest('fixture corpus'), source_commit='test-commit', runner_identity='pytest', signing_key=key)
    return key


def test_byte_tokenizer_is_lossless_and_has_no_unknown_token():
    tokenizer = ByteTokenizer(); text = "HIM → φ\ncode:\treturn 1"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.manifest()["unknown_token"] is None


def test_parameter_count_is_weights_not_tokens():
    model = OpenHIM(OpenHIMConfig(context_length=4, embedding_dim=8, hidden_dim=12))
    assert model.num_parameters == sum(value.size for value in model.weights.values())
    assert model.num_parameters > 272


def test_checkpoint_round_trip(tmp_path):
    model = OpenHIM(OpenHIMConfig(context_length=4, embedding_dim=8, hidden_dim=12))
    key = _save(model, tmp_path)
    loaded = OpenHIM.load(tmp_path, runner_signing_key=key)
    assert loaded.num_parameters == model.num_parameters
    assert loaded.config == model.config
    assert loaded.verified_checkpoint["verified"] is True


def test_native_generator_reports_exact_byte_token_usage(tmp_path, monkeypatch):
    model = OpenHIM(OpenHIMConfig(context_length=4, embedding_dim=8, hidden_dim=12))
    key = _save(model, tmp_path)
    monkeypatch.setenv('AURO_TRAINING_RECEIPT_HMAC_KEY', key)
    result = NativeOpenWeightGenerator(str(tmp_path), runner_signing_key=key)([{"role": "user", "content": "hello"}], {"max_tokens": 4, "temperature": 0.5})
    assert result["provider"] == "repository-native-open-weights"
    assert result["usage"]["total_tokens"] == result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"]
