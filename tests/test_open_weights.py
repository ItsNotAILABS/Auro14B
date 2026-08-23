from auro_native_llm.open_weights import (
    BYTE_OFFSET,
    BYTE_VOCAB_END,
    LEGACY_CONTROL_TOKENS,
    RELEASE_EXTENSION_TOKENS,
    ByteTokenizer,
    OpenHIM,
    OpenHIMConfig,
    corpus_digest,
)
from auro_native_llm.production_fleet.runtime import NativeOpenWeightGenerator


def _save(model, directory, key='test-runner-key'):
    model.save(directory, {"open_weights": True}, corpus_sha256=corpus_digest('fixture corpus'), source_commit='test-commit', runner_identity='pytest', signing_key=key)
    return key


def test_byte_tokenizer_is_lossless_and_has_no_unknown_token():
    tokenizer = ByteTokenizer(); text = "HIM → φ\ncode:\treturn 1"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.manifest()["unknown_token"] is None


def test_tokenizer_v2_preserves_all_legacy_control_and_byte_ids():
    legacy = ByteTokenizer("v1")
    release = ByteTokenizer("v2")
    assert BYTE_OFFSET == len(LEGACY_CONTROL_TOKENS) == 16
    assert BYTE_VOCAB_END == 272
    for index, token in enumerate(LEGACY_CONTROL_TOKENS):
        assert legacy.control_token_ids[token] == index
        assert release.control_token_ids[token] == index
    text = "same byte IDs: φ"
    assert legacy.encode(text) == release.encode(text)
    assert max(legacy.encode(text)) < BYTE_VOCAB_END


def test_tokenizer_v2_appends_load_bearing_controls_after_byte_range():
    release = ByteTokenizer("v2")
    assert tuple(release.control_token_ids[token] for token in RELEASE_EXTENSION_TOKENS) == (272, 273, 274)
    assert release.vocab_size == 275
    manifest = release.manifest()
    assert manifest["legacy_control_ids_preserved"] is True
    assert manifest["legacy_byte_ids_preserved"] is True


def test_parameter_count_is_weights_not_tokens():
    model = OpenHIM(OpenHIMConfig(context_length=4, embedding_dim=8, hidden_dim=12))
    assert model.num_parameters == sum(value.size for value in model.weights.values())
    assert model.num_parameters > 272


def test_checkpoint_round_trip_uses_release_tokenizer_by_default(tmp_path):
    model = OpenHIM(OpenHIMConfig(context_length=4, embedding_dim=8, hidden_dim=12))
    assert model.tokenizer.version == "v2"
    key = _save(model, tmp_path)
    loaded = OpenHIM.load(tmp_path, runner_signing_key=key)
    assert loaded.num_parameters == model.num_parameters
    assert loaded.config == model.config
    assert loaded.tokenizer.version == "v2"
    assert loaded.verified_checkpoint["verified"] is True
    assert loaded.verified_checkpoint["tokenizer_version"] == "v2"


def test_explicit_v1_checkpoint_remains_loadable(tmp_path):
    config = OpenHIMConfig(context_length=4, embedding_dim=8, hidden_dim=12, tokenizer_version="v1")
    model = OpenHIM(config)
    key = _save(model, tmp_path)
    loaded = OpenHIM.load(tmp_path, runner_signing_key=key)
    assert loaded.config.tokenizer_version == "v1"
    assert loaded.tokenizer.vocab_size == 272
    assert loaded.verified_checkpoint["tokenizer_version"] == "v1"


def test_native_generator_reports_exact_byte_token_usage(tmp_path, monkeypatch):
    model = OpenHIM(OpenHIMConfig(context_length=4, embedding_dim=8, hidden_dim=12))
    key = _save(model, tmp_path)
    monkeypatch.setenv('AURO_TRAINING_RECEIPT_HMAC_KEY', key)
    result = NativeOpenWeightGenerator(str(tmp_path), runner_signing_key=key)([{"role": "user", "content": "hello"}], {"max_tokens": 4, "temperature": 0.5})
    assert result["provider"] == "repository-native-open-weights"
    assert result["usage"]["total_tokens"] == result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"]
