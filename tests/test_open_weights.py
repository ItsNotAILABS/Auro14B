from auro_native_llm.open_weights import ByteTokenizer, OpenHIM, OpenHIMConfig
from auro_native_llm.production_fleet.runtime import NativeOpenWeightGenerator

def test_byte_tokenizer_is_lossless_and_has_no_unknown_token():
 tok=ByteTokenizer(); text="HIM → φ\ncode:\treturn 1"
 assert tok.decode(tok.encode(text))==text
 assert tok.manifest()["unknown_token"] is None

def test_parameter_count_is_weights_not_tokens():
 model=OpenHIM(OpenHIMConfig(context_length=4,embedding_dim=8,hidden_dim=12))
 assert model.num_parameters==sum(x.size for x in model.weights.values())
 assert model.num_parameters>272

def test_checkpoint_round_trip(tmp_path):
 model=OpenHIM(OpenHIMConfig(context_length=4,embedding_dim=8,hidden_dim=12))
 model.save(tmp_path,{"open_weights":True}); loaded=OpenHIM.load(tmp_path)
 assert loaded.num_parameters==model.num_parameters
 assert loaded.config==model.config

def test_native_generator_reports_exact_byte_token_usage(tmp_path):
 model=OpenHIM(OpenHIMConfig(context_length=4,embedding_dim=8,hidden_dim=12)); model.save(tmp_path,{"open_weights":True})
 result=NativeOpenWeightGenerator(str(tmp_path))([{"role":"user","content":"hello"}],{"max_tokens":4,"temperature":.5})
 assert result["provider"]=="repository-native-open-weights"
 assert result["usage"]["total_tokens"]==result["usage"]["prompt_tokens"]+result["usage"]["completion_tokens"]
