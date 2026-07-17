from auro_native_llm.production_fleet.server import token_authorized

def test_execution_token_is_fail_closed():
 assert token_authorized("Bearer secret","") is False
 assert token_authorized("","secret") is False
 assert token_authorized("secret","secret") is False

def test_execution_token_accepts_exact_bearer_only():
 assert token_authorized("Bearer secret","secret") is True
 assert token_authorized("Bearer wrong","secret") is False
