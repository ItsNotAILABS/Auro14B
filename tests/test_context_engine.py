from auro_native_llm.production_fleet.context_engine import ContextEngine

def test_million_token_logical_store_retrieves_bounded_working_set(tmp_path):
    engine=ContextEngine(tmp_path/"context.sqlite",default_budget=1200)
    text=("alpha architecture establishes owner controlled signing and receipt lineage. "*60_000)
    result=engine.ingest(text,source="architecture.md",chunk_tokens=700,overlap_tokens=60)
    pack=engine.retrieve("owner signing receipt",token_budget=1200)
    assert result["tokens"]>=1_000_000
    assert pack.logical_tokens>=1_000_000
    assert 0<pack.injected_tokens<=1200
    assert "architecture.md" in pack.context
    assert len(pack.receipt_hash)==64

def test_context_is_persistent_deduplicated_and_ranked(tmp_path):
    path=tmp_path/"context.sqlite"
    first=ContextEngine(path)
    a=first.ingest("Python context virtualization and ranking",source="a",importance=.9)
    assert first.ingest("Python context virtualization and ranking",source="a")["deduplicated"]
    first.close()
    second=ContextEngine(path)
    pack=second.retrieve("Python ranking")
    assert pack.hits[0].document_id==a["document_id"]
    assert second.stats()["documents"]==1

def test_context_secret_scan_is_fail_closed(tmp_path):
    engine=ContextEngine(tmp_path/"context.sqlite")
    try:engine.ingest("-----BEGIN PRIVATE KEY-----\nsecret",source="bad")
    except ValueError as exc:assert "secret" in str(exc)
    else:raise AssertionError("secret-bearing context must be rejected")
