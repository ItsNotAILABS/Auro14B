from auro_native_llm.production_fleet.browser_gateway import BrowserTaskBroker

def test_browser_task_round_trip_is_persistent_and_receipted(tmp_path):
 broker=BrowserTaskBroker(tmp_path/"tasks.json");created=broker.enqueue("research",{"objective":"study memory"});claimed=broker.claim("chrome-1")
 assert claimed["id"]==created["id"] and claimed["status"]=="claimed"
 done=broker.complete(created["id"],{"finding":"memory is persistent"});assert done["status"]=="completed" and len(done["receipt_hash"])==64
 restored=BrowserTaskBroker(tmp_path/"tasks.json");assert restored.get(created["id"])["receipt_hash"]==done["receipt_hash"]

def test_browser_gateway_rejects_unknown_work(tmp_path):
 broker=BrowserTaskBroker(tmp_path/"tasks.json")
 try:broker.enqueue("arbitrary-shell",{})
 except ValueError as exc:assert "unsupported" in str(exc)
 else:raise AssertionError("unknown browser work must be denied")
