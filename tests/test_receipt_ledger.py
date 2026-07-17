import json
from auro_native_llm.production_fleet.receipts import ReceiptLedger

def test_receipts_form_verified_chain(tmp_path=None):
 ledger=ReceiptLedger(); first=ledger.record("capability","brain.state",True,{"beat":1}); second=ledger.record("capability","memory.rank",True,{"top":0})
 assert first.previous_hash=="GENESIS"
 assert second.previous_hash==first.receipt_hash
 assert ledger.verify()["valid"] is True

def test_persisted_chain_reloads(tmp_path=None):
 import tempfile
 from pathlib import Path
 with tempfile.TemporaryDirectory() as d:
  path=Path(d)/"ledger.jsonl"; ledger=ReceiptLedger(path); ledger.record("test","one",True,{"x":1})
  loaded=ReceiptLedger(path); assert loaded.verify()["count"]==1

def test_tampered_chain_is_rejected():
 import tempfile
 from pathlib import Path
 with tempfile.TemporaryDirectory() as d:
  path=Path(d)/"ledger.jsonl"; ReceiptLedger(path).record("test","one",True,{"x":1})
  data=json.loads(path.read_text()); data["subject"]="changed"; path.write_text(json.dumps(data)+"\n")
  try: ReceiptLedger(path)
  except ValueError: pass
  else: raise AssertionError("tampered ledger must fail")
