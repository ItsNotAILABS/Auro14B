import hashlib, tempfile
from pathlib import Path
from zipfile import ZipFile
from auro_native_llm.production_fleet.wallet import PaperWallet
from auro_native_llm.production_fleet.office import NativeOffice
from auro_native_llm.production_fleet.vault import IntegrityVault

def test_wallet_is_double_entry_paper_only():
 w=PaperWallet(); w.fund("AURO", "100.00"); w.transfer("AURO","AGENT:MATHESIS","12.50")
 assert w.balance("AURO")=="87.50000000" and w.balance("AGENT:MATHESIS")=="12.50000000"
 status=w.verify(); assert status["valid"] and status["live_custody"] is False

def test_wallet_blocks_overdraft():
 w=PaperWallet()
 try: w.transfer("AURO","AGENT",1)
 except ValueError as exc: assert "insufficient" in str(exc)
 else: raise AssertionError("paper wallet must block overdraft")

def test_office_delivers_real_openable_formats():
 with tempfile.TemporaryDirectory() as d:
  bundle=NativeOffice().create_bundle(d,"Auro Delivery",[{"heading":"Outcome","body":"The internal suite delivered this packet."}],[["Item","Status"],["DOCX","PASS"]])
  paths={Path(x["path"]).suffix:Path(x["path"]) for x in bundle["files"]}
  with ZipFile(paths[".docx"]) as z: assert "word/document.xml" in z.namelist()
  with ZipFile(paths[".xlsx"]) as z: assert "xl/worksheets/sheet1.xml" in z.namelist()
  assert paths[".pdf"].read_bytes().startswith(b"%PDF-1.4")
  for record in bundle["files"]: assert hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()==record["sha256"]

def test_vault_is_integrity_storage_not_secret_custody():
 with tempfile.TemporaryDirectory() as d:
  vault=IntegrityVault(d); record=vault.put("report.txt",b"evidence", "text/plain")
  assert vault.verify(record)["valid"] and record["secret_custody"] is False and record["encrypted"] is False
