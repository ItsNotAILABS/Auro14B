from auro_native_llm.production_fleet.organ_sdk import AuroOrganSDK

class FakeMat:
    def call(self,name,arguments): return {"name":name,"arguments":arguments}
class FakeCaps:
    def create_session(self,**kwargs): return {"session":kwargs}
class FakeHTTP: pass

def sdk():
    value=AuroOrganSDK.__new__(AuroOrganSDK); value.matdaemon=FakeMat(); value.capsula=FakeCaps(); return value

def test_manifest_names_four_organs():
    assert set(sdk().manifest()["organs"]) == {"brain","nova","matdaemon","capsula"}

def test_bounded_dispatch():
    out=sdk().execute({"tool":"matdaemon","arguments":{"name":"matdaemon_backend_status","arguments":{}}})
    assert out["ok"] is True
    out=sdk().execute({"tool":"capsula","arguments":{"operation":"create_session","parameters":{"runtime":"python"}}})
    assert out["output"]["session"]["runtime"] == "python"

def test_contract_is_machine_promptable():
    contract=sdk().action_contract()
    assert contract["matdaemon"]["arguments"]["name"].startswith("matdaemon_")
    assert "deploy_plan" in contract["capsula"]["arguments"]["operation"]
