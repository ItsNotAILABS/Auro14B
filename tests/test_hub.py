"""Tests for the hub module: server, sessions, connectors, schema."""

from mesie.hub.server import ResearchHub, HubConfig
from mesie.hub.session import HubSession, SessionManager
from mesie.hub.connectors import CLIConnector, JupyterConnector, WebSocketConnector, ConnectorRegistry
from mesie.hub.schema import HubSchema, ToolSchema, LabSchema


class TestSessionManager:
    def test_create_session(self):
        mgr = SessionManager()
        session = mgr.create(user="test_user")
        assert session.user == "test_user"
        assert session.session_id

    def test_get_session(self):
        mgr = SessionManager()
        session = mgr.create()
        retrieved = mgr.get(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_close_session(self):
        mgr = SessionManager()
        session = mgr.create()
        assert mgr.close(session.session_id) is True
        assert mgr.get(session.session_id) is None

    def test_record_action(self):
        session = HubSession()
        session.record_action("test.action", {"key": "value"}, "result")
        assert len(session.history) == 1
        assert session.history[0]["action"] == "test.action"


class TestConnectors:
    def test_cli_connector(self):
        conn = CLIConnector()
        assert conn.name == "cli"
        assert conn.is_connected()
        result = conn.send({"action": "test"})
        assert result["status"] == "dispatched"

    def test_jupyter_connector(self):
        def callback(msg):
            return {"echo": msg}

        conn = JupyterConnector(hub_callback=callback)
        assert conn.name == "jupyter"
        result = conn.send({"hello": "world"})
        assert result["echo"]["hello"] == "world"

    def test_websocket_connector(self):
        conn = WebSocketConnector()
        assert not conn.is_connected()
        conn.connect()
        assert conn.is_connected()
        result = conn.send({"data": 1})
        assert result["status"] == "queued"

    def test_connector_registry(self):
        registry = ConnectorRegistry()
        registry.register(CLIConnector())
        registry.register(JupyterConnector())
        assert "cli" in registry.names
        assert "jupyter" in registry.names


class TestHubSchema:
    def test_register_and_list_tools(self):
        schema = HubSchema()
        schema.register_tool(ToolSchema(
            tool_id="test-tool", name="Test", description="A test tool",
        ))
        tools = schema.list_tools()
        assert len(tools) == 1
        assert tools[0]["tool_id"] == "test-tool"

    def test_openapi_spec(self):
        schema = HubSchema()
        schema.register_lab(LabSchema(
            domain="test", name="Test Lab", capabilities=["op1"],
        ))
        spec = schema.openapi_spec()
        assert spec["openapi"] == "3.1.0"
        assert "/labs/test" in spec["paths"]


class TestResearchHub:
    def test_hub_info(self):
        hub = ResearchHub()
        hub.start()
        info = hub.info()
        assert "MESIE Research Hub" == info["name"]
        assert "spectral" in info["labs"]

    def test_run_lab(self):
        hub = ResearchHub()
        result = hub.run_lab("chemistry", "formula_parse", formula="NaCl")
        assert result["status"] == "success"
        assert result["data"]["elements"]["Na"] == 1

    def test_research(self):
        hub = ResearchHub()
        result = hub.research("What is spectral analysis?")
        assert result["status"] in ("completed", "failed")

    def test_handle_request_lab(self):
        hub = ResearchHub()
        result = hub.handle_request({
            "type": "lab",
            "params": {"domain": "physics", "operation": "constants", "name": "h"},
        })
        assert result["status"] == "success"
        assert result["data"]["value"] == 6.62607015e-34

    def test_handle_request_info(self):
        hub = ResearchHub()
        hub.start()
        result = hub.handle_request({"type": "info", "params": {}})
        assert "labs" in result

    def test_session_tracking(self):
        hub = ResearchHub()
        session = hub.create_session(user="scientist")
        hub.run_lab("earth", "seismic_magnitude", session_id=session.session_id, magnitude=5.0)
        retrieved = hub.get_session(session.session_id)
        assert len(retrieved.history) == 1
