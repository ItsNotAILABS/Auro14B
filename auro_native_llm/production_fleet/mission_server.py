"""Auro production API with durable multi-task mission routes.

This handler subclasses the existing Auro/HIM/council server so mission support
is additive. Existing response, council, model, context, and receipt routes are
preserved unchanged.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
from typing import Any
from urllib.parse import unquote, urlsplit

from .server import Handler, ThreadingHTTPServer, production_mode, production_security_status
from .task_service import MissionAuthorizationError, MissionService


class MissionHandler(Handler):
    mission_service: MissionService | None = None
    server_version = "AuroHIM/2.3-missions"

    @classmethod
    def get_mission_service(cls) -> MissionService:
        if cls.mission_service is None:
            cls.mission_service = MissionService.from_env(cls.get_council_service())
        return cls.mission_service

    def do_GET(self):
        path = urlsplit(self.path).path
        if not path.startswith("/v1/missions"):
            return super().do_GET()
        self.request_id = self._request_id()
        self._require_api_auth()
        operator_id, organization_id = self._mission_identity()
        service = self.get_mission_service()
        try:
            if path == "/v1/missions/status":
                return self._json(200, service.status())
            if path == "/v1/missions":
                return self._json(
                    200,
                    {
                        "schema": "auro.mission.list.v1",
                        "missions": service.list(
                            operator_id=operator_id,
                            organization_id=organization_id,
                            limit=50,
                        ),
                    },
                )
            parts = [part for part in path.split("/") if part]
            if len(parts) >= 3:
                mission_id = parts[2]
                if len(parts) >= 5 and parts[3] == "artifacts":
                    relative = unquote("/".join(parts[4:]))
                    artifact = service.artifact_path(
                        mission_id,
                        relative,
                        operator_id=operator_id,
                        organization_id=organization_id,
                    )
                    content_type = mimetypes.guess_type(artifact.name)[0] or "application/octet-stream"
                    return self._bytes(200, content_type, artifact.read_bytes())
                if len(parts) == 3:
                    return self._json(
                        200,
                        service.get(
                            mission_id,
                            operator_id=operator_id,
                            organization_id=organization_id,
                        ),
                    )
            raise self._mission_error(404, "mission_route_not_found", "Mission route not found")
        except MissionAuthorizationError as exc:
            raise self._mission_error(403, "mission_forbidden", str(exc)) from exc
        except KeyError as exc:
            raise self._mission_error(404, "mission_not_found", str(exc)) from exc
        except FileNotFoundError as exc:
            raise self._mission_error(404, "artifact_not_found", str(exc)) from exc

    def do_POST(self):
        path = urlsplit(self.path).path
        if not path.startswith("/v1/missions"):
            return super().do_POST()
        self.request_id = self._request_id()
        self._require_api_auth()
        operator_id, organization_id = self._mission_identity()
        service = self.get_mission_service()
        body = self._body()
        try:
            if path == "/v1/missions":
                mission = service.create(
                    body,
                    operator_id=operator_id,
                    organization_id=organization_id,
                )
                return self._json(201, mission)
            parts = [part for part in path.split("/") if part]
            if len(parts) == 4:
                mission_id, action = parts[2], parts[3]
                self._require_execution_auth()
                if action == "run":
                    result = service.run(
                        mission_id,
                        body,
                        operator_id=operator_id,
                        organization_id=organization_id,
                    )
                    return self._json(200, result)
                if action in {"pause", "resume", "cancel"}:
                    result = service.transition(
                        mission_id,
                        action,
                        operator_id=operator_id,
                        organization_id=organization_id,
                    )
                    return self._json(200, result)
            raise self._mission_error(404, "mission_route_not_found", "Mission route not found")
        except MissionAuthorizationError as exc:
            raise self._mission_error(403, "mission_forbidden", str(exc)) from exc
        except KeyError as exc:
            raise self._mission_error(404, "mission_not_found", str(exc)) from exc

    def _mission_identity(self) -> tuple[str, str]:
        operator = self.headers.get("x-auro-operator-id", "").strip()
        organization = self.headers.get("x-auro-organization-id", "").strip()
        if not operator or not organization:
            raise self._mission_error(
                401,
                "mission_identity_required",
                "x-auro-operator-id and x-auro-organization-id are required",
            )
        return operator, organization

    @staticmethod
    def _mission_error(status: int, code: str, message: str):
        from .server import ApiError

        return ApiError(status, code, message)


def discovery() -> dict[str, Any]:
    return {
        "schema": "auro.mission-api.v1",
        "create": "POST /v1/missions",
        "list": "GET /v1/missions",
        "status": "GET /v1/missions/status",
        "inspect": "GET /v1/missions/{mission_id}",
        "run_burst": "POST /v1/missions/{mission_id}/run",
        "pause": "POST /v1/missions/{mission_id}/pause",
        "resume": "POST /v1/missions/{mission_id}/resume",
        "cancel": "POST /v1/missions/{mission_id}/cancel",
        "artifact": "GET /v1/missions/{mission_id}/artifacts/{relative_path}",
        "identity_headers": [
            "x-auro-operator-id",
            "x-auro-organization-id",
        ],
        "mutation_auth": "x-auro-execution-token or execution bearer token",
        "long_running": "call run in bounded bursts; state, retries, and artifacts persist",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve Auro14B/HIM, council, and durable mission APIs"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    security = production_security_status(args.host)
    if production_mode() and not security["ready"]:
        missing = [name for name, ok in security["secret_checks"].items() if not ok]
        detail = ", ".join(missing) if missing else "production secrets must be distinct"
        raise SystemExit(f"AURO production security configuration invalid: {detail}")
    server = ThreadingHTTPServer((args.host, args.port), MissionHandler)
    print(json.dumps({"service": "auro-mission-api", "listen": f"http://{args.host}:{args.port}", "routes": discovery()}))
    server.serve_forever()


if __name__ == "__main__":
    main()
