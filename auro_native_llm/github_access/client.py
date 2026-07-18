"""GitHub access layer — local `gh` + MCP-compatible status.

Sign-in is done with the GitHub CLI (browser OAuth). MCP server `github`
uses the session already linked to Grok / GitHub App credentials.

Never print full tokens.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _run(cmd: List[str], timeout: float = 30.0) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except FileNotFoundError:
        return 127, "", "gh not found"
    except Exception as exc:
        return 1, "", str(exc)[:300]


@dataclass
class GitHubAccess:
    """Inspect and use authenticated GitHub from the machine."""

    gh_path: str = field(default_factory=lambda: shutil.which("gh") or "")

    def status(self) -> Dict[str, Any]:
        if not self.gh_path:
            return {
                "ok": False,
                "signed_in": False,
                "status": "TECHNICALLY_UNAVAILABLE",
                "error": "GitHub CLI (gh) not on PATH",
                "sign_in": "Install: https://cli.github.com/ then run: gh auth login",
                "sign_up": "https://github.com/signup",
                "mcp": {
                    "server": "github",
                    "note": "Grok MCP GitHub server still works if linked in Grok settings",
                },
            }
        code, out, err = _run([self.gh_path, "auth", "status"])
        signed_in = code == 0 and "Logged in" in (out + err)
        who = self.whoami() if signed_in else {}
        return {
            "ok": signed_in,
            "signed_in": signed_in,
            "status": "AVAILABLE" if signed_in else "NOT_CURRENTLY_CONFIGURED",
            "cli": self.gh_path,
            "auth_status_text": (out or err)[:800],
            "user": who,
            "sign_in_commands": [
                "gh auth login -h github.com -p https -w",
                "gh auth login -h github.com -p ssh -w",
                "gh auth refresh -h github.com -s repo,read:org,gist",
            ],
            "sign_up": "https://github.com/signup",
            "mcp": {
                "server": "github",
                "tools_examples": [
                    "github__get_me",
                    "github__list_pull_requests",
                    "github__create_pull_request",
                    "github__search_users",
                    "github__list_notifications",
                ],
                "note": (
                    "MCP is already connected in this Grok session when status is ready. "
                    "CLI and MCP may share the same GitHub identity."
                ),
            },
            "scopes_hint": "repo, read:org, gist, admin:public_key (as granted)",
        }

    def whoami(self) -> Dict[str, Any]:
        code, out, err = _run([self.gh_path, "api", "user"])
        if code != 0:
            return {"ok": False, "error": err or out}
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            return {"ok": False, "error": "bad json", "raw": out[:200]}
        return {
            "ok": True,
            "login": data.get("login"),
            "id": data.get("id"),
            "name": data.get("name"),
            "company": data.get("company"),
            "html_url": data.get("html_url"),
            "public_repos": data.get("public_repos"),
        }

    def orgs(self) -> Dict[str, Any]:
        code, out, err = _run([self.gh_path, "api", "user/orgs"])
        if code != 0:
            return {"ok": False, "error": err or out}
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            return {"ok": False, "error": "bad json"}
        return {"ok": True, "orgs": [o.get("login") for o in data]}

    def list_repos(self, owner: Optional[str] = None, limit: int = 15) -> Dict[str, Any]:
        if owner:
            code, out, err = _run(
                [self.gh_path, "repo", "list", owner, "--limit", str(limit), "--json", "name,url,isPrivate,updatedAt"]
            )
        else:
            code, out, err = _run(
                [self.gh_path, "repo", "list", "--limit", str(limit), "--json", "name,url,isPrivate,updatedAt"]
            )
        if code != 0:
            return {"ok": False, "error": err or out}
        try:
            repos = json.loads(out) if out else []
        except json.JSONDecodeError:
            return {"ok": False, "error": "bad json", "raw": out[:300]}
        return {"ok": True, "owner": owner or "authenticated", "repos": repos}

    def login_hint(self) -> Dict[str, Any]:
        return {
            "sign_in": {
                "browser_oauth": "gh auth login -h github.com -p https -w",
                "ssh": "gh auth login -h github.com -p ssh -w",
                "token": "gh auth login -h github.com -p https -t  # paste PAT",
            },
            "sign_up": {
                "web": "https://github.com/signup",
                "after_signup": "gh auth login -h github.com -p https -w",
            },
            "mcp": {
                "grok": "Ensure GitHub MCP / connector is enabled in Grok settings and authorized",
                "verify": "Ask agent to call github__get_me",
            },
            "security": "Never paste PATs into chat logs; use gh auth login or OS keyring.",
        }


def github_status() -> Dict[str, Any]:
    return GitHubAccess().status()
