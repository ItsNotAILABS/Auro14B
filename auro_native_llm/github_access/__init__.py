"""GitHub access via gh CLI + MCP (sign-in status, whoami, orgs, repos)."""

from auro_native_llm.github_access.client import GitHubAccess, github_status

__all__ = ["GitHubAccess", "github_status"]
