"""Workers for AURO durable task runs.

The council worker handles analysis, research, writing, review, and synthesis.
It does not claim shell, code execution, builds, deployments, or tests. Those
capabilities belong to separately governed POCKET Agent or runtime-cell workers.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Mapping, Sequence

from .task_runtime import DurableTaskRuntime


COUNCIL_CAPABILITIES = (
    "analysis",
    "research",
    "write",
    "review",
    "synthesis",
    "artifact-validation",
)
COUNCIL_KINDS = {"analysis", "research", "write", "review", "synthesize", "other"}


def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _bounded_dependency_context(run: Mapping[str, Any], step: Mapping[str, Any]) -> list[dict[str, Any]]:
    dependencies = set(step.get("dependencies") or [])
    artifact_by_step: dict[str, list[dict[str, Any]]] = {}
    for artifact in run.get("artifacts") or []:
        artifact_by_step.setdefault(str(artifact.get("step_id")), []).append(
            {
                "artifact_id": artifact.get("artifact_id"),
                "name": artifact.get("name"),
                "media_type": artifact.get("media_type"),
                "bytes": artifact.get("bytes"),
                "sha256": artifact.get("sha256"),
            }
        )
    output = []
    for candidate in run.get("steps") or []:
        if candidate.get("step_id") not in dependencies:
            continue
        output.append(
            {
                "step_id": candidate.get("step_id"),
                "title": candidate.get("title"),
                "status": candidate.get("status"),
                "output": candidate.get("output"),
                "validation": candidate.get("validation"),
                "artifacts": artifact_by_step.get(str(candidate.get("step_id")), []),
            }
        )
    encoded = json.dumps(output, ensure_ascii=False)
    if len(encoded) > 48_000:
        encoded = encoded[:47_999] + "…"
        return [{"truncated_dependency_context": encoded}]
    return output


@dataclass
class CouncilStepExecutor:
    council_service: Any

    @property
    def capabilities(self) -> tuple[str, ...]:
        return COUNCIL_CAPABILITIES

    def execute(
        self,
        run: Mapping[str, Any],
        step: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        if not getattr(self.council_service, "configured", False):
            raise RuntimeError("Auro-2B council is not configured")
        if str(step.get("kind")) not in COUNCIL_KINDS:
            raise RuntimeError(f"council worker does not execute task kind {step.get('kind')}")
        required = set(step.get("required_capabilities") or [])
        unsupported = required - set(self.capabilities)
        if unsupported:
            raise RuntimeError(f"council worker lacks required capabilities: {sorted(unsupported)}")

        prompt = {
            "instruction": (
                "Complete one bounded task. Return a user-deliverable result, concise reasoning "
                "summary, decisions, evidence references, limitations, and recommended next actions. "
                "Do not expose hidden chain-of-thought. Do not claim tools, code, tests, builds, "
                "deployments, memory, or training executed unless the dependency evidence says so."
            ),
            "run": {
                "run_id": run.get("run_id"),
                "objective": run.get("objective"),
                "quality_mode": run.get("quality_mode"),
                "plan_sha256": run.get("plan_sha256"),
                "reasoning_summary": (run.get("plan") or {}).get("reasoning_summary", []),
                "assumptions": (run.get("plan") or {}).get("assumptions", []),
            },
            "step": {
                "step_id": step.get("step_id"),
                "title": step.get("title"),
                "objective": step.get("objective"),
                "kind": step.get("kind"),
                "reasoning_depth": step.get("reasoning_depth"),
                "artifact_contract": step.get("artifact_contract", []),
                "validation_contract": step.get("validation_contract", {}),
            },
            "dependency_context": _bounded_dependency_context(run, step),
        }
        response = self.council_service.respond(json.dumps(prompt, ensure_ascii=False))
        text = str(response.get("text") or "").strip()
        structured = response.get("structured_answer")
        if not isinstance(structured, Mapping):
            structured = {"answer": text}
        answer = str(structured.get("answer") or text).strip()
        if not answer:
            raise RuntimeError("council returned no deliverable answer")

        output = {
            "summary": answer,
            "reasoning_summary": _strings(
                structured.get("reasoning_summary")
                or structured.get("key_points")
                or []
            ),
            "decisions": _strings(structured.get("recommendations") or []),
            "limitations": _strings(structured.get("caveats") or []),
            "evidence_refs": _strings(structured.get("citations") or []),
            "confidence": structured.get("confidence"),
            "council": {
                "turn_id": response.get("turn_id"),
                "evidence_class": response.get("evidence_class"),
                "runtime_receipt_sha256": (response.get("runtime_receipt") or {}).get(
                    "receipt_sha256"
                ),
                "release_evidence_ready": response.get("release_evidence_ready", False),
                "blockers": response.get("blockers", []),
                "composition_is_not_one_checkpoint": True,
            },
            "private_chain_of_thought_exported": False,
        }

        artifacts: list[dict[str, Any]] = []
        requirements = list(step.get("artifact_contract") or [])
        if not requirements:
            requirements = [
                {
                    "name": f"{step.get('step_id')}-RESULT.md",
                    "media_type": "text/markdown",
                    "required": False,
                }
            ]
        for requirement in requirements:
            name = str(requirement.get("name") or "RESULT.md")
            media_type = str(requirement.get("media_type") or "text/markdown")
            if media_type == "application/json":
                artifacts.append(
                    {
                        "name": name,
                        "media_type": media_type,
                        "json": output,
                        "metadata": {
                            "generated_by": "Auro-2B council worker",
                            "step_id": step.get("step_id"),
                        },
                    }
                )
            elif media_type.startswith("text/") or name.lower().endswith(
                (".md", ".txt", ".py", ".js", ".ts", ".jsonl", ".yaml", ".yml")
            ):
                markdown = [
                    f"# {step.get('title')}",
                    "",
                    answer,
                    "",
                    "## Reasoning summary",
                    *[f"- {item}" for item in output["reasoning_summary"]],
                    "",
                    "## Decisions",
                    *[f"- {item}" for item in output["decisions"]],
                    "",
                    "## Limitations",
                    *[f"- {item}" for item in output["limitations"]],
                    "",
                    "## Evidence",
                    *[f"- {item}" for item in output["evidence_refs"]],
                    "",
                    "_Generated by the configured AURO council. This artifact does not prove external execution._",
                ]
                artifacts.append(
                    {
                        "name": name,
                        "media_type": media_type,
                        "content": "\n".join(markdown).strip() + "\n",
                        "metadata": {
                            "generated_by": "Auro-2B council worker",
                            "step_id": step.get("step_id"),
                        },
                    }
                )
            else:
                raise RuntimeError(
                    f"council worker cannot create binary artifact contract {name} ({media_type})"
                )

        validation = {
            "passed": True,
            "validator": "council-response-and-artifact-contract",
            "external_execution_proven": False,
            "model_quality_proven": False,
            "private_chain_of_thought_exported": False,
        }
        return output, artifacts, validation


class CouncilTaskWorker:
    """Lease and complete reasoning-oriented tasks through the Auro council."""

    def __init__(
        self,
        runtime: DurableTaskRuntime,
        executor: CouncilStepExecutor,
        *,
        run_id: str,
        principal_id: str,
        organization_id: str | None = None,
        worker_id: str = "auro-council-worker",
        lease_seconds: int = 900,
    ) -> None:
        self.runtime = runtime
        self.executor = executor
        self.run_id = run_id
        self.principal_id = principal_id
        self.organization_id = organization_id
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds

    def run_once(self) -> dict[str, Any] | None:
        step = self.runtime.claim_step(
            self.run_id,
            worker_id=self.worker_id,
            capabilities=self.executor.capabilities,
            lease_seconds=self.lease_seconds,
        )
        if step is None:
            return None
        try:
            run = self.runtime.get_run(
                self.run_id,
                principal_id=self.principal_id,
                organization_id=self.organization_id,
            )
            self.runtime.progress(
                self.run_id,
                step["step_id"],
                worker_id=self.worker_id,
                lease_token=step["lease_token"],
                progress={
                    "phase": "council-reasoning",
                    "message": "Auro council is producing the bounded deliverable.",
                    "private_chain_of_thought_exported": False,
                },
            )
            output, artifacts, validation = self.executor.execute(run, step)
            completed = self.runtime.complete_step(
                self.run_id,
                step["step_id"],
                worker_id=self.worker_id,
                lease_token=step["lease_token"],
                output=output,
                artifacts=artifacts,
                validation=validation,
            )
            return {"ok": True, "step": completed}
        except Exception as exc:
            failed = self.runtime.fail_step(
                self.run_id,
                step["step_id"],
                worker_id=self.worker_id,
                lease_token=step["lease_token"],
                error=f"{type(exc).__name__}: {str(exc)[:1000]}",
                retry_delay_seconds=30,
            )
            return {"ok": False, "step": failed, "error": str(exc)[:1000]}

    def run_until_idle(
        self,
        *,
        max_steps: int = 100,
        idle_rounds: int = 2,
        idle_sleep_seconds: float = 0.25,
    ) -> dict[str, Any]:
        completed: list[dict[str, Any]] = []
        idle = 0
        for _ in range(max(1, int(max_steps))):
            state = self.runtime.get_run(
                self.run_id,
                principal_id=self.principal_id,
                organization_id=self.organization_id,
            )
            if state["status"] in {"succeeded", "partial", "failed", "cancelled"}:
                break
            result = self.run_once()
            if result is None:
                idle += 1
                if idle >= max(1, int(idle_rounds)):
                    break
                time.sleep(max(0.0, float(idle_sleep_seconds)))
                continue
            idle = 0
            completed.append(result)
        final = self.runtime.get_run(
            self.run_id,
            principal_id=self.principal_id,
            organization_id=self.organization_id,
        )
        return {
            "schema": "auro.task-worker.run.v1",
            "run_id": self.run_id,
            "worker_id": self.worker_id,
            "completed_attempts": completed,
            "final_status": final["status"],
            "progress": final["progress"],
        }
