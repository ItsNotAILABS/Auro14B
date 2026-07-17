"""Scriptural Systems Architecture — executable doctrine for Auro LLMs.

Libraries:
  - canon: load/hash doctrine (scripture that binds)
  - gates: five-gate machine
  - executor: symbolic execution of ops against canon
  - memory: scriptural memory embedded in model runtime
  - governance: inner AI governance (refuse / allow / annotate)
  - substrate: live runtime wrapping generate / train / dispatch

Principle: sufficiently integrated symbols construct behavior, memory,
relationships, and possible world — they do not merely describe them.
"""

from auro_native_llm.scripture.canon import Canon, Article, load_canon, default_canon_path
from auro_native_llm.scripture.executor import (
    ScripturalExecutor,
    ExecutionVerdict,
    Operation,
)
from auro_native_llm.scripture.gates import GateId, GateMachine, GateResult
from auro_native_llm.scripture.memory import ScripturalMemory, MemoryRecord
from auro_native_llm.scripture.governance import InnerGovernance, GovernanceDecision
from auro_native_llm.scripture.substrate import ScripturalSubstrate, SubstrateResult
from auro_native_llm.scripture.rules_engine import RulesEngine, RuleVerdict
from auro_native_llm.scripture.process_model import ProcessModel
from auro_native_llm.scripture.hooks import EnforcementHooks, HookContext, HookResult
from auro_native_llm.scripture.agent_loop import (
    StructuredCognitiveLoop,
    CognitiveLoopResult,
)
from auro_native_llm.scripture.constitutional import (
    ConstitutionalEngine,
    ConstitutionalResult,
    hybrid_pipeline,
)

__all__ = [
    "Article",
    "Canon",
    "CognitiveLoopResult",
    "ConstitutionalEngine",
    "ConstitutionalResult",
    "EnforcementHooks",
    "ExecutionVerdict",
    "GateId",
    "GateMachine",
    "GateResult",
    "GovernanceDecision",
    "HookContext",
    "HookResult",
    "InnerGovernance",
    "MemoryRecord",
    "Operation",
    "ProcessModel",
    "RuleVerdict",
    "RulesEngine",
    "ScripturalExecutor",
    "ScripturalMemory",
    "ScripturalSubstrate",
    "StructuredCognitiveLoop",
    "SubstrateResult",
    "default_canon_path",
    "hybrid_pipeline",
    "load_canon",
]
