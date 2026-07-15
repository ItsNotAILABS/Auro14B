"""Tokenomics Measurement and Benchmarking Framework.

Formalizes cognitive resource allocation measurement through five components:
    - Token Value Function (TV)
    - Cognitive Return Metrics (CRPT)
    - Salience Allocation Equations
    - Compression Efficiency Metrics
    - Benchmark Tasks comparing tokenomic and non-tokenomic systems

Key Components:
    - TokenValueFunction: Evaluate token contribution to task quality
    - CognitiveReturnMetrics: Measure useful cognition per token
    - SalienceAllocator: Rank and allocate token budget by importance
    - CompressionEfficiencyMetrics: Audit compression fidelity
    - TokenomicBenchmark: Compare tokenomic vs non-tokenomic systems
    - RuntimeMeasurementLoop: Self-evaluating feedback loop
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Enumerations
# =============================================================================


class TaskClass(Enum):
    """Benchmark task categories."""
    INVOICE_EXECUTION = "invoice_execution"
    ESTIMATING = "estimating"
    CASHFLOW_DECISION = "cashflow_decision"
    PROPOSAL_GENERATION = "proposal_generation"
    RESEARCH_SYNTHESIS = "research_synthesis"
    ARCHITECTURE_DESIGN = "architecture_design"
    RED_TEAM_REVIEW = "red_team_review"
    MEMORY_CONSOLIDATION = "memory_consolidation"


class EvaluationCriterion(Enum):
    """Evaluation criteria for mature tokenomic systems."""
    COGNITIVE_RETURN_PER_TOKEN = "cognitive_return_per_token"
    COMPRESSION_FIDELITY = "compression_fidelity"
    ACTION_CONVERSION_RATE = "action_conversion_rate"
    RISK_PRESERVATION = "risk_preservation"
    REUSE_EXTRACTION_RATE = "reuse_extraction_rate"
    CONTEXT_HYGIENE = "context_hygiene"
    ADAPTIVE_DEPTH_ACCURACY = "adaptive_depth_accuracy"
    ERROR_AVOIDANCE = "error_avoidance"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TokenValueWeights:
    """Task-specific weighting coefficients for token value computation.

    Attributes:
        w_d: Weight for decision value.
        w_a: Weight for action usefulness.
        w_r: Weight for risk reduction.
        w_c: Weight for compression contribution.
        w_m: Weight for memory/reuse value.
        w_n: Weight for noise penalty.
    """
    w_d: float = 1.0
    w_a: float = 1.0
    w_r: float = 1.0
    w_c: float = 1.0
    w_m: float = 1.0
    w_n: float = 1.0


@dataclass
class TokenScores:
    """Scores for a single token or token group.

    Attributes:
        decision_value: Decision quality contributed (D_t).
        action_usefulness: Action usefulness (A_t).
        risk_reduction: Risk reduction (R_t).
        compression_contribution: Compression contribution (C_t).
        memory_value: Memory or reuse value (M_t).
        noise: Noise, redundancy, or attention waste (N_t).
    """
    decision_value: float = 0.0
    action_usefulness: float = 0.0
    risk_reduction: float = 0.0
    compression_contribution: float = 0.0
    memory_value: float = 0.0
    noise: float = 0.0


@dataclass
class CognitiveReturnScores:
    """Cognitive return category scores (0-5 scale each).

    Attributes:
        decision_quality: Did the response improve the actual decision?
        actionability: Can the user or system act immediately?
        risk_control: Did the response identify or reduce failure modes?
        reuse_value: Did it create reusable rules, templates, or artifacts?
        learning_gain: Did the interaction improve future system behavior?
    """
    decision_quality: float = 0.0
    actionability: float = 0.0
    risk_control: float = 0.0
    reuse_value: float = 0.0
    learning_gain: float = 0.0

    def total(self) -> float:
        """Compute total cognitive return."""
        return (
            self.decision_quality
            + self.actionability
            + self.risk_control
            + self.reuse_value
            + self.learning_gain
        )


@dataclass
class SalienceWeights:
    """Task-specific weights for salience scoring.

    Attributes:
        alpha: Urgency weight.
        beta: Risk/consequence weight.
        gamma: Mission relevance weight.
        delta: Time sensitivity weight.
        epsilon: Novelty/uncertainty weight.
        zeta: Known context penalty weight.
    """
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 1.0
    epsilon: float = 1.0
    zeta: float = 1.0


@dataclass
class SalienceItem:
    """An information unit to be scored for salience allocation.

    Attributes:
        item_id: Unique identifier.
        urgency: Urgency score.
        risk: Risk or consequence score.
        mission_relevance: Mission relevance score.
        time_sensitivity: Time sensitivity score.
        novelty: Novelty or uncertainty score.
        known_context: Already-known or settled context score.
    """
    item_id: str
    urgency: float = 0.0
    risk: float = 0.0
    mission_relevance: float = 0.0
    time_sensitivity: float = 0.0
    novelty: float = 0.0
    known_context: float = 0.0


@dataclass
class CompressionResult:
    """Result of compression efficiency evaluation.

    Attributes:
        information_retained: Preservation of task-relevant content.
        action_clarity: Clarity of next step or decision.
        risk_preserved: Preservation of caution and constraints.
        output_tokens: Total output tokens used.
        efficiency: Computed compression efficiency factor.
    """
    information_retained: float = 0.0
    action_clarity: float = 0.0
    risk_preserved: float = 0.0
    output_tokens: int = 1
    efficiency: float = 0.0


@dataclass
class BenchmarkResult:
    """Result of a tokenomic benchmark comparison.

    Attributes:
        task_class: The task class evaluated.
        score_tokenomic: Score for the tokenomic system.
        tokens_tokenomic: Tokens used by the tokenomic system.
        score_baseline: Score for the non-tokenomic baseline.
        tokens_baseline: Tokens used by the baseline.
        tokenomic_gain: Computed tokenomic efficiency gain.
    """
    task_class: TaskClass
    score_tokenomic: float = 0.0
    tokens_tokenomic: int = 1
    score_baseline: float = 0.0
    tokens_baseline: int = 1
    tokenomic_gain: float = 0.0


@dataclass
class RuntimeLoopState:
    """State tracked by the runtime measurement loop.

    Attributes:
        task_classification: Classified task type.
        risk_estimate: Estimated task risk (0-1).
        complexity_estimate: Estimated complexity (0-1).
        salience_rankings: Ranked salience items.
        token_budget: Allocated total budget.
        modules_recruited: Active modules/agents.
        compression_quality: Post-generation compression score.
        cognitive_return: Post-generation CR score.
        wasted_tokens: Detected wasted tokens.
        extracted_rules: Reusable rules extracted.
        policy_update: Suggested policy adjustment.
    """
    task_classification: Optional[TaskClass] = None
    risk_estimate: float = 0.0
    complexity_estimate: float = 0.0
    salience_rankings: List[Tuple[str, float]] = field(default_factory=list)
    token_budget: int = 0
    modules_recruited: List[str] = field(default_factory=list)
    compression_quality: float = 0.0
    cognitive_return: float = 0.0
    wasted_tokens: int = 0
    extracted_rules: List[str] = field(default_factory=list)
    policy_update: Optional[Dict[str, Any]] = None


# =============================================================================
# Token Value Function (Section 15.1)
# =============================================================================


class TokenValueFunction:
    """Compute token value based on contribution to task quality.

    Each emitted token is treated as a unit of compute, attention,
    memory surface, and action influence. A token has positive value
    when it improves decision quality, enables action, reduces risk,
    compresses useful knowledge, or creates reusable memory.

    TV(t) = w_d*D_t + w_a*A_t + w_r*R_t + w_c*C_t + w_m*M_t - w_n*N_t

    Attributes:
        weights: Task-specific weighting coefficients.
    """

    def __init__(self, weights: Optional[TokenValueWeights] = None):
        """Initialize with optional custom weights.

        Args:
            weights: Token value weights. Uses equal weights if None.
        """
        self.weights = weights or TokenValueWeights()

    def compute(self, scores: TokenScores) -> float:
        """Compute token value for given scores.

        Args:
            scores: Token contribution scores.

        Returns:
            Token value (positive = useful, negative = wasteful).
        """
        w = self.weights
        return (
            w.w_d * scores.decision_value
            + w.w_a * scores.action_usefulness
            + w.w_r * scores.risk_reduction
            + w.w_c * scores.compression_contribution
            + w.w_m * scores.memory_value
            - w.w_n * scores.noise
        )

    def compute_batch(self, token_scores: List[TokenScores]) -> np.ndarray:
        """Compute token values for a sequence of tokens.

        Args:
            token_scores: List of token scores.

        Returns:
            Array of token values.
        """
        return np.array([self.compute(s) for s in token_scores])

    def total_value(self, token_scores: List[TokenScores]) -> float:
        """Compute total value across all tokens.

        Args:
            token_scores: List of token scores.

        Returns:
            Sum of all token values.
        """
        return float(self.compute_batch(token_scores).sum())

    def mean_value(self, token_scores: List[TokenScores]) -> float:
        """Compute mean token value.

        Args:
            token_scores: List of token scores.

        Returns:
            Mean token value.
        """
        values = self.compute_batch(token_scores)
        if len(values) == 0:
            return 0.0
        return float(values.mean())


# =============================================================================
# Cognitive Return Metrics (Section 15.2)
# =============================================================================


class CognitiveReturnMetrics:
    """Measure Cognitive Return Per Token (CRPT).

    CRPT = (DQ + ACT + RISK + REUSE + LEARN) / TotalTokens

    This rewards systems that produce compact but useful outputs and
    penalizes long outputs that don't improve action or judgment.
    """

    def cognitive_return(self, scores: CognitiveReturnScores) -> float:
        """Compute total cognitive return.

        Args:
            scores: Category scores (0-5 each).

        Returns:
            Total cognitive return (0-25 range).
        """
        return scores.total()

    def crpt(
        self,
        scores: CognitiveReturnScores,
        prompt_tokens: int,
        output_tokens: int,
    ) -> float:
        """Compute Cognitive Return Per Token.

        Args:
            scores: Cognitive return category scores.
            prompt_tokens: Number of prompt tokens.
            output_tokens: Number of output tokens.

        Returns:
            CRPT score.
        """
        total_tokens = prompt_tokens + output_tokens
        if total_tokens == 0:
            return 0.0
        return scores.total() / total_tokens

    def compare(
        self,
        scores_a: CognitiveReturnScores,
        tokens_a: int,
        scores_b: CognitiveReturnScores,
        tokens_b: int,
    ) -> float:
        """Compare CRPT between two systems.

        Args:
            scores_a: System A cognitive return scores.
            tokens_a: System A total tokens.
            scores_b: System B cognitive return scores.
            tokens_b: System B total tokens.

        Returns:
            Difference (positive = B is better).
        """
        crpt_a = scores_a.total() / tokens_a if tokens_a > 0 else 0.0
        crpt_b = scores_b.total() / tokens_b if tokens_b > 0 else 0.0
        return crpt_b - crpt_a


# =============================================================================
# Salience Allocation (Section 15.3)
# =============================================================================


class SalienceAllocator:
    """Rank information units and allocate token budget.

    S_i = α*U_i + β*R_i + γ*M_i + δ*T_i + ε*N_i - ζ*K_i
    B_i = B_total * (S_i / ΣS)

    Prevents low-value context from consuming high-value token space.

    Attributes:
        weights: Salience scoring weights.
    """

    def __init__(self, weights: Optional[SalienceWeights] = None):
        """Initialize with optional custom weights.

        Args:
            weights: Salience weights. Uses equal weights if None.
        """
        self.weights = weights or SalienceWeights()

    def score(self, item: SalienceItem) -> float:
        """Compute salience score for an information unit.

        Args:
            item: Information unit with feature scores.

        Returns:
            Salience score (can be negative for low-value items).
        """
        w = self.weights
        return (
            w.alpha * item.urgency
            + w.beta * item.risk
            + w.gamma * item.mission_relevance
            + w.delta * item.time_sensitivity
            + w.epsilon * item.novelty
            - w.zeta * item.known_context
        )

    def rank(self, items: List[SalienceItem]) -> List[Tuple[str, float]]:
        """Rank items by salience score (highest first).

        Args:
            items: Information units to rank.

        Returns:
            Sorted list of (item_id, score) tuples.
        """
        scored = [(item.item_id, self.score(item)) for item in items]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def allocate_budget(
        self,
        items: List[SalienceItem],
        total_budget: int,
    ) -> Dict[str, int]:
        """Allocate token budget proportionally to salience.

        B_i = B_total * (S_i / ΣS)

        Items with non-positive salience receive zero budget.

        Args:
            items: Information units.
            total_budget: Total available output token budget.

        Returns:
            Mapping of item_id to allocated token budget.
        """
        scores = [(item.item_id, max(0.0, self.score(item))) for item in items]
        total_salience = sum(s for _, s in scores)

        if total_salience == 0:
            # Equal allocation when all scores are zero
            per_item = total_budget // len(items) if items else 0
            return {item.item_id: per_item for item in items}

        allocation: Dict[str, int] = {}
        allocated = 0
        for item_id, s in scores:
            budget = int(total_budget * (s / total_salience))
            allocation[item_id] = budget
            allocated += budget

        # Distribute remainder to highest-salience item
        remainder = total_budget - allocated
        if remainder > 0 and scores:
            top_id = max(scores, key=lambda x: x[1])[0]
            allocation[top_id] += remainder

        return allocation


# =============================================================================
# Compression Efficiency Metrics (Section 15.4)
# =============================================================================


class CompressionEfficiencyMetrics:
    """Audit compression quality and fidelity.

    CEF = (InformationRetained + ActionClarity + RiskPreserved) / OutputTokens

    Good compression reduces surface length while preserving correct action.
    Bad compression merely deletes context and increases operational risk.
    """

    def compute(
        self,
        information_retained: float,
        action_clarity: float,
        risk_preserved: float,
        output_tokens: int,
    ) -> CompressionResult:
        """Compute compression efficiency factor.

        Args:
            information_retained: Preservation of task-relevant content (0-5).
            action_clarity: Clarity of next step or decision (0-5).
            risk_preserved: Preservation of caution/constraints (0-5).
            output_tokens: Total output tokens used.

        Returns:
            CompressionResult with computed efficiency.
        """
        if output_tokens <= 0:
            output_tokens = 1
        efficiency = (
            information_retained + action_clarity + risk_preserved
        ) / output_tokens
        return CompressionResult(
            information_retained=information_retained,
            action_clarity=action_clarity,
            risk_preserved=risk_preserved,
            output_tokens=output_tokens,
            efficiency=efficiency,
        )

    def passes_tokenomic_test(self, result: CompressionResult, threshold: float = 0.5) -> bool:
        """Check if compressed output passes the tokenomic quality test.

        A compressed output passes only if the user or downstream system
        can still act correctly (adequate action clarity and risk preservation).

        Args:
            result: Compression result to evaluate.
            threshold: Minimum acceptable score for each dimension (0-5).

        Returns:
            True if compression preserves adequate action and risk info.
        """
        return (
            result.action_clarity >= threshold
            and result.risk_preserved >= threshold
            and result.information_retained >= threshold
        )

    def compare(
        self,
        original_tokens: int,
        compressed_tokens: int,
        information_retained: float,
        action_clarity: float,
        risk_preserved: float,
    ) -> Dict[str, float]:
        """Compare original vs compressed output.

        Args:
            original_tokens: Tokens in original output.
            compressed_tokens: Tokens in compressed output.
            information_retained: Preservation score (0-5).
            action_clarity: Action clarity score (0-5).
            risk_preserved: Risk preservation score (0-5).

        Returns:
            Dictionary with compression ratio and efficiency metrics.
        """
        ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        result = self.compute(
            information_retained, action_clarity, risk_preserved, compressed_tokens
        )
        return {
            "compression_ratio": ratio,
            "token_reduction": 1.0 - ratio,
            "efficiency": result.efficiency,
            "passes_test": float(self.passes_tokenomic_test(result)),
        }


# =============================================================================
# Tokenomic Benchmark (Section 15.5)
# =============================================================================


class TokenomicBenchmark:
    """Compare tokenomic vs non-tokenomic systems on benchmark tasks.

    Score = DQ + ACT + RISK + REUSE + ACCURACY - WASTE
    TokenomicGain = (Score_B / Tokens_B) - (Score_A / Tokens_A)

    A tokenomic system is superior when it produces equal or higher
    task score with fewer tokens, or significantly higher task score
    with a justified increase in tokens.
    """

    def task_score(
        self,
        decision_quality: float,
        actionability: float,
        risk_control: float,
        reuse_value: float,
        accuracy: float,
        waste: float,
    ) -> float:
        """Compute benchmark task score.

        Score = DQ + ACT + RISK + REUSE + ACCURACY - WASTE

        Args:
            decision_quality: Decision quality score (0-5).
            actionability: Actionability score (0-5).
            risk_control: Risk control score (0-5).
            reuse_value: Reusable value score (0-5).
            accuracy: Factual/procedural correctness (0-5).
            waste: Unnecessary token expenditure (0-5).

        Returns:
            Net task score.
        """
        return (
            decision_quality
            + actionability
            + risk_control
            + reuse_value
            + accuracy
            - waste
        )

    def tokenomic_gain(
        self,
        score_tokenomic: float,
        tokens_tokenomic: int,
        score_baseline: float,
        tokens_baseline: int,
    ) -> float:
        """Compute tokenomic efficiency gain.

        TokenomicGain = (Score_B / Tokens_B) - (Score_A / Tokens_A)

        Args:
            score_tokenomic: Score for tokenomic system (B).
            tokens_tokenomic: Tokens used by tokenomic system.
            score_baseline: Score for baseline system (A).
            tokens_baseline: Tokens used by baseline system.

        Returns:
            Tokenomic gain (positive = tokenomic system is superior).
        """
        eff_b = score_tokenomic / tokens_tokenomic if tokens_tokenomic > 0 else 0.0
        eff_a = score_baseline / tokens_baseline if tokens_baseline > 0 else 0.0
        return eff_b - eff_a

    def run_comparison(
        self,
        task_class: TaskClass,
        scores_tokenomic: Dict[str, float],
        tokens_tokenomic: int,
        scores_baseline: Dict[str, float],
        tokens_baseline: int,
    ) -> BenchmarkResult:
        """Run a full benchmark comparison for a task.

        Args:
            task_class: Task category.
            scores_tokenomic: Score components for tokenomic system.
            tokens_tokenomic: Tokens used by tokenomic system.
            scores_baseline: Score components for baseline system.
            tokens_baseline: Tokens used by baseline system.

        Returns:
            BenchmarkResult with computed gain.
        """
        score_t = self.task_score(
            decision_quality=scores_tokenomic.get("decision_quality", 0.0),
            actionability=scores_tokenomic.get("actionability", 0.0),
            risk_control=scores_tokenomic.get("risk_control", 0.0),
            reuse_value=scores_tokenomic.get("reuse_value", 0.0),
            accuracy=scores_tokenomic.get("accuracy", 0.0),
            waste=scores_tokenomic.get("waste", 0.0),
        )
        score_b = self.task_score(
            decision_quality=scores_baseline.get("decision_quality", 0.0),
            actionability=scores_baseline.get("actionability", 0.0),
            risk_control=scores_baseline.get("risk_control", 0.0),
            reuse_value=scores_baseline.get("reuse_value", 0.0),
            accuracy=scores_baseline.get("accuracy", 0.0),
            waste=scores_baseline.get("waste", 0.0),
        )
        gain = self.tokenomic_gain(score_t, tokens_tokenomic, score_b, tokens_baseline)
        return BenchmarkResult(
            task_class=task_class,
            score_tokenomic=score_t,
            tokens_tokenomic=tokens_tokenomic,
            score_baseline=score_b,
            tokens_baseline=tokens_baseline,
            tokenomic_gain=gain,
        )

    def is_superior(self, result: BenchmarkResult) -> bool:
        """Determine if tokenomic system is superior in this benchmark.

        Args:
            result: Benchmark comparison result.

        Returns:
            True if tokenomic system demonstrates positive gain.
        """
        return result.tokenomic_gain > 0


# =============================================================================
# Runtime Measurement Loop (Section 15.6)
# =============================================================================


class RuntimeMeasurementLoop:
    """Self-evaluating runtime feedback loop for tokenomic systems.

    Implements the 11-step measurement loop:
    1. Classify the task
    2. Estimate task risk and complexity
    3. Rank salience targets
    4. Allocate token budget
    5. Recruit only necessary modules or agents
    6. Generate the response or artifact
    7. Audit compression quality
    8. Score cognitive return
    9. Detect wasted tokens
    10. Extract reusable rules or memory
    11. Update future token allocation policy

    Attributes:
        salience_allocator: Salience scoring engine.
        token_value_fn: Token value calculator.
        cognitive_metrics: CRPT calculator.
        compression_metrics: Compression auditor.
        history: List of loop states from past interactions.
    """

    def __init__(
        self,
        salience_weights: Optional[SalienceWeights] = None,
        token_value_weights: Optional[TokenValueWeights] = None,
    ):
        """Initialize the runtime loop.

        Args:
            salience_weights: Custom salience weights.
            token_value_weights: Custom token value weights.
        """
        self.salience_allocator = SalienceAllocator(salience_weights)
        self.token_value_fn = TokenValueFunction(token_value_weights)
        self.cognitive_metrics = CognitiveReturnMetrics()
        self.compression_metrics = CompressionEfficiencyMetrics()
        self.history: List[RuntimeLoopState] = []

    def classify_task(self, task_description: str) -> TaskClass:
        """Classify a task into a benchmark category.

        Uses keyword matching for classification. Override for
        ML-based classification.

        Args:
            task_description: Natural language task description.

        Returns:
            Best-matching TaskClass.
        """
        desc = task_description.lower()
        keywords = {
            TaskClass.INVOICE_EXECUTION: ["invoice", "payment", "billing", "hours"],
            TaskClass.ESTIMATING: ["estimate", "pricing", "quote", "labor"],
            TaskClass.CASHFLOW_DECISION: ["cashflow", "cash flow", "schedule", "payment"],
            TaskClass.PROPOSAL_GENERATION: ["proposal", "client", "pitch"],
            TaskClass.RESEARCH_SYNTHESIS: ["research", "paper", "synthesis", "literature"],
            TaskClass.ARCHITECTURE_DESIGN: ["architecture", "design", "module", "interface"],
            TaskClass.RED_TEAM_REVIEW: ["red team", "failure", "vulnerability", "attack"],
            TaskClass.MEMORY_CONSOLIDATION: ["memory", "template", "reusable", "consolidat"],
        }
        best_match = TaskClass.RESEARCH_SYNTHESIS
        best_count = 0
        for task_class, kws in keywords.items():
            count = sum(1 for kw in kws if kw in desc)
            if count > best_count:
                best_count = count
                best_match = task_class
        return best_match

    def estimate_risk_complexity(
        self, task_class: TaskClass, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, float]:
        """Estimate task risk and complexity.

        Args:
            task_class: Classified task type.
            context: Optional additional context.

        Returns:
            Tuple of (risk_estimate, complexity_estimate) both 0-1.
        """
        risk_map = {
            TaskClass.INVOICE_EXECUTION: 0.7,
            TaskClass.ESTIMATING: 0.5,
            TaskClass.CASHFLOW_DECISION: 0.9,
            TaskClass.PROPOSAL_GENERATION: 0.3,
            TaskClass.RESEARCH_SYNTHESIS: 0.2,
            TaskClass.ARCHITECTURE_DESIGN: 0.6,
            TaskClass.RED_TEAM_REVIEW: 0.8,
            TaskClass.MEMORY_CONSOLIDATION: 0.2,
        }
        complexity_map = {
            TaskClass.INVOICE_EXECUTION: 0.6,
            TaskClass.ESTIMATING: 0.5,
            TaskClass.CASHFLOW_DECISION: 0.7,
            TaskClass.PROPOSAL_GENERATION: 0.4,
            TaskClass.RESEARCH_SYNTHESIS: 0.6,
            TaskClass.ARCHITECTURE_DESIGN: 0.8,
            TaskClass.RED_TEAM_REVIEW: 0.7,
            TaskClass.MEMORY_CONSOLIDATION: 0.3,
        }
        risk = risk_map.get(task_class, 0.5)
        complexity = complexity_map.get(task_class, 0.5)
        return risk, complexity

    def run(
        self,
        task_description: str,
        salience_items: List[SalienceItem],
        total_budget: int,
        available_modules: Optional[List[str]] = None,
    ) -> RuntimeLoopState:
        """Execute the pre-generation phase of the measurement loop.

        Steps 1-5: classify, estimate, rank, allocate, recruit.

        Args:
            task_description: Natural language task.
            salience_items: Information units to rank.
            total_budget: Total available token budget.
            available_modules: Pool of modules to recruit from.

        Returns:
            RuntimeLoopState for the pre-generation phase.
        """
        # Step 1: Classify
        task_class = self.classify_task(task_description)

        # Step 2: Estimate risk and complexity
        risk, complexity = self.estimate_risk_complexity(task_class)

        # Step 3: Rank salience targets
        rankings = self.salience_allocator.rank(salience_items)

        # Step 4: Allocate token budget
        allocation = self.salience_allocator.allocate_budget(
            salience_items, total_budget
        )

        # Step 5: Recruit necessary modules
        modules = available_modules or []
        # Recruit more modules for higher complexity/risk
        recruit_ratio = min(1.0, risk + complexity) / 2.0
        num_recruit = max(1, int(len(modules) * recruit_ratio))
        recruited = modules[:num_recruit]

        state = RuntimeLoopState(
            task_classification=task_class,
            risk_estimate=risk,
            complexity_estimate=complexity,
            salience_rankings=rankings,
            token_budget=total_budget,
            modules_recruited=recruited,
        )
        return state

    def evaluate(
        self,
        state: RuntimeLoopState,
        token_scores: List[TokenScores],
        cognitive_scores: CognitiveReturnScores,
        output_tokens: int,
        prompt_tokens: int,
        information_retained: float = 3.0,
        action_clarity: float = 3.0,
        risk_preserved: float = 3.0,
    ) -> RuntimeLoopState:
        """Execute the post-generation evaluation phase.

        Steps 7-11: audit, score, detect waste, extract, update.

        Args:
            state: Pre-generation loop state.
            token_scores: Per-token value scores.
            cognitive_scores: Cognitive return scores.
            output_tokens: Number of output tokens generated.
            prompt_tokens: Number of prompt tokens.
            information_retained: Compression info retention (0-5).
            action_clarity: Compression action clarity (0-5).
            risk_preserved: Compression risk preservation (0-5).

        Returns:
            Updated RuntimeLoopState with evaluation results.
        """
        # Step 7: Audit compression quality
        compression = self.compression_metrics.compute(
            information_retained, action_clarity, risk_preserved, output_tokens
        )
        state.compression_quality = compression.efficiency

        # Step 8: Score cognitive return
        crpt = self.cognitive_metrics.crpt(
            cognitive_scores, prompt_tokens, output_tokens
        )
        state.cognitive_return = crpt

        # Step 9: Detect wasted tokens
        values = self.token_value_fn.compute_batch(token_scores)
        wasted = int(np.sum(values < 0))
        state.wasted_tokens = wasted

        # Step 10: Extract reusable rules (placeholder - override for real extraction)
        if cognitive_scores.reuse_value >= 3.0:
            state.extracted_rules = ["high_reuse_interaction"]

        # Step 11: Update future policy
        state.policy_update = {
            "crpt": crpt,
            "waste_ratio": wasted / len(token_scores) if token_scores else 0.0,
            "compression_efficiency": compression.efficiency,
            "adjust_budget": crpt > 0.1,
        }

        self.history.append(state)
        return state

    def average_crpt(self) -> float:
        """Compute average CRPT across all historical interactions.

        Returns:
            Mean CRPT score.
        """
        if not self.history:
            return 0.0
        return float(np.mean([s.cognitive_return for s in self.history]))

    def improvement_trend(self) -> float:
        """Compute improvement trend in CRPT over time.

        Returns:
            Slope of CRPT over interactions (positive = improving).
        """
        if len(self.history) < 2:
            return 0.0
        crpts = [s.cognitive_return for s in self.history]
        x = np.arange(len(crpts))
        coeffs = np.polyfit(x, crpts, 1)
        return float(coeffs[0])
