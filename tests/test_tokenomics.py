"""Tests for the Tokenomics Measurement and Benchmarking Framework."""

import numpy as np
import pytest

from mesie.cognitive.tokenomics import (
    BenchmarkResult,
    CognitiveReturnMetrics,
    CognitiveReturnScores,
    CompressionEfficiencyMetrics,
    CompressionResult,
    EvaluationCriterion,
    RuntimeMeasurementLoop,
    SalienceAllocator,
    SalienceItem,
    SalienceWeights,
    TaskClass,
    TokenomicBenchmark,
    TokenScores,
    TokenValueFunction,
    TokenValueWeights,
)


# =============================================================================
# Token Value Function Tests
# =============================================================================


class TestTokenValueFunction:
    """Tests for TokenValueFunction."""

    def test_compute_positive_value(self):
        tvf = TokenValueFunction()
        scores = TokenScores(
            decision_value=3.0,
            action_usefulness=2.0,
            risk_reduction=1.0,
            compression_contribution=1.0,
            memory_value=1.0,
            noise=0.0,
        )
        assert tvf.compute(scores) == 8.0

    def test_compute_negative_value(self):
        tvf = TokenValueFunction()
        scores = TokenScores(noise=10.0)
        assert tvf.compute(scores) == -10.0

    def test_compute_with_custom_weights(self):
        weights = TokenValueWeights(w_d=2.0, w_a=0.0, w_r=0.0, w_c=0.0, w_m=0.0, w_n=1.0)
        tvf = TokenValueFunction(weights)
        scores = TokenScores(decision_value=5.0, noise=3.0)
        assert tvf.compute(scores) == 7.0  # 2*5 - 1*3

    def test_compute_batch(self):
        tvf = TokenValueFunction()
        batch = [
            TokenScores(decision_value=1.0),
            TokenScores(decision_value=2.0),
            TokenScores(noise=1.0),
        ]
        values = tvf.compute_batch(batch)
        assert len(values) == 3
        assert values[0] == 1.0
        assert values[1] == 2.0
        assert values[2] == -1.0

    def test_total_value(self):
        tvf = TokenValueFunction()
        batch = [
            TokenScores(decision_value=3.0),
            TokenScores(decision_value=2.0),
        ]
        assert tvf.total_value(batch) == 5.0

    def test_mean_value(self):
        tvf = TokenValueFunction()
        batch = [
            TokenScores(decision_value=4.0),
            TokenScores(decision_value=2.0),
        ]
        assert tvf.mean_value(batch) == 3.0

    def test_mean_value_empty(self):
        tvf = TokenValueFunction()
        assert tvf.mean_value([]) == 0.0


# =============================================================================
# Cognitive Return Metrics Tests
# =============================================================================


class TestCognitiveReturnMetrics:
    """Tests for CognitiveReturnMetrics."""

    def test_cognitive_return_total(self):
        crm = CognitiveReturnMetrics()
        scores = CognitiveReturnScores(
            decision_quality=4.0,
            actionability=3.0,
            risk_control=2.0,
            reuse_value=3.0,
            learning_gain=1.0,
        )
        assert crm.cognitive_return(scores) == 13.0

    def test_crpt_calculation(self):
        crm = CognitiveReturnMetrics()
        scores = CognitiveReturnScores(
            decision_quality=5.0,
            actionability=5.0,
            risk_control=5.0,
            reuse_value=5.0,
            learning_gain=5.0,
        )
        # 25 / (100 + 100) = 0.125
        assert crm.crpt(scores, 100, 100) == 0.125

    def test_crpt_zero_tokens(self):
        crm = CognitiveReturnMetrics()
        scores = CognitiveReturnScores(decision_quality=5.0)
        assert crm.crpt(scores, 0, 0) == 0.0

    def test_compare_systems(self):
        crm = CognitiveReturnMetrics()
        scores_a = CognitiveReturnScores(decision_quality=3.0, actionability=2.0)
        scores_b = CognitiveReturnScores(decision_quality=4.0, actionability=4.0)
        # B is better: (8/50) - (5/100) = 0.16 - 0.05 = 0.11
        diff = crm.compare(scores_a, 100, scores_b, 50)
        assert diff > 0


# =============================================================================
# Salience Allocator Tests
# =============================================================================


class TestSalienceAllocator:
    """Tests for SalienceAllocator."""

    def test_score_basic(self):
        allocator = SalienceAllocator()
        item = SalienceItem(
            item_id="task1",
            urgency=5.0,
            risk=3.0,
            mission_relevance=4.0,
            time_sensitivity=2.0,
            novelty=1.0,
            known_context=2.0,
        )
        # 5 + 3 + 4 + 2 + 1 - 2 = 13
        assert allocator.score(item) == 13.0

    def test_score_with_weights(self):
        weights = SalienceWeights(alpha=2.0, beta=0.0, gamma=0.0, delta=0.0, epsilon=0.0, zeta=0.0)
        allocator = SalienceAllocator(weights)
        item = SalienceItem(item_id="t", urgency=3.0)
        assert allocator.score(item) == 6.0

    def test_rank_ordering(self):
        allocator = SalienceAllocator()
        items = [
            SalienceItem(item_id="low", urgency=1.0),
            SalienceItem(item_id="high", urgency=5.0),
            SalienceItem(item_id="mid", urgency=3.0),
        ]
        ranked = allocator.rank(items)
        assert ranked[0][0] == "high"
        assert ranked[1][0] == "mid"
        assert ranked[2][0] == "low"

    def test_allocate_budget_proportional(self):
        allocator = SalienceAllocator()
        items = [
            SalienceItem(item_id="a", urgency=3.0),
            SalienceItem(item_id="b", urgency=1.0),
        ]
        allocation = allocator.allocate_budget(items, 100)
        # a should get ~75, b should get ~25
        assert allocation["a"] > allocation["b"]
        assert allocation["a"] + allocation["b"] == 100

    def test_allocate_budget_zero_salience(self):
        allocator = SalienceAllocator()
        items = [
            SalienceItem(item_id="a"),
            SalienceItem(item_id="b"),
        ]
        allocation = allocator.allocate_budget(items, 100)
        assert allocation["a"] == 50
        assert allocation["b"] == 50


# =============================================================================
# Compression Efficiency Tests
# =============================================================================


class TestCompressionEfficiencyMetrics:
    """Tests for CompressionEfficiencyMetrics."""

    def test_compute_efficiency(self):
        cem = CompressionEfficiencyMetrics()
        result = cem.compute(4.0, 4.0, 4.0, 100)
        assert result.efficiency == pytest.approx(0.12)
        assert result.output_tokens == 100

    def test_passes_test_good(self):
        cem = CompressionEfficiencyMetrics()
        result = CompressionResult(
            information_retained=4.0,
            action_clarity=3.0,
            risk_preserved=3.0,
            output_tokens=50,
            efficiency=0.2,
        )
        assert cem.passes_tokenomic_test(result, threshold=2.0) is True

    def test_passes_test_bad(self):
        cem = CompressionEfficiencyMetrics()
        result = CompressionResult(
            information_retained=1.0,
            action_clarity=0.2,
            risk_preserved=0.1,
            output_tokens=50,
            efficiency=0.026,
        )
        assert cem.passes_tokenomic_test(result, threshold=0.5) is False

    def test_compare_outputs(self):
        cem = CompressionEfficiencyMetrics()
        comparison = cem.compare(
            original_tokens=200,
            compressed_tokens=50,
            information_retained=4.0,
            action_clarity=4.0,
            risk_preserved=4.0,
        )
        assert comparison["compression_ratio"] == 0.25
        assert comparison["token_reduction"] == 0.75
        assert comparison["efficiency"] > 0


# =============================================================================
# Tokenomic Benchmark Tests
# =============================================================================


class TestTokenomicBenchmark:
    """Tests for TokenomicBenchmark."""

    def test_task_score(self):
        bench = TokenomicBenchmark()
        score = bench.task_score(4.0, 3.0, 3.0, 2.0, 4.0, 1.0)
        assert score == 15.0  # 4+3+3+2+4-1

    def test_tokenomic_gain_positive(self):
        bench = TokenomicBenchmark()
        # Tokenomic: 15/50=0.3, Baseline: 12/100=0.12
        gain = bench.tokenomic_gain(15.0, 50, 12.0, 100)
        assert gain > 0

    def test_tokenomic_gain_negative(self):
        bench = TokenomicBenchmark()
        # Tokenomic: 5/200=0.025, Baseline: 12/100=0.12
        gain = bench.tokenomic_gain(5.0, 200, 12.0, 100)
        assert gain < 0

    def test_run_comparison(self):
        bench = TokenomicBenchmark()
        result = bench.run_comparison(
            task_class=TaskClass.INVOICE_EXECUTION,
            scores_tokenomic={
                "decision_quality": 4.0,
                "actionability": 4.0,
                "risk_control": 3.0,
                "reuse_value": 2.0,
                "accuracy": 5.0,
                "waste": 1.0,
            },
            tokens_tokenomic=80,
            scores_baseline={
                "decision_quality": 3.0,
                "actionability": 3.0,
                "risk_control": 2.0,
                "reuse_value": 1.0,
                "accuracy": 4.0,
                "waste": 3.0,
            },
            tokens_baseline=200,
        )
        assert result.task_class == TaskClass.INVOICE_EXECUTION
        assert result.tokenomic_gain > 0
        assert bench.is_superior(result)

    def test_is_superior_false(self):
        bench = TokenomicBenchmark()
        result = BenchmarkResult(
            task_class=TaskClass.ESTIMATING,
            tokenomic_gain=-0.05,
        )
        assert bench.is_superior(result) is False


# =============================================================================
# Runtime Measurement Loop Tests
# =============================================================================


class TestRuntimeMeasurementLoop:
    """Tests for RuntimeMeasurementLoop."""

    def test_classify_task(self):
        loop = RuntimeMeasurementLoop()
        assert loop.classify_task("process invoice and apply payment") == TaskClass.INVOICE_EXECUTION
        assert loop.classify_task("design system architecture modules") == TaskClass.ARCHITECTURE_DESIGN
        assert loop.classify_task("red team review for failure modes") == TaskClass.RED_TEAM_REVIEW

    def test_estimate_risk_complexity(self):
        loop = RuntimeMeasurementLoop()
        risk, complexity = loop.estimate_risk_complexity(TaskClass.CASHFLOW_DECISION)
        assert risk == 0.9
        assert complexity == 0.7

    def test_run_pregeneration(self):
        loop = RuntimeMeasurementLoop()
        items = [
            SalienceItem(item_id="urgent", urgency=5.0, risk=4.0),
            SalienceItem(item_id="routine", urgency=1.0, risk=1.0),
        ]
        state = loop.run(
            task_description="process invoice payment",
            salience_items=items,
            total_budget=500,
            available_modules=["billing", "math", "memory", "formatting"],
        )
        assert state.task_classification == TaskClass.INVOICE_EXECUTION
        assert state.risk_estimate > 0
        assert len(state.salience_rankings) == 2
        assert state.salience_rankings[0][0] == "urgent"

    def test_evaluate_postgeneration(self):
        loop = RuntimeMeasurementLoop()
        items = [SalienceItem(item_id="main", urgency=3.0)]
        state = loop.run("research synthesis paper", items, 1000)

        token_scores = [
            TokenScores(decision_value=2.0),
            TokenScores(decision_value=3.0),
            TokenScores(noise=5.0),  # wasted token
        ]
        cognitive = CognitiveReturnScores(
            decision_quality=4.0,
            actionability=3.0,
            risk_control=2.0,
            reuse_value=4.0,
            learning_gain=3.0,
        )
        state = loop.evaluate(
            state=state,
            token_scores=token_scores,
            cognitive_scores=cognitive,
            output_tokens=200,
            prompt_tokens=100,
        )
        assert state.cognitive_return > 0
        assert state.wasted_tokens == 1
        assert state.compression_quality > 0
        assert state.policy_update is not None

    def test_average_crpt(self):
        loop = RuntimeMeasurementLoop()
        assert loop.average_crpt() == 0.0

        # Add some history
        items = [SalienceItem(item_id="x", urgency=2.0)]
        for _ in range(3):
            state = loop.run("estimate pricing", items, 500)
            loop.evaluate(
                state=state,
                token_scores=[TokenScores(decision_value=2.0)],
                cognitive_scores=CognitiveReturnScores(decision_quality=4.0, actionability=3.0),
                output_tokens=100,
                prompt_tokens=50,
            )
        assert loop.average_crpt() > 0

    def test_improvement_trend(self):
        loop = RuntimeMeasurementLoop()
        assert loop.improvement_trend() == 0.0


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum completeness."""

    def test_task_classes(self):
        assert len(TaskClass) == 8

    def test_evaluation_criteria(self):
        assert len(EvaluationCriterion) == 8
