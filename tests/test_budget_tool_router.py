import unittest

import numpy as np

from budget_tool_router.router import _DiagonalLinUCB, _DisjointLinUCB

from budget_tool_router import (
    BudgetAwareToolRouter,
    NoFeasibleToolError,
    ToolSpec,
)
from budget_tool_router.runner import ToolCallResult, execute_with_fallback


def context(index: int, dimension: int = 384) -> np.ndarray:
    value = np.zeros(dimension)
    value[index] = 1.0
    return value


class BudgetAwareToolRouterTests(unittest.TestCase):
    def make_router(self) -> BudgetAwareToolRouter:
        return BudgetAwareToolRouter(
            [
                ToolSpec(
                    "a",
                    initial_cost=1.0,
                    fixed_cost=1.0,
                    initial_latency=0.2,
                    latency_range=0.1,
                    prior_count=100,
                ),
                ToolSpec(
                    "b",
                    initial_cost=1.0,
                    fixed_cost=1.0,
                    initial_latency=0.2,
                    latency_range=0.1,
                    prior_count=100,
                ),
            ],
            alpha=0.1,
        )

    def test_reward_model_learns_contextual_passes(self) -> None:
        router = self.make_router()
        x = context(3)
        for _ in range(20):
            router.observe("a", x, passed=True, observed_cost=1.0, observed_latency=0.2)
            router.observe("b", x, passed=False, observed_cost=1.0, observed_latency=0.2)

        decision = router.select(x, remaining_budget=2.0, latency_sla=1.0)
        self.assertEqual(decision.tool_name, "a")
        self.assertGreater(decision.pass_mean, 0.8)
        self.assertLessEqual(decision.pass_mean, 1.0)
        self.assertLessEqual(decision.pass_ucb, 1.0)

    def test_budget_and_latency_filter_candidates(self) -> None:
        router = BudgetAwareToolRouter(
            [
                ToolSpec("expensive", 5.0, 0.1, fixed_cost=5.0, latency_range=0.01, prior_count=100),
                ToolSpec("slow", 1.0, 20.0, fixed_cost=1.0, latency_range=0.01, prior_count=100),
                ToolSpec("valid", 1.0, 0.1, fixed_cost=1.0, latency_range=0.01, prior_count=100),
            ]
        )
        decision = router.select(context(0), remaining_budget=2.0, latency_sla=1.0)
        self.assertEqual(decision.tool_name, "valid")

    def test_no_feasible_tool(self) -> None:
        router = self.make_router()
        with self.assertRaises(NoFeasibleToolError):
            router.select(context(0), remaining_budget=0.5)

    def test_unknown_resources_require_runtime_estimates(self) -> None:
        router = BudgetAwareToolRouter(
            [ToolSpec("unknown", None, None, cost_range=0.1,
                      latency_range=10.0, prior_count=0)]
        )
        with self.assertRaises(NoFeasibleToolError):
            router.select(context(0), remaining_budget=0.01, latency_sla=2.0)
        decision = router.select(
            context(0), remaining_budget=0.01, latency_sla=2.0,
            runtime_estimates={"unknown": {
                "estimated_cost": 0.002,
                "estimated_latency": 0.5,
            }},
        )
        self.assertEqual(decision.tool_name, "unknown")
        self.assertEqual(decision.cost_ucb, 0.002)

    def test_free_tool_with_unknown_latency_has_bounded_score(self) -> None:
        router = BudgetAwareToolRouter(
            [ToolSpec("free", 0.0, None, prior_count=1)],
            alpha=0.35, cost_weight=0.5, latency_weight=0.5,
            cost_scale=0.012, latency_scale=3.0,
        )
        decision = router.select(context(0), remaining_budget=1.0, latency_sla=5.0)
        self.assertGreaterEqual(decision.score, 0.0)
        self.assertLessEqual(decision.score, 1.0)

    def test_score_uses_additive_resource_penalties(self) -> None:
        router = BudgetAwareToolRouter(
            [ToolSpec("tool", 0.004, 1.2, fixed_cost=0.004, prior_count=1)],
            alpha=0.82,
            cost_weight=0.25,
            latency_weight=0.15,
            cost_scale=0.01,
            latency_scale=3.0,
        )
        decision = router.select(context(0), remaining_budget=1.0, latency_sla=5.0)

        expected_penalty = 0.25 * (0.004 / 0.01)
        expected_penalty += 0.15 * (1.2 / 3.0)
        self.assertAlmostEqual(decision.score, decision.pass_ucb - expected_penalty)

    def test_lqm_score_matches_latency_quality_formula(self) -> None:
        router = BudgetAwareToolRouter(
            [ToolSpec("tool", 0.0, 2.0, fixed_cost=0.0, prior_count=1)],
            context_dimension=4, alpha=0.4, diagonal_covariance=True,
            cost_weight=0.5, latency_weight=0.5,
            lqm_latency_reference=4.0, lqm_deflation=2.0,
        )
        decision = router.select(
            np.array([1.0, 0.0, 0.0, 0.0]),
            remaining_budget=1.0,
            latency_sla=5.0,
        )
        expected = (
            decision.pass_mean / (1.0 + decision.latency_ucb / 4.0)
            + 0.4 * decision.pass_uncertainty
        )
        self.assertAlmostEqual(decision.lqm_score, expected)

    def test_lqm_deflates_exploration_for_quality_dominated_arm(self) -> None:
        router = BudgetAwareToolRouter(
            [
                ToolSpec("strong", 0.0, 1.0, fixed_cost=0.0, prior_count=1),
                ToolSpec("weak", 0.0, 1.0, fixed_cost=0.0, prior_count=1),
            ],
            context_dimension=4, alpha=0.5, diagonal_covariance=True,
            cost_weight=0.5, latency_weight=0.5,
            lqm_latency_reference=2.0, lqm_deflation=4.0,
        )
        x = np.array([1.0, 0.0, 0.0, 0.0])
        for _ in range(20):
            router.observe_reward("strong", x, passed=True)
            router.observe_reward("weak", x, passed=False)
        decisions = {item.tool_name: item for item in router.rank(x, remaining_budget=1.0)}
        strong, weak = decisions["strong"], decisions["weak"]
        undeflated_weak = (
            weak.pass_mean / (1.0 + weak.latency_ucb / 2.0)
            + 0.5 * weak.pass_uncertainty
        )
        self.assertGreater(strong.pass_mean, weak.pass_mean)
        self.assertLess(weak.lqm_score, undeflated_weak)
        self.assertGreater(strong.lqm_score, weak.lqm_score)

    def test_resources_use_separate_exponential_moving_averages(self) -> None:
        router = BudgetAwareToolRouter(
            [ToolSpec("tool", 10.0, 10.0, prior_count=1)],
            alpha=0.0,
            cost_weight=0.5,
            latency_weight=0.5,
            cost_scale=10.0,
            latency_scale=10.0,
        )

        router.observe_resources("tool", observed_cost=30.0, observed_latency=30.0)
        decision = router.select(context(0), remaining_budget=100.0, latency_sla=100.0)

        self.assertAlmostEqual(decision.cost_ucb, 11.0)  # 10 + (1/20) * 20
        self.assertAlmostEqual(decision.latency_ucb, 10.02)  # 10 + (1/1000) * 20

    def test_pass_model_forgets_old_observations(self) -> None:
        router = BudgetAwareToolRouter(
            [ToolSpec("tool", 0.0, 0.1, fixed_cost=0.0, prior_count=1)],
            alpha=0.0,
            diagonal_covariance=True,
            pass_ema_alpha=1.0 / 1000.0,
        )
        x = context(0)
        for _ in range(1000):
            router.observe_reward("tool", x, passed=True)
        mean_before_failures = router.select(x, remaining_budget=1.0).pass_mean

        for _ in range(1000):
            router.observe_reward("tool", x, passed=False)
        mean_after_failures = router.select(x, remaining_budget=1.0).pass_mean

        self.assertGreater(mean_before_failures, 0.99)
        self.assertLess(mean_after_failures, 0.4)

    def test_greedy_fallback_uses_distinct_tools(self) -> None:
        router = self.make_router()
        calls = []

        def call_tool(name: str) -> ToolCallResult:
            calls.append(name)
            return ToolCallResult(
                passed=len(calls) == 2,
                cost=1.0,
                latency=0.2,
                output="done" if len(calls) == 2 else None,
            )

        result = execute_with_fallback(
            router,
            context(0),
            budget=2.0,
            latency_sla=1.0,
            call_tool=call_tool,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.output, "done")
        self.assertEqual(len(set(calls)), 2)
        self.assertAlmostEqual(result.remaining_budget, 0.0)

    def test_failed_attempt_updates_context_before_reranking(self) -> None:
        router = self.make_router()
        seen_contexts = []
        original_select = router.select

        def recording_select(current_context, **kwargs):
            seen_contexts.append(np.asarray(current_context).copy())
            return original_select(current_context, **kwargs)

        router.select = recording_select
        calls = 0

        def call_tool(name: str) -> ToolCallResult:
            nonlocal calls
            calls += 1
            return ToolCallResult(calls == 2, 1.0, 0.2)

        updates = []

        def update_context(current, attempt):
            updates.append(attempt.decision.tool_name)
            return context(1)

        result = execute_with_fallback(
            router,
            context(0),
            budget=2.0,
            latency_sla=1.0,
            call_tool=call_tool,
            context_updater=update_context,
        )
        self.assertTrue(result.passed)
        self.assertEqual(len(updates), 1)
        np.testing.assert_array_equal(seen_contexts[0], context(0))
        np.testing.assert_array_equal(seen_contexts[1], context(1))

    def test_context_updater_rejects_wrong_dimension(self) -> None:
        router = self.make_router()
        with self.assertRaisesRegex(ValueError, "changed context dimension"):
            execute_with_fallback(
                router,
                context(0),
                budget=2.0,
                latency_sla=1.0,
                max_attempts=2,
                call_tool=lambda name: ToolCallResult(False, 1.0, 0.2),
                context_updater=lambda current, attempt: np.ones(12),
            )

    def test_diagonal_discount_keeps_ridge_in_unobserved_direction(self) -> None:
        model = _DiagonalLinUCB(dimension=2, regularization=1.0, discount=0.99)
        observed = np.array([1.0, 0.0])
        unobserved = np.array([0.0, 1.0])
        for _ in range(2_000):
            model.update(observed, 1.0)

        self.assertAlmostEqual(model.a_diag[1], 1.0)
        _, uncertainty, _ = model.estimate(unobserved, alpha=0.35)
        self.assertAlmostEqual(uncertainty, 1.0)

    def test_full_discount_keeps_ridge_in_unobserved_direction(self) -> None:
        model = _DisjointLinUCB(dimension=2, regularization=1.0, discount=0.99)
        observed = np.array([1.0, 0.0])
        unobserved = np.array([0.0, 1.0])
        for _ in range(200):
            model.update(observed, 1.0)

        self.assertAlmostEqual(model.a_matrix[1, 1], 1.0)
        _, uncertainty, _ = model.estimate(unobserved, alpha=0.35)
        self.assertAlmostEqual(uncertainty, 1.0)


if __name__ == "__main__":
    unittest.main()
