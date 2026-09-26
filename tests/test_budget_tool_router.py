import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
