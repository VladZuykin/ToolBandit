"""Execution helper implementing budget-limited greedy fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

from .router import BudgetAwareToolRouter, Decision, NoFeasibleToolError


@dataclass(frozen=True)
class ToolCallResult:
    passed: bool
    cost: float
    latency: float
    output: object = None


@dataclass(frozen=True)
class Attempt:
    decision: Decision
    result: ToolCallResult
    budget_after: float


@dataclass(frozen=True)
class RouteResult:
    passed: bool
    output: object
    remaining_budget: float
    attempts: List[Attempt]


def execute_with_fallback(
    router: BudgetAwareToolRouter,
    context: Sequence[float] | np.ndarray,
    *,
    budget: float,
    call_tool: Callable[[str], ToolCallResult],
    candidates: Optional[Iterable[str]] = None,
    latency_sla: Optional[float] = None,
    max_attempts: Optional[int] = None,
    context_updater: Optional[
        Callable[[np.ndarray, Attempt], Sequence[float] | np.ndarray]
    ] = None,
) -> RouteResult:
    """Call distinct tools greedily until success or no feasible tool remains.

    After a failed call, ``context_updater`` may construct the next context
    from the current embedding and the complete attempt. A production updater
    can re-embed the original request together with the tool output/error;
    omitting it preserves the original fixed-context behavior.
    """

    candidate_list = list(candidates) if candidates is not None else router.tool_names
    limit = max_attempts if max_attempts is not None else len(candidate_list)
    if limit < 0:
        raise ValueError("max_attempts must be non-negative")

    remaining = budget
    current_context = np.asarray(context, dtype=np.float64).reshape(-1)
    excluded: set[str] = set()
    attempts: List[Attempt] = []

    while len(attempts) < limit:
        try:
            decision = router.select(
                current_context,
                remaining_budget=remaining,
                latency_sla=latency_sla,
                candidates=candidate_list,
                excluded=excluded,
            )
        except NoFeasibleToolError:
            break

        result = call_tool(decision.tool_name)
        if result.cost < 0 or result.cost > remaining + 1e-12:
            raise ValueError("tool returned an invalid cost for the remaining budget")
        router.observe(
            decision.tool_name,
            current_context,
            passed=result.passed,
            observed_cost=result.cost,
            observed_latency=result.latency,
        )
        remaining -= result.cost
        attempt = Attempt(decision, result, remaining)
        attempts.append(attempt)
        excluded.add(decision.tool_name)
        if result.passed:
            return RouteResult(True, result.output, remaining, attempts)
        if context_updater is not None:
            updated = np.asarray(
                context_updater(current_context.copy(), attempt), dtype=np.float64
            ).reshape(-1)
            if updated.shape != current_context.shape:
                raise ValueError(
                    "context_updater changed context dimension from "
                    f"{current_context.shape} to {updated.shape}"
                )
            if not np.all(np.isfinite(updated)) or np.linalg.norm(updated) == 0:
                raise ValueError("context_updater returned an invalid context")
            current_context = updated

    return RouteResult(False, None, remaining, attempts)
