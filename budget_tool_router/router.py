"""Budget-Aware Greedy LinUCB adapted to tool selection.

The reward model learns contextual pass/fail outcomes. Cost and latency are
kept outside the reward model: they constrain feasibility and are subtracted
from the optimistic reward as normalized resource penalties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


class NoFeasibleToolError(RuntimeError):
    """Raised when no candidate satisfies budget and latency constraints."""


@dataclass(frozen=True)
class ToolSpec:
    """Static tool configuration.

    Set ``fixed_cost`` when a tool has a known price. Otherwise ``initial_cost``
    is used as a prior until observed costs arrive. Latency is learned online,
    starting from ``initial_latency``.
    """

    name: str
    initial_cost: Optional[float]
    initial_latency: Optional[float]
    fixed_cost: Optional[float] = None
    cost_range: float = 1.0
    latency_range: float = 10.0
    prior_count: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")
        if self.initial_cost is not None and self.initial_cost < 0:
            raise ValueError("cost must be non-negative")
        if self.initial_latency is not None and self.initial_latency < 0:
            raise ValueError("cost and latency must be non-negative")
        if self.fixed_cost is not None and self.fixed_cost < 0:
            raise ValueError("fixed_cost must be non-negative")
        if self.cost_range <= 0 or self.latency_range <= 0:
            raise ValueError("confidence ranges must be positive")
        if self.prior_count < 0:
            raise ValueError("prior_count must be non-negative")
        if (self.initial_cost is not None or self.initial_latency is not None) and self.prior_count <= 0:
            raise ValueError("known initial estimates require a positive prior_count")


@dataclass
class _RunningEstimate:
    mean: Optional[float]
    count: float
    value_range: float
    smoothing: float

    def __post_init__(self) -> None:
        if not 0 < self.smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1]")

    def update(self, value: float) -> None:
        if value < 0 or not math.isfinite(value):
            raise ValueError("observation must be a finite non-negative number")
        if self.count <= 0 or self.mean is None:
            self.mean = value
            self.count = 1.0
        else:
            self.count += 1.0
            self.mean += self.smoothing * (value - self.mean)

    def radius(self, *, step: int, delta: float) -> float:
        if self.count <= 0 or self.mean is None:
            return math.inf
        # Anytime Hoeffding-style confidence radius for a bounded quantity.
        log_term = math.log(max(2.0, 2.0 * (step + 1) ** 2 / delta))
        return self.value_range * math.sqrt(log_term / (2.0 * self.count))


@dataclass
class _DisjointLinUCB:
    dimension: int
    regularization: float
    a_inv: np.ndarray = field(init=False, repr=False)
    a_diag: np.ndarray = field(init=False, repr=False)
    b: np.ndarray = field(init=False, repr=False)
    observations: int = 0

    def __post_init__(self) -> None:
        self.a_inv = np.eye(self.dimension, dtype=np.float64) / self.regularization
        # Track diag(A) alongside A^-1. This costs O(d) memory and lets us
        # compare the full model with the diagonal approximation trained on
        # exactly the same observations.
        self.a_diag = np.full(self.dimension, self.regularization, dtype=np.float64)
        self.b = np.zeros(self.dimension, dtype=np.float64)

    def estimate(self, context: np.ndarray, alpha: float) -> tuple[float, float, float]:
        theta = self.a_inv @ self.b
        mean = float(context @ theta)
        uncertainty = math.sqrt(max(0.0, float(context @ self.a_inv @ context)))
        return mean, uncertainty, mean + alpha * uncertainty

    def update(self, context: np.ndarray, reward: float) -> None:
        # Sherman-Morrison rank-one update: O(d^2), avoiding a fresh inverse.
        ax = self.a_inv @ context
        denominator = 1.0 + float(context @ ax)
        self.a_inv -= np.outer(ax, ax) / denominator
        self.a_diag += context * context
        self.b += reward * context
        self.observations += 1


@dataclass
class _DiagonalLinUCB:
    dimension: int
    regularization: float
    a_diag: np.ndarray = field(init=False, repr=False)
    b: np.ndarray = field(init=False, repr=False)
    observations: int = 0

    def __post_init__(self) -> None:
        self.a_diag = np.full(self.dimension, self.regularization, dtype=np.float64)
        self.b = np.zeros(self.dimension, dtype=np.float64)

    def estimate(self, context: np.ndarray, alpha: float) -> tuple[float, float, float]:
        inverse_diagonal = 1.0 / self.a_diag
        mean = float(context @ (inverse_diagonal * self.b))
        uncertainty = math.sqrt(max(0.0, float((context * context) @ inverse_diagonal)))
        return mean, uncertainty, mean + alpha * uncertainty

    def update(self, context: np.ndarray, reward: float) -> None:
        self.a_diag += context * context
        self.b += reward * context
        self.observations += 1


@dataclass
class _ToolState:
    spec: ToolSpec
    reward_model: _DisjointLinUCB | _DiagonalLinUCB
    cost: _RunningEstimate
    latency: _RunningEstimate


@dataclass(frozen=True)
class Decision:
    tool_name: str
    score: float
    pass_mean: float
    pass_uncertainty: float
    pass_ucb: float
    cost_lcb: float
    cost_ucb: float
    latency_lcb: float
    latency_ucb: float
    remaining_budget: float


class BudgetAwareToolRouter:
    """Score tools by optimistic pass rate minus normalized resource use.

    Selection policy::

        PassUCB(query, a)
        - cost_weight * CostUCB(a) / cost_scale
        - latency_weight * LatencyUCB(a) / latency_scale

    subject to ``CostUCB <= remaining_budget`` and, when supplied,
    ``LatencyUCB <= latency_sla``.
    """

    def __init__(
        self,
        tools: Sequence[ToolSpec],
        *,
        context_dimension: int = 384,
        alpha: float = 1.0,
        regularization: float = 1.0,
        confidence_delta: float = 0.05,
        diagonal_covariance: bool = False,
        cost_weight: float = 1.0,
        latency_weight: float = 0.0,
        cost_scale: float = 1.0,
        latency_scale: float = 1.0,
        cost_ema_alpha: float = 1.0 / 20.0,
        latency_ema_alpha: float = 1.0 / 1000.0,
    ) -> None:
        if context_dimension <= 0:
            raise ValueError("context_dimension must be positive")
        if alpha < 0 or regularization <= 0:
            raise ValueError("invalid LinUCB parameters")
        if not 0 < confidence_delta < 1:
            raise ValueError("confidence_delta must be in (0, 1)")
        if cost_weight < 0 or latency_weight < 0 or cost_weight + latency_weight <= 0:
            raise ValueError("resource weights must be non-negative and not both zero")
        if cost_scale <= 0 or latency_scale <= 0:
            raise ValueError("resource scales must be positive")
        if not 0 < cost_ema_alpha <= 1 or not 0 < latency_ema_alpha <= 1:
            raise ValueError("EMA coefficients must be in (0, 1]")

        self.context_dimension = context_dimension
        self.alpha = alpha
        self.regularization = regularization
        self.confidence_delta = confidence_delta
        self.diagonal_covariance = diagonal_covariance
        self.cost_weight = cost_weight
        self.latency_weight = latency_weight
        self.cost_scale = cost_scale
        self.latency_scale = latency_scale
        self.cost_ema_alpha = cost_ema_alpha
        self.latency_ema_alpha = latency_ema_alpha
        self.step = 0
        self._states: Dict[str, _ToolState] = {}

        for spec in tools:
            if spec.name in self._states:
                raise ValueError(f"duplicate tool name: {spec.name}")
            self._states[spec.name] = _ToolState(
                spec=spec,
                reward_model=(
                    _DiagonalLinUCB(context_dimension, regularization)
                    if diagonal_covariance
                    else _DisjointLinUCB(context_dimension, regularization)
                ),
                cost=_RunningEstimate(
                    spec.initial_cost,
                    spec.prior_count if spec.initial_cost is not None else 0.0,
                    spec.cost_range,
                    cost_ema_alpha,
                ),
                latency=_RunningEstimate(
                    spec.initial_latency,
                    spec.prior_count if spec.initial_latency is not None else 0.0,
                    spec.latency_range,
                    latency_ema_alpha,
                ),
            )
    @property
    def tool_names(self) -> List[str]:
        return list(self._states)

    def upsert_tool(self, spec: ToolSpec) -> None:
        """Add a tool or update resource configuration without losing reward learning."""
        existing = self._states.get(spec.name)
        if existing is None:
            self._states[spec.name] = _ToolState(
                spec=spec,
                reward_model=(
                    _DiagonalLinUCB(self.context_dimension, self.regularization)
                    if self.diagonal_covariance
                    else _DisjointLinUCB(self.context_dimension, self.regularization)
                ),
                cost=_RunningEstimate(spec.initial_cost,
                    spec.prior_count if spec.initial_cost is not None else 0.0,
                    spec.cost_range, self.cost_ema_alpha),
                latency=_RunningEstimate(spec.initial_latency,
                    spec.prior_count if spec.initial_latency is not None else 0.0,
                    spec.latency_range, self.latency_ema_alpha),
            )
            return
        existing.spec = spec
        existing.cost.value_range = spec.cost_range
        existing.latency.value_range = spec.latency_range
        if spec.fixed_cost is not None:
            existing.cost.mean = spec.fixed_cost
            existing.cost.count = max(existing.cost.count, spec.prior_count)
        elif existing.cost.count == 0 and spec.initial_cost is not None:
            existing.cost.mean = spec.initial_cost
            existing.cost.count = spec.prior_count
        if existing.latency.count == 0 and spec.initial_latency is not None:
            existing.latency.mean = spec.initial_latency
            existing.latency.count = spec.prior_count

    def remove_tool(self, name: str) -> bool:
        """Remove a tool from future ranking; returns whether it existed."""
        return self._states.pop(name, None) is not None

    def _context(self, context: Sequence[float] | np.ndarray) -> np.ndarray:
        vector = np.asarray(context, dtype=np.float64).reshape(-1)
        if vector.shape != (self.context_dimension,):
            raise ValueError(
                f"expected context shape ({self.context_dimension},), got {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("context contains non-finite values")
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError("context must not be the zero vector")
        # Stable confidence geometry even if an encoder was not configured to
        # normalize its embeddings.
        return vector / norm

    def _bounds(
        self, state: _ToolState, runtime: Optional[Mapping[str, float]] = None
    ) -> tuple[float, float, float, float]:
        runtime = runtime or {}
        spec = state.spec
        if "estimated_cost" in runtime:
            cost_lcb = max(0.0, float(runtime["estimated_cost"]))
            cost_ucb = cost_lcb
        elif spec.fixed_cost is not None:
            cost_lcb = cost_ucb = spec.fixed_cost
        elif state.cost.mean is None:
            cost_lcb, cost_ucb = 0.0, math.inf
        else:
            cost_lcb = cost_ucb = state.cost.mean
        if "estimated_latency" in runtime:
            latency_lcb = max(0.0, float(runtime["estimated_latency"]))
            latency_ucb = latency_lcb
        elif state.latency.mean is None:
            # Cold start: unknown latency is allowed once so real observations
            # can bootstrap the estimate. It is not treated as an SLA failure.
            latency_lcb = latency_ucb = 0.0
        else:
            latency_lcb = latency_ucb = state.latency.mean
        return cost_lcb, cost_ucb, latency_lcb, latency_ucb

    def rank(
        self,
        context: Sequence[float] | np.ndarray,
        *,
        remaining_budget: float,
        latency_sla: Optional[float] = None,
        candidates: Optional[Iterable[str]] = None,
        excluded: Iterable[str] = (),
        runtime_estimates: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> List[Decision]:
        """Return feasible candidates ordered from best to worst."""

        if remaining_budget < 0:
            raise ValueError("remaining_budget must be non-negative")
        if latency_sla is not None and latency_sla < 0:
            raise ValueError("latency_sla must be non-negative")
        x = self._context(context)
        names = list(candidates) if candidates is not None else self.tool_names
        excluded_set = set(excluded)
        decisions: List[Decision] = []

        for name in names:
            if name in excluded_set:
                continue
            if name not in self._states:
                raise KeyError(f"unknown tool: {name}")
            state = self._states[name]
            runtime = None if runtime_estimates is None else runtime_estimates.get(name)
            cost_lcb, cost_ucb, latency_lcb, latency_ucb = self._bounds(state, runtime)
            if cost_ucb > remaining_budget:
                continue
            if latency_sla is not None and latency_ucb > latency_sla:
                continue

            raw_mean, uncertainty, raw_pass_ucb = state.reward_model.estimate(x, self.alpha)
            # Rewards are binary pass/fail, so expose and score probability-like
            # estimates within their meaningful range. Keep raw uncertainty visible.
            mean = min(1.0, max(0.0, raw_mean))
            pass_ucb = min(1.0, max(mean, raw_pass_ucb))
            resource_penalty = (
                self.cost_weight * cost_ucb / self.cost_scale
                + self.latency_weight * latency_ucb / self.latency_scale
            )
            score = pass_ucb - resource_penalty
            decisions.append(
                Decision(
                    tool_name=name,
                    score=score,
                    pass_mean=mean,
                    pass_uncertainty=uncertainty,
                    pass_ucb=pass_ucb,
                    cost_lcb=cost_lcb,
                    cost_ucb=cost_ucb,
                    latency_lcb=latency_lcb,
                    latency_ucb=latency_ucb,
                    remaining_budget=remaining_budget,
                )
            )

        decisions.sort(key=lambda item: (-item.score, item.tool_name))
        return decisions

    def select(self, *args, **kwargs) -> Decision:
        """Select the highest-ranked feasible tool."""

        ranked = self.rank(*args, **kwargs)
        if not ranked:
            raise NoFeasibleToolError("no tool satisfies budget and latency constraints")
        return ranked[0]

    def observe(
        self,
        tool_name: str,
        context: Sequence[float] | np.ndarray,
        *,
        passed: bool,
        observed_cost: float,
        observed_latency: float,
    ) -> None:
        """Update reward, cost, and latency after one real or simulated call."""

        if tool_name not in self._states:
            raise KeyError(f"unknown tool: {tool_name}")
        self.observe_resources(tool_name, observed_cost=observed_cost,
                               observed_latency=observed_latency)
        self.observe_reward(tool_name, context, passed=passed)

    def observe_resources(self, tool_name: str, *, observed_cost: float,
                          observed_latency: float) -> None:
        """Learn real resource usage independently of Judge availability."""
        if tool_name not in self._states:
            raise KeyError(f"unknown tool: {tool_name}")
        state = self._states[tool_name]
        if state.spec.fixed_cost is None:
            state.cost.update(observed_cost)
        elif not math.isclose(
            observed_cost, state.spec.fixed_cost, rel_tol=1e-6, abs_tol=1e-9
        ):
            raise ValueError(
                f"observed cost {observed_cost} differs from fixed cost "
                f"{state.spec.fixed_cost} for {tool_name}"
            )
        state.latency.update(observed_latency)
        self.step += 1

    def observe_reward(self, tool_name: str, context: Sequence[float] | np.ndarray,
                       *, passed: bool) -> None:
        """Learn pass/fail reward after asynchronous judging."""
        if tool_name not in self._states:
            raise KeyError(f"unknown tool: {tool_name}")
        x = self._context(context)
        self._states[tool_name].reward_model.update(x, float(passed))

    def snapshot(self) -> dict:
        """Return JSON-serializable learned state summaries."""

        result = {
            "step": self.step,
            "context_dimension": self.context_dimension,
            "covariance": "diagonal" if self.diagonal_covariance else "full",
            "tools": {},
        }
        for name, state in self._states.items():
            result["tools"][name] = {
                "reward_observations": state.reward_model.observations,
                "cost_mean": state.spec.fixed_cost
                if state.spec.fixed_cost is not None
                else state.cost.mean,
                "cost_observations": state.cost.count,
                "latency_mean": state.latency.mean,
                "latency_observations": state.latency.count,
            }
        return result

    def save_linucb_state(self, path: str | Path) -> Path:
        """Save complete reward-model state as an uncompressed NPZ archive.

        The archive contains A^-1, b and diag(A) for every tool. Uncompressed
        storage is intentional: it avoids spending a long time compressing
        roughly one gigabyte of dense floating-point matrices.
        """

        if self.diagonal_covariance:
            raise ValueError("full-matrix export is unavailable for a diagonal router")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        tools: list[dict[str, object]] = []
        for index, (name, state) in enumerate(self._states.items()):
            prefix = f"tool_{index:03d}"
            model = state.reward_model
            arrays[f"{prefix}_a_inv"] = model.a_inv
            arrays[f"{prefix}_a_diag"] = model.a_diag
            arrays[f"{prefix}_b"] = model.b
            tools.append(
                {
                    "name": name,
                    "prefix": prefix,
                    "observations": model.observations,
                }
            )
        metadata = {
            "format_version": 1,
            "context_dimension": self.context_dimension,
            "regularization": self.regularization,
            "step": self.step,
            "tools": tools,
            "matrix_meaning": "a_inv is full inverse(A); a_diag is diag(A)",
        }
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
        np.savez(destination, **arrays)
        return destination
