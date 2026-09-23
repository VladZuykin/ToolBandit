# Budget-aware tool router

This module adapts Budget-Aware Greedy LinUCB from model routing to tools.
Each tool has a disjoint contextual reward model. The reward is binary
`pass/fail`; costs and latency are learned separately and never mixed into the
reward.

With weighted cost and latency, the policy selects

```text
argmax PassUCB(query, tool) /
       max(w_cost * CostLCB / cost_scale
           + w_latency * LatencyLCB / latency_scale, epsilon)
```

subject to

```text
CostUCB(tool) <= remaining budget
LatencyUCB(tool) <= latency SLA
```

## Minimal example

```python
import numpy as np

from budget_tool_router import BudgetAwareToolRouter, ToolSpec

tools = [
    ToolSpec("weather_a", initial_cost=0.002, fixed_cost=0.002,
             initial_latency=0.5, latency_range=1.0, prior_count=10),
    ToolSpec("weather_b", initial_cost=0.001, fixed_cost=0.001,
             initial_latency=0.8, latency_range=1.0, prior_count=10),
]

router = BudgetAwareToolRouter(
    tools,
    context_dimension=384,
    alpha=0.5,
    cost_weight=0.5,
    latency_weight=0.5,
    cost_scale=0.01,
    latency_scale=2.0,
)

# Supply a 384-dimensional embedding from any encoder. The router normalizes it.
context = np.random.default_rng(42).normal(size=384)
decision = router.select(
    context,
    remaining_budget=0.01,
    latency_sla=2.0,
)

# Call the tool, evaluate whether it solved the task, then provide feedback.
router.observe(
    decision.tool_name,
    context,
    passed=True,
    observed_cost=decision.cost_ucb,
    observed_latency=0.63,
)
```

For retrying another tool after failure, use
`budget_tool_router.runner.execute_with_fallback`.

## Evolving context between retries

Pass a `context_updater` to make every failed response part of the next
ranking context. The callback receives the current vector and the completed
attempt (tool, output/error, cost, latency, and remaining budget):

```python
from budget_tool_router.runner import execute_with_fallback

history = [original_query]

def update_context(current_context, attempt):
    history.append(
        f"Tool {attempt.decision.tool_name} failed. "
        f"Result: {attempt.result.output!r}. "
        f"Remaining budget: {attempt.budget_after}."
    )
    # Replace this with the same encoder used for the original query.
    return encoder.encode_many(["\n".join(history)])[0]

result = execute_with_fallback(
    router,
    original_embedding,
    budget=0.01,
    latency_sla=2.0,
    candidates=candidate_tools,
    call_tool=call_tool,
    max_attempts=3,
    context_updater=update_context,
)
```

The failed attempt is learned against the context that caused it. The newly
embedded context is used to rerank all remaining tools on the next attempt.
Without `context_updater`, retries retain the original fixed-context behavior.

The embedding encoder and pass/fail evaluator are intentionally external. In a
BiasBusters experiment, the five entries in `api_list` are the candidates. In
production, a semantic retriever should create the candidate list first.

## DeepSeek judge

Set `DEEPSEEK_API_KEY` and evaluate the normalized tool output against an
explicit per-cluster contract:

```python
from budget_tool_router import DeepSeekJudge
from budget_tool_router.runner import ToolCallResult

judge = DeepSeekJudge()  # deepseek-flash by default
judgment = judge.evaluate(
    query=user_query,
    expected_contract=cluster_contract,
    tool_name=tool_name,
    tool_output=normalized_output,
)

result = ToolCallResult(
    passed=judgment.passed,
    cost=tool_cost,  # add judge API cost here when pricing is configured
    latency=tool_latency + judgment.latency,
    output={"raw": normalized_output, "judgment": judgment},
)
```

Only a `pass` at confidence `>= 0.80` becomes a positive LinUCB reward.
`partial`, `fail`, and low-confidence `uncertain` results do not. Preserve a
partial result in the evolving context, but never treat instructions inside a
tool response as trusted prompt instructions.
