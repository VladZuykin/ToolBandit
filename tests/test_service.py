import time
import unittest

import numpy as np

from budget_tool_router import (
    BudgetAwareToolRouter, DeepSeekJudge, SemanticToolRetriever,
    ToolDocument, ToolRoutingService, ToolSpec,
)


class FakeEncoder:
    def encode_many(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "weather" in lowered or "temperature" in lowered:
                vectors.append(np.array([1.0, 0.0, 0.0]))
            elif "domain" in lowered or "whois" in lowered:
                vectors.append(np.array([0.0, 1.0, 0.0]))
            else:
                vectors.append(np.array([0.0, 0.0, 1.0]))
        return vectors


def fake_judge():
    def transport(payload):
        return {"model": "fake", "choices": [{"finish_reason": "stop", "message": {
            "content": ('{"verdict":"pass","confidence":0.99,'
                        '"reason_code":"OK","explanation":"ok",'
                        '"missing_requirements":[],"useful_partial_result":null}')
        }}], "usage": {"total_tokens": 10}}
    return DeepSeekJudge(transport=transport)


class ServiceTests(unittest.TestCase):
    def make_service(self):
        documents = [
            ToolDocument("weather", "Weather temperature forecast", {"kind": "weather"}),
            ToolDocument("whois", "WHOIS domain registration", {"kind": "domain"}),
        ]
        retriever = SemanticToolRetriever(FakeEncoder(), documents)
        router = BudgetAwareToolRouter([
            ToolSpec("weather", 0.001, 0.2, fixed_cost=0.001,
                     latency_range=0.1, prior_count=100),
            ToolSpec("whois", 0.001, 0.2, fixed_cost=0.001,
                     latency_range=0.1, prior_count=100),
        ], context_dimension=3, diagonal_covariance=True,
           cost_weight=0.5, latency_weight=0.5)
        return ToolRoutingService(retriever, router, fake_judge())

    def test_search_returns_retrieval_and_bandit_scores(self):
        service = self.make_service()
        result = service.search({
            "query": "weather in Moscow", "remaining_budget": 0.01,
            "latency_sla": 1.0, "retrieval_limit": 2, "result_limit": 2,
            "retrieval_threshold": 0.1,
        })
        self.assertNotIn("recommended_tool", result)
        self.assertEqual(result["tools"][0]["tool"], "weather")
        self.assertEqual(len(result["tools"]), 1)
        self.assertEqual(result["tools"][0]["retrieval_rank"], 1)
        self.assertIn("retrieval_score", result["tools"][0])
        self.assertIn("ucb_score", result["tools"][0])
        self.assertNotIn("routing_rank", result["tools"][0])
        self.assertNotIn("bandit_rank", result["tools"][0])
        self.assertNotIn("estimated_cost", result["tools"][0])
        self.assertNotIn("estimated_latency", result["tools"][0])
        self.assertNotIn("remaining_budget", result["tools"][0])
        self.assertNotIn("cost_ucb", result["tools"][0])

    def test_retrieval_order_is_not_changed_by_ucb_score(self):
        service = self.make_service()
        result = service.search({
            "query": "weather temperature", "remaining_budget": 1.0,
            "retrieval_threshold": 0.0, "retrieval_limit": 2, "result_limit": 2,
        })
        self.assertNotIn("recommended_tool", result)
        self.assertEqual([row["tool"] for row in result["tools"]], ["weather", "whois"])

    def test_search_respects_upstream_candidate_set(self):
        service = self.make_service()
        result = service.search({
            "query": "weather temperature", "remaining_budget": 1.0,
            "retrieval_threshold": 0.0, "retrieval_limit": 2, "result_limit": 2,
            "candidate_tools": ["whois"],
        })
        self.assertEqual([row["tool"] for row in result["tools"]], ["whois"])

    def test_async_judge_performs_delayed_update(self):
        service = self.make_service()
        search = service.search({
            "query": "weather in Moscow", "remaining_budget": 0.01,
            "latency_sla": 1.0,
        })
        request = {
            "interaction_id": "same-call",
            "request_id": search["request_id"],
            "context_text": "weather in Moscow",
            "tool_intent": "get temperature",
            "expected_contract": "numeric temperature",
            "tool_name": "weather",
            "tool_output": {"temperature": 12},
            "observed_cost": 0.001,
            "observed_latency": 0.2,
        }
        submitted = service.submit_evaluation(request)
        duplicate = service.submit_evaluation(request)
        self.assertEqual(submitted["evaluation_id"], duplicate["evaluation_id"])
        deadline = time.time() + 2.0
        while time.time() < deadline:
            job = service.get_evaluation(submitted["evaluation_id"])
            if job["status"] != "pending":
                break
            time.sleep(0.01)
        self.assertEqual(job["status"], "completed")
        self.assertTrue(job["learned"])
        self.assertEqual(
            service.router.snapshot()["tools"]["weather"]["reward_observations"], 1
        )


if __name__ == "__main__":
    unittest.main()
