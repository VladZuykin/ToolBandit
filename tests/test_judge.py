import unittest

from budget_tool_router import DeepSeekJudge


class DeepSeekJudgeTests(unittest.TestCase):
    def test_parses_high_confidence_pass_and_requests_json(self):
        captured = {}

        def transport(payload):
            captured.update(payload)
            return {
                "model": "deepseek-flash",
                "choices": [{"finish_reason": "stop", "message": {"content": (
                    '{"verdict":"pass","confidence":0.96,'
                    '"reason_code":"COMPLETE","explanation":"ok",'
                    '"missing_requirements":[],"useful_partial_result":null}'
                )}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            }

        judge = DeepSeekJudge(transport=transport)
        result = judge.evaluate(query="weather", expected_contract="temperature",
                                tool_name="weather-api", tool_output={"temperature": 10})
        self.assertTrue(result.passed)
        self.assertEqual(result.total_tokens, 120)
        self.assertEqual(captured["response_format"], {"type": "json_object"})

    def test_low_confidence_pass_becomes_uncertain(self):
        def transport(payload):
            return {"choices": [{"finish_reason": "stop", "message": {"content": (
                '{"verdict":"pass","confidence":0.51,'
                '"reason_code":"MAYBE","explanation":"unclear",'
                '"missing_requirements":[],"useful_partial_result":null}'
            )}}]}

        result = DeepSeekJudge(transport=transport).evaluate(
            query="q", expected_contract="c", tool_name="t", tool_output="o")
        self.assertEqual(result.verdict, "uncertain")
        self.assertFalse(result.passed)

    def test_untrusted_output_is_delimited_and_truncated(self):
        captured = {}

        def transport(payload):
            captured.update(payload)
            return {"choices": [{"finish_reason": "stop", "message": {"content": (
                '{"verdict":"fail","confidence":1.0,'
                '"reason_code":"BAD","explanation":"bad",'
                '"missing_requirements":["x"],"useful_partial_result":null}'
            )}}]}

        judge = DeepSeekJudge(transport=transport, max_input_chars=10)
        judge.evaluate(query="q", expected_contract="c", tool_name="t", tool_output="x" * 100)
        prompt = captured["messages"][1]["content"]
        self.assertIn("[TRUNCATED]", prompt)
        self.assertNotIn("x" * 11, prompt)

    def test_retries_once_when_output_token_limit_is_reached(self):
        calls = []

        def transport(payload):
            calls.append(payload["max_tokens"])
            if len(calls) == 1:
                return {"choices": [{"finish_reason": "length", "message": {"content": "{"}}]}
            return {"choices": [{"finish_reason": "stop", "message": {"content": (
                '{"verdict":"pass","confidence":0.99,'
                '"reason_code":"OK","explanation":"ok",'
                '"missing_requirements":[],"useful_partial_result":null}'
            )}}]}

        result = DeepSeekJudge(transport=transport, max_output_tokens=400).evaluate(
            query="q", expected_contract="c", tool_name="t", tool_output={"ok": True})
        self.assertTrue(result.passed)
        self.assertEqual(calls, [400, 800])


if __name__ == "__main__":
    unittest.main()
