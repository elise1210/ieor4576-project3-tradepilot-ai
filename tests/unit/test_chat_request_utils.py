import unittest

from app.chat_request_utils import resolve_effective_query


class ChatRequestUtilsTests(unittest.TestCase):
    def test_resolve_effective_query_uses_direct_query(self):
        result = resolve_effective_query(query="Should I buy Apple today?")

        self.assertEqual(result, "Should I buy Apple today?")

    def test_resolve_effective_query_builds_time_horizon_follow_up(self):
        result = resolve_effective_query(
            original_query="Should I buy Apple today?",
            clarification_type="time_horizon",
            clarification_value="short_term",
        )

        self.assertEqual(
            result,
            "Should I buy Apple today?\n\nClarification: I want a short-term trading view.",
        )

    def test_resolve_effective_query_raises_when_missing_inputs(self):
        with self.assertRaises(ValueError):
            resolve_effective_query()


if __name__ == "__main__":
    unittest.main()
