import unittest

from app.prompts.critic_prompt import build_critic_system_prompt, build_critic_user_instructions
from app.prompts.decision_prompt import build_decision_system_prompt, build_decision_user_instructions
from app.prompts.planner_prompt import build_planner_system_prompt, build_planner_user_instructions
from app.prompts.research_prompt import build_research_system_prompt, build_research_user_instructions
from app.skills.schema import format_skill_schema, get_skill_schema


class PromptSchemaTests(unittest.TestCase):
    def test_news_skill_schema_contains_expected_fields(self):
        schema = get_skill_schema("news")

        self.assertIsNotNone(schema)
        self.assertIn("purpose", schema)
        self.assertIn("inputs", schema)
        self.assertIn("outputs", schema)
        self.assertIn("article_count", schema["outputs"])

    def test_formatted_market_schema_mentions_requested_date(self):
        rendered = format_skill_schema("market")

        self.assertIn("lookback_days", rendered)
        self.assertIn("requested_date", rendered)
        self.assertIn("history", rendered)

    def test_research_prompt_includes_skill_schema_guidance(self):
        prompt = build_research_system_prompt()

        self.assertIn("Available skill schemas", prompt)
        self.assertIn("Skill: news", prompt)
        self.assertIn("Skill: market", prompt)
        self.assertIn("sentiment should follow news", prompt)

    def test_critic_prompt_includes_semantic_critic_guidance(self):
        prompt = build_critic_system_prompt()

        self.assertIn("semantic evidence critic", prompt)
        self.assertIn("Available skill schemas", prompt)
        self.assertIn("thin evidence", prompt)
        self.assertIn("Return exactly these top-level fields", prompt)
        self.assertIn("follow_up_steps should usually be non-empty", prompt)
        self.assertIn("Example 1:", prompt)
        self.assertIn('"skill": "news"', prompt)

    def test_prompt_instruction_lists_are_non_empty(self):
        self.assertTrue(build_research_user_instructions())
        self.assertTrue(build_critic_user_instructions())
        self.assertTrue(build_planner_user_instructions())
        self.assertTrue(build_decision_user_instructions())

    def test_planner_prompt_mentions_json_and_schema_fields(self):
        prompt = build_planner_system_prompt()

        self.assertIn("return JSON only", prompt)
        self.assertIn("Allowed intent values", prompt)
        self.assertIn("Return exactly these fields", prompt)

    def test_decision_prompt_mentions_json_and_allowed_output_shapes(self):
        prompt = build_decision_system_prompt()

        self.assertIn("Return JSON only", prompt)
        self.assertIn("single-stock decisions return", prompt)
        self.assertIn("comparison decisions return", prompt)


if __name__ == "__main__":
    unittest.main()
