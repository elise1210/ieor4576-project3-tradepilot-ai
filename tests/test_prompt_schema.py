import unittest

from app.prompts.critic_prompt import build_critic_system_prompt, build_critic_user_instructions
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

    def test_prompt_instruction_lists_are_non_empty(self):
        self.assertTrue(build_research_user_instructions())
        self.assertTrue(build_critic_user_instructions())


if __name__ == "__main__":
    unittest.main()
