"""Offline tests for the schema-visible OpenClaw meal-report adapter."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
import subprocess
from typing import Any
import unittest

from nutrition_optimizer.openclaw_meal_report import OpenClawMealReportError, OpenClawMealReportInterpreter
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import FoodResolutionRequest, LocalFDFoodResolver, ResolvedFood
from nutrition_optimizer.meal_report import MealPlan, PlannedMealItem
from tests.test_food_resolution import DAY1, T1, component, mapped, menu_day


def envelope(payload: object) -> dict[str, object]:
    return {"ok": True, "toolName": "llm-task", "source": "plugin", "output": {"details": {"json": payload, "provider": "openai", "model": "gpt-5.6-luna"}}}


class Runner:
    def __init__(self, payload: object) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.stdout = json.dumps(envelope(payload))

    def __call__(self, args: list[str], **kwargs: Any):
        self.calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, self.stdout, "")


class OpenClawMealReportTests(unittest.TestCase):
    def make_plan(self):
        from tempfile import TemporaryDirectory
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        catalog = OfficialNutritionCatalog(Path(directory.name) / "nutrition.sqlite3")
        self.addCleanup(catalog.close)
        catalog.synchronize_fd_refresh(
            mapped(menu_day(DAY1, 3, "Dinner", ((component(1, "Chicken Tenders"), "Grill"),))),
            requested_start=DAY1,
            requested_end=DAY1,
            observed_at=T1,
        )
        food = LocalFDFoodResolver(catalog).resolve(
            FoodResolutionRequest("Chicken Tenders", DAY1, meal="Dinner")
        )
        assert isinstance(food, ResolvedFood)
        return MealPlan(DAY1, "Dinner", (PlannedMealItem(food, Decimal("1"), "3 tenders"),))

    def test_uses_luna_high_schema_visible_plan_references_and_no_nutrition(self) -> None:
        runner = Runner({"planned_items": [{"plan_item_id": "item_1", "reference_text": "tenders", "action": "eaten", "quantity_relation": "as_recommended", "quantity_text": None}], "additional_foods": [], "unresolved_statements": []})
        interpreter = OpenClawMealReportInterpreter("/test-only/openclaw", runner=runner)
        result = interpreter.interpret(self.make_plan(), "I ate the tenders")
        self.assertEqual(result.planned_items[0].plan_item_id, "item_1")
        params = json.loads(runner.calls[0][0][-1])
        args = params["args"]
        self.assertEqual(args["model"], "openai/gpt-5.6-luna")
        self.assertEqual(args["thinking"], "high")
        self.assertIn("OUTPUT CONTRACT", args["prompt"])
        self.assertIn('"plan_item_id"', args["prompt"])
        self.assertIn('"quantity_relation"', args["prompt"])
        visible = json.loads(args["input"])
        self.assertEqual(visible["plan_items"][0]["plan_item_id"], "item_1")
        self.assertNotIn("calories", args["input"])
        self.assertNotIn("component_identity", args["input"])
        self.assertNotIn("recommended_official_servings", args["input"])

    def test_malformed_output_is_sanitized(self) -> None:
        interpreter = OpenClawMealReportInterpreter(
            "/test-only/openclaw",
            runner=Runner({"planned_items": [], "additional_foods": [], "unresolved_statements": [], "unexpected": True}),
        )
        with self.assertRaisesRegex(OpenClawMealReportError, "invalid output"):
            interpreter.interpret(self.make_plan(), "private user report")

    def test_contextual_turn_exposes_station_and_draft_questions_without_nutrition(self) -> None:
        runner = Runner(
            {
                "intent": "clarification_answer",
                "report_scope": "partial",
                "location_plan_item_id": None,
                "location_food_text": None,
                "planned_items": [],
                "additional_foods": [],
                "unresolved_statements": [],
            }
        )
        interpreter = OpenClawMealReportInterpreter("/test-only/openclaw", runner=runner)
        interpreter.interpret_with_context(
            self.make_plan(),
            "1/4",
            {
                "draft": {
                    "state": "awaiting_clarification",
                    "resolved_items": [],
                    "outstanding_questions": [
                        {"reason": "portion_semantic_matcher_unavailable", "plan_item_id": "item_1"}
                    ],
                    "last_clarification": "How much Chicken Tenders did you eat?",
                }
            },
        )

        params = json.loads(runner.calls[0][0][-1])
        visible = json.loads(params["args"]["input"])
        self.assertEqual(visible["plan_items"][0]["station_name"], "Grill")
        self.assertEqual(visible["conversation_context"]["draft"]["outstanding_questions"][0]["plan_item_id"], "item_1")
        rendered = params["args"]["input"]
        for forbidden in ("calories", "recommended_official_servings", "nutrition_snapshot_id"):
            self.assertNotIn(forbidden, rendered)

    def test_missing_or_inconsistent_quantity_relation_is_sanitized(self) -> None:
        for item in (
            {
                "plan_item_id": "item_1",
                "reference_text": "tenders",
                "action": "eaten",
                "quantity_text": None,
            },
            {
                "plan_item_id": "item_1",
                "reference_text": "tenders",
                "action": "eaten",
                "quantity_relation": "as_recommended",
                "quantity_text": "two servings",
            },
        ):
            with self.subTest(item=item):
                interpreter = OpenClawMealReportInterpreter(
                    "/test-only/openclaw",
                    runner=Runner(
                        {
                            "planned_items": [item],
                            "additional_foods": [],
                            "unresolved_statements": [],
                        }
                    ),
                )
                with self.assertRaisesRegex(OpenClawMealReportError, "invalid output"):
                    interpreter.interpret(self.make_plan(), "private user report")

    def test_adapter_has_no_direct_api_key_diningbucket_or_sol_dependency(self) -> None:
        import nutrition_optimizer.openclaw_meal_report as module
        source = Path(module.__file__).read_text(encoding="utf-8")
        for forbidden in ("OPENAI_API_KEY", "DiningBucket", "gpt-5.6-sol", "OpenAI("):
            self.assertNotIn(forbidden, source)

    def test_no_active_request_schema_routes_positive_and_non_request_wording_without_menu_authority(self) -> None:
        cases = (
            (
                "I want pizza",
                {
                    "intent": "meal_request",
                    "request_mode": "targeted",
                    "requested_food_texts": ["pizza"],
                    "requested_meal": None,
                },
                "meal_request",
            ),
            (
                "I had pizza",
                {
                    "intent": "meal_report",
                    "request_mode": None,
                    "requested_food_texts": [],
                    "requested_meal": None,
                },
                "meal_report",
            ),
            (
                "I don't want pizza",
                {
                    "intent": "replacement_request",
                    "request_mode": None,
                    "requested_food_texts": [],
                    "requested_meal": None,
                },
                "replacement_request",
            ),
            (
                "Where is pizza?",
                {
                    "intent": "location_question",
                    "request_mode": None,
                    "requested_food_texts": [],
                    "requested_meal": None,
                },
                "location_question",
            ),
            (
                "I want something lighter for dinner",
                {
                    "intent": "meal_request",
                    "request_mode": "whole_meal",
                    "requested_food_texts": [],
                    "requested_meal": "dinner",
                },
                "meal_request",
            ),
        )
        for text, payload, expected_intent in cases:
            with self.subTest(text=text):
                runner = Runner(payload)
                interpreter = OpenClawMealReportInterpreter("/test-only/openclaw", runner=runner)
                result = interpreter.interpret_meal_request(text)

                self.assertEqual(result.intent, expected_intent)
                visible = json.loads(json.loads(runner.calls[0][0][-1])["args"]["input"])
                self.assertEqual(visible, {"user_message": text})
                rendered = runner.calls[0][0][-1]
                self.assertNotIn("component_identity", rendered)
                self.assertNotIn("calories", rendered)

    def test_report_routing_exposes_only_message_and_keeps_brunch_distinct(self) -> None:
        runner = Runner(
            {
                "intent": "report_or_clarification",
                "explicit_meal_slot": "brunch",
            }
        )
        interpreter = OpenClawMealReportInterpreter("/test-only/openclaw", runner=runner)

        result = interpreter.interpret_report_routing("At brunch I ate chicken")

        self.assertEqual(result.explicit_meal_slot, "brunch")
        params = json.loads(runner.calls[0][0][-1])
        visible = json.loads(params["args"]["input"])
        self.assertEqual(visible, {"user_message": "At brunch I ate chicken"})
        rendered = runner.calls[0][0][-1]
        for forbidden in ("plan_items", "component_identity", "calories", "fd:2"):
            self.assertNotIn(forbidden, rendered)

    def test_report_routing_rejects_requested_meal_as_report_scope(self) -> None:
        interpreter = OpenClawMealReportInterpreter(
            "/test-only/openclaw",
            runner=Runner(
                {
                    "intent": "other_or_unclear",
                    "explicit_meal_slot": "lunch",
                }
            ),
        )

        with self.assertRaisesRegex(OpenClawMealReportError, "invalid output"):
            interpreter.interpret_report_routing("I want chicken for lunch")


if __name__ == "__main__":
    unittest.main()
