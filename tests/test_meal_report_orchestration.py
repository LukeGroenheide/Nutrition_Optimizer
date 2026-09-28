"""Focused application coverage for inbound known-meal report processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nutrition_optimizer.application import (
    MealReportNoActivePlanError,
    MealReportOrchestrator,
)
from nutrition_optimizer.durable_state import DurableMealState
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import FoodResolutionRequest, LocalFDFoodResolver, ResolvedFood
from nutrition_optimizer.meal_report import MealPlan, MealReportReconciler, PlannedMealItem
from nutrition_optimizer.meal_report_structured import MealReportSemanticResult
from nutrition_optimizer.nutrition import DailyLedger, IntakeEntry
from nutrition_optimizer.portion_interpretation import (
    NaturalPortionInterpreter,
    PortionInterpretationRequest,
    PortionSemanticDecision,
)
from tests.test_food_resolution import component, mapped, menu_day


DAY = date(2026, 8, 25)
OBSERVED_AT = datetime(2026, 8, 25, 12, tzinfo=timezone.utc)


def food(
    component_id: int,
    name: str,
    *,
    calories: str = "100",
    protein: str = "20",
    carbohydrates: str = "10",
    fat: str = "5",
    fiber: str = "1",
    serving_quantity: str = "1",
    serving_unit: str = "Each",
) -> dict[str, object]:
    value = component(component_id, name, protein=protein)
    value.update(
        {
            "recipePortionSize": serving_quantity,
            "recipePortionSizeUnit": serving_unit,
            "calories": calories,
            "protein": protein,
            "carbohydrates": carbohydrates,
            "fat": fat,
            "dietaryFiber": fiber,
            "dietaryFiberUOM": "g",
        }
    )
    return value


def semantic(
    planned: list[dict[str, object]] | None = None,
    additional: list[dict[str, object]] | None = None,
    unresolved: list[dict[str, object]] | None = None,
) -> MealReportSemanticResult:
    planned_items = [
        (
            item
            if "quantity_relation" in item
            else {
                **item,
                "quantity_relation": (
                    None
                    if item.get("action") == "skipped"
                    else "as_recommended"
                    if item.get("quantity_text") is None
                    else "modified"
                ),
            }
        )
        for item in (planned or [])
    ]
    return MealReportSemanticResult.model_validate(
        {
            "planned_items": planned_items,
            "additional_foods": additional or [],
            "unresolved_statements": unresolved or [],
        }
    )


@dataclass
class FakeMealReportSemanticInterpreter:
    result: object
    calls: list[tuple[MealPlan, str]] = field(default_factory=list)

    def interpret(self, plan: MealPlan, user_text: str) -> object:
        self.calls.append((plan, user_text))
        return self.result


@dataclass
class FakePortionMatcher:
    result: PortionSemanticDecision
    calls: list[PortionInterpretationRequest] = field(default_factory=list)

    def decide(self, request: PortionInterpretationRequest) -> PortionSemanticDecision:
        self.calls.append(request)
        return self.result


class MealReportOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog = OfficialNutritionCatalog(Path(self.directory.name) / "nutrition.sqlite3")
        self.addCleanup(self.catalog.close)
        self.catalog.synchronize_fd_refresh(
            mapped(
                menu_day(
                    DAY,
                    1,
                    "Breakfast",
                    (
                        (food(1, "Eggs"), "Grill"),
                        (food(2, "Cereal", serving_quantity="1", serving_unit="Cup"), "Cold Bar"),
                        (food(3, "Broccoli"), "Vegetables"),
                        (
                            food(
                                4,
                                "Chicken Tender",
                                serving_quantity="2",
                                serving_unit="Each",
                            ),
                            "Grill",
                        ),
                    ),
                )
            ),
            requested_start=DAY,
            requested_end=DAY,
            observed_at=OBSERVED_AT,
        )
        self.state = DurableMealState(self.catalog)
        self.resolver = LocalFDFoodResolver(self.catalog)

    def resolved(self, name: str) -> ResolvedFood:
        result = self.resolver.resolve(FoodResolutionRequest(name, DAY, meal=1))
        assert isinstance(result, ResolvedFood)
        return result

    def breakfast_plan(self) -> MealPlan:
        return MealPlan(
            DAY,
            1,
            (
                PlannedMealItem(self.resolved("Eggs"), Decimal("1"), "two eggs"),
                PlannedMealItem(self.resolved("Cereal"), Decimal("1.5"), "a bowl of cereal"),
                PlannedMealItem(self.resolved("Broccoli"), Decimal("1"), "some broccoli"),
            ),
        )

    def orchestrator(
        self,
        result: MealReportSemanticResult,
        *,
        portion_interpreter: NaturalPortionInterpreter | None = None,
    ) -> tuple[MealReportOrchestrator, FakeMealReportSemanticInterpreter]:
        interpreter = FakeMealReportSemanticInterpreter(result)
        reconciler = MealReportReconciler(
            interpreter,
            self.resolver,
            portion_interpreter or NaturalPortionInterpreter(),
        )
        return MealReportOrchestrator(self.catalog, reconciler), interpreter

    def application_count(self) -> int:
        return int(
            self.catalog._connection.execute(
                "SELECT COUNT(*) FROM meal_report_applications"
            ).fetchone()[0]
        )

    def test_happy_path_uses_the_active_plan_and_preserves_exact_history(self) -> None:
        plan = self.breakfast_plan()
        saved = self.state.save_meal_plan(plan, plan_id="breakfast-active")
        orchestrator, _ = self.orchestrator(
            semantic(
                [
                    {
                        "plan_item_id": "item_1",
                        "reference_text": "the eggs",
                        "action": "eaten",
                        "quantity_text": None,
                    },
                    {
                        "plan_item_id": "item_2",
                        "reference_text": "the cereal",
                        "action": "eaten",
                        "quantity_text": None,
                    },
                    {
                        "plan_item_id": "item_3",
                        "reference_text": "the broccoli",
                        "action": "skipped",
                        "quantity_text": None,
                    },
                ]
            )
        )

        result = orchestrator.process(
            DAY,
            " breakfast ",
            "I ate the recommended amounts of eggs and cereal but skipped the broccoli.",
            "inbound-breakfast-1",
        )

        self.assertEqual(result.outcome, "applied")
        self.assertEqual(result.persisted_plan.plan_id, saved.plan_id)
        self.assertEqual(result.persisted_plan.status, "applied")
        self.assertEqual(result.message, "I understood that as:\n- Eggs\n- Cereal\n\nSkipped:\n- Broccoli")
        assert result.reconciled_report is not None
        self.assertEqual(result.reconciled_report.unspecified_items, ())
        self.assertEqual(
            [item.official_servings for item in result.reconciled_report.eaten_items],
            [Decimal("1"), Decimal("1.5")],
        )
        intake = self.state.get_daily_intake(DAY)
        self.assertEqual([entry.plan_item_id for entry in intake], ["item_1", "item_2"])
        self.assertEqual([entry.official_servings for entry in intake], [Decimal("1"), Decimal("1.5")])
        self.assertEqual(
            [entry.occurrence.nutrition_snapshot_id for entry in intake],
            [plan.items[0].food.nutrition_snapshot_id, plan.items[1].food.nutrition_snapshot_id],
        )
        expected_ledger = DailyLedger(
            (
                IntakeEntry(plan.items[0].record, Decimal("1")),
                IntakeEntry(plan.items[1].record, Decimal("1.5")),
            )
        )
        self.assertEqual(result.updated_ledger, expected_ledger)
        self.assertIsNone(self.state.load_active_meal_plan(DAY, "Breakfast"))
        self.assertEqual(self.application_count(), 1)

    def test_semantic_quantity_estimate_creates_no_authoritative_intake(self) -> None:
        tender_plan = MealPlan(
            DAY,
            1,
            (
                PlannedMealItem(
                    self.resolved("Chicken Tender"),
                    Decimal("1"),
                    "two chicken tenders",
                ),
            ),
        )
        self.state.save_meal_plan(tender_plan)
        matcher = FakePortionMatcher(
            PortionSemanticDecision("estimate", Decimal("0.5"), "high", "one_tender")
        )
        orchestrator, _ = self.orchestrator(
            semantic(
                [
                    {
                        "plan_item_id": "item_1",
                        "reference_text": "one tender",
                        "action": "eaten",
                        "quantity_text": "one tender",
                    }
                ]
            ),
            portion_interpreter=NaturalPortionInterpreter(semantic_matcher=matcher),
        )

        result = orchestrator.process(
            DAY,
            1,
            "I only ate one tender.",
            "inbound-tender-1",
        )

        assert result.reconciled_report is not None
        self.assertEqual(len(matcher.calls), 1)
        self.assertEqual(result.outcome, "clarification_required")
        self.assertEqual(result.reconciled_report.eaten_items, ())
        self.assertEqual(
            result.reconciled_report.clarification_items[0].reason,
            "semantic_quantity_not_authoritative",
        )
        self.assertEqual(self.state.get_daily_intake(DAY), ())

    def test_safely_applicable_partial_report_leaves_unspecified_items_unlogged(self) -> None:
        plan = self.breakfast_plan()
        self.state.save_meal_plan(plan)
        orchestrator, _ = self.orchestrator(
            semantic(
                [
                    {
                        "plan_item_id": "item_1",
                        "reference_text": "the eggs",
                        "action": "eaten",
                        "quantity_text": None,
                    }
                ]
            )
        )

        result = orchestrator.process(DAY, "breakfast", "I ate all of the eggs.", "partial-event")

        assert result.reconciled_report is not None
        self.assertEqual(
            [item.display_name for item in result.reconciled_report.unspecified_items],
            ["Cereal", "Broccoli"],
        )
        self.assertEqual([entry.record.name for entry in self.state.get_daily_intake(DAY)], ["Eggs"])
        self.assertIn("I left the other planned items unlogged.", result.message)
        self.assertEqual(result.persisted_plan.status, "applied")

    def test_same_source_event_replays_after_completion_without_a_second_reconciliation(self) -> None:
        plan = self.breakfast_plan()
        self.state.save_meal_plan(plan)
        orchestrator, interpreter = self.orchestrator(
            semantic(
                [
                    {
                        "plan_item_id": "item_1",
                        "reference_text": "eggs",
                        "action": "eaten",
                        "quantity_text": None,
                    },
                    {
                        "plan_item_id": "item_2",
                        "reference_text": "cereal",
                        "action": "eaten",
                        "quantity_text": None,
                    },
                    {
                        "plan_item_id": "item_3",
                        "reference_text": "broccoli",
                        "action": "skipped",
                        "quantity_text": None,
                    },
                ]
            )
        )

        first = orchestrator.process(DAY, 1, "all of the recommended eggs and cereal; skipped broccoli", "retry-event")
        replay = orchestrator.process(DAY, "Breakfast", "all of the recommended eggs and cereal; skipped broccoli", "retry-event")

        self.assertEqual(first.outcome, "applied")
        self.assertFalse(first.already_applied)
        self.assertEqual(replay.outcome, "replayed")
        self.assertTrue(replay.already_applied)
        self.assertEqual(replay.message, "Got it — that meal report was already logged.")
        self.assertEqual(len(interpreter.calls), 1)
        self.assertEqual(self.application_count(), 1)
        self.assertEqual(len(self.state.get_daily_intake(DAY)), 2)
        loaded = self.state.load_meal_plan(first.persisted_plan.plan_id)
        assert loaded is not None
        self.assertEqual(loaded.status, "applied")

    def test_clarification_persists_nothing_and_leaves_the_plan_active(self) -> None:
        plan = self.breakfast_plan()
        saved = self.state.save_meal_plan(plan)
        orchestrator, _ = self.orchestrator(
            semantic(
                [
                    {
                        "plan_item_id": "item_1",
                        "reference_text": "some eggs",
                        "action": "eaten",
                        "quantity_text": "some",
                    }
                ]
            )
        )

        result = orchestrator.process(DAY, 1, "I ate some of it.", "clarify-event")

        self.assertEqual(result.outcome, "clarification_required")
        self.assertTrue(result.clarification_required)
        self.assertIsNone(result.application)
        self.assertIsNone(result.updated_ledger)
        self.assertEqual(self.state.get_daily_intake(DAY), ())
        self.assertEqual(self.application_count(), 0)
        active = self.state.load_active_meal_plan(DAY, "Breakfast")
        self.assertEqual(active, saved)
        self.assertIn("How much Eggs did you eat?", result.message)
        assert result.reconciled_report is not None
        self.assertEqual(result.reconciled_report.skipped_items, ())
        self.assertEqual(result.reconciled_report.unspecified_items, plan.items)

    def test_wrong_service_context_never_falls_back_to_another_active_plan(self) -> None:
        self.state.save_meal_plan(self.breakfast_plan())
        orchestrator, _ = self.orchestrator(semantic())

        with self.assertRaisesRegex(MealReportNoActivePlanError, "no active meal plan"):
            orchestrator.process(DAY, "dinner", "I ate it.", "wrong-meal-event")
        with self.assertRaisesRegex(MealReportNoActivePlanError, "no active meal plan"):
            orchestrator.process(date(2026, 8, 26), "breakfast", "I ate it.", "wrong-day-event")
        self.assertEqual(self.state.get_daily_intake(DAY), ())
        self.assertEqual(self.application_count(), 0)


if __name__ == "__main__":
    unittest.main()
