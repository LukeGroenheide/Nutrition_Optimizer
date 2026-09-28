"""Offline tests for read-only known-meal report reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import nutrition_optimizer.meal_report as meal_report
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import FoodResolutionRequest, LocalFDFoodResolver, ResolvedFood
from nutrition_optimizer.meal_report import MealPlan, MealReportReconciler, PlannedMealItem
from nutrition_optimizer.meal_report_structured import MealReportSemanticResult
from nutrition_optimizer.nutrition import DailyLedger
from nutrition_optimizer.portion_interpretation import (
    NaturalPortionInterpreter,
    PortionInterpretationRequest,
    PortionSemanticDecision,
)
from tests.test_food_resolution import DAY1, T1, component, mapped, menu_day


def semantic(
    planned: list[dict[str, object]] | None = None,
    additional: list[dict[str, object]] | None = None,
    unresolved: list[dict[str, object]] | None = None,
):
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
class FakeSemanticInterpreter:
    result: object

    def __post_init__(self) -> None:
        self.calls: list[tuple[MealPlan, str]] = []

    def interpret(self, plan: MealPlan, user_text: str):
        self.calls.append((plan, user_text))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


@dataclass
class FakePortionMatcher:
    result: object

    def __post_init__(self) -> None:
        self.calls: list[PortionInterpretationRequest] = []

    def decide(self, request: PortionInterpretationRequest):
        self.calls.append(request)
        return self.result


class MealReportReconcilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog = OfficialNutritionCatalog(Path(self.directory.name) / "nutrition.sqlite3")
        self.addCleanup(self.catalog.close)
        recipes = (
            (component(1, "Chicken Tenders"), "Grill"),
            (component(2, "Cereal"), "Cold Bar"),
            (component(3, "Broccoli"), "Vegetables"),
            (component(4, "Eggs"), "Grill"),
            (component(5, "Sausage"), "Grill"),
            (component(6, "Cavatappi and Cheese"), "Pasta"),
            (component(7, "Mashed Potatoes"), "Sides"),
        )
        self.catalog.synchronize_fd_refresh(
            mapped(menu_day(DAY1, 3, "Dinner", recipes)),
            requested_start=DAY1,
            requested_end=DAY1,
            observed_at=T1,
        )
        self.resolver = LocalFDFoodResolver(self.catalog)
        self.foods = {
            name: self._food(name)
            for name in (
                "Chicken Tenders", "Cereal", "Broccoli", "Eggs", "Sausage",
                "Cavatappi and Cheese", "Mashed Potatoes",
            )
        }

    def _food(self, name: str) -> ResolvedFood:
        result = self.resolver.resolve(FoodResolutionRequest(name, DAY1, meal="Dinner"))
        assert isinstance(result, ResolvedFood)
        return result

    def plan(self, names: tuple[str, ...] = ("Chicken Tenders", "Cereal", "Broccoli")) -> MealPlan:
        text = {
            "Chicken Tenders": "3 chicken tenders",
            "Cereal": "a bowl of cereal",
            "Broccoli": "some broccoli",
            "Eggs": "two eggs",
            "Sausage": "a sausage patty",
            "Cavatappi and Cheese": "some mac and cheese",
        }
        quantities = {"Cereal": Decimal("1.5")}
        return MealPlan(
            DAY1,
            "Dinner",
            tuple(
                PlannedMealItem(self.foods[name], quantities.get(name, Decimal("1")), text[name])
                for name in names
            ),
        )

    def reconcile(self, result: object, *, plan: MealPlan | None = None, portion=None, user_text="I ate what you recommended"):
        return MealReportReconciler(
            FakeSemanticInterpreter(result),
            self.resolver,
            portion or NaturalPortionInterpreter(),
        ).reconcile(plan or self.plan(), user_text)

    def test_explicit_attestation_inherits_exact_decimal_without_portion_call(self) -> None:
        plan = self.plan()
        matcher = FakePortionMatcher(PortionSemanticDecision("estimate", Decimal("9"), "low", "unused"))
        result = self.reconcile(
            semantic([{"plan_item_id": "item_2", "reference_text": "the cereal", "action": "eaten", "quantity_text": None}]),
            plan=plan,
            portion=NaturalPortionInterpreter(semantic_matcher=matcher),
        )
        self.assertEqual(result.eaten_items[0].official_servings, Decimal("1.5"))
        self.assertEqual(result.eaten_items[0].quantity_source, "planned_quantity")
        self.assertEqual(matcher.calls, [])
        self.assertEqual(result.unspecified_items, (plan.items[0], plan.items[2]))

    def test_skipped_is_not_an_eaten_zero_quantity(self) -> None:
        result = self.reconcile(semantic([{"plan_item_id": "item_3", "reference_text": "the broccoli", "action": "skipped", "quantity_text": None}]))
        self.assertEqual(result.eaten_items, ())
        self.assertEqual(result.skipped_items[0].plan_item.record.name, "Broccoli")
        self.assertEqual(result.unspecified_items[0].record.name, "Chicken Tenders")

    def test_all_marks_all_planned_items_eaten_at_planned_quantities(self) -> None:
        result = self.reconcile(semantic([
            {"plan_item_id": f"item_{number}", "reference_text": "everything", "action": "eaten", "quantity_text": None}
            for number in range(1, 4)
        ]))
        self.assertEqual([item.official_servings for item in result.eaten_items], [Decimal("1"), Decimal("1.5"), Decimal("1")])
        self.assertEqual(result.skipped_items, ())
        self.assertEqual(result.unspecified_items, ())

    def test_everything_except_preserves_explicit_skip(self) -> None:
        result = self.reconcile(semantic([
            {"plan_item_id": "item_1", "reference_text": "everything", "action": "eaten", "quantity_text": None},
            {"plan_item_id": "item_2", "reference_text": "everything", "action": "eaten", "quantity_text": None},
            {"plan_item_id": "item_3", "reference_text": "except broccoli", "action": "skipped", "quantity_text": None},
        ]))
        self.assertEqual(len(result.eaten_items), 2)
        self.assertEqual(result.skipped_items[0].plan_item.record.name, "Broccoli")

    def test_semantic_portion_estimate_is_not_authoritative(self) -> None:
        plan = self.plan()
        matcher = FakePortionMatcher(PortionSemanticDecision("estimate", Decimal("0.5"), "medium", "one_tender"))
        result = self.reconcile(
            semantic([{"plan_item_id": "item_1", "reference_text": "only one tender", "action": "eaten", "quantity_text": "one tender"}]),
            plan=plan,
            portion=NaturalPortionInterpreter(semantic_matcher=matcher),
            user_text="I ate only one tender",
        )
        self.assertEqual(len(matcher.calls), 1)
        self.assertEqual(result.eaten_items, ())
        self.assertIn(plan.items[0], result.unspecified_items)
        self.assertEqual(
            result.clarification_items[0].reason,
            "semantic_quantity_not_authoritative",
        )

    def test_ambiguous_explicit_override_fails_closed_for_that_item(self) -> None:
        result = self.reconcile(semantic([{"plan_item_id": "item_1", "reference_text": "some tenders", "action": "eaten", "quantity_text": "some"}]), user_text="I ate some tenders")
        self.assertEqual(result.eaten_items, ())
        self.assertIn(self.plan().items[0], result.unspecified_items)
        self.assertEqual(result.clarification_items[0].reason, "portion_semantic_matcher_unavailable")

    def test_multiple_eaten_and_skipped_items_are_distinct(self) -> None:
        plan = self.plan(("Eggs", "Cereal", "Sausage"))
        result = self.reconcile(semantic([
            {"plan_item_id": "item_1", "reference_text": "eggs", "action": "eaten", "quantity_text": None},
            {"plan_item_id": "item_2", "reference_text": "cereal", "action": "eaten", "quantity_text": None},
            {"plan_item_id": "item_3", "reference_text": "sausage", "action": "skipped", "quantity_text": None},
        ]), plan=plan)
        self.assertEqual([item.record.name for item in result.eaten_items], ["Eggs", "Cereal"])
        self.assertEqual([item.plan_item.record.name for item in result.skipped_items], ["Sausage"])

    def test_alias_is_safe_only_through_supplied_plan_item_id(self) -> None:
        plan = self.plan(("Cavatappi and Cheese",))
        result = self.reconcile(semantic([{"plan_item_id": "item_1", "reference_text": "mac and cheese", "action": "eaten", "quantity_text": None}]), plan=plan)
        self.assertEqual(result.eaten_items[0].record.name, "Cavatappi and Cheese")

    def test_unknown_plan_id_and_contradiction_fail_closed(self) -> None:
        unknown = self.reconcile(semantic([{"plan_item_id": "item_99", "reference_text": "food", "action": "eaten", "quantity_text": None}]))
        contradictory = self.reconcile(semantic([
            {"plan_item_id": "item_1", "reference_text": "tenders", "action": "eaten", "quantity_text": None},
            {"plan_item_id": "item_1", "reference_text": "tenders", "action": "skipped", "quantity_text": None},
        ]))
        self.assertEqual(unknown.eaten_items, ())
        self.assertEqual(unknown.clarification_items[0].reason, "unknown_plan_item_reference")
        self.assertEqual(contradictory.skipped_items, ())
        self.assertEqual(contradictory.clarification_items[0].reason, "contradictory_planned_report")

    def test_ambiguous_statement_is_preserved_for_clarification(self) -> None:
        result = self.reconcile(semantic(unresolved=[
            {"reference_text": "the side", "reason": "ambiguous_reference"}
        ]))
        self.assertEqual(result.unspecified_items, self.plan().items)
        self.assertEqual(result.clarification_items[0].reason, "ambiguous_reference")

    def test_duplicate_identical_mentions_do_not_duplicate_proposal(self) -> None:
        item = {"plan_item_id": "item_1", "reference_text": "tenders", "action": "eaten", "quantity_text": None}
        result = self.reconcile(semantic([item, item]))
        self.assertEqual(len(result.eaten_items), 1)

    def test_unplanned_food_resolves_locally_at_same_date_and_meal(self) -> None:
        result = self.reconcile(semantic([], [{"food_text": "mashed potatoes", "quantity_text": "2 servings"}]), user_text="I ate 2 servings of mashed potatoes")
        extra = result.unplanned_items[0]
        self.assertEqual(extra.resolved_food.record.name, "Mashed Potatoes")
        self.assertEqual(extra.official_servings, Decimal("2"))
        self.assertEqual(extra.quantity_source, "explicit_deterministic")

    def test_unplanned_unresolved_or_quantityless_food_requires_clarification(self) -> None:
        unresolved = self.reconcile(semantic([], [{"food_text": "banana", "quantity_text": None}]))
        known_without_quantity = self.reconcile(
            semantic([], [{"food_text": "mashed potatoes", "quantity_text": None}]),
            user_text="I ate mashed potatoes",
        )
        self.assertIsNone(unresolved.unplanned_items[0].resolved_food)
        self.assertEqual(unresolved.clarification_items[0].reason, "unplanned_food_unresolved")
        self.assertIsNotNone(known_without_quantity.unplanned_items[0].resolved_food)
        self.assertIsNone(known_without_quantity.unplanned_items[0].official_servings)
        self.assertEqual(known_without_quantity.clarification_items[0].reason, "unplanned_quantity_required")

    def test_malformed_or_transport_semantics_fail_closed_without_mutation(self) -> None:
        plan = self.plan()
        malformed = self.reconcile(object(), plan=plan)
        failed = self.reconcile(RuntimeError("private transport"), plan=plan)
        self.assertEqual(malformed.unspecified_items, plan.items)
        self.assertEqual(failed.clarification_items[0].reason, "meal_report_semantic_failed")
        self.assertEqual(DailyLedger().entries, ())

    def test_exact_record_and_decimal_are_preserved_without_arithmetic(self) -> None:
        plan = self.plan()
        result = self.reconcile(semantic([{"plan_item_id": "item_1", "reference_text": "the chicken", "action": "eaten", "quantity_text": None}]), plan=plan)
        self.assertIs(result.eaten_items[0].record, plan.items[0].food.nutrition_record)
        self.assertEqual(result.eaten_items[0].official_servings, Decimal("1"))
        self.assertNotIsInstance(result.eaten_items[0].official_servings, float)
        self.assertEqual(plan.items[0].recommended_official_servings, Decimal("1"))

    def test_module_has_no_ledger_persistence_messaging_live_fd_or_diningbucket_dependency(self) -> None:
        source = Path(meal_report.__file__).read_text(encoding="utf-8")
        for forbidden in ("DailyLedger", "RecordIntakeCommand", "BlueBubbles", "FDMealPlannerClient", "DiningBucket"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
