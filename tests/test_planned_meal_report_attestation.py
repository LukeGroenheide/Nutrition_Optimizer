"""Regression coverage for plan-scoped meal-report quantity attestations."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nutrition_optimizer.application import MealReportOrchestrator
from nutrition_optimizer.durable_state import DurableMealState
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import (
    FoodResolutionCandidate,
    FoodResolutionRequest,
    FoodSemanticDecision,
    LocalFDFoodResolver,
    ResolvedFood,
)
from nutrition_optimizer.meal_report import MealPlan, MealReportReconciler, PlannedMealItem
from nutrition_optimizer.meal_report_structured import (
    MealReportSemanticResult,
    MealReportStructuredValidationError,
    interpret_meal_report_structured_result,
)
from nutrition_optimizer.portion_interpretation import (
    NaturalPortionInterpreter,
    PortionInterpretationRequest,
    PortionSemanticDecision,
    UnresolvedPortion,
)
from tests.test_food_resolution import DAY1, T1, component, mapped, menu_day


def recipe(
    component_id: int,
    name: str,
    *,
    serving_quantity: str,
    serving_unit: str,
) -> dict[str, object]:
    value = component(component_id, name)
    value["recipePortionSize"] = serving_quantity
    value["recipePortionSizeUnit"] = serving_unit
    return value


def semantic(
    *,
    planned: list[dict[str, object]] | None = None,
    additional: list[dict[str, object]] | None = None,
    unresolved: list[dict[str, object]] | None = None,
) -> MealReportSemanticResult:
    return MealReportSemanticResult.model_validate(
        {
            "planned_items": planned or [],
            "additional_foods": additional or [],
            "unresolved_statements": unresolved or [],
        }
    )


@dataclass
class FixedMealReportSemanticInterpreter:
    result: object
    calls: list[tuple[MealPlan, str]] = field(default_factory=list)

    def interpret(self, plan: MealPlan, user_text: str) -> object:
        self.calls.append((plan, user_text))
        return self.result


@dataclass
class FixedFoodSemanticMatcher:
    result: object
    calls: list[tuple[FoodResolutionRequest, tuple[FoodResolutionCandidate, ...]]] = field(
        default_factory=list
    )

    def decide(
        self,
        request: FoodResolutionRequest,
        candidates: tuple[FoodResolutionCandidate, ...],
    ) -> object:
        self.calls.append((request, candidates))
        return self.result


@dataclass
class FixedPortionMatcher:
    result: object
    calls: list[PortionInterpretationRequest] = field(default_factory=list)

    def decide(self, request: PortionInterpretationRequest) -> object:
        self.calls.append(request)
        return self.result


class PlannedMealReportAttestationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog = OfficialNutritionCatalog(Path(self.directory.name) / "nutrition.sqlite3")
        self.addCleanup(self.catalog.close)
        self.catalog.synchronize_fd_refresh(
            mapped(
                menu_day(
                    DAY1,
                    3,
                    "Dinner",
                    (
                        (
                            recipe(
                                1,
                                "Barbecue Pulled Pork",
                                serving_quantity="6",
                                serving_unit="Ounce Cooked Weight",
                            ),
                            "Smokehouse",
                        ),
                        (
                            recipe(
                                2,
                                "Jackfruit BBQ",
                                serving_quantity="6",
                                serving_unit="Ounce",
                            ),
                            "Smokehouse",
                        ),
                        (
                            recipe(
                                3,
                                "Mediterranean Sweet Potato Salad",
                                serving_quantity="3",
                                serving_unit="Ounce",
                            ),
                            "Salad Bar",
                        ),
                        (
                            recipe(
                                4,
                                "Cheddar Garlic Biscuits",
                                serving_quantity="1",
                                serving_unit="Each",
                            ),
                            "Bakery",
                        ),
                        (
                            recipe(
                                5,
                                "Buttermilk Biscuits",
                                serving_quantity="1",
                                serving_unit="Each",
                            ),
                            "Bakery",
                        ),
                        (
                            recipe(
                                6,
                                "Garlic Bread",
                                serving_quantity="1",
                                serving_unit="Each",
                            ),
                            "Bakery",
                        ),
                    ),
                )
            ),
            requested_start=DAY1,
            requested_end=DAY1,
            observed_at=T1,
        )
        self.resolver = LocalFDFoodResolver(self.catalog)
        self.foods = {
            name: self._food(name)
            for name in (
                "Barbecue Pulled Pork",
                "Jackfruit BBQ",
                "Mediterranean Sweet Potato Salad",
                "Cheddar Garlic Biscuits",
                "Buttermilk Biscuits",
                "Garlic Bread",
            )
        }
        self.plan = MealPlan(
            DAY1,
            "Dinner",
            (
                PlannedMealItem(
                    self.foods["Barbecue Pulled Pork"],
                    Decimal("2.00"),
                    "about 2 scoops",
                ),
                PlannedMealItem(
                    self.foods["Jackfruit BBQ"],
                    Decimal("2.00"),
                    "about 2 scoops or 1 ladle",
                ),
                PlannedMealItem(
                    self.foods["Mediterranean Sweet Potato Salad"],
                    Decimal("2.00"),
                    "roughly 1-2 serving-spoon portions",
                ),
            ),
        )

    def _food(self, name: str) -> ResolvedFood:
        result = self.resolver.resolve(FoodResolutionRequest(name, DAY1, meal="Dinner"))
        assert isinstance(result, ResolvedFood)
        return result

    def reconcile(
        self,
        result: object,
        *,
        resolver: LocalFDFoodResolver | None = None,
        portion_interpreter: NaturalPortionInterpreter | None = None,
        user_text: str = "test report",
    ):
        return MealReportReconciler(
            FixedMealReportSemanticInterpreter(result),
            resolver or self.resolver,
            portion_interpreter or NaturalPortionInterpreter(),
        ).reconcile(self.plan, user_text)

    def test_serving_line_equivalence_cannot_attest_to_plan_quantity(self) -> None:
        state = DurableMealState(self.catalog)
        state.save_meal_plan(self.plan, plan_id="attestation-plan")
        persisted = state.load_meal_plan("attestation-plan")
        assert persisted is not None
        salad = persisted.plan.items[2]
        portion_matcher = FixedPortionMatcher(
            PortionSemanticDecision("estimate", Decimal("9"), "low", "unused")
        )
        reconciled = MealReportReconciler(
            FixedMealReportSemanticInterpreter(
                semantic(
                    planned=[
                        {
                            "plan_item_id": "item_3",
                            "reference_text": "a scoop of the salad",
                            "action": "eaten",
                            "quantity_relation": "as_recommended",
                            "quantity_text": None,
                        }
                    ]
                )
            ),
            self.resolver,
            NaturalPortionInterpreter(semantic_matcher=portion_matcher),
        ).reconcile(persisted.plan, "I ate a scoop of the salad")

        self.assertEqual(salad.recommended_official_servings, Decimal("2.00"))
        self.assertEqual(salad.natural_quantity_text, "roughly 1-2 serving-spoon portions")
        self.assertEqual(reconciled.eaten_items, ())
        self.assertEqual(reconciled.clarification_items[0].reason, "plan_attestation_required")
        self.assertEqual(reconciled.clarification_items[0].original_user_phrase, "a scoop")
        self.assertEqual(portion_matcher.calls, [])
        self.assertIsInstance(
            NaturalPortionInterpreter().interpret(
                PortionInterpretationRequest(salad.food, "a scoop")
            ),
            UnresolvedPortion,
        )

    def test_sandwich_count_does_not_attest_to_the_planned_pork_quantity(self) -> None:
        reconciled = self.reconcile(
            semantic(
                planned=[
                    {
                        "plan_item_id": "item_1",
                        "reference_text": "2 barbecue pulled pork sandwiches",
                        "action": "eaten",
                        "quantity_relation": "modified",
                        "quantity_text": "2 sandwiches",
                    }
                ]
            )
        )

        self.assertEqual(reconciled.eaten_items, ())
        self.assertIn(self.plan.items[0], reconciled.unspecified_items)
        self.assertEqual(len(reconciled.clarification_items), 1)
        clarification = reconciled.clarification_items[0]
        self.assertIs(clarification.plan_item, self.plan.items[0])
        self.assertEqual(reconciled.unplanned_items, ())
        self.assertNotIn("Garlic Bread", [item.record.name for item in reconciled.eaten_items])

    def test_explicitly_skipped_planned_item_needs_no_quantity_clarification(self) -> None:
        reconciled = self.reconcile(
            semantic(
                planned=[
                    {
                        "plan_item_id": "item_2",
                        "reference_text": "skipped the Jackfruit BBQ",
                        "action": "skipped",
                        "quantity_relation": None,
                        "quantity_text": None,
                    }
                ]
            )
        )

        self.assertEqual(reconciled.eaten_items, ())
        self.assertEqual(reconciled.skipped_items[0].plan_item, self.plan.items[1])
        self.assertEqual(reconciled.clarification_items, ())

    def test_unique_unplanned_alias_uses_only_the_authoritative_current_menu_food(self) -> None:
        biscuit = self.foods["Cheddar Garlic Biscuits"]
        matcher = FixedFoodSemanticMatcher(
            FoodSemanticDecision("match", biscuit.source_identifier.value, "unique menu alias")
        )
        reconciled = self.reconcile(
            semantic(
                additional=[
                    {
                        "food_text": "biscuit",
                        "quantity_text": "half of one",
                    }
                ]
            ),
            resolver=LocalFDFoodResolver(self.catalog, semantic_matcher=matcher),
            user_text="I ate half of one cheddar garlic biscuit",
        )

        unplanned = reconciled.unplanned_items[0]
        assert unplanned.resolved_food is not None
        self.assertEqual(unplanned.resolved_food.source_identifier, biscuit.source_identifier)
        self.assertEqual(unplanned.resolved_food.nutrition_snapshot_id, biscuit.nutrition_snapshot_id)
        self.assertEqual(unplanned.resolved_food.record, biscuit.record)
        self.assertEqual(unplanned.official_servings, Decimal("0.5"))
        self.assertEqual(unplanned.quantity_source, "explicit_deterministic")
        self.assertEqual(unplanned.resolved_food.record.serving.unit, "Each")
        self.assertEqual(len(matcher.calls), 1)
        self.assertIn(
            biscuit.source_identifier.value,
            [candidate.source_identifier.value for candidate in matcher.calls[0][1]],
        )

    def test_ambiguous_unplanned_alias_does_not_guess_a_menu_food(self) -> None:
        matcher = FixedFoodSemanticMatcher(
            FoodSemanticDecision("ambiguous", None, "multiple possible biscuits")
        )
        reconciled = self.reconcile(
            semantic(
                additional=[
                    {
                        "food_text": "biscuit",
                        "quantity_text": "half of one",
                    }
                ]
            ),
            resolver=LocalFDFoodResolver(self.catalog, semantic_matcher=matcher),
        )

        self.assertIsNone(reconciled.unplanned_items[0].resolved_food)
        self.assertEqual(reconciled.clarification_items[0].reason, "unplanned_food_ambiguous")
        self.assertEqual(len(matcher.calls), 1)

    def test_mixed_report_keeps_uncalibrated_pork_and_salad_unresolved(self) -> None:
        biscuit = self.foods["Cheddar Garlic Biscuits"]
        food_matcher = FixedFoodSemanticMatcher(
            FoodSemanticDecision("match", biscuit.source_identifier.value, "unique menu alias")
        )
        semantic_result = semantic(
            planned=[
                {
                    "plan_item_id": "item_1",
                    "reference_text": "2 barbecue pulled pork sandwiches",
                    "action": "eaten",
                    "quantity_relation": "modified",
                    "quantity_text": "2 sandwiches",
                },
                {
                    "plan_item_id": "item_2",
                    "reference_text": "skipped the Jackfruit BBQ",
                    "action": "skipped",
                    "quantity_relation": None,
                    "quantity_text": None,
                },
                {
                    "plan_item_id": "item_3",
                    "reference_text": "1 scoop of Mediterranean Sweet Potato Salad",
                    "action": "eaten",
                    "quantity_relation": "as_recommended",
                    "quantity_text": None,
                },
            ],
            additional=[
                {
                    "food_text": "biscuit",
                    "quantity_text": "half of one",
                }
            ],
        )
        interpreter = FixedMealReportSemanticInterpreter(semantic_result)
        reconciler = MealReportReconciler(
            interpreter,
            LocalFDFoodResolver(self.catalog, semantic_matcher=food_matcher),
            NaturalPortionInterpreter(),
        )
        state = DurableMealState(self.catalog)
        saved = state.save_meal_plan(self.plan, plan_id="mixed-attestation-plan")
        result = MealReportOrchestrator(self.catalog, reconciler).process(
            DAY1,
            "Dinner",
            "I ate 2 barbecue pulled pork sandwiches, skipped the Jackfruit BBQ, "
            "1 scoop of Mediterranean Sweet Potato Salad, and half of one cheddar garlic biscuit",
            "mixed-attestation-event",
        )

        self.assertEqual(result.outcome, "clarification_required")
        assert result.reconciled_report is not None
        report = result.reconciled_report
        self.assertEqual([item.plan_item for item in report.eaten_items], [])
        self.assertEqual([item.plan_item for item in report.skipped_items], [report.plan.items[1]])
        assert report.unplanned_items[0].resolved_food is not None
        self.assertEqual(
            report.unplanned_items[0].resolved_food.source_identifier,
            biscuit.source_identifier,
        )
        self.assertEqual(report.unplanned_items[0].resolved_food.record, biscuit.record)
        self.assertEqual(report.unplanned_items[0].official_servings, Decimal("0.5"))
        self.assertEqual(report.unspecified_items, (report.plan.items[0], report.plan.items[2]))
        self.assertEqual(len(report.clarification_items), 2)
        self.assertIs(report.clarification_items[0].plan_item, report.plan.items[0])
        self.assertIn("Barbecue Pulled Pork", result.message)
        self.assertIn("about 2 scoops", result.message)
        self.assertIn("Mediterranean Sweet Potato Salad", result.message)
        self.assertNotIn("Jackfruit BBQ", result.message)
        self.assertNotIn("biscuit", result.message.casefold())
        self.assertEqual(state.get_daily_intake(DAY1), ())
        self.assertEqual(
            self.catalog._connection.execute(
                "SELECT COUNT(*) FROM meal_report_applications"
            ).fetchone()[0],
            0,
        )
        self.assertEqual(state.load_active_meal_plan(DAY1, "Dinner"), saved)

    def test_invalid_or_missing_quantity_relation_fails_closed_at_the_schema_boundary(self) -> None:
        base = {
            "plan_item_id": "item_1",
            "reference_text": "the pork",
            "action": "eaten",
            "quantity_text": None,
        }
        for planned_item in (
            base,
            {**base, "quantity_relation": "as_recommended", "quantity_text": "two servings"},
            {**base, "quantity_relation": "modified"},
        ):
            with self.subTest(planned_item=planned_item):
                with self.assertRaises(MealReportStructuredValidationError):
                    interpret_meal_report_structured_result(
                        {
                            "planned_items": [planned_item],
                            "additional_foods": [],
                            "unresolved_statements": [],
                        }
                    )

    def test_model_visible_plan_context_has_no_authoritative_quantity_or_nutrition(self) -> None:
        visible = self.plan.model_visible_items()
        self.assertEqual(visible[2]["display_quantity"], "roughly 1-2 serving-spoon portions")
        rendered = str(visible)
        for forbidden in (
            "recommended_official_servings",
            "nutrition_snapshot_id",
            "calories",
            "12",
            "6 Ounce",
        ):
            self.assertNotIn(forbidden, rendered)


if __name__ == "__main__":
    unittest.main()
