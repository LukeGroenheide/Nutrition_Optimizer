"""Focused coverage for deterministic recommendation preparation."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import nutrition_optimizer.application as application
from nutrition_optimizer.application import (
    DAILY_CALORIES_KCAL_ENV_VAR,
    DAILY_CARBOHYDRATES_G_ENV_VAR,
    DAILY_DIETARY_FIBER_MINIMUM_G_ENV_VAR,
    DAILY_FAT_G_ENV_VAR,
    DAILY_PROTEIN_G_ENV_VAR,
    MealRecommendationOrchestrator,
    MealRecommendationRequestError,
    MealRecommendationUnavailableError,
    ProductionNutritionConfigurationError,
    load_production_daily_targets,
)
from nutrition_optimizer.durable_state import DurableMealState
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import FoodResolutionRequest, LocalFDFoodResolver, ResolvedFood
from nutrition_optimizer.meal_report import (
    MealPlan,
    PlannedMealItem,
    ProposedEatenMealItem,
    ReconciledMealReport,
)
from nutrition_optimizer.nutrition import DailyLedger, DailyMinimums, DailyTargets
from nutrition_optimizer.recommendation_rendering import (
    RecommendedPortionRenderer,
    RecommendedPortionSemanticResult,
    format_meal_message,
)
from tests.test_food_resolution import component, mapped, menu_day


DAY = date(2026, 8, 25)
OBSERVED_AT = datetime(2026, 8, 25, 12, tzinfo=timezone.utc)


def targets(*, fiber_minimum: Decimal | None = None) -> DailyTargets:
    return DailyTargets(
        calories_kcal=Decimal("1000"),
        protein_g=Decimal("100"),
        carbohydrates_g=Decimal("100"),
        fat_g=Decimal("50"),
        minimums=DailyMinimums(fiber_minimum),
    )


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
    serving_unit: str = "Cup",
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


def production_environment() -> dict[str, str]:
    return {
        DAILY_CALORIES_KCAL_ENV_VAR: "3200",
        DAILY_PROTEIN_G_ENV_VAR: "125",
        DAILY_CARBOHYDRATES_G_ENV_VAR: "450",
        DAILY_FAT_G_ENV_VAR: "100",
        DAILY_DIETARY_FIBER_MINIMUM_G_ENV_VAR: "38",
    }


class FixedPracticalDescriptorRenderer:
    def __init__(self, descriptor: str) -> None:
        self.descriptor = descriptor
        self.calls = []

    def render(self, request):
        self.calls.append(request)
        return RecommendedPortionSemanticResult(
            natural_descriptor=self.descriptor,
            confidence="medium",
            presentation_practicality="practical",
        )


class ProductionDailyTargetsTests(unittest.TestCase):
    def test_all_required_targets_load_as_exact_decimals(self) -> None:
        loaded = load_production_daily_targets(production_environment())

        self.assertEqual(loaded.calories_kcal, Decimal("3200"))
        self.assertEqual(loaded.protein_g, Decimal("125"))
        self.assertEqual(loaded.carbohydrates_g, Decimal("450"))
        self.assertEqual(loaded.fat_g, Decimal("100"))
        self.assertEqual(loaded.minimums.dietary_fiber_g, Decimal("38"))

    def test_missing_required_target_never_falls_back(self) -> None:
        for variable_name in production_environment():
            with self.subTest(variable_name=variable_name):
                environment = production_environment()
                del environment[variable_name]
                with self.assertRaisesRegex(
                    ProductionNutritionConfigurationError, variable_name
                ):
                    load_production_daily_targets(environment)

        with self.assertRaises(ProductionNutritionConfigurationError):
            load_production_daily_targets({})

    def test_malformed_or_non_positive_targets_fail_explicitly(self) -> None:
        invalid_values = ("not-a-number", "NaN", "Infinity", "-1", "0")
        for variable_name in production_environment():
            for invalid_value in invalid_values:
                with self.subTest(variable_name=variable_name, invalid_value=invalid_value):
                    environment = production_environment()
                    environment[variable_name] = invalid_value
                    with self.assertRaisesRegex(
                        ProductionNutritionConfigurationError, variable_name
                    ):
                        load_production_daily_targets(environment)


class MealRecommendationOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "state" / "nutrition.sqlite3"
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self.catalog.close)
        self.state = DurableMealState(self.catalog)
        self.orchestrator = MealRecommendationOrchestrator(self.catalog)

    def sync(self, *menu_days: dict[str, object]) -> None:
        self.catalog.synchronize_fd_refresh(
            mapped(*menu_days),
            requested_start=DAY,
            requested_end=DAY,
            observed_at=OBSERVED_AT,
        )

    def plan_count(self) -> int:
        return int(
            self.catalog._connection.execute("SELECT COUNT(*) FROM meal_plans").fetchone()[0]
        )

    def intake_count(self) -> int:
        return int(
            self.catalog._connection.execute(
                "SELECT COUNT(*) FROM accepted_intake_entries"
            ).fetchone()[0]
        )

    def accept_breakfast(self, *, servings: Decimal = Decimal("1")) -> DailyLedger:
        resolved = LocalFDFoodResolver(self.catalog).resolve(
            FoodResolutionRequest("Earlier Breakfast", DAY, meal=1)
        )
        assert isinstance(resolved, ResolvedFood)
        plan = MealPlan(
            DAY,
            1,
            (PlannedMealItem(resolved, servings, "one earlier breakfast serving"),),
        )
        persisted = self.state.save_meal_plan(plan)
        report = ReconciledMealReport(
            plan,
            (
                ProposedEatenMealItem(
                    plan.items[0], servings, "planned_quantity", "earlier breakfast"
                ),
            ),
            (),
            (),
            (),
            (),
        )
        self.state.apply_reconciled_meal_report(
            persisted,
            report,
            source_event_id="test-earlier-breakfast",
            recorded_at=OBSERVED_AT,
        )
        return self.state.load_daily_ledger(DAY)

    def test_prepares_current_menu_with_durable_ledger_and_exact_quantity_chain(self) -> None:
        self.sync(
            menu_day(
                DAY,
                1,
                "Breakfast",
                ((food(1, "Earlier Breakfast", calories="300", protein="40", carbohydrates="45", fat="10", fiber="4"), "Breakfast"),),
            ),
            menu_day(
                DAY,
                3,
                "Dinner",
                (
                    (food(2, "Chicken Tenders", calories="300", protein="30", carbohydrates="15", fat="9", fiber="1", serving_quantity="3", serving_unit="Each"), "Grill"),
                    (food(3, "Brown Rice", calories="200", protein="4", carbohydrates="45", fat="1", fiber="2"), "Sides"),
                    (food(4, "Broccoli", calories="50", protein="4", carbohydrates="10", fat="0.5", fiber="5"), "Vegetables"),
                ),
            ),
        )
        expected_ledger = self.accept_breakfast()
        calls: list[tuple[date, int, DailyTargets, DailyLedger]] = []
        original_optimize = application.LocalMealOptimizer.optimize_meal

        def tracked_optimize(
            optimizer,
            service_date: date,
            meal: str | int,
            supplied_targets: DailyTargets,
            current_ledger: DailyLedger,
            *,
            policy=None,
        ):
            calls.append((service_date, meal, supplied_targets, current_ledger))
            return original_optimize(
                optimizer,
                service_date,
                meal,
                supplied_targets,
                current_ledger,
                policy=policy,
            )

        supplied_targets = targets(fiber_minimum=Decimal("20"))
        with patch.object(application.LocalMealOptimizer, "optimize_meal", tracked_optimize):
            prepared = self.orchestrator.prepare(DAY, "Dinner", supplied_targets)

        self.assertEqual(calls, [(DAY, 3, supplied_targets, expected_ledger)])
        self.assertEqual(prepared.current_ledger, expected_ledger)
        self.assertEqual(prepared.recommendation.meal, 3)
        self.assertTrue(prepared.recommendation.items)
        dinner_occurrences = self.catalog.list_current_meal_occurrences(DAY, meal=3)
        dinner_ids = {occurrence.occurrence_id for occurrence in dinner_occurrences}
        self.assertTrue(
            {item.occurrence.occurrence_id for item in prepared.recommendation.items}
            <= dinner_ids
        )

        loaded = self.state.load_meal_plan(prepared.plan_id)
        self.assertEqual(loaded, prepared.persisted_plan)
        assert loaded is not None
        self.assertEqual(prepared.message, format_meal_message(loaded.plan))
        self.assertEqual(loaded.plan, prepared.rendering.meal_plan)
        self.assertEqual(len(loaded.plan.items), len(prepared.recommendation.items))
        for recommendation_item, rendered_item, planned_item in zip(
            prepared.recommendation.items,
            prepared.rendering.rendered_items,
            loaded.plan.items,
            strict=True,
        ):
            self.assertEqual(
                planned_item.recommended_official_servings,
                recommendation_item.official_servings,
            )
            self.assertEqual(
                rendered_item.rendering.physical_quantity,
                recommendation_item.physical_quantity,
            )
            self.assertEqual(
                planned_item.food.occurrence.occurrence_id,
                recommendation_item.occurrence.occurrence_id,
            )
            self.assertEqual(
                planned_item.food.nutrition_snapshot_id,
                recommendation_item.occurrence.nutrition_snapshot_id,
            )
            self.assertEqual(planned_item.record, recommendation_item.record)

    def test_unapproved_semantic_visual_is_not_requested(self) -> None:
        self.sync(
            menu_day(
                DAY,
                3,
                "Dinner",
                (
                    (
                        food(
                            1,
                            "Roasted Sausage, Greens, and Beans",
                            serving_quantity="4",
                            serving_unit="Ounce",
                        ),
                        "Hot Bar",
                    ),
                ),
            )
        )
        semantic = FixedPracticalDescriptorRenderer("about 2 scoops or 1 ladle")
        orchestrator = MealRecommendationOrchestrator(
            self.catalog,
            RecommendedPortionRenderer(semantic),
        )

        prepared = orchestrator.prepare(DAY, "dinner", targets())

        self.assertEqual(len(semantic.calls), 0)
        self.assertEqual(
            prepared.message,
            "For dinner, get:\n- about 8 oz total of roasted sausage, greens, and beans",
        )
        recommendation_item = prepared.recommendation.items[0]
        rendering = prepared.rendering.renderings[0]
        planned_item = prepared.persisted_plan.plan.items[0]
        self.assertEqual(rendering.semantic_descriptor_disposition, "not_requested")
        self.assertEqual(rendering.physical_quantity, recommendation_item.physical_quantity)
        self.assertEqual(
            planned_item.recommended_official_servings,
            recommendation_item.official_servings,
        )
        self.assertEqual(planned_item.food.occurrence, recommendation_item.occurrence)
        self.assertEqual(
            planned_item.food.nutrition_snapshot_id,
            recommendation_item.occurrence.nutrition_snapshot_id,
        )

    def test_changed_durable_earlier_intake_changes_later_preparation(self) -> None:
        self.sync(
            menu_day(
                DAY,
                1,
                "Breakfast",
                ((food(1, "Earlier Breakfast", calories="300", protein="80", carbohydrates="100", fat="50"), "Breakfast"),),
            ),
            menu_day(
                DAY,
                3,
                "Dinner",
                ((food(2, "Protein Plate", calories="100", protein="20", carbohydrates="0", fat="0"), "Grill"),),
            ),
        )

        before_intake = self.orchestrator.prepare(DAY, "dinner", targets())
        self.accept_breakfast()
        after_intake = self.orchestrator.prepare(DAY, "dinner", targets())

        self.assertEqual(before_intake.current_ledger, DailyLedger())
        self.assertEqual(len(after_intake.current_ledger.entries), 1)
        self.assertEqual(
            before_intake.recommendation.items[0].official_servings,
            Decimal("2.00"),
        )
        self.assertEqual(
            after_intake.recommendation.items[0].official_servings,
            Decimal("1.00"),
        )
        self.assertGreater(
            before_intake.recommendation.items[0].official_servings,
            after_intake.recommendation.items[0].official_servings,
        )

    def test_empty_current_menu_returns_explicit_result_without_persisting(self) -> None:
        self.sync(
            menu_day(
                DAY,
                3,
                "Dinner",
                ((food(1, "Dinner Only"), "Grill"),),
            )
        )
        plans_before = self.plan_count()
        intake_before = self.intake_count()

        with self.assertRaisesRegex(
            MealRecommendationUnavailableError, "no current local menu is available"
        ) as raised:
            self.orchestrator.prepare(DAY, "breakfast", targets())

        self.assertEqual(raised.exception.outcome, "empty_menu")
        self.assertEqual(raised.exception.recommendation.items, ())
        self.assertEqual(self.plan_count(), plans_before)
        self.assertEqual(self.intake_count(), intake_before)

    def test_meal_names_and_fd_ids_canonicalize_to_the_same_preparation_path(self) -> None:
        self.sync(
            menu_day(
                DAY,
                3,
                "Dinner",
                ((food(1, "Dinner Food"), "Grill"),),
            )
        )

        numeric = self.orchestrator.prepare(DAY, 3, targets())
        semantic = self.orchestrator.prepare(DAY, " Dinner ", targets())

        self.assertEqual(numeric.recommendation.meal, 3)
        self.assertEqual(semantic.recommendation.meal, 3)
        self.assertEqual(numeric.message, semantic.message)
        self.assertEqual(
            [item.occurrence.occurrence_id for item in numeric.recommendation.items],
            [item.occurrence.occurrence_id for item in semantic.recommendation.items],
        )
        plans_before = self.plan_count()
        with self.assertRaises(MealRecommendationRequestError):
            self.orchestrator.prepare(DAY, "snack", targets())
        self.assertEqual(self.plan_count(), plans_before)

    def test_successful_repreparation_supersedes_only_the_prior_active_context_plan(self) -> None:
        self.sync(
            menu_day(
                DAY,
                3,
                "Dinner",
                ((food(1, "Dinner Food"), "Grill"),),
            )
        )

        first = self.orchestrator.prepare(DAY, "dinner", targets())
        second = self.orchestrator.prepare(DAY, 3, targets())

        first_history = self.state.load_meal_plan(first.plan_id)
        second_history = self.state.load_meal_plan(second.plan_id)
        assert first_history is not None
        assert second_history is not None
        self.assertEqual(first_history.status, "superseded")
        self.assertEqual(second_history.status, "active")
        self.assertEqual(self.state.load_active_meal_plan(DAY, "Dinner"), second.persisted_plan)

    def test_failed_replacement_preparation_does_not_supersede_the_prior_plan(self) -> None:
        self.sync(
            menu_day(
                DAY,
                3,
                "Dinner",
                ((food(1, "Dinner Food"), "Grill"),),
            )
        )
        first = self.orchestrator.prepare(DAY, "dinner", targets())

        with patch.object(application, "render_meal_recommendation", side_effect=RuntimeError("forced")):
            with self.assertRaisesRegex(RuntimeError, "forced"):
                self.orchestrator.prepare(DAY, "dinner", targets())

        first_history = self.state.load_meal_plan(first.plan_id)
        assert first_history is not None
        self.assertEqual(first_history.status, "active")
        self.assertEqual(self.state.load_active_meal_plan(DAY, 3), first.persisted_plan)

    def test_render_failure_cannot_leave_a_partial_plan(self) -> None:
        self.sync(
            menu_day(
                DAY,
                3,
                "Dinner",
                ((food(1, "Dinner Food"), "Grill"),),
            )
        )

        with patch.object(application, "render_meal_recommendation", side_effect=RuntimeError("forced")):
            with self.assertRaisesRegex(RuntimeError, "forced"):
                self.orchestrator.prepare(DAY, "dinner", targets())

        self.assertEqual(self.plan_count(), 0)
        self.assertEqual(self.intake_count(), 0)

    def test_orchestration_has_no_transport_or_semantic_runtime_dependency(self) -> None:
        source = inspect.getsource(application)
        for forbidden in ("OpenClaw", "OpenAI", "BlueBubbles", "DiningBucket", "Luna"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
