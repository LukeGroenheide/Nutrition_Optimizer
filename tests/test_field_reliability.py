"""Field-use coverage for personal foods, presentation, and eligibility."""

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nutrition_optimizer.durable_state import DurableMealState
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import LocalFDFoodResolver
from nutrition_optimizer.meal_optimizer import LocalMealOptimizer
from nutrition_optimizer.meal_report import MealReportReconciler
from nutrition_optimizer.nutrition import DailyLedger, DailyTargets, Serving
from nutrition_optimizer.personal_foods import is_personally_excluded
from nutrition_optimizer.portion_interpretation import NaturalPortionInterpreter
from nutrition_optimizer.presentation_binding import PresentationKind
from nutrition_optimizer.recommendation_rendering import (
    RecommendedPortionRenderRequest, RecommendedPortionRenderer,
    format_meal_message, render_meal_recommendation,
)
from tests.test_food_resolution import mapped, menu_day
from tests.test_meal_optimizer import occurrence, record, targets
from tests.test_meal_report_conversation import ScriptedInterpreter, eaten, food, semantic, skipped


DAY = date(2026, 9, 29)
TARGETS = DailyTargets(Decimal("2000"), Decimal("80"), Decimal("250"), Decimal("70"))


class FieldReliabilityTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.catalog = OfficialNutritionCatalog(Path(directory.name) / "state.sqlite3")
        self.addCleanup(self.catalog.close)
        self.catalog.synchronize_fd_refresh(mapped(menu_day(DAY, 1, "Breakfast", (
            (food(903, "Biscuits and Gravy", serving_quantity="10", serving_unit="Ounce"), "Global Bowls"),
        ))), requested_start=DAY, requested_end=DAY,
            observed_at=datetime(2026, 9, 29, tzinfo=timezone.utc))
        self.state = DurableMealState(self.catalog)

    def _biscuit_plan(self):
        recommendation = LocalMealOptimizer(self.catalog).optimize_meal(
            DAY, 1, TARGETS, DailyLedger(),
        )
        self.assertTrue(recommendation.is_recommendation)
        self.assertEqual([item.record.name for item in recommendation.items], ["Biscuit"])
        self.assertEqual(recommendation.items[0].official_servings, Decimal("2"))
        return render_meal_recommendation(recommendation, RecommendedPortionRenderer()).meal_plan

    def test_potato_and_rice_visuals_keep_exact_weight_and_safe_fallback(self):
        renderer = RecommendedPortionRenderer()
        potato = renderer.render(RecommendedPortionRenderRequest(
            "Roasted Red Potatoes", Serving(Decimal("6"), "Ounce", "6 Ounce"), Decimal("1"),
        ))
        self.assertEqual(potato.natural_quantity_text, "about 2 fist-sized portions")
        self.assertEqual(potato.physical_quantity.amount, Decimal("6"))
        self.assertEqual(potato.semantic_descriptor_disposition, "deterministic_visual")
        rice = renderer.render(RecommendedPortionRenderRequest(
            "Steamed Rice", Serving(Decimal("4"), "Ounce", "4 Ounce"), Decimal("1"),
        ))
        self.assertEqual(rice.natural_quantity_text, "about 1 scoop-sized portion")
        green_beans = renderer.render(RecommendedPortionRenderRequest(
            "Garlic Green Beans", Serving(Decimal("4"), "Ounce", "4 Ounce"), Decimal("1"),
        ))
        self.assertEqual(green_beans.natural_quantity_text, "about 1 fist-sized portion")
        unsupported = renderer.render(RecommendedPortionRenderRequest(
            "Mystery Casserole", Serving(Decimal("6"), "Ounce", "6 Ounce"), Decimal("1"),
        ))
        self.assertEqual(unsupported.natural_quantity_text, "about 6 oz of mystery casserole")
        for name in ("Cream of Potato Soup", "Mashed Potatoes", "Roasted Vegetable Quinoa Salad"):
            with self.subTest(name=name):
                result = renderer.render(RecommendedPortionRenderRequest(
                    name, Serving(Decimal("6"), "Ounce", "6 Ounce"), Decimal("1"),
                ))
                self.assertNotIn("fist-sized", result.natural_quantity_text)

    def test_biscuit_estimate_is_separate_persistable_and_counted(self):
        parent = self.catalog.list_current_meal_occurrences(DAY, meal=1)[0]
        original = parent.nutrition_record
        plan = self._biscuit_plan()
        biscuit = plan.items[0]
        self.assertEqual(biscuit.record.name, "Biscuit")
        self.assertEqual(biscuit.record.provenance.provider, "personal_derived_estimate")
        self.assertEqual(biscuit.record.provenance.record_type, "approximate_personal_estimate")
        self.assertIn(parent.content_signature, biscuit.record.provenance.source_reference)
        nutrients = biscuit.record.nutrients
        self.assertEqual((nutrients.calories_kcal, nutrients.protein_g,
                          nutrients.carbohydrates_g, nutrients.fat_g, nutrients.dietary_fiber_g),
                         (Decimal("200"), Decimal("5"), Decimal("24"), Decimal("10"), Decimal("1")))
        self.assertEqual(biscuit.natural_quantity_text, "2 biscuits")
        self.assertEqual(biscuit.presentation_binding.kind, PresentationKind.EXACT_AUTHORITATIVE)
        self.assertIn("2 biscuits", format_meal_message(plan))
        self.assertNotIn("personal nutrition estimate", format_meal_message(plan))
        self.assertEqual(self.catalog.list_current_meal_occurrences(DAY, meal=1)[0].nutrition_record, original)
        self.assertEqual(self.catalog.ensure_personal_biscuit_occurrence(parent).occurrence_id,
                         biscuit.food.occurrence.occurrence_id)
        saved = self.state.save_meal_plan(plan)
        self.assertEqual(self.state.load_meal_plan(saved.plan_id).plan.items[0].record, biscuit.record)

    def test_biscuit_reports_use_planned_counts(self):
        plan = self._biscuit_plan()
        reconciler = MealReportReconciler(ScriptedInterpreter({}), LocalFDFoodResolver(self.catalog),
                                          NaturalPortionInterpreter())
        cases = (
            ("I ate 1 biscuit", eaten("item_1", "1 biscuit", relation="modified", quantity_text="1 biscuit"), Decimal("1")),
            ("I ate 2 biscuits", eaten("item_1", "2 biscuits", relation="modified", quantity_text="2 biscuits"), Decimal("2")),
            ("I ate two biscuits", eaten("item_1", "I ate two biscuits"), Decimal("2")),
            ("I ate everything you recommended. I only had one biscuit", eaten("item_1", "I only had one biscuit", relation="modified", quantity_text="one biscuit"), Decimal("1")),
            ("I ate both biscuits", eaten("item_1", "both biscuits"), Decimal("2")),
            ("I ate all the biscuits", eaten("item_1", "all the biscuits"), Decimal("2")),
        )
        for text, item, amount in cases:
            with self.subTest(text=text):
                result = reconciler.reconcile_semantic(plan, semantic(planned=[item]), text)
                self.assertEqual(result.eaten_items[0].official_servings, amount)
                self.assertEqual(result.clarification_items, ())
        skipped_report = reconciler.reconcile_semantic(
            plan, semantic(planned=[skipped("item_1", "biscuits")]), "I skipped the biscuits",
        )
        self.assertEqual(len(skipped_report.skipped_items), 1)
        self.assertEqual(skipped_report.eaten_items, ())
        persisted = self.state.save_meal_plan(plan)
        one = reconciler.reconcile_semantic(plan, semantic(planned=[cases[0][1]]), cases[0][0])
        self.state.apply_reconciled_meal_report(persisted, one, source_event_id="one-biscuit")
        self.assertEqual(self.state.get_daily_intake(DAY)[0].official_servings, Decimal("1"))
        self.assertEqual(self.state.load_recommendation_ledger(DAY).total_consumed_nutrients.calories_kcal,
                         Decimal("200"))

    def test_multiple_composite_occurrences_offer_one_bounded_biscuit_candidate(self):
        self.catalog.synchronize_fd_refresh(mapped(menu_day(DAY, 1, "Breakfast", (
            (food(903, "Biscuits and Gravy", serving_quantity="10", serving_unit="Ounce"), "Global Bowls"),
            (food(904, "Biscuits and Gravy", serving_quantity="10", serving_unit="Ounce"), "Grill"),
        ))), requested_start=DAY, requested_end=DAY,
            observed_at=datetime(2026, 9, 29, 1, tzinfo=timezone.utc))
        recommendation = LocalMealOptimizer(self.catalog).optimize_meal(
            DAY, 1, TARGETS, DailyLedger(),
        )
        self.assertEqual(recommendation.diagnostics.candidate_occurrences, 1)
        self.assertEqual([item.record.name for item in recommendation.items], ["Biscuit"])
        self.assertLessEqual(recommendation.items[0].official_servings, Decimal("2"))

    def test_personal_exclusions_are_food_based_and_keep_normal_foods(self):
        for name, excluded in (
            ("Vegan Scrambled Eggs", True), ("Vegan Scrambled 'Eggs'", True),
            ("Plant-Based Egg Substitute", True),
            ("JUST Egg", True), ("Scrambled Eggs", False),
            ("Scrambled Eggs with Vegan Cheese", False),
            ("Vegan Cheese Eggs", False),
            ("Vegan Sausage Patties", True), ("Plant-Based Breakfast Sausage", True),
            ("Meatless Breakfast Patties", True), ("Pork Sausage Patties", False),
        ):
            with self.subTest(name=name):
                self.assertEqual(is_personally_excluded(record(name)), excluded)
        class Menu:
            def __init__(self, foods):
                self.foods = foods
            def list_current_meal_occurrences(self, service_date, meal=None):
                return self.foods
        for unwanted, normal in (("Vegan Scrambled Eggs", "Scrambled Eggs"),
                                 ("Vegan Sausage Patties", "Pork Sausage Patties")):
            with self.subTest(unwanted=unwanted):
                foods = (occurrence(1, record(unwanted), meal="lunch"),
                         occurrence(2, record(normal), meal="lunch"))
                result = LocalMealOptimizer(Menu(foods)).optimize_meal(
                    date(2026, 8, 25), "lunch", targets(), DailyLedger(),
                )
                self.assertEqual([item.record.name for item in result.items], [normal])


if __name__ == "__main__":
    unittest.main()
