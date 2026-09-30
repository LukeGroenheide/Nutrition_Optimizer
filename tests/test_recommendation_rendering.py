"""Offline tests for exact recommendation presentation and plan adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
import inspect
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any
import unittest

import nutrition_optimizer.recommendation_rendering as recommendation_rendering
from nutrition_optimizer.durable_state import DurableMealState
from nutrition_optimizer.presentation_binding import PresentationKind
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import (
    FoodResolutionRequest,
    LocalFDFoodResolver,
    ResolvedFood,
)
from nutrition_optimizer.meal_optimizer import (
    MealOptimizationDiagnostics,
    MealRecommendation,
    RecommendedMealItem,
)
from nutrition_optimizer.meal_report import MealPlan, MealReportReconciler, PlannedMealItem
from nutrition_optimizer.meal_report_structured import MealReportSemanticResult
from nutrition_optimizer.nutrition import (
    DailyLedger,
    DailyTargets,
    NutrientProfile,
    NutritionProvenance,
    NutritionRecord,
    Serving,
    SourceIdentifier,
    calculate_daily_balance,
)
from nutrition_optimizer.openclaw_recommendation_rendering import (
    OpenClawRecommendedPortionRenderer,
)
from nutrition_optimizer.physical_quantity import (
    PhysicalQuantityError,
    official_servings_for_physical_count,
)
from nutrition_optimizer.portion_interpretation import (
    NaturalPortionInterpreter,
    PortionInterpretationRequest,
    PortionSemanticDecision,
    UnresolvedPortion,
)
from tests.test_food_resolution import DAY1, T1, component, mapped, menu_day


ZERO_NUTRIENTS = NutrientProfile(
    calories_kcal=Decimal("0"),
    protein_g=Decimal("0"),
    carbohydrates_g=Decimal("0"),
    fat_g=Decimal("0"),
    sodium_mg=Decimal("0"),
    dietary_fiber_g=Decimal("0"),
)


def record(
    name: str,
    *,
    quantity: str = "1",
    unit: str = "Serving",
    source_value: str = "1",
) -> NutritionRecord:
    return NutritionRecord(
        name=name,
        serving=Serving(
            quantity=Decimal(quantity),
            unit=unit,
            text=f"{quantity} {unit}",
        ),
        nutrients=NutrientProfile(
            calories_kcal=Decimal("100"),
            protein_g=Decimal("10"),
            carbohydrates_g=Decimal("5"),
            fat_g=Decimal("2"),
            sodium_mg=Decimal("50"),
            dietary_fiber_g=Decimal("1"),
        ),
        provenance=NutritionProvenance(
            provider="test",
            retrieved_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            identifiers=(SourceIdentifier("component", source_value),),
        ),
    )


def occurrence(
    identifier: int,
    food: NutritionRecord,
    *,
    meal_period_id: int | str = 3,
    meal_period_name: str = "Dinner",
):
    from nutrition_optimizer.fdmealplanner.catalog import FDMenuOccurrence

    return FDMenuOccurrence(
        occurrence_id=identifier,
        occurrence_key=f"occurrence-{identifier}",
        service_date=DAY1,
        meal_period_id=meal_period_id,
        meal_period_name=meal_period_name,
        station_concept_id=40,
        station_name="Grill",
        source_identifier=food.provenance.identifiers[0],
        content_signature=f"signature-{identifier}",
        nutrition_snapshot_id=identifier,
        nutrition_record=food,
        menu_detail_id=f"detail-{identifier}",
        menu_id=3,
        first_observed_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        last_observed_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )


def resolved_food_for_occurrence(occurrence_value) -> ResolvedFood:
    return ResolvedFood(
        original_food_text=occurrence_value.nutrition_record.name,
        occurrence=occurrence_value,
        equivalent_occurrences=(occurrence_value,),
        source_identifier=occurrence_value.source_identifier,
        content_signature=occurrence_value.content_signature,
        nutrition_snapshot_id=occurrence_value.nutrition_snapshot_id,
        nutrition_record=occurrence_value.nutrition_record,
        resolution_method="exact_name",
    )


def recommendation(*items: RecommendedMealItem) -> MealRecommendation:
    targets = DailyTargets(
        calories_kcal=Decimal("2000"),
        protein_g=Decimal("100"),
        carbohydrates_g=Decimal("200"),
        fat_g=Decimal("70"),
    )
    return MealRecommendation(
        service_date=DAY1,
        meal="Dinner",
        items=tuple(items),
        projected_meal_nutrition=ZERO_NUTRIENTS,
        projected_daily_total=ZERO_NUTRIENTS,
        projected_balance=calculate_daily_balance(targets, ZERO_NUTRIENTS),
        objective_score=Decimal("0"),
        diagnostics=MealOptimizationDiagnostics(
            candidate_occurrences=len(items),
            eligible_candidates=len(items),
            duplicate_occurrences_collapsed=0,
            excluded_wrong_context=0,
            excluded_missing_required_nutrition=0,
            excluded_invalid_serving=0,
            search_states_evaluated=1,
            baseline_objective_score=Decimal("0"),
            stage_fraction=Decimal("1"),
            outcome="recommended",
        ),
    )


@dataclass
class FakeSemanticRenderer:
    result: object

    def __post_init__(self) -> None:
        self.calls: list[recommendation_rendering.RecommendedPortionRenderRequest] = []

    def render(self, request):
        self.calls.append(request)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def semantic_result(
    quantity: str | None,
    *,
    confidence: str = "medium",
    practicality: str = "practical",
):
    return recommendation_rendering.RecommendedPortionSemanticResult(
        natural_descriptor=quantity,
        confidence=confidence,  # type: ignore[arg-type]
        presentation_practicality=practicality,  # type: ignore[arg-type]
    )


CHICKEN_TWO_ITEM_SERVINGS = official_servings_for_physical_count(
    Serving(Decimal("3"), "Each", "3 Each"),
    Decimal("2"),
)


class RecommendationPortionRenderingTests(unittest.TestCase):
    def test_scrambled_eggs_use_approximate_visual_without_changing_weight(self) -> None:
        for name in ("Eggs", "Scrambled Eggs"):
            with self.subTest(name=name):
                food = record(name, quantity="4", unit="Ounce")
                item = RecommendedMealItem(occurrence(91, food), food, Decimal("2"))
                rendered = recommendation_rendering.render_meal_recommendation(
                    recommendation(item), recommendation_rendering.RecommendedPortionRenderer()
                )
                self.assertEqual(
                    rendered.message,
                    f"For dinner, get:\n- about 2 baseball-sized portions of {name.casefold()}",
                )
                self.assertEqual(
                    rendered.renderings[0].canonical_quantity_text,
                    f"about 8 oz of {name.casefold()}",
                )
                self.assertEqual(rendered.renderings[0].physical_quantity.amount, Decimal("8"))
                self.assertEqual(rendered.meal_plan.items[0].recommended_official_servings, Decimal("2"))
                self.assertEqual(rendered.meal_plan.items[0].presentation_binding.kind, PresentationKind.DESCRIPTIVE_ONLY)

    def test_exact_egg_count_still_wins_over_visual_fallback(self) -> None:
        request = recommendation_rendering.RecommendedPortionRenderRequest(
            "Eggs", Serving(Decimal("1"), "Each", "1 Each"), Decimal("2")
        )
        rendered = recommendation_rendering.RecommendedPortionRenderer().render(request)
        self.assertEqual(rendered.natural_quantity_text, "2 eggs")
        self.assertEqual(rendered.semantic_descriptor_disposition, "not_requested")

    def test_plain_pasta_visual_keeps_weight_and_cannot_reverse_authorize(self) -> None:
        food = record("Corn Spaghetti", quantity="4", unit="Ounce")
        occurrence_value = occurrence(91, food)
        renderer = recommendation_rendering.RecommendedPortionRenderer()
        for multiplier, expected in ((Decimal("1"), "about 1 scoop-sized portion"),
                                     (Decimal("2"), "about 2 scoop-sized portions")):
            with self.subTest(multiplier=multiplier):
                item = RecommendedMealItem(occurrence_value, food, multiplier)
                rendered = recommendation_rendering.render_meal_recommendation(
                    recommendation(item), renderer,
                )
                plan_item = rendered.meal_plan.items[0]
                self.assertEqual(plan_item.natural_quantity_text, expected)
                self.assertEqual(rendered.message, f"For dinner, get:\n- {expected} of corn spaghetti")
                self.assertEqual(rendered.renderings[0].canonical_quantity_text,
                                 f"about {4 * multiplier} oz of corn spaghetti")
                self.assertEqual(rendered.renderings[0].physical_quantity.amount, 4 * multiplier)
                self.assertEqual(plan_item.recommended_official_servings, multiplier)
                self.assertEqual(plan_item.presentation_binding.kind, PresentationKind.DESCRIPTIVE_ONLY)
                self.assertIsNone(plan_item.presentation_binding.calibration)
                self.assertIsInstance(NaturalPortionInterpreter().interpret(PortionInterpretationRequest(
                    resolved_food_for_occurrence(occurrence_value), "1 scoop of spaghetti",
                    plan_item.presentation_binding, multiplier,
                )), UnresolvedPortion)

    def test_mac_and_cheese_uses_scoop_visual(self) -> None:
        for multiplier, expected in ((Decimal("1"), "about 1 scoop-sized portion"),
                                     (Decimal("2"), "about 2 scoop-sized portions")):
            with self.subTest(multiplier=multiplier):
                request = recommendation_rendering.RecommendedPortionRenderRequest(
                    "Mac and Cheese", Serving(Decimal("4"), "Ounce", "4 Ounce"), multiplier
                )
                rendered = recommendation_rendering.RecommendedPortionRenderer().render(request)
                self.assertEqual(rendered.natural_quantity_text, expected)
                self.assertEqual(rendered.canonical_quantity_text,
                                 f"about {4 * multiplier} oz total of mac and cheese")
                self.assertEqual(rendered.semantic_descriptor_disposition, "deterministic_visual")

    def test_mixed_pasta_keeps_existing_canonical_fallback(self) -> None:
        for name in ("Chicken Florentine Pasta Bake", "Cajun Chicken Pasta",
                     "Corn Penne in Cheese Sauce", "Cheese Ravioli", "Lasagna",
                     "Chicken Noodle Soup"):
            with self.subTest(name=name):
                request = recommendation_rendering.RecommendedPortionRenderRequest(
                    name, Serving(Decimal("4"), "Ounce", "4 Ounce"), Decimal("2")
                )
                rendered = recommendation_rendering.RecommendedPortionRenderer().render(request)
                self.assertEqual(rendered.natural_quantity_text, f"about 8 oz of {name.casefold()}")
                self.assertEqual(rendered.semantic_descriptor_disposition, "not_requested")

    def test_two_physical_tenders_use_the_exact_derived_multiplier(self) -> None:
        food = record("Chicken Tenders", quantity="3", unit="Each")
        item = RecommendedMealItem(
            occurrence(1, food),
            food,
            CHICKEN_TWO_ITEM_SERVINGS,
        )
        renderer = recommendation_rendering.RecommendedPortionRenderer()

        rendered = renderer.render(
            recommendation_rendering.RecommendedPortionRenderRequest(
                food.name,
                food.serving,
                item.official_servings,
            )
        )

        self.assertEqual(rendered.natural_quantity_text, "2 chicken tenders")
        self.assertEqual(item.physical_quantity.amount, Decimal("2"))
        self.assertEqual(item.official_servings, CHICKEN_TWO_ITEM_SERVINGS)
        self.assertEqual(rendered.physical_quantity, item.physical_quantity)
        self.assertEqual(rendered.rendering_method, "deterministic")
        self.assertEqual(rendered.confidence, "high")

    def test_half_cup_times_one_point_two_five_is_exactly_five_eighths(self) -> None:
        request = recommendation_rendering.RecommendedPortionRenderRequest(
            "Mexican Rice",
            Serving(Decimal("0.5"), "Cup", "0.5 Cup"),
            Decimal("1.25"),
        )

        rendered = recommendation_rendering.RecommendedPortionRenderer().render(request)

        self.assertEqual(rendered.physical_quantity.amount, Decimal("0.625"))
        self.assertEqual(rendered.natural_quantity_text, "⅝ cup of mexican rice")
        self.assertNotIn("1¼ cups", rendered.natural_quantity_text)

    def test_fractional_count_multiplier_is_rejected_instead_of_rounded(self) -> None:
        with self.assertRaises(PhysicalQuantityError):
            recommendation_rendering.RecommendedPortionRenderRequest(
                "Chicken Tenders",
                Serving(Decimal("3"), "Each", "3 Each"),
                Decimal("0.75"),
            )

    def test_one_each_and_multi_each_exact_count_math_is_deterministic(self) -> None:
        one_each = recommendation_rendering.RecommendedPortionRenderer().render(
            recommendation_rendering.RecommendedPortionRenderRequest(
                "Baked Potato",
                Serving(Decimal("1"), "Each", "1 Each"),
                Decimal("2.00"),
            )
        )
        multi_each = recommendation_rendering.RecommendedPortionRenderer().render(
            recommendation_rendering.RecommendedPortionRenderRequest(
                "Chicken Tenders",
                Serving(Decimal("3"), "Each", "3 Each"),
                Decimal("1.00"),
            )
        )

        self.assertEqual(one_each.natural_quantity_text, "2 baked potatoes")
        self.assertEqual(multi_each.natural_quantity_text, "3 chicken tenders")
        self.assertEqual(one_each.presentation_practicality, "practical")

    def test_slice_and_stick_units_use_deterministic_generic_unit_logic(self) -> None:
        slice_rendering = recommendation_rendering.RecommendedPortionRenderer().render(
            recommendation_rendering.RecommendedPortionRenderRequest(
                "Cheese Pizza",
                Serving(Decimal("1"), "Slice", "1 Slice"),
                Decimal("2.00"),
            )
        )
        stick_rendering = recommendation_rendering.RecommendedPortionRenderer().render(
            recommendation_rendering.RecommendedPortionRenderRequest(
                "Breadsticks",
                Serving(Decimal("2"), "Sticks", "2 Sticks"),
                Decimal("1.00"),
            )
        )

        self.assertEqual(slice_rendering.natural_quantity_text, "2 slices of cheese pizza")
        self.assertEqual(stick_rendering.natural_quantity_text, "2 sticks of breadsticks")
        self.assertEqual(slice_rendering.rendering_method, "deterministic")
        self.assertEqual(stick_rendering.rendering_method, "deterministic")

    def test_weight_without_a_useful_descriptor_keeps_canonical_text(self) -> None:
        fake = FakeSemanticRenderer(
            semantic_result(
                None,
                practicality="awkward",
            )
        )
        request = recommendation_rendering.RecommendedPortionRenderRequest(
            "Beef",
            Serving(Decimal("4"), "Ounce", "4 Ounce"),
            Decimal("0.75"),
            approved_descriptive_options=("about 2 scoops",),
        )

        rendering = recommendation_rendering.RecommendedPortionRenderer(fake).render(request)

        self.assertEqual(rendering.presentation_practicality, "awkward")
        self.assertEqual(rendering.physical_quantity.amount, Decimal("3.00"))
        self.assertEqual(rendering.natural_quantity_text, "about 3 oz of beef")
        self.assertEqual(rendering.semantic_descriptor_disposition, "not_useful")

    def test_discrete_and_natural_volume_quantities_do_not_invoke_semantic_augmentation(self) -> None:
        fake = FakeSemanticRenderer(
            semantic_result(
                "about 2 scoops",
                confidence="medium",
            )
        )
        renderer = recommendation_rendering.RecommendedPortionRenderer(fake)
        count = recommendation_rendering.RecommendedPortionRenderRequest(
            "Chicken Tenders",
            Serving(Decimal("3"), "Each", "3 Each"),
            CHICKEN_TWO_ITEM_SERVINGS,
        )
        volume = recommendation_rendering.RecommendedPortionRenderRequest(
            "Cereal",
            Serving(Decimal("1"), "Cup", "1 Cup"),
            Decimal("1.5"),
        )

        count_result = renderer.render(count)
        result = renderer.render(volume)

        self.assertEqual(count_result.natural_quantity_text, "2 chicken tenders")
        self.assertEqual(result.natural_quantity_text, "1½ cups of cereal")
        self.assertEqual(result.confidence, "high")
        self.assertEqual(result.rendering_method, "deterministic")
        self.assertEqual(fake.calls, [])

        weight_fake = FakeSemanticRenderer(
            semantic_result(
                "about 2 scoops or 1 ladle",
                confidence="medium",
            )
        )
        weight = recommendation_rendering.RecommendedPortionRenderRequest(
            "Roasted Chicken",
            Serving(Decimal("4"), "Ounce Cooked Weight", "4 Ounce Cooked Weight"),
            Decimal("1.25"),
            approved_descriptive_options=("about 2 scoops or 1 ladle",),
        )
        weight_result = recommendation_rendering.RecommendedPortionRenderer(weight_fake).render(weight)
        self.assertEqual(weight_result.confidence, "medium")
        self.assertEqual(weight_result.presentation_practicality, "practical")
        self.assertEqual(weight_result.canonical_quantity_text, "about 5 oz cooked of roasted chicken")
        self.assertEqual(weight_result.semantic_descriptor, "about 2 scoops or 1 ladle")
        self.assertEqual(weight_result.natural_quantity_text, "about 2 scoops or 1 ladle")
        self.assertEqual(weight_result.semantic_descriptor_disposition, "accepted")
        self.assertEqual(weight_result.physical_quantity, weight.physical_quantity)

    def test_unknown_unit_fails_closed_even_with_semantic_renderer(self) -> None:
        with self.assertRaises(
            recommendation_rendering.RecommendedPortionRenderingUnavailableError
        ):
            recommendation_rendering.RecommendedPortionRenderer().render(
                recommendation_rendering.RecommendedPortionRenderRequest(
                    "Mystery Food",
                    Serving(Decimal("1"), "Portion", "1 Portion"),
                    Decimal("1.5"),
                )
            )

    def test_semantic_output_cannot_replace_food_identity_or_authoritative_field(self) -> None:
        replacement = FakeSemanticRenderer(
            semantic_result("about 2 burgers")
        )
        request = recommendation_rendering.RecommendedPortionRenderRequest(
            "Chicken Tenders",
            Serving(Decimal("4"), "Ounce", "4 Ounce"),
            Decimal("0.75"),
            approved_descriptive_options=("about 2 scoops",),
        )
        rendering = recommendation_rendering.RecommendedPortionRenderer(replacement).render(request)

        self.assertNotIn("recommended_official_servings", {
            field.name for field in recommendation_rendering.RecommendedPortionRendering.__dataclass_fields__.values()
        })
        self.assertEqual(request.recommended_official_servings, Decimal("0.75"))
        self.assertEqual(rendering.physical_quantity, request.physical_quantity)
        self.assertEqual(rendering.natural_quantity_text, "about 3 oz of chicken tenders")
        self.assertEqual(rendering.semantic_descriptor_disposition, "rejected")
        self.assertIsNone(rendering.semantic_descriptor)

    def test_semantic_descriptor_cannot_replace_canonical_measurement(self) -> None:
        replacement = FakeSemanticRenderer(
            semantic_result("about 8 oz")
        )
        request = recommendation_rendering.RecommendedPortionRenderRequest(
            "Roasted Chicken",
            Serving(Decimal("4"), "Ounce", "4 Ounce"),
            Decimal("1.25"),
            approved_descriptive_options=("about 2 scoops",),
        )

        rendering = recommendation_rendering.RecommendedPortionRenderer(replacement).render(request)
        self.assertEqual(rendering.natural_quantity_text, "about 5 oz of roasted chicken")
        self.assertEqual(rendering.canonical_quantity_text, "about 5 oz of roasted chicken")
        self.assertEqual(rendering.semantic_descriptor_disposition, "rejected")

    def test_useless_practical_descriptor_forms_fall_back_without_guessing_a_conversion(self) -> None:
        request = recommendation_rendering.RecommendedPortionRenderRequest(
            "Roasted Chicken",
            Serving(Decimal("4"), "Ounce", "4 Ounce"),
            Decimal("2"),
            approved_descriptive_options=("about 2 scoops",),
        )
        for descriptor in (
            "about 1 serving",
            "about a medium portion",
            "about 9 scoops",
            "about 2 scoops of spaghetti",
        ):
            with self.subTest(descriptor=descriptor):
                rendering = recommendation_rendering.RecommendedPortionRenderer(
                    FakeSemanticRenderer(semantic_result(descriptor))
                ).render(request)
                self.assertEqual(rendering.natural_quantity_text, "about 8 oz of roasted chicken")
                self.assertEqual(rendering.semantic_descriptor_disposition, "rejected")

    def test_semantic_descriptor_must_match_approved_option_exactly(self) -> None:
        request = recommendation_rendering.RecommendedPortionRenderRequest(
            "Roasted Chicken",
            Serving(Decimal("4"), "Ounce", "4 Ounce"),
            Decimal("2"),
            approved_descriptive_options=("about 2 scoops",),
        )
        for altered in ("about 3 scoops", "About 2 Scoops", "about 2  scoops"):
            with self.subTest(altered=altered):
                rendering = recommendation_rendering.RecommendedPortionRenderer(
                    FakeSemanticRenderer(semantic_result(altered))
                ).render(request)
                self.assertEqual(rendering.semantic_descriptor_disposition, "rejected")
                self.assertEqual(rendering.natural_quantity_text, "about 8 oz of roasted chicken")

    def test_malformed_semantic_output_and_transport_failure_fall_back_to_canonical_text(self) -> None:
        weight_request = recommendation_rendering.RecommendedPortionRenderRequest(
            "Cereal",
            Serving(Decimal("4"), "Ounce", "4 Ounce"),
            Decimal("1.5"),
            approved_descriptive_options=("about 2 scoops",),
        )
        malformed = FakeSemanticRenderer({"natural_descriptor": "a bowl"})
        malformed_rendering = recommendation_rendering.RecommendedPortionRenderer(
            malformed
        ).render(weight_request)
        self.assertEqual(malformed_rendering.natural_quantity_text, "about 6 oz of cereal")
        self.assertEqual(malformed_rendering.semantic_descriptor_disposition, "rejected")

        transport = FakeSemanticRenderer(RuntimeError("private transport details"))
        transport_rendering = recommendation_rendering.RecommendedPortionRenderer(
            transport
        ).render(weight_request)
        self.assertEqual(transport_rendering.natural_quantity_text, "about 6 oz of cereal")
        self.assertEqual(transport_rendering.semantic_descriptor_disposition, "unavailable")

    def test_weight_based_composite_dish_without_approved_visual_uses_canonical_text(self) -> None:
        food = record(
            "Roasted Sausage, Greens, and Beans",
            quantity="4",
            unit="Ounce",
        )
        item = RecommendedMealItem(occurrence(8, food), food, Decimal("2"))
        rendered = recommendation_rendering.render_meal_recommendation(
            recommendation(item),
            recommendation_rendering.RecommendedPortionRenderer(
                FakeSemanticRenderer(semantic_result("about 2 scoops or 1 ladle"))
            ),
        )

        item_rendering = rendered.renderings[0]
        self.assertEqual(
            item_rendering.natural_quantity_text,
            "about 8 oz total of roasted sausage, greens, and beans",
        )
        self.assertEqual(
            item_rendering.canonical_quantity_text,
            "about 8 oz total of roasted sausage, greens, and beans",
        )
        self.assertEqual(item_rendering.physical_quantity, item.physical_quantity)
        self.assertEqual(
            rendered.message,
            "For dinner, get:\n- about 8 oz total of roasted sausage, greens, and beans",
        )

    def test_weight_based_composite_dish_falls_back_to_an_unambiguous_total(self) -> None:
        request = recommendation_rendering.RecommendedPortionRenderRequest(
            "Roasted Sausage, Greens, and Beans",
            Serving(Decimal("4"), "Ounce", "4 Ounce"),
            Decimal("2"),
        )

        rendering = recommendation_rendering.RecommendedPortionRenderer().render(request)

        self.assertEqual(
            rendering.natural_quantity_text,
            "about 8 oz total of roasted sausage, greens, and beans",
        )
        self.assertEqual(rendering.canonical_quantity_text, rendering.natural_quantity_text)
        self.assertEqual(rendering.semantic_descriptor_disposition, "not_requested")

    def test_renderer_does_not_call_natural_portion_interpreter_backwards(self) -> None:
        source = inspect.getsource(recommendation_rendering)
        self.assertNotIn("portion_interpretation import", source)
        self.assertNotIn("NaturalPortionInterpreter(", source)
        self.assertNotIn("DailyTargets", source)
        self.assertNotIn("scale_nutrients", source)
        self.assertNotIn("BlueBubbles", source)
        self.assertNotIn("DiningBucket", source)
        self.assertNotIn("Sol", source)


class MealRecommendationAdapterTests(unittest.TestCase):
    def test_numeric_fd_meal_ids_render_as_human_readable_names(self) -> None:
        for meal_id, meal_name in ((1, "breakfast"), (2, "lunch"), (3, "dinner")):
            with self.subTest(meal_id=meal_id):
                food = record(meal_name.title(), source_value=str(meal_id))
                meal_occurrence = occurrence(
                    meal_id,
                    food,
                    meal_period_id=meal_id,
                    meal_period_name=meal_name.title(),
                )
                plan = MealPlan(
                    DAY1,
                    meal_id,
                    (
                        PlannedMealItem(
                            resolved_food_for_occurrence(meal_occurrence),
                            Decimal("1"),
                            f"one {meal_name} serving",
                        ),
                    ),
                )

                self.assertEqual(
                    recommendation_rendering.format_meal_message(plan).splitlines()[0],
                    f"For {meal_name}, get:",
                )

    def test_unknown_meal_message_keeps_existing_normalized_fallback(self) -> None:
        food = record("Snack", source_value="4")
        meal_occurrence = occurrence(
            4,
            food,
            meal_period_id=4,
            meal_period_name="Snack",
        )
        plan = MealPlan(
            DAY1,
            "Snack",
            (PlannedMealItem(resolved_food_for_occurrence(meal_occurrence), Decimal("1"), "a snack"),),
        )

        self.assertEqual(
            recommendation_rendering.format_meal_message(plan).splitlines()[0],
            "For snack, get:",
        )

    def test_recommendation_becomes_plan_with_exact_food_links_and_decimal_quantities(self) -> None:
        chicken = record("Chicken Tenders", quantity="3", unit="Each", source_value="1")
        sausage = record(
            "Roasted Sausage, Greens, and Beans",
            quantity="4",
            unit="Ounce",
            source_value="2",
        )
        chicken_item = RecommendedMealItem(
            occurrence(1, chicken), chicken, CHICKEN_TWO_ITEM_SERVINGS
        )
        sausage_item = RecommendedMealItem(occurrence(2, sausage), sausage, Decimal("2.00"))
        rec = recommendation(chicken_item, sausage_item)
        fake = FakeSemanticRenderer(
            semantic_result("about 2 scoops or 1 ladle", confidence="medium")
        )

        rendered = recommendation_rendering.render_meal_recommendation(
            rec,
            recommendation_rendering.RecommendedPortionRenderer(fake),
        )
        plan = rendered.meal_plan

        self.assertEqual(
            [item.recommended_official_servings for item in plan.items],
            [CHICKEN_TWO_ITEM_SERVINGS, Decimal("2.00")],
        )
        self.assertEqual(plan.items[0].presentation_binding.kind, PresentationKind.EXACT_AUTHORITATIVE)
        self.assertEqual(plan.items[1].presentation_binding.kind, PresentationKind.EXACT_AUTHORITATIVE)
        self.assertIsNone(plan.items[1].presentation_binding.calibration)
        self.assertEqual(plan.items[0].natural_quantity_text, "2 chicken tenders")
        self.assertEqual(
            plan.items[1].natural_quantity_text,
            "about 8 oz total of roasted sausage, greens, and beans",
        )
        self.assertIsNone(plan.items[1].display_food_name)
        self.assertEqual(plan.items[0].food.occurrence, chicken_item.occurrence)
        self.assertEqual(plan.items[1].food.occurrence, sausage_item.occurrence)
        self.assertEqual(plan.items[0].record, chicken_item.record)
        self.assertEqual(plan.items[1].record, sausage_item.record)
        self.assertEqual(
            plan.items[1].food.nutrition_snapshot_id,
            sausage_item.occurrence.nutrition_snapshot_id,
        )
        self.assertEqual(
            plan.items[1].food.content_signature,
            sausage_item.occurrence.content_signature,
        )
        self.assertEqual(plan.items[1].record.nutrients, sausage_item.record.nutrients)
        self.assertEqual(rendered.renderings[0].physical_quantity, chicken_item.physical_quantity)
        self.assertEqual(rendered.renderings[1].physical_quantity, sausage_item.physical_quantity)
        self.assertEqual(
            [item.food.source_identifier for item in plan.items],
            [chicken_item.occurrence.source_identifier, sausage_item.occurrence.source_identifier],
        )
        self.assertEqual(
            rendered.renderings[1].canonical_quantity_text,
            "about 8 oz total of roasted sausage, greens, and beans",
        )
        self.assertEqual(
            rendered.message,
            "For dinner, get:\n- 2 chicken tenders\n"
            "- about 8 oz total of roasted sausage, greens, and beans",
        )
        self.assertNotIn("calorie", rendered.message.casefold())
        self.assertNotIn("protein", rendered.message.casefold())
        self.assertNotIn("scoop", rendered.message)
        self.assertEqual(rendered.practicality_counts(), {"practical": 2, "awkward": 0, "unclear": 0})

    def test_empty_or_partial_adaptation_fails_closed(self) -> None:
        empty = recommendation_rendering.RecommendedPortionRenderer()
        no_items = MealRecommendation(
            service_date=DAY1,
            meal="Dinner",
            items=(),
            projected_meal_nutrition=ZERO_NUTRIENTS,
            projected_daily_total=ZERO_NUTRIENTS,
            projected_balance=calculate_daily_balance(
                DailyTargets(Decimal("1"), Decimal("1"), Decimal("1"), Decimal("1")),
                ZERO_NUTRIENTS,
            ),
            objective_score=Decimal("0"),
            diagnostics=MealOptimizationDiagnostics(
                0, 0, 0, 0, 0, 0, 0, Decimal("0"), Decimal("1"), "empty_menu"
            ),
        )
        with self.assertRaises(recommendation_rendering.MealRecommendationRenderingError):
            recommendation_rendering.render_meal_recommendation(no_items, empty)


class DurableRecommendationRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "state" / "nutrition.sqlite3"
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self.catalog.close)
        chicken = component(1, "Chicken Tenders")
        chicken.update({"recipePortionSize": "3", "recipePortionSizeUnit": "Each"})
        sausage = component(2, "Roasted Sausage, Greens, and Beans")
        sausage.update({"recipePortionSize": "4", "recipePortionSizeUnit": "Ounce"})
        self.catalog.synchronize_fd_refresh(
            mapped(menu_day(DAY1, 3, "Dinner", ((chicken, "Grill"), (sausage, "Hot Bar")))),
            requested_start=DAY1,
            requested_end=DAY1,
            observed_at=T1,
        )
        occurrences = self.catalog.list_current_meal_occurrences(DAY1, meal="Dinner")
        self.occurrences = {item.nutrition_record.name: item for item in occurrences}

    def plan_recommendation(self) -> MealRecommendation:
        chicken_occurrence = self.occurrences["Chicken Tenders"]
        sausage_occurrence = self.occurrences["Roasted Sausage, Greens, and Beans"]
        return recommendation(
            RecommendedMealItem(
                chicken_occurrence,
                chicken_occurrence.nutrition_record,
                CHICKEN_TWO_ITEM_SERVINGS,
            ),
            RecommendedMealItem(
                sausage_occurrence,
                sausage_occurrence.nutrition_record,
                Decimal("2.00"),
            ),
        )

    def rendered(self):
        fake = FakeSemanticRenderer(
            semantic_result("about 2 scoops or 1 ladle", confidence="medium")
        )
        return recommendation_rendering.render_meal_recommendation(
            self.plan_recommendation(),
            recommendation_rendering.RecommendedPortionRenderer(fake),
        )

    def test_explicit_persistence_uses_existing_durable_state_and_reloads_decimal(self) -> None:
        state = DurableMealState(self.catalog)
        semantic_renderer = FakeSemanticRenderer(
            semantic_result("about 2 scoops or 1 ladle", confidence="medium")
        )
        persisted = recommendation_rendering.persist_meal_recommendation(
            self.plan_recommendation(),
            recommendation_rendering.RecommendedPortionRenderer(semantic_renderer),
            state,
            plan_id="rendered-dinner",
            created_at=T1,
        )
        self.assertEqual(
            persisted.plan.items[0].recommended_official_servings,
            CHICKEN_TWO_ITEM_SERVINGS,
        )
        self.assertEqual(persisted.plan.items[1].recommended_official_servings, Decimal("2.00"))
        self.assertEqual(self.catalog._connection.execute("SELECT COUNT(*) FROM meal_plans").fetchone()[0], 1)

        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        loaded = DurableMealState(self.catalog).load_meal_plan("rendered-dinner")
        assert loaded is not None
        self.assertEqual(
            [item.recommended_official_servings for item in loaded.plan.items],
            [CHICKEN_TWO_ITEM_SERVINGS, Decimal("2.00")],
        )
        self.assertEqual(loaded.plan.items[0].natural_quantity_text, "2 chicken tenders")
        self.assertEqual(
            loaded.plan.items[1].natural_quantity_text,
            "about 8 oz total of roasted sausage, greens, and beans",
        )
        self.assertIsNone(loaded.plan.items[1].display_food_name)
        self.assertEqual(
            [item.food.occurrence.occurrence_id for item in loaded.plan.items],
            [
                self.occurrences["Chicken Tenders"].occurrence_id,
                self.occurrences["Roasted Sausage, Greens, and Beans"].occurrence_id,
            ],
        )

    def test_everything_report_inherits_exact_optimizer_decimals_after_persistence(self) -> None:
        rendered = self.rendered()
        state = DurableMealState(self.catalog)
        persisted = state.save_meal_plan(rendered.plan, plan_id="round-trip")
        semantic = MealReportSemanticResult.model_validate(
            {
                "planned_items": [
                    {
                        "plan_item_id": "item_1",
                        "reference_text": "everything",
                        "action": "eaten",
                        "quantity_relation": "as_recommended",
                        "quantity_text": None,
                    },
                    {
                        "plan_item_id": "item_2",
                        "reference_text": "everything",
                        "action": "eaten",
                        "quantity_relation": "as_recommended",
                        "quantity_text": None,
                    },
                ],
                "additional_foods": [],
                "unresolved_statements": [],
            }
        )

        class FixedSemantic:
            def interpret(self, plan, user_text):
                return semantic

        loaded = state.load_meal_plan("round-trip")
        assert loaded is not None
        reconciled = MealReportReconciler(
            FixedSemantic(),
            LocalFDFoodResolver(self.catalog),
            NaturalPortionInterpreter(),
        ).reconcile(loaded.plan, "I ate everything")

        self.assertEqual(
            [item.official_servings for item in reconciled.eaten_items],
            [CHICKEN_TWO_ITEM_SERVINGS, Decimal("2.00")],
        )
        self.assertEqual(
            [item.quantity_source for item in reconciled.eaten_items],
            ["planned_quantity", "planned_quantity"],
        )
        applied = state.apply_reconciled_meal_report(
            loaded,
            reconciled,
            source_event_id="round-trip-report",
        )
        self.assertEqual(
            [entry.official_servings for entry in applied.accepted_intake_entries],
            [CHICKEN_TWO_ITEM_SERVINGS, Decimal("2.00")],
        )

    def test_semantic_quantity_override_does_not_become_authoritative(self) -> None:
        rendered = self.rendered()
        plan = rendered.plan
        semantic = MealReportSemanticResult.model_validate(
            {
                "planned_items": [
                    {
                        "plan_item_id": "item_1",
                        "reference_text": "I only ate one tender",
                        "action": "eaten",
                        "quantity_relation": "modified",
                        "quantity_text": "one tender",
                    },
                ],
                "additional_foods": [],
                "unresolved_statements": [],
            }
        )

        @dataclass
        class FakePortionMatcher:
            def decide(self, request: PortionInterpretationRequest):
                return PortionSemanticDecision(
                    "estimate",
                    Decimal("0.333333333333333333"),
                    "medium",
                    "one tender",
                )

        class FixedSemantic:
            def interpret(self, plan, user_text):
                return semantic

        result = MealReportReconciler(
            FixedSemantic(),
            LocalFDFoodResolver(self.catalog),
            NaturalPortionInterpreter(semantic_matcher=FakePortionMatcher()),
        ).reconcile(plan, "I only ate one tender")

        self.assertEqual(result.eaten_items, ())
        self.assertIn(plan.items[0], result.unspecified_items)
        self.assertEqual(
            result.clarification_items[0].reason,
            "semantic_quantity_not_authoritative",
        )


def openclaw_envelope(payload: object) -> dict[str, object]:
    return {
        "ok": True,
        "toolName": "llm-task",
        "source": "plugin",
        "output": {
            "details": {
                "json": payload,
                "provider": "openai",
                "model": "gpt-5.6-luna",
            }
        },
    }


class OpenClawRunner:
    def __init__(self, payload: object, *, returncode: int = 0) -> None:
        self.payload = payload
        self.returncode = returncode
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, args: list[str], **kwargs: Any):
        self.calls.append((args, kwargs))
        stdout = json.dumps(openclaw_envelope(self.payload))
        return subprocess.CompletedProcess(args, self.returncode, stdout, "private stderr")


class OpenClawRecommendationRenderingTests(unittest.TestCase):
    @staticmethod
    def request() -> recommendation_rendering.RecommendedPortionRenderRequest:
        return recommendation_rendering.RecommendedPortionRenderRequest(
            "Corn Spaghetti",
            Serving(Decimal("4"), "Ounce", "4 Ounce"),
            Decimal("2.00"),
            station_name="Pasta",
            approved_descriptive_options=("about 2 scoops or 1 ladle",),
        )

    def test_payload_schema_and_runtime_are_narrow_and_luna_configured(self) -> None:
        runner = OpenClawRunner(
            {
                "natural_descriptor": "about 2 scoops or 1 ladle",
                "confidence": "medium",
                "presentation_practicality": "practical",
            }
        )
        adapter = OpenClawRecommendedPortionRenderer(
            "/test-only/openclaw",
            runner=runner,
        )

        result = adapter.render(self.request())

        self.assertEqual(result.natural_descriptor, "about 2 scoops or 1 ladle")
        params = json.loads(runner.calls[0][0][runner.calls[0][0].index("--params") + 1])
        args = params["args"]
        self.assertEqual(args["provider"], "openai")
        self.assertEqual(args["model"], "openai/gpt-5.6-luna")
        self.assertEqual(args["thinking"], "high")
        self.assertEqual(args["schema"]["additionalProperties"], False)
        self.assertNotIn("natural_quantity_text", args["schema"]["properties"])
        self.assertNotIn("natural_display_name", args["schema"]["properties"])
        input_payload = json.loads(args["input"])
        self.assertEqual(
            input_payload["canonical_physical_quantity"]["description"],
            "about 8 oz",
        )
        self.assertEqual(input_payload["canonical_physical_quantity"]["amount"], "8.00")
        self.assertNotIn("recommended_official_servings", input_payload)
        self.assertNotIn("official_serving", input_payload)
        self.assertNotIn("station_name", input_payload)
        for forbidden in (
            "calories",
            "protein",
            "carbohydrates",
            "fat",
            "fiber",
            "targets",
            "objective",
            "component_id",
            "occurrence_id",
            "nutrition_snapshot_id",
            "source_identifier",
            "station_name",
        ):
            self.assertNotIn(forbidden, args["input"])
        self.assertIn("natural_descriptor", args["prompt"])
        self.assertIn("do not calculate", args["prompt"].casefold())

    def test_malformed_structured_output_and_transport_failure_are_sanitized(self) -> None:
        malformed = OpenClawRecommendedPortionRenderer(
            "/test-only/openclaw",
            runner=OpenClawRunner(
                {
                    "natural_descriptor": "about 2 scoops",
                    # The model has no field through which it can rename the
                    # food. A returned display name is an invalid extra field.
                    "natural_display_name": "Different Food",
                    "confidence": "medium",
                    "presentation_practicality": "practical",
                }
            ),
        )
        with self.assertRaisesRegex(
            recommendation_rendering.RecommendedPortionRenderingOutputError,
            "invalid output",
        ):
            malformed.render(self.request())

        failed = OpenClawRecommendedPortionRenderer(
            "/test-only/openclaw",
            runner=OpenClawRunner({}, returncode=1),
        )
        with self.assertRaisesRegex(
            recommendation_rendering.RecommendedPortionRenderingTransportError,
            "failed",
        ) as raised:
            failed.render(self.request())
        self.assertNotIn("private stderr", str(raised.exception))

    def test_adapter_has_no_direct_food_selection_persistence_or_message_transport(self) -> None:
        source = Path(__import__(
            "nutrition_optimizer.openclaw_recommendation_rendering",
            fromlist=["x"],
        ).__file__).read_text(encoding="utf-8")
        for forbidden in (
            "OfficialNutritionCatalog",
            "DailyLedger",
            "NutritionRecord",
            "BlueBubbles",
            "DiningBucket",
            "gpt-5.6-sol",
            "OpenAI(",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
