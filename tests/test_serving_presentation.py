"""Offline regression coverage for calibrated Phelps serving presentation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nutrition_optimizer.application import MealRecommendationOrchestrator
from nutrition_optimizer.durable_state import DurableMealState
from nutrition_optimizer.fdmealplanner import FDMenuOccurrence, OfficialNutritionCatalog
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
from nutrition_optimizer.meal_recommendation_replacement import (
    format_replacement_recommendation_message,
)
from nutrition_optimizer.meal_report import MealPlan, MealReportReconciler, PlannedMealItem
from nutrition_optimizer.meal_report_conversation import MealReportConversationOrchestrator
from nutrition_optimizer.meal_report_structured import MealReportSemanticResult
from nutrition_optimizer.nutrition import (
    DailyTargets,
    NutrientProfile,
    NutritionProvenance,
    NutritionRecord,
    Serving,
    SourceIdentifier,
    calculate_daily_balance,
    scale_nutrients,
)
from nutrition_optimizer.portion_interpretation import (
    InterpretedPortion,
    NaturalPortionInterpreter,
    PortionInterpretationRequest,
    UnresolvedPortion,
)
from nutrition_optimizer.recommendation_rendering import (
    RecommendedPortionRenderRequest,
    RecommendedPortionRenderer,
    render_meal_recommendation,
)
from nutrition_optimizer.presentation_binding import freeze_presentation, PresentationKind
from nutrition_optimizer.serving_presentation import (
    DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS,
)
from tests.test_food_resolution import component, mapped, menu_day


DAY = date(2026, 9, 3)
OBSERVED_AT = datetime(2026, 9, 3, 17, tzinfo=timezone.utc)
CALIBRATED_SOURCE = SourceIdentifier("component", "7:181:62889")


def _record(
    name: str = "Baked Sweet Potatoes-Master",
    *,
    source_identifier: SourceIdentifier = CALIBRATED_SOURCE,
    serving: Serving | None = None,
) -> NutritionRecord:
    return NutritionRecord(
        name=name,
        serving=serving or Serving(Decimal("1"), "Each", "1 Each"),
        nutrients=NutrientProfile(
            calories_kcal=Decimal("100"),
            protein_g=Decimal("2"),
            carbohydrates_g=Decimal("20"),
            fat_g=Decimal("1"),
            sodium_mg=Decimal("20"),
            dietary_fiber_g=Decimal("4"),
        ),
        provenance=NutritionProvenance(
            provider="test",
            retrieved_at=OBSERVED_AT,
            identifiers=(source_identifier,),
        ),
    )


def _occurrence(
    record: NutritionRecord,
    *,
    occurrence_id: int = 1,
) -> FDMenuOccurrence:
    return FDMenuOccurrence(
        occurrence_id=occurrence_id,
        occurrence_key=f"sweet-potato-{occurrence_id}",
        service_date=DAY,
        meal_period_id=2,
        meal_period_name="Lunch",
        station_concept_id=40,
        station_name="Homestyle",
        source_identifier=record.provenance.identifiers[0],
        content_signature=f"signature-{occurrence_id}",
        nutrition_snapshot_id=occurrence_id,
        nutrition_record=record,
        menu_detail_id=f"detail-{occurrence_id}",
        menu_id=2,
        first_observed_at=OBSERVED_AT,
        last_observed_at=OBSERVED_AT,
    )


def _resolved(record: NutritionRecord, *, occurrence_id: int = 1) -> ResolvedFood:
    occurrence = _occurrence(record, occurrence_id=occurrence_id)
    return ResolvedFood(
        original_food_text=record.name,
        occurrence=occurrence,
        equivalent_occurrences=(occurrence,),
        source_identifier=occurrence.source_identifier,
        content_signature=occurrence.content_signature,
        nutrition_snapshot_id=occurrence.nutrition_snapshot_id,
        nutrition_record=record,
        resolution_method="exact_name",
    )


def _recommendation(item: RecommendedMealItem) -> MealRecommendation:
    targets = DailyTargets(
        calories_kcal=Decimal("2000"),
        protein_g=Decimal("100"),
        carbohydrates_g=Decimal("250"),
        fat_g=Decimal("70"),
    )
    zero = NutrientProfile(
        calories_kcal=Decimal("0"),
        protein_g=Decimal("0"),
        carbohydrates_g=Decimal("0"),
        fat_g=Decimal("0"),
        sodium_mg=Decimal("0"),
        dietary_fiber_g=Decimal("0"),
    )
    return MealRecommendation(
        service_date=DAY,
        meal=2,
        items=(item,),
        projected_meal_nutrition=zero,
        projected_daily_total=zero,
        projected_balance=calculate_daily_balance(targets, zero),
        objective_score=Decimal("0"),
        diagnostics=MealOptimizationDiagnostics(
            candidate_occurrences=1,
            eligible_candidates=1,
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


def _modified_semantic(quantity_text: str, *, scope: str = "complete") -> MealReportSemanticResult:
    return MealReportSemanticResult.model_validate(
        {
            "intent": "meal_report",
            "report_scope": scope,
            "location_plan_item_id": None,
            "location_food_text": None,
            "replacement_mode": None,
            "replacement_plan_item_ids": [],
            "planned_items": [
                {
                    "plan_item_id": "item_1",
                    "reference_text": "the sweet potato",
                    "action": "eaten",
                    "quantity_relation": "modified",
                    "quantity_text": quantity_text,
                    "comparison_plan_item_id": None,
                }
            ],
            "additional_foods": [],
            "unresolved_statements": [],
        }
    )


@dataclass
class _NoAdditionalFoodResolver:
    def resolve(self, request):
        raise AssertionError(f"unexpected unplanned-food lookup: {request}")


@dataclass
class _FailingSemanticRenderer:
    calls: int = 0

    def render(self, request):
        del request
        self.calls += 1
        raise AssertionError("calibrated count wording must not invoke semantic rendering")


@dataclass
class _FixedSemanticInterpreter:
    result: MealReportSemanticResult

    def interpret(self, plan, user_text):
        del plan, user_text
        return self.result


@dataclass
class _ScriptedSemanticInterpreter:
    responses: dict[str, MealReportSemanticResult]

    def interpret(self, plan, user_text):
        del plan
        return self.responses[user_text]

    def interpret_with_context(self, plan, user_text, conversation_context):
        del plan, conversation_context
        return self.responses[user_text]


@dataclass
class _FixedPlanResolver:
    persisted_plan: object

    def resolve(self, source_event_id: str):
        del source_event_id
        return self.persisted_plan


class ServingPresentationTests(unittest.TestCase):
    def _calibration(self):
        calibration = DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS.find(
            CALIBRATED_SOURCE,
            Serving(Decimal("1.0"), "Each", "1.0 Each"),
        )
        assert calibration is not None
        return calibration

    def test_calibration_is_stable_identity_and_serving_guarded(self) -> None:
        registry = DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS
        self.assertIsNotNone(
            registry.find(
                CALIBRATED_SOURCE,
                Serving(Decimal("1"), "Each", "1 Each"),
            )
        )
        self.assertIsNone(
            registry.find(
                SourceIdentifier("component", "7:181:99999"),
                Serving(Decimal("1"), "Each", "1 Each"),
            )
        )
        self.assertIsNone(
            registry.find(
                CALIBRATED_SOURCE,
                Serving(Decimal("1"), "Cup", "1 Cup"),
            )
        )

    def test_calibration_follows_stable_identity_across_occurrences(self) -> None:
        record = _record()
        renderings = []
        for occurrence_id in (1, 99):
            item = RecommendedMealItem(
                _occurrence(record, occurrence_id=occurrence_id),
                record,
                Decimal("2"),
            )
            renderings.append(
                render_meal_recommendation(
                    _recommendation(item),
                    RecommendedPortionRenderer(),
                ).meal_plan.items[0].natural_quantity_text
            )
        self.assertEqual(
            renderings,
            [
                "8 sweet potato quarter pieces (2 whole sweet potatoes)",
                "8 sweet potato quarter pieces (2 whole sweet potatoes)",
            ],
        )

    def test_calibrated_outbound_text_and_display_name_keep_official_quantity(self) -> None:
        semantic = _FailingSemanticRenderer()
        request = RecommendedPortionRenderRequest(
            "Baked Sweet Potatoes-Master",
            Serving(Decimal("1"), "Each", "1 Each"),
            Decimal("2.00"),
            source_identifier=CALIBRATED_SOURCE,
        )
        rendering = RecommendedPortionRenderer(semantic).render(request)

        self.assertEqual(
            rendering.natural_quantity_text,
            "8 sweet potato quarter pieces (2 whole sweet potatoes)",
        )
        self.assertEqual(rendering.natural_display_name, "Baked Sweet Potatoes")
        self.assertEqual(rendering.physical_quantity, request.physical_quantity)
        self.assertEqual(request.recommended_official_servings, Decimal("2.00"))
        self.assertEqual(semantic.calls, 0)

    def test_fractional_calibrated_rendering_uses_exact_decimal_arithmetic(self) -> None:
        calibration = self._calibration()
        cases = {
            Decimal("0.25"): "1 sweet potato quarter piece",
            Decimal("0.5"): "2 sweet potato quarter pieces",
            Decimal("1.5"): "6 sweet potato quarter pieces (1.5 whole sweet potatoes)",
            Decimal("2.00"): "8 sweet potato quarter pieces (2 whole sweet potatoes)",
        }
        for official_servings, expected in cases.items():
            with self.subTest(official_servings=official_servings):
                presentation = calibration.render(official_servings)
                self.assertEqual(presentation.natural_quantity_text, expected)
                self.assertIsInstance(presentation.presentation_amount, Decimal)
                self.assertEqual(
                    presentation.presentation_amount,
                    official_servings * Decimal("4"),
                )

    def test_similarly_named_or_unrelated_each_foods_do_not_inherit_quarter_pieces(self) -> None:
        same_name_other_identity = RecommendedPortionRenderRequest(
            "Baked Sweet Potatoes-Master",
            Serving(Decimal("1"), "Each", "1 Each"),
            Decimal("2"),
            source_identifier=SourceIdentifier("component", "7:181:99999"),
        )
        chicken = RecommendedPortionRenderRequest(
            "Chicken Tenders",
            Serving(Decimal("3"), "Each", "3 Each"),
            Decimal("1"),
            source_identifier=SourceIdentifier("component", "7:181:2"),
        )
        renderer = RecommendedPortionRenderer()

        same_name = renderer.render(same_name_other_identity)
        tenders = renderer.render(chicken)
        self.assertNotIn("quarter", same_name.natural_quantity_text)
        self.assertIsNone(same_name.natural_display_name)
        self.assertEqual(tenders.natural_quantity_text, "3 chicken tenders")

    def test_inbound_calibrated_phrases_map_to_exact_official_servings(self) -> None:
        food = _resolved(_record())
        interpreter = NaturalPortionInterpreter()
        self.assertIsInstance(
            interpreter.interpret(PortionInterpretationRequest(food, "3 quarter pieces")),
            UnresolvedPortion,
        )
        calibration = self._calibration()
        binding = freeze_presentation(
            food,
            Decimal("2"),
            calibration.render(Decimal("2")).natural_quantity_text,
            PresentationKind.REVERSIBLE_CALIBRATED,
            calibration,
        )
        cases = {
            "3 quarter pieces": Decimal("0.75"),
            "one quarter piece": Decimal("0.25"),
            "1/4 of a sweet potato": Decimal("0.25"),
            "half a sweet potato": Decimal("0.5"),
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                result = interpreter.interpret(
                    PortionInterpretationRequest(
                        food,
                        phrase,
                        presentation_binding=binding,
                        planned_official_servings=Decimal("2"),
                    )
                )
                self.assertIsInstance(result, InterpretedPortion)
                self.assertIsInstance(result.estimated_official_servings, Decimal)
                self.assertEqual(result.estimated_official_servings, expected)
                self.assertEqual(result.interpretation_method, "calibrated_presentation")
                self.assertEqual(result.confidence, "high")

    def test_inbound_calibration_does_not_leak_to_another_each_food(self) -> None:
        unrelated = _resolved(
            _record(
                "Unrelated Sweet Potato",
                source_identifier=SourceIdentifier("component", "7:181:10000"),
            )
        )
        result = NaturalPortionInterpreter().interpret(
            PortionInterpretationRequest(unrelated, "3 quarter pieces")
        )
        self.assertIsInstance(result, UnresolvedPortion)
        self.assertEqual(result.reason, "portion_semantic_matcher_unavailable")

    def test_rendered_plan_and_replacement_message_share_calibrated_presentation(self) -> None:
        record = _record()
        item = RecommendedMealItem(_occurrence(record), record, Decimal("2.00"))
        rendered = render_meal_recommendation(
            _recommendation(item),
            RecommendedPortionRenderer(),
        )

        self.assertEqual(
            rendered.meal_plan.items[0].natural_quantity_text,
            "8 sweet potato quarter pieces (2 whole sweet potatoes)",
        )
        self.assertEqual(
            rendered.meal_plan.items[0].display_food_name,
            "Baked Sweet Potatoes",
        )
        self.assertEqual(
            rendered.meal_plan.items[0].recommended_official_servings,
            Decimal("2.00"),
        )
        self.assertNotIn("Master", rendered.message)
        self.assertEqual(
            format_replacement_recommendation_message(rendered.meal_plan),
            "Updated recommendation:\n" + rendered.message,
        )

    def test_replacement_preparation_uses_the_shared_calibrated_renderer(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        with OfficialNutritionCatalog(Path(directory.name) / "replacement.sqlite3") as catalog:
            pork = component(1, "Pork")
            pork.update({"recipePortionSize": "1", "recipePortionSizeUnit": "Each"})
            sweet_potato = component(62889, "Baked Sweet Potatoes-Master")
            sweet_potato.update(
                {
                    "recipePortionSize": "1",
                    "recipePortionSizeUnit": "Each",
                    "dietaryFiber": "4",
                    "dietaryFiberUOM": "g",
                }
            )
            catalog.synchronize_fd_refresh(
                mapped(
                    menu_day(
                        DAY,
                        2,
                        "Lunch",
                        ((pork, "Homestyle"), (sweet_potato, "Homestyle")),
                    )
                ),
                requested_start=DAY,
                requested_end=DAY,
                observed_at=OBSERVED_AT,
            )
            state = DurableMealState(catalog)
            resolver = LocalFDFoodResolver(catalog)
            original_food = resolver.resolve(FoodResolutionRequest("Pork", DAY, meal=2))
            assert isinstance(original_food, ResolvedFood)
            original = state.save_meal_plan(
                MealPlan(
                    DAY,
                    2,
                    (PlannedMealItem(original_food, Decimal("1"), "1 pork"),),
                )
            )
            replacement = MealRecommendationOrchestrator(catalog).prepare_replacement(
                original,
                ("item_1",),
                whole_meal=False,
                targets=DailyTargets(
                    calories_kcal=Decimal("2000"),
                    protein_g=Decimal("100"),
                    carbohydrates_g=Decimal("250"),
                    fat_g=Decimal("70"),
                ),
            )

        item = replacement.rendering.meal_plan.items[0]
        self.assertEqual(item.food.source_identifier, CALIBRATED_SOURCE)
        self.assertIn("sweet potato quarter pieces", item.natural_quantity_text)
        self.assertEqual(item.display_food_name, "Baked Sweet Potatoes")

    def test_reconciliation_uses_official_servings_not_display_piece_count(self) -> None:
        food = _resolved(_record())
        plan = MealPlan(
            DAY,
            2,
            (
                PlannedMealItem(
                    food,
                    Decimal("2.00"),
                    "8 sweet potato quarter pieces (2 whole sweet potatoes)",
                    "Baked Sweet Potatoes",
                    freeze_presentation(food, Decimal("2.00"),
                        "8 sweet potato quarter pieces (2 whole sweet potatoes)",
                        PresentationKind.REVERSIBLE_CALIBRATED,
                        DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS.calibrations[0]),
                ),
            ),
        )
        reconciled = MealReportReconciler(
            _FixedSemanticInterpreter(_modified_semantic("3 quarter pieces")),
            _NoAdditionalFoodResolver(),
            NaturalPortionInterpreter(),
        ).reconcile(plan, "I ate 3 quarter pieces")

        self.assertEqual(reconciled.clarification_items, ())
        eaten = reconciled.eaten_items[0]
        self.assertEqual(eaten.official_servings, Decimal("0.75"))
        self.assertEqual(eaten.quantity_source, "explicit_deterministic")
        self.assertEqual(
            scale_nutrients(eaten.record.nutrients, eaten.official_servings).calories_kcal,
            Decimal("75.00"),
        )


class ServingPresentationConversationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog = OfficialNutritionCatalog(Path(self.directory.name) / "nutrition.sqlite3")
        self.addCleanup(self.catalog.close)
        sweet_potato = component(62889, "Baked Sweet Potatoes-Master")
        sweet_potato.update(
            {
                "recipePortionSize": "1",
                "recipePortionSizeUnit": "Each",
                "dietaryFiber": "4",
                "dietaryFiberUOM": "g",
            }
        )
        self.catalog.synchronize_fd_refresh(
            mapped(menu_day(DAY, 2, "Lunch", ((sweet_potato, "Homestyle"),))),
            requested_start=DAY,
            requested_end=DAY,
            observed_at=OBSERVED_AT,
        )
        self.state = DurableMealState(self.catalog)
        resolver = LocalFDFoodResolver(self.catalog)
        resolved = resolver.resolve(
            FoodResolutionRequest("Baked Sweet Potatoes-Master", DAY, meal=2)
        )
        assert isinstance(resolved, ResolvedFood)
        self.persisted_plan = self.state.save_meal_plan(
            MealPlan(
                DAY,
                2,
                (
                    PlannedMealItem(
                        resolved,
                        Decimal("2.00"),
                        "8 sweet potato quarter pieces (2 whole sweet potatoes)",
                        "Baked Sweet Potatoes",
                        freeze_presentation(resolved, Decimal("2.00"),
                            "8 sweet potato quarter pieces (2 whole sweet potatoes)",
                            PresentationKind.REVERSIBLE_CALIBRATED,
                            DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS.calibrations[0]),
                    ),
                ),
            ),
            plan_id="calibrated-lunch",
            created_at=OBSERVED_AT,
        )

    def test_draft_follow_up_accepts_calibrated_quarter_piece_quantity(self) -> None:
        initial = _modified_semantic("some", scope="partial")
        follow_up = _modified_semantic("3 quarter pieces", scope="partial")
        complete = MealReportSemanticResult.model_validate(
            {
                "intent": "clarification_answer",
                "report_scope": "complete",
                "location_plan_item_id": None,
                "location_food_text": None,
                "replacement_mode": None,
                "replacement_plan_item_ids": [],
                "planned_items": [],
                "additional_foods": [],
                "unresolved_statements": [],
            }
        )
        reconciler = MealReportReconciler(
            _ScriptedSemanticInterpreter(
                {
                    "I had some sweet potato": initial,
                    "3 quarter pieces": follow_up,
                    "that's all": complete,
                }
            ),
            LocalFDFoodResolver(self.catalog),
            NaturalPortionInterpreter(),
        )
        conversation = MealReportConversationOrchestrator(self.state, reconciler)
        context = _FixedPlanResolver(self.persisted_plan)

        unresolved = conversation.process(
            chat_guid="chat-calibrated",
            user_text="I had some sweet potato",
            source_event_id="calibrated-initial",
            context_resolver=context,
        )
        self.assertEqual(unresolved.outcome, "clarification_required")
        self.assertIn("Baked Sweet Potatoes", unresolved.message)
        self.assertEqual(self.state.load_meal_plan("calibrated-lunch").status, "active")

        resolved = conversation.process(
            chat_guid="chat-calibrated",
            user_text="3 quarter pieces",
            source_event_id="calibrated-follow-up",
            context_resolver=context,
        )
        self.assertEqual(resolved.outcome, "clarification_required")
        self.assertEqual(self.state.load_meal_plan("calibrated-lunch").status, "active")

        applied = conversation.process(
            chat_guid="chat-calibrated",
            user_text="that's all",
            source_event_id="calibrated-complete",
            context_resolver=context,
        )
        self.assertEqual(applied.outcome, "applied")
        self.assertEqual(self.state.load_meal_plan("calibrated-lunch").status, "applied")
        connection = self.catalog._connection
        assert connection is not None
        row = connection.execute(
            "SELECT official_servings, quantity_source FROM accepted_intake_entries"
        ).fetchone()
        self.assertEqual(tuple(row), ("0.75", "explicit_deterministic"))


if __name__ == "__main__":
    unittest.main()
