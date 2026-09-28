"""Phase 2C end-to-end rendering and frozen reverse-quantity behavior."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from nutrition_optimizer.durable_state import DurableMealState, PersistedMealPlan
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import (
    FoodResolutionRequest,
    LocalFDFoodResolver,
    ResolvedFood,
)
from nutrition_optimizer.meal_optimizer import RecommendedMealItem
from nutrition_optimizer.meal_report import (
    MealPlan,
    MealReportReconciler,
    PlannedMealItem,
)
from nutrition_optimizer.meal_report_conversation import (
    MealReportConversationOrchestrator,
)
from nutrition_optimizer.portion_interpretation import NaturalPortionInterpreter
from nutrition_optimizer.recommendation_rendering import (
    RecommendedPortionRenderer,
    render_meal_recommendation,
)
from nutrition_optimizer.serving_presentation import (
    DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS,
    PresentationCalibrationRegistry,
)
from tests.test_food_resolution import component, mapped, menu_day
from tests.test_meal_report_conversation import ScriptedInterpreter, eaten, semantic, skipped
from tests.test_meal_report import semantic as report_semantic
from tests import test_serving_presentation as serving_fixtures


@dataclass
class _PlanResolver:
    plan: PersistedMealPlan

    def resolve(self, source_event_id: str) -> PersistedMealPlan:
        del source_event_id
        return self.plan


class Phase2CPresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "phase2c.sqlite3"
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self._close)
        spaghetti = component(75146, "Corn Spaghetti")
        spaghetti.update(recipePortionSize="4", recipePortionSizeUnit="Ounce")
        potato = component(62889, "Baked Sweet Potatoes-Master")
        potato.update(recipePortionSize="1", recipePortionSizeUnit="Each")
        eggs = component(91234, "Scrambled Eggs")
        eggs.update(recipePortionSize="4", recipePortionSizeUnit="Ounce")
        self.catalog.synchronize_fd_refresh(
            mapped(
                menu_day(
                    serving_fixtures.DAY,
                    2,
                    "Lunch",
                    ((spaghetti, "Pasta"), (potato, "Homestyle"), (eggs, "Grill")),
                )
            ),
            requested_start=serving_fixtures.DAY,
            requested_end=serving_fixtures.DAY,
            observed_at=serving_fixtures.OBSERVED_AT,
        )
        self.state = DurableMealState(self.catalog)
        resolver = LocalFDFoodResolver(self.catalog)
        self.spaghetti = resolver.resolve(
            FoodResolutionRequest("Corn Spaghetti", serving_fixtures.DAY, meal=2)
        )
        self.potato = resolver.resolve(
            FoodResolutionRequest(
                "Baked Sweet Potatoes-Master",
                serving_fixtures.DAY,
                meal=2,
            )
        )
        self.eggs = resolver.resolve(
            FoodResolutionRequest("Scrambled Eggs", serving_fixtures.DAY, meal=2)
        )
        assert isinstance(self.spaghetti, ResolvedFood)
        assert isinstance(self.potato, ResolvedFood)
        assert isinstance(self.eggs, ResolvedFood)

    def _close(self) -> None:
        if getattr(self, "catalog", None) is not None:
            self.catalog.close()

    def _recommendation(self, official_servings: Decimal = Decimal("2")):
        item = RecommendedMealItem(
            self.potato.occurrence,
            self.potato.nutrition_record,
            official_servings,
        )
        return serving_fixtures._recommendation(item)

    def _rendered_plan(
        self,
        *,
        registry: PresentationCalibrationRegistry = (
            DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS
        ),
        official_servings: Decimal = Decimal("2"),
        semantic_renderer: object | None = None,
    ) -> MealPlan:
        with patch(
            "nutrition_optimizer.recommendation_rendering."
            "DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS",
            registry,
        ):
            return render_meal_recommendation(
                self._recommendation(official_servings),
                RecommendedPortionRenderer(semantic_renderer),
            ).meal_plan

    @staticmethod
    def _report(plan: MealPlan, phrase: str):
        return MealReportReconciler(
            serving_fixtures._FixedSemanticInterpreter(
                serving_fixtures._modified_semantic(phrase)
            ),
            serving_fixtures._NoAdditionalFoodResolver(),
            NaturalPortionInterpreter(),
        ).reconcile(plan, f"I ate {phrase}")

    def _semantic_report(self, plan: MealPlan, text: str, output):
        return MealReportReconciler(
            ScriptedInterpreter({text: output}),
            LocalFDFoodResolver(self.catalog),
            NaturalPortionInterpreter(),
        ).reconcile(plan, text)

    def _conversation_report(
        self, persisted: PersistedMealPlan, text: str, output, guid: str,
        *, chat_guid: str = "grounding-chat",
    ):
        return MealReportConversationOrchestrator(
            self.state,
            MealReportReconciler(
                ScriptedInterpreter({text: output}),
                LocalFDFoodResolver(self.catalog),
                NaturalPortionInterpreter(),
            ),
        ).process(
            chat_guid=chat_guid,
            user_text=text,
            source_event_id=guid,
            context_resolver=_PlanResolver(persisted),
        )

    def _rounded_registry(self) -> PresentationCalibrationRegistry:
        base = DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS.calibrations[0]
        rounded = replace(
            base,
            calibration_id="test-only-rounded-scoop",
            presentation_units_per_official_serving=Decimal("0.9"),
            presentation_unit_singular="test scoop",
            presentation_unit_plural="test scoops",
            presentation_aliases=("test scoop", "test scoops"),
            whole_items_per_official_serving=None,
            whole_item_singular=None,
            whole_item_plural=None,
            whole_item_aliases=(),
            display_rounding_increment=Decimal("1"),
        )
        return PresentationCalibrationRegistry((rounded,))

    def _scoop_plan(self) -> MealPlan:
        base = self._rounded_registry().calibrations[0]
        scoop = replace(
            base,
            presentation_unit_singular="scoop",
            presentation_unit_plural="scoops",
            presentation_aliases=("scoop", "scoops"),
        )
        return self._rendered_plan(registry=PresentationCalibrationRegistry((scoop,)))

    def _two_food_plan(self) -> MealPlan:
        potato = self._rendered_plan().items[0]
        return MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "4 oz spaghetti"), potato,
        ))

    def _assert_complete_rejected(
        self, plan: MealPlan, text: str, output, case_id: str,
    ):
        persisted = self.state.save_meal_plan(plan, plan_id=case_id)
        result = self._conversation_report(
            persisted, text, output, case_id + "-guid", chat_guid=case_id + "-chat",
        )
        self.assertEqual(result.outcome, "clarification_required")
        self.assertIsNotNone(result.draft)
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())
        self.assertEqual(self.catalog._connection.execute(
            "SELECT count(*) FROM accepted_intake_entries"
        ).fetchone()[0], 0)
        self.assertEqual(
            self.catalog._connection.execute(
                "SELECT count(*) FROM meal_report_applications"
            ).fetchone()[0],
            0,
        )
        return result

    def _assert_unresolved_after_reopen(
        self, draft_id: str, *, plan_item_id: str | None = None,
    ) -> None:
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        reopened = self.state.load_meal_report_draft(draft_id)
        self.assertIsNotNone(reopened)
        facts = (
            reopened.unplanned_items if plan_item_id is None else
            tuple(fact for fact in reopened.planned_items
                  if fact.plan_item_id == plan_item_id)
        )
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].quantity_status, "unresolved")
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())
        self.assertEqual(self.catalog._connection.execute(
            "SELECT count(*) FROM meal_report_applications"
        ).fetchone()[0], 0)

    def test_sweet_potato_render_and_two_quarter_piece_reverse(self) -> None:
        plan = self._rendered_plan()
        item = plan.items[0]
        self.assertEqual(
            item.natural_quantity_text,
            "8 sweet potato quarter pieces (2 whole sweet potatoes)",
        )
        self.assertEqual(
            item.presentation_binding.calibration.calibration_id,
            "phelps-baked-sweet-potato-quarter",
        )
        report = self._report(plan, "2 quarter pieces")
        self.assertEqual(report.clarification_items, ())
        self.assertEqual(report.eaten_items[0].official_servings, Decimal("0.5"))

    def test_scrambled_egg_visual_stays_non_authoritative_after_reopen(self) -> None:
        item = RecommendedMealItem(
            self.eggs.occurrence, self.eggs.nutrition_record, Decimal("2")
        )
        plan = render_meal_recommendation(
            serving_fixtures._recommendation(item), RecommendedPortionRenderer()
        ).meal_plan
        self.assertEqual(plan.items[0].natural_quantity_text, "about 2 baseball-sized portions")
        self.assertEqual(plan.items[0].presentation_binding.kind.value, "descriptive_only")
        saved = self.state.save_meal_plan(plan)
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        loaded = self.state.load_meal_plan(saved.plan_id)
        assert loaded is not None
        self.assertIsNone(loaded.plan.items[0].presentation_binding.calibration)
        report = self._report(loaded.plan, "1 baseball-sized portion")
        self.assertEqual(report.eaten_items, ())
        self.assertEqual(len(report.clarification_items), 1)

    def test_rounded_display_never_becomes_reverse_arithmetic(self) -> None:
        plan = self._rendered_plan(registry=self._rounded_registry())
        item = plan.items[0]
        self.assertEqual(item.natural_quantity_text, "about 2 test scoops")
        calibration = item.presentation_binding.calibration
        self.assertEqual(
            calibration.presentation_units_per_official_serving,
            Decimal("0.9"),
        )
        saved = self.state.save_meal_plan(plan, plan_id="rounded-plan")
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        loaded = self.state.load_meal_plan(saved.plan_id)
        assert loaded is not None
        self.assertEqual(loaded.plan.items[0].presentation_binding, item.presentation_binding)
        one_scoop = self._report(loaded.plan, "1 test scoop")
        self.assertEqual(
            one_scoop.eaten_items[0].official_servings,
            Decimal("1.11111111111111111111111111111111111"),
        )

    def test_small_calibrated_amount_does_not_round_up_to_a_full_unit(self) -> None:
        calibration = self._rounded_registry().calibrations[0]
        presentation = calibration.render(Decimal("0.1"))
        self.assertEqual(presentation.natural_quantity_text, "0.09 test scoops")
        self.assertEqual(
            calibration.presentation_units_per_official_serving,
            Decimal("0.9"),
        )

    def test_calibrated_amounts_below_equal_and_above_plan_are_proportional(self) -> None:
        plan = self._rendered_plan(registry=self._rounded_registry())
        for phrase, expected in (
            ("half a test scoop", Decimal("0.555555555555555555555555555555555556")),
            ("1.8 test scoops", Decimal("2")),
            ("3 test scoops", Decimal("3.33333333333333333333333333333333333")),
        ):
            with self.subTest(phrase=phrase):
                report = self._report(plan, phrase)
                self.assertEqual(report.eaten_items[0].official_servings, expected)
        self.assertGreater(
            self._report(plan, "3 test scoops").eaten_items[0].official_servings,
            plan.items[0].recommended_official_servings,
        )

    def test_only_frozen_aliases_resolve(self) -> None:
        plan = self._rendered_plan(registry=self._rounded_registry())
        self.assertEqual(
            self._report(plan, "1 test scoop").eaten_items[0].quantity_source,
            "explicit_deterministic",
        )
        for phrase in ("1 scoop", "1 ladle", "one serving spoon"):
            with self.subTest(phrase=phrase):
                report = self._report(plan, phrase)
                self.assertEqual(report.eaten_items, ())
                self.assertEqual(len(report.clarification_items), 1)

    def test_unplanned_potato_has_no_visual_reverse_authority(self) -> None:
        from tests.test_meal_report import FakeSemanticInterpreter, semantic as report_semantic

        plan = MealPlan(
            serving_fixtures.DAY,
            2,
            (PlannedMealItem(self.spaghetti, Decimal("1"), "about 4 oz of corn spaghetti"),),
        )
        report = MealReportReconciler(
            FakeSemanticInterpreter(
                report_semantic(
                    [],
                    [{"food_text": "Baked Sweet Potatoes-Master", "quantity_text": "2 quarter pieces"}],
                )
            ),
            LocalFDFoodResolver(self.catalog),
            NaturalPortionInterpreter(),
        ).reconcile(plan, "I ate 2 quarter pieces of baked sweet potato")
        self.assertEqual(report.eaten_items, ())
        self.assertEqual(report.unplanned_items[0].resolved_food.source_identifier, self.potato.source_identifier)
        self.assertIsNone(report.unplanned_items[0].official_servings)
        self.assertEqual(len(report.clarification_items), 1)

    def test_relative_quantity_ignores_visual_rounding(self) -> None:
        plan = self._rendered_plan(registry=self._rounded_registry())
        for phrase, expected in (
            ("half of that", Decimal("1")),
            ("quarter of that", Decimal("0.5")),
            ("50%", Decimal("1")),
        ):
            with self.subTest(phrase=phrase):
                interpreter = ScriptedInterpreter(
                    {
                        phrase: semantic(
                            planned=[
                                eaten(
                                    "item_1",
                                    phrase,
                                    relation="fraction_of_recommended",
                                    quantity_text=phrase,
                                )
                            ]
                        )
                    }
                )
                report = MealReportReconciler(
                    interpreter,
                    serving_fixtures._NoAdditionalFoodResolver(),
                    NaturalPortionInterpreter(),
                ).reconcile(plan, phrase)
                self.assertEqual(report.eaten_items[0].official_servings, expected)

    def test_complete_physical_half_cannot_become_half_the_plan(self) -> None:
        plan = self._rendered_plan(official_servings=Decimal("2"))
        text = "I ate half a sweet potato"
        output = semantic(scope="complete", planned=[eaten(
            "item_1", text, relation="fraction_of_recommended", quantity_text="half",
        )])
        self._assert_complete_rejected(plan, text, output, "physical-half")

    def test_complete_explicit_plan_half_uses_exact_planned_amount(self) -> None:
        plan = self._rendered_plan(official_servings=Decimal("2"))
        text = "I ate half of what you recommended for sweet potato"
        output = semantic(scope="complete", planned=[eaten(
            "item_1", text, relation="fraction_of_recommended",
            quantity_text="half of what you recommended",
        )])
        persisted = self.state.save_meal_plan(plan, plan_id="explicit-plan-half")
        result = self._conversation_report(persisted, text, output, "explicit-plan-half-guid")
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY)[0].official_servings,
                         Decimal("1"))

    def test_complete_local_plan_fraction_and_item_all_are_independent(self) -> None:
        plan = self._two_food_plan()
        text = "I ate half of the recommended spaghetti and all the sweet potato"
        output = semantic(scope="complete", planned=[
            eaten("item_1", "half of the recommended spaghetti",
                  relation="fraction_of_recommended", quantity_text="half"),
            eaten("item_2", "all the sweet potato"),
        ])
        persisted = self.state.save_meal_plan(plan, plan_id="local-plan-relations")
        result = self._conversation_report(persisted, text, output, "local-plan-relations-guid")
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(
            [(entry.plan_item_id, entry.official_servings)
             for entry in self.state.get_daily_intake(serving_fixtures.DAY)],
            [("item_1", Decimal("0.5")), ("item_2", Decimal("2"))],
        )

    def test_complete_percentage_needs_its_plan_referent(self) -> None:
        plan = self._rendered_plan()
        text = "I ate 25% of the recommended sweet potato amount"
        output = semantic(scope="complete", planned=[eaten(
            "item_1", text, relation="fraction_of_recommended", quantity_text="25%",
        )])
        persisted = self.state.save_meal_plan(plan, plan_id="explicit-plan-percent")
        result = self._conversation_report(persisted, text, output, "explicit-plan-percent-guid")
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY)[0].official_servings,
                         Decimal("0.5"))

    def test_complete_physical_decimal_fraction_cannot_borrow_plan(self) -> None:
        plan = self._rendered_plan()
        text = "I ate .5 of the sweet potato"
        output = semantic(scope="complete", planned=[eaten(
            "item_1", text, relation="fraction_of_recommended", quantity_text=".5",
        )])
        self._assert_complete_rejected(plan, text, output, "physical-decimal-fraction")

    def test_complete_unplanned_physical_fraction_cannot_borrow_plan_cue(self) -> None:
        plan = self._rendered_plan()
        text = "I ate half an egg as recommended"
        output = semantic(scope="complete", planned=[eaten(
            "item_1", text, relation="fraction_of_recommended", quantity_text="half",
        )])
        self._assert_complete_rejected(plan, text, output, "foreign-physical-fraction")

    def test_complete_fraction_cannot_move_from_potato_to_spaghetti(self) -> None:
        for index, text in enumerate((
            "I ate half the sweet potato and spaghetti",
            "I ate spaghetti and half the sweet potato",
        )):
            with self.subTest(text=text):
                output = semantic(scope="complete", planned=[eaten(
                    "item_1", "spaghetti", relation="fraction_of_recommended",
                    quantity_text="half",
                )])
                rejected = self._assert_complete_rejected(
                    self._two_food_plan(), text, output, f"wrong-fraction-{index}",
                )
                self.assertEqual(rejected.draft.planned_items[0].plan_item_id, "item_1")
                self.assertEqual(rejected.draft.planned_items[0].quantity_status, "unresolved")
                draft_id = rejected.draft.draft_id
                self.catalog.close()
                self.catalog = OfficialNutritionCatalog(self.path)
                self.state = DurableMealState(self.catalog)
                reopened = self.state.load_meal_report_draft(draft_id)
                self.assertEqual(reopened.planned_items[0].quantity_status, "unresolved")
                followup = "I ate 2 quarter pieces of sweet potato"
                result = self._conversation_report(
                    self.state.load_meal_plan(f"wrong-fraction-{index}"),
                    followup,
                    semantic(scope="complete", planned=[eaten(
                        "item_2", followup, relation="modified",
                        quantity_text="2 quarter pieces",
                    )]),
                    f"wrong-fraction-{index}-followup",
                    chat_guid=f"wrong-fraction-{index}-chat",
                )
                self.assertEqual(result.outcome, "clarification_required")
                facts = {fact.plan_item_id: fact for fact in result.draft.planned_items}
                self.assertEqual(facts["item_1"].quantity_status, "unresolved")
                self.assertEqual(facts["item_2"].official_servings, Decimal("0.5"))
                self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())

    def test_repeated_same_food_amounts_cannot_follow_model_choice(self) -> None:
        text = "I ate 1 serving of sweet potato. Later I had 3 servings of sweet potato"
        for index, selected in enumerate(("1 serving", "3 servings")):
            with self.subTest(selected=selected):
                result = self._assert_complete_rejected(
                    self._rendered_plan(), text,
                    semantic(scope="complete", planned=[eaten(
                        "item_1", text, relation="modified", quantity_text=selected,
                    )]),
                    f"repeated-planned-{index}",
                )
                self.assertEqual(result.draft.planned_items[0].quantity_status, "unresolved")

    def test_unnamed_followup_servings_cannot_follow_model_choice(self) -> None:
        for text_index, text in enumerate((
            "I ate 1 serving of sweet potato. Later I had 3 more servings.",
            "I ate 1 serving. Later I had 3 more servings.",
        )):
            for selected_index, selected in enumerate(("1 serving", "3 more servings")):
                with self.subTest(text=text, selected=selected):
                    case_id = f"unnamed-planned-{text_index}-{selected_index}"
                    result = self._assert_complete_rejected(
                        self._rendered_plan(), text,
                        semantic(scope="complete", planned=[eaten(
                            "item_1", text, relation="modified", quantity_text=selected,
                        )]),
                        case_id,
                    )
                    self.assertEqual(result.draft.planned_items[0].quantity_status,
                                     "unresolved")
                    self._assert_unresolved_after_reopen(
                        result.draft.draft_id, plan_item_id="item_1",
                    )

    def test_quantity_evidence_without_a_count_cannot_follow_model_choice(self) -> None:
        for wording in ("twice that", "double that", "another serving", "same again"):
            for named in (True, False):
                text = (
                    "I ate 1 serving" + (" of sweet potato" if named else "")
                    + f". Later I had {wording}."
                )
                for selected in ("1 serving", wording):
                    with self.subTest(text=text, selected=selected):
                        case_id = f"quantity-evidence-{wording.replace(' ', '-')}-{named}-{selected.replace(' ', '-')}"
                        result = self._assert_complete_rejected(
                            self._rendered_plan(), text,
                            semantic(scope="complete", planned=[eaten(
                                "item_1", text, relation="modified", quantity_text=selected,
                            )]),
                            case_id,
                        )
                        self.assertEqual(result.draft.planned_items[0].quantity_status,
                                         "unresolved")
                        self._assert_unresolved_after_reopen(
                            result.draft.draft_id, plan_item_id="item_1",
                        )

    def test_unaccounted_same_food_clauses_cannot_follow_model_choice(self) -> None:
        for wording in (
            "3 more servings", "twice that", "double that", "another serving",
            "the rest", "a second helping",
        ):
            for selected in ("1 serving", wording):
                with self.subTest(wording=wording, selected=selected):
                    action = "finished" if wording == "the rest" else (
                        "took" if wording == "a second helping" else "had"
                    )
                    text = (
                        "I ate 1 serving of sweet potato. "
                        f"Later I {action} {wording}."
                    )
                    case_id = (
                        f"unaccounted-clause-{wording.replace(' ', '-')}-"
                        f"{selected.replace(' ', '-')}"
                    )
                    result = self._assert_complete_rejected(
                        self._rendered_plan(), text,
                        semantic(scope="complete", planned=[eaten(
                            "item_1", text, relation="modified", quantity_text=selected,
                        )]),
                        case_id,
                    )
                    self.assertEqual(result.draft.planned_items[0].quantity_status,
                                     "unresolved")
                    self._assert_unresolved_after_reopen(
                        result.draft.draft_id, plan_item_id="item_1",
                    )

    def test_unaccounted_clause_stays_pending_after_other_food_completes(self) -> None:
        for index, wording in enumerate(("the rest", "a second helping")):
            with self.subTest(wording=wording):
                action = "finished" if wording == "the rest" else "took"
                text = f"I ate 1 serving of sweet potato. Later I {action} {wording}."
                case_id = f"unaccounted-durable-{index}"
                persisted = self.state.save_meal_plan(
                    self._two_food_plan(), plan_id=case_id,
                )
                first = self._conversation_report(
                    persisted, text,
                    semantic(scope="complete", planned=[eaten(
                        "item_2", text, relation="modified", quantity_text="1 serving",
                    )]),
                    case_id + "-first", chat_guid=case_id + "-chat",
                )
                self.assertEqual(first.outcome, "clarification_required")
                self._assert_unresolved_after_reopen(
                    first.draft.draft_id, plan_item_id="item_2",
                )
                followup = "I ate 4 oz spaghetti"
                second = self._conversation_report(
                    self.state.load_meal_plan(case_id), followup,
                    semantic(scope="complete", planned=[eaten(
                        "item_1", followup, relation="modified", quantity_text="4 oz",
                    )]),
                    case_id + "-second", chat_guid=case_id + "-chat",
                )
                self.assertEqual(second.outcome, "clarification_required")
                facts = {fact.plan_item_id: fact for fact in second.draft.planned_items}
                self.assertEqual(facts["item_1"].quantity_status, "resolved")
                self.assertEqual(facts["item_2"].quantity_status, "unresolved")
                self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())
                self.assertEqual(self.catalog._connection.execute(
                    "SELECT count(*) FROM accepted_intake_entries"
                ).fetchone()[0], 0)
                self.assertEqual(self.catalog._connection.execute(
                    "SELECT count(*) FROM meal_report_applications"
                ).fetchone()[0], 0)

    def test_unaccounted_clause_before_or_after_other_authority_relations(self) -> None:
        cases = (
            (
                "I finished the rest. I ate 1 serving of sweet potato.",
                "modified", "1 serving",
            ),
            (
                "Actually I finished the rest. I ate 1 serving of sweet potato.",
                "modified", "1 serving",
            ),
            (
                "I ate half of what you recommended for sweet potato. "
                "Later I finished the rest.",
                "fraction_of_recommended", "half of what you recommended",
            ),
            (
                "I ate sweet potato as recommended. Later I took a second helping.",
                "as_recommended", None,
            ),
        )
        for index, (text, relation, quantity) in enumerate(cases):
            with self.subTest(text=text):
                result = self._assert_complete_rejected(
                    self._rendered_plan(), text,
                    semantic(scope="complete", planned=[eaten(
                        "item_1", text, relation=relation, quantity_text=quantity,
                    )]),
                    f"unaccounted-other-relation-{index}",
                )
                self.assertEqual(result.draft.planned_items[0].quantity_status,
                                 "unresolved")

    def test_elliptical_named_food_followup_cannot_leave_first_amount(self) -> None:
        text = "I ate 1 serving of sweet potato. Later another sweet potato."
        result = self._assert_complete_rejected(
            self._rendered_plan(), text,
            semantic(scope="complete", planned=[eaten(
                "item_1", text, relation="modified", quantity_text="1 serving",
            )]),
            "elliptical-named-food-followup",
        )
        self.assertEqual(result.draft.planned_items[0].quantity_status, "unresolved")
        self._assert_unresolved_after_reopen(
            result.draft.draft_id, plan_item_id="item_1",
        )

    def test_physical_amount_and_plan_fraction_cannot_follow_model_choice(self) -> None:
        text = (
            "I ate 1 serving of sweet potato. "
            "Later I had half of what you recommended."
        )
        for index, (relation, selected) in enumerate((
            ("modified", "1 serving"),
            ("fraction_of_recommended", "half of what you recommended"),
        )):
            with self.subTest(selected=selected):
                result = self._assert_complete_rejected(
                    self._rendered_plan(official_servings=Decimal("4")), text,
                    semantic(scope="complete", planned=[eaten(
                        "item_1", text, relation=relation, quantity_text=selected,
                    )]),
                    f"physical-and-plan-fraction-{index}",
                )
                self.assertEqual(result.draft.planned_items[0].quantity_status,
                                 "unresolved")
                self._assert_unresolved_after_reopen(
                    result.draft.draft_id, plan_item_id="item_1",
                )

    def test_two_plan_relative_claims_cannot_follow_model_choice(self) -> None:
        text = (
            "I ate half of what you recommended for sweet potato. "
            "Later I had quarter of what you recommended."
        )
        for index, selected in enumerate((
            "half of what you recommended",
            "quarter of what you recommended",
        )):
            with self.subTest(selected=selected):
                result = self._assert_complete_rejected(
                    self._rendered_plan(official_servings=Decimal("4")), text,
                    semantic(scope="complete", planned=[eaten(
                        "item_1", text, relation="fraction_of_recommended",
                        quantity_text=selected,
                    )]),
                    f"two-plan-fractions-{index}",
                )
                self.assertEqual(result.draft.planned_items[0].quantity_status,
                                 "unresolved")
                self._assert_unresolved_after_reopen(
                    result.draft.draft_id, plan_item_id="item_1",
                )

    def test_other_persons_followup_quantity_does_not_compete(self) -> None:
        for index, wording in enumerate(("twice that", "another serving")):
            with self.subTest(wording=wording):
                text = f"I ate 1 serving of sweet potato. My friend had {wording}."
                persisted = self.state.save_meal_plan(
                    self._rendered_plan(), plan_id=f"other-person-amount-{index}",
                )
                result = self._conversation_report(
                    persisted, text,
                    semantic(scope="complete", planned=[eaten(
                        "item_1", text, relation="modified", quantity_text="1 serving",
                    )]),
                    f"other-person-amount-{index}-guid",
                )
                self.assertEqual(result.outcome, "applied")
                self.assertEqual(
                    self.state.get_daily_intake(serving_fixtures.DAY)[-1].official_servings,
                    Decimal("1"),
                )

    def test_other_persons_unaccounted_clause_does_not_compete(self) -> None:
        text = "I ate 1 serving of sweet potato. My friend finished the rest."
        persisted = self.state.save_meal_plan(
            self._rendered_plan(), plan_id="friend-finished-rest",
        )
        result = self._conversation_report(
            persisted, text,
            semantic(scope="complete", planned=[eaten(
                "item_1", text, relation="modified", quantity_text="1 serving",
            )]),
            "friend-finished-rest-guid",
        )
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(
            self.state.get_daily_intake(serving_fixtures.DAY)[0].official_servings,
            Decimal("1"),
        )

    def test_possessive_food_is_not_an_other_person_referent(self) -> None:
        text = "I ate 1 serving of sweet potato. My sweet potato was gone later."
        result = self._assert_complete_rejected(
            self._rendered_plan(), text,
            semantic(scope="complete", planned=[eaten(
                "item_1", text, relation="modified", quantity_text="1 serving",
            )]),
            "possessive-food-unaccounted",
        )
        self.assertEqual(result.draft.planned_items[0].quantity_status,
                         "unresolved")

    def test_local_correction_is_authoritative_when_model_selects_final_literal(self) -> None:
        for index, (text, selected, expected) in enumerate((
            ("I ate 1 serving of sweet potato, actually 2 servings",
             "2 servings", Decimal("2")),
            ("I ate 2 quarter pieces, actually 3",
             "3", Decimal("0.75")),
        )):
            with self.subTest(text=text):
                persisted = self.state.save_meal_plan(
                    self._rendered_plan(), plan_id=f"selected-final-correction-{index}",
                )
                result = self._conversation_report(
                    persisted, text,
                    semantic(scope="complete", planned=[eaten(
                        "item_1", text, relation="modified", quantity_text=selected,
                    )]),
                    f"selected-final-correction-{index}-guid",
                )
                self.assertEqual(result.outcome, "applied")
                self.assertEqual(
                    self.state.get_daily_intake(serving_fixtures.DAY)[-1].official_servings,
                    expected,
                )

    def test_later_quantity_named_for_other_food_stays_local(self) -> None:
        text = "I ate 1 serving of sweet potato. Later I had 4 oz spaghetti."
        persisted = self.state.save_meal_plan(
            self._two_food_plan(), plan_id="distinct-food-followup",
        )
        result = self._conversation_report(
            persisted, text,
            semantic(scope="complete", planned=[
                eaten("item_2", text, relation="modified", quantity_text="1 serving"),
                eaten("item_1", text, relation="modified", quantity_text="4 oz"),
            ]),
            "distinct-food-followup-guid",
        )
        self.assertEqual(result.outcome, "clarification_required")
        facts = {fact.plan_item_id: fact for fact in result.draft.planned_items}
        self.assertEqual(facts["item_2"].official_servings, Decimal("1"))
        self.assertEqual(facts["item_2"].quantity_status, "resolved")
        self.assertEqual(facts["item_1"].quantity_status, "unresolved")
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())

    def test_bare_unnamed_followup_amount_cannot_hide(self) -> None:
        for index, text in enumerate((
            "I ate 1 serving of sweet potato. Later I had 3 more.",
            "I ate 1 serving. Later I had three more.",
            "I ate 1 serving of sweet potato. Later I had another 3.",
            "I ate 1 serving of sweet potato. Later I had 3 additional servings.",
            "I ate 1 serving of sweet potato. Later I had 3 extra.",
            "I ate 1 serving of sweet potato. Later I had 3 additional.",
        )):
            with self.subTest(text=text):
                result = self._assert_complete_rejected(
                    self._rendered_plan(), text,
                    semantic(scope="complete", planned=[eaten(
                        "item_1", text, relation="modified", quantity_text="1 serving",
                    )]),
                    f"bare-unnamed-planned-{index}",
                )
                self.assertEqual(result.draft.planned_items[0].quantity_status,
                                 "unresolved")
                self._assert_unresolved_after_reopen(
                    result.draft.draft_id, plan_item_id="item_1",
                )

    def test_repeated_unplanned_food_amounts_cannot_follow_model_choice(self) -> None:
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "4 oz spaghetti"),
        ))
        text = (
            "I ate 1 serving of baked sweet potato. "
            "Later I had 3 servings of baked sweet potato"
        )
        for index, selected in enumerate(("1 serving", "3 servings")):
            with self.subTest(selected=selected):
                result = self._assert_complete_rejected(
                    plan, text,
                    report_semantic([], [{
                        "food_text": "Baked Sweet Potatoes-Master",
                        "quantity_text": selected,
                    }]),
                    f"repeated-unplanned-{index}",
                )
                self.assertEqual(result.draft.unplanned_items[0].quantity_status, "unresolved")

    def test_unplanned_unnamed_followup_cannot_follow_model_choice(self) -> None:
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "4 oz spaghetti"),
        ))
        for text_index, text in enumerate((
            "I ate 1 serving of baked sweet potato. Later I had 3 more servings.",
            "I ate 1 serving of baked sweet potato. Later I had 3 more.",
        )):
            selections = ("1 serving", "3 more servings") if text_index == 0 else (
                "1 serving", "3 more",
            )
            for selected_index, selected in enumerate(selections):
                with self.subTest(text=text, selected=selected):
                    result = self._assert_complete_rejected(
                        plan, text,
                        report_semantic([], [{
                            "food_text": "Baked Sweet Potatoes-Master",
                            "quantity_text": selected,
                        }]),
                        f"unnamed-unplanned-{text_index}-{selected_index}",
                    )
                    self.assertEqual(result.draft.unplanned_items[0].quantity_status,
                                     "unresolved")
                    self._assert_unresolved_after_reopen(result.draft.draft_id)

    def test_unplanned_quantity_evidence_without_a_count_clarifies(self) -> None:
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "4 oz spaghetti"),
        ))
        for wording in ("twice that", "double that", "another serving"):
            text = f"I ate 1 serving of baked sweet potato. Later I had {wording}."
            for selected in ("1 serving", wording):
                with self.subTest(text=text, selected=selected):
                    case_id = f"unplanned-quantity-evidence-{wording.replace(' ', '-')}-{selected.replace(' ', '-')}"
                    result = self._assert_complete_rejected(
                        plan, text,
                        report_semantic([], [{
                            "food_text": "Baked Sweet Potatoes-Master",
                            "quantity_text": selected,
                        }]),
                        case_id,
                    )
                    self.assertEqual(result.draft.unplanned_items[0].quantity_status,
                                     "unresolved")
                    self._assert_unresolved_after_reopen(result.draft.draft_id)

    def test_ambiguous_unnamed_followup_stays_unresolved(self) -> None:
        text = (
            "I ate 1 serving of sweet potato and spaghetti. "
            "Later I had 3 more servings."
        )
        for index, selected in enumerate(("1 serving", "3 more servings")):
            with self.subTest(selected=selected):
                result = self._assert_complete_rejected(
                    self._two_food_plan(), text,
                    semantic(scope="complete", planned=[eaten(
                        "item_2", text, relation="modified", quantity_text=selected,
                    )]),
                    f"ambiguous-unnamed-{index}",
                )
                facts = {fact.plan_item_id: fact for fact in result.draft.planned_items}
                self.assertEqual(facts["item_2"].quantity_status, "unresolved")
                self._assert_unresolved_after_reopen(
                    result.draft.draft_id, plan_item_id="item_2",
                )

    def test_unnamed_followup_after_two_named_amounts_cannot_choose_a_food(self) -> None:
        text = (
            "I ate 1 serving of sweet potato and 2 servings of spaghetti. "
            "Later I had 3 more servings."
        )
        result = self._assert_complete_rejected(
            self._two_food_plan(), text,
            semantic(scope="complete", planned=[
                eaten("item_2", text, relation="modified", quantity_text="1 serving"),
                eaten("item_1", text, relation="modified", quantity_text="2 servings"),
            ]),
            "ambiguous-two-named-foods",
        )
        facts = {fact.plan_item_id: fact for fact in result.draft.planned_items}
        self.assertEqual(facts["item_1"].quantity_status, "unresolved")
        self.assertEqual(facts["item_2"].quantity_status, "unresolved")
        self._assert_unresolved_after_reopen(
            result.draft.draft_id, plan_item_id="item_2",
        )

    def test_unnamed_followup_stays_pending_after_other_food_completion(self) -> None:
        text = "I ate 1 serving of sweet potato. Later I had 3 more servings."
        persisted = self.state.save_meal_plan(
            self._two_food_plan(), plan_id="unnamed-followup-durable",
        )
        first = self._conversation_report(
            persisted, text,
            semantic(scope="complete", planned=[eaten(
                "item_2", text, relation="modified", quantity_text="1 serving",
            )]),
            "unnamed-followup-first", chat_guid="unnamed-followup-chat",
        )
        self.assertEqual(first.outcome, "clarification_required")
        self._assert_unresolved_after_reopen(
            first.draft.draft_id, plan_item_id="item_2",
        )
        followup = "I ate 4 oz spaghetti"
        result = self._conversation_report(
            self.state.load_meal_plan("unnamed-followup-durable"),
            followup,
            semantic(scope="complete", planned=[eaten(
                "item_1", followup, relation="modified", quantity_text="4 oz",
            )]),
            "unnamed-followup-spaghetti", chat_guid="unnamed-followup-chat",
        )
        self.assertEqual(result.outcome, "clarification_required")
        facts = {fact.plan_item_id: fact for fact in result.draft.planned_items}
        self.assertEqual(facts["item_1"].official_servings, Decimal("1"))
        self.assertEqual(facts["item_2"].quantity_status, "unresolved")
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())
        self.assertEqual(self.catalog._connection.execute(
            "SELECT count(*) FROM accepted_intake_entries"
        ).fetchone()[0], 0)
        self.assertEqual(self.catalog._connection.execute(
            "SELECT count(*) FROM meal_report_applications"
        ).fetchone()[0], 0)

    def test_unnamed_followup_uses_one_active_draft_referent_only_to_block(self) -> None:
        persisted = self.state.save_meal_plan(
            self._two_food_plan(), plan_id="unnamed-draft-referent",
        )
        first_text = "I ate 4 oz spaghetti and some sweet potato"
        first = self._conversation_report(
            persisted, first_text,
            semantic(scope="complete", planned=[
                eaten("item_1", first_text, relation="modified", quantity_text="4 oz"),
                eaten("item_2", first_text, relation="modified", quantity_text="some"),
            ]),
            "unnamed-draft-first", chat_guid="unnamed-draft-chat",
        )
        self.assertEqual(first.outcome, "clarification_required")
        text = "I ate 1 serving. Later I had 3 extra."
        second = self._conversation_report(
            persisted, text,
            semantic(scope="complete", planned=[eaten(
                "item_2", text, relation="modified", quantity_text="1 serving",
            )]),
            "unnamed-draft-second", chat_guid="unnamed-draft-chat",
        )
        self.assertEqual(second.outcome, "clarification_required")
        facts = {fact.plan_item_id: fact for fact in second.draft.planned_items}
        self.assertEqual(facts["item_1"].official_servings, Decimal("1"))
        self.assertEqual(facts["item_2"].quantity_status, "unresolved")
        self._assert_unresolved_after_reopen(
            second.draft.draft_id, plan_item_id="item_2",
        )

    def test_named_spaghetti_quantity_does_not_compete_with_potato(self) -> None:
        text = "I ate 1 serving of sweet potato. Later spaghetti was 3 servings."
        result = self._assert_complete_rejected(
            self._two_food_plan(), text,
            semantic(scope="complete", planned=[
                eaten("item_2", text, relation="modified", quantity_text="1 serving"),
                eaten("item_1", text, relation="modified", quantity_text="3 servings"),
            ]),
            "named-spaghetti-separate",
        )
        facts = {fact.plan_item_id: fact for fact in result.draft.planned_items}
        self.assertEqual(facts["item_2"].official_servings, Decimal("1"))
        self.assertEqual(facts["item_1"].quantity_status, "unresolved")

    def test_unplanned_followup_amount_without_repeated_name_clarifies(self) -> None:
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "4 oz spaghetti"),
        ))
        text = "I ate 1 serving of baked sweet potato. Later I had 3 servings"
        result = self._assert_complete_rejected(
            plan, text,
            report_semantic([], [{
                "food_text": "Baked Sweet Potatoes-Master",
                "quantity_text": "1 serving",
            }]),
            "unplanned-unnamed-followup",
        )
        self.assertEqual(result.draft.unplanned_items[0].quantity_status, "unresolved")

    def test_repeated_whole_food_counts_do_not_hide_from_unit_scanner(self) -> None:
        text = "I ate one sweet potato. Later I had two sweet potatoes"
        result = self._assert_complete_rejected(
            self._rendered_plan(), text,
            semantic(scope="complete", planned=[eaten(
                "item_1", text, relation="modified", quantity_text="one sweet potato",
            )]),
            "repeated-whole-potatoes",
        )
        self.assertEqual(result.draft.planned_items[0].quantity_status, "unresolved")

    def test_later_unbounded_amount_cannot_leave_first_serving_authoritative(self) -> None:
        text = "I ate 1 serving of sweet potato. Later I had some sweet potato"
        result = self._assert_complete_rejected(
            self._rendered_plan(), text,
            semantic(scope="complete", planned=[eaten(
                "item_1", text, relation="modified", quantity_text="1 serving",
            )]),
            "later-unbounded-potato",
        )
        self.assertEqual(result.draft.planned_items[0].quantity_status, "unresolved")

    def test_repeated_frozen_quarter_pieces_cannot_select_one_amount(self) -> None:
        text = (
            "I ate 2 quarter pieces of sweet potato. "
            "Later I had 3 quarter pieces of sweet potato"
        )
        result = self._assert_complete_rejected(
            self._rendered_plan(), text,
            semantic(scope="complete", planned=[eaten(
                "item_1", text, relation="modified", quantity_text="2 quarter pieces",
            )]),
            "repeated-quarter-pieces",
        )
        self.assertEqual(result.draft.planned_items[0].quantity_status, "unresolved")

    def test_additive_or_separate_same_food_mentions_clarify(self) -> None:
        for index, text in enumerate((
            "I ate 1 serving of sweet potato plus 3 servings of sweet potato",
            "I ate 1 serving of sweet potato and then 3 servings of sweet potato",
            "I ate 1 serving of sweet potato then 3 servings of sweet potato",
            "I ate 1 serving of sweet potato. Afterward I had 3 servings of sweet potato",
            "I ate 1 serving of sweet potato. Afterwards I had 3 servings of sweet potato",
            "I ate 1 serving of sweet potato. Subsequently I had 3 servings of sweet potato",
            "I ate 1 serving of sweet potato and another 3 servings of sweet potato",
            "I ate 1 serving of sweet potato and also 3 servings of sweet potato",
        )):
            with self.subTest(text=text):
                result = self._assert_complete_rejected(
                    self._rendered_plan(), text,
                    semantic(scope="complete", planned=[eaten(
                        "item_1", text, relation="modified", quantity_text="1 serving",
                    )]),
                    f"repeated-connective-{index}",
                )
                self.assertEqual(result.draft.planned_items[0].quantity_status, "unresolved")

    def test_other_food_correction_cannot_choose_potato_amount(self) -> None:
        text = (
            "I ate 1 serving of sweet potato, actually 2 servings of spaghetti, "
            "then 3 servings of sweet potato"
        )
        for index, selected in enumerate(("1 serving", "3 servings")):
            with self.subTest(selected=selected):
                result = self._assert_complete_rejected(
                    self._two_food_plan(), text,
                    semantic(scope="complete", planned=[eaten(
                        "item_2", text, relation="modified", quantity_text=selected,
                    )]),
                    f"other-food-correction-{index}",
                )
                self.assertEqual(result.draft.planned_items[0].quantity_status, "unresolved")

    def test_bare_digit_correction_inherits_only_frozen_local_unit(self) -> None:
        text = "I ate 2 quarter pieces of sweet potato, actually 3"
        plan = self._rendered_plan()
        persisted = self.state.save_meal_plan(plan, plan_id="bare-quarter-correction")
        output = semantic(scope="complete", planned=[eaten(
            "item_1", text, relation="modified", quantity_text="2 quarter pieces",
        )])
        result = self._conversation_report(
            persisted, text, output, "bare-quarter-correction-guid",
        )
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY)[0].official_servings,
                         Decimal("0.75"))

    def test_single_local_serving_still_applies(self) -> None:
        text = "I ate 1 serving of sweet potato"
        persisted = self.state.save_meal_plan(self._rendered_plan(), plan_id="single-serving")
        result = self._conversation_report(
            persisted, text,
            semantic(scope="complete", planned=[eaten(
                "item_1", text, relation="modified", quantity_text="1 serving",
            )]),
            "single-serving-guid",
        )
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY)[0].official_servings,
                         Decimal("1"))

    def test_complete_item_attestation_stays_in_its_clause(self) -> None:
        for index, (text, resolved_id) in enumerate((
            ("I ate spaghetti as recommended and sweet potato", "item_1"),
            ("I ate the sweet potato like you recommended and spaghetti", "item_2"),
        )):
            with self.subTest(text=text):
                plan = self._two_food_plan()
                output = semantic(scope="complete", planned=[
                    eaten("item_1", "spaghetti"), eaten("item_2", "sweet potato"),
                ])
                persisted = self.state.save_meal_plan(plan, plan_id=f"item-attest-{index}")
                result = self._conversation_report(
                    persisted, text, output, f"item-attest-{index}-guid",
                    chat_guid=f"item-attest-{index}-chat",
                )
                self.assertEqual(result.outcome, "clarification_required")
                facts = {fact.plan_item_id: fact for fact in result.draft.planned_items}
                self.assertEqual(facts[resolved_id].quantity_status, "resolved")
                other_id = "item_2" if resolved_id == "item_1" else "item_1"
                self.assertEqual(facts[other_id].quantity_status, "unresolved")
                self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())
                self.assertEqual(self.catalog._connection.execute(
                    "SELECT count(*) FROM meal_report_applications"
                ).fetchone()[0], 0)

    def test_complete_unplanned_food_cannot_attest_to_only_plan_item(self) -> None:
        plan = self._rendered_plan()
        text = "I ate eggs as recommended"
        output = semantic(scope="complete", planned=[eaten("item_1", text)])
        self._assert_complete_rejected(plan, text, output, "foreign-attestation")

    def test_everything_for_one_item_cannot_expand_to_whole_meal(self) -> None:
        plan = self._two_food_plan()
        text = "I ate everything you recommended for spaghetti, and sweet potato"
        output = semantic(scope="complete", planned=[
            eaten("item_1", "everything you recommended for spaghetti"),
            eaten("item_2", "sweet potato"),
        ])
        persisted = self.state.save_meal_plan(plan, plan_id="item-everything")
        result = self._conversation_report(persisted, text, output, "item-everything-guid")
        self.assertEqual(result.outcome, "clarification_required")
        facts = {fact.plan_item_id: fact for fact in result.draft.planned_items}
        self.assertEqual(facts["item_1"].quantity_status, "resolved")
        self.assertEqual(facts["item_2"].quantity_status, "unresolved")
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())

    def test_complete_explicit_whole_meal_attests_both_items(self) -> None:
        plan = self._two_food_plan()
        text = "I ate everything you recommended"
        output = semantic(scope="complete", planned=[
            eaten("item_1", text), eaten("item_2", text),
        ])
        persisted = self.state.save_meal_plan(plan, plan_id="whole-meal-attest")
        result = self._conversation_report(persisted, text, output, "whole-meal-attest-guid")
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(
            [(entry.plan_item_id, entry.official_servings)
             for entry in self.state.get_daily_intake(serving_fixtures.DAY)],
            [("item_1", Decimal("1")), ("item_2", Decimal("2"))],
        )

    def test_item_attestation_and_other_local_quantity_both_apply(self) -> None:
        plan = self._two_food_plan()
        text = "I ate spaghetti as recommended and 2 quarter pieces sweet potato"
        output = semantic(scope="complete", planned=[
            eaten("item_1", "spaghetti as recommended"),
            eaten("item_2", "2 quarter pieces sweet potato", relation="modified",
                  quantity_text="2 quarter pieces"),
        ])
        persisted = self.state.save_meal_plan(plan, plan_id="local-attest-and-quantity")
        result = self._conversation_report(
            persisted, text, output, "local-attest-and-quantity-guid",
        )
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(
            [(entry.plan_item_id, entry.official_servings)
             for entry in self.state.get_daily_intake(serving_fixtures.DAY)],
            [("item_1", Decimal("1")), ("item_2", Decimal("0.5"))],
        )

    def test_no_food_skip_does_not_cancel_other_item_attestation(self) -> None:
        plan = self._two_food_plan()
        text = "I ate spaghetti as recommended and no sweet potato"
        output = semantic(scope="complete", planned=[
            eaten("item_1", "spaghetti as recommended"),
            skipped("item_2", "no sweet potato"),
        ])
        persisted = self.state.save_meal_plan(plan, plan_id="attest-and-skip")
        result = self._conversation_report(persisted, text, output, "attest-and-skip-guid")
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(
            [(entry.plan_item_id, entry.official_servings)
             for entry in self.state.get_daily_intake(serving_fixtures.DAY)],
            [("item_1", Decimal("1"))],
        )

    def test_complete_literal_correction_overrides_model_selected_first_amount(self) -> None:
        for index, (text, first, expected) in enumerate((
            ("I ate 1 serving of sweet potato, actually 2 servings",
             "1 serving", Decimal("2")),
            ("I ate 2 quarter pieces, actually 3 quarter pieces",
             "2 quarter pieces", Decimal("0.75")),
        )):
            with self.subTest(text=text):
                plan = self._rendered_plan()
                output = semantic(scope="complete", planned=[eaten(
                    "item_1", text, relation="modified", quantity_text=first,
                )])
                persisted = self.state.save_meal_plan(plan, plan_id=f"literal-correction-{index}")
                result = self._conversation_report(
                    persisted, text, output, f"literal-correction-{index}-guid",
                    chat_guid=f"literal-correction-{index}-chat",
                )
                self.assertEqual(result.outcome, "applied")
                self.assertEqual(
                    self.state.get_daily_intake(serving_fixtures.DAY)[-1].official_servings,
                    expected,
                )

    def test_complete_competing_amounts_without_correction_clarify(self) -> None:
        plan = self._rendered_plan()
        text = "I ate 1 serving of sweet potato, 2 servings"
        output = semantic(scope="complete", planned=[eaten(
            "item_1", text, relation="modified", quantity_text="1 serving",
        )])
        self._assert_complete_rejected(plan, text, output, "uncorrected-amounts")

    def test_complete_unparsed_later_correction_cannot_leave_earlier_amount(self) -> None:
        for index, (text, first) in enumerate((
            ("I ate 1 serving of sweet potato, actually three", "1 serving"),
            ("I ate one sweet potato, actually two", "one"),
        )):
            with self.subTest(text=text):
                output = semantic(scope="complete", planned=[eaten(
                    "item_1", text, relation="modified", quantity_text=first,
                )])
                self._assert_complete_rejected(
                    self._rendered_plan(), text, output, f"unparsed-correction-{index}",
                )

    def test_complete_unplanned_correction_overrides_model_selected_first_amount(self) -> None:
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "4 oz spaghetti"),
        ))
        text = "I ate 1 serving of baked sweet potato, actually 2 servings"
        output = report_semantic([], [{
            "food_text": "Baked Sweet Potatoes-Master", "quantity_text": "1 serving",
        }])
        persisted = self.state.save_meal_plan(plan, plan_id="corrected-unplanned")
        result = self._conversation_report(
            persisted, text, output, "corrected-unplanned-guid",
        )
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(
            [(entry.plan_item_id, entry.official_servings)
             for entry in self.state.get_daily_intake(serving_fixtures.DAY)],
            [(None, Decimal("2"))],
        )

    def test_complete_unplanned_competing_or_unparsed_correction_clarifies(self) -> None:
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "4 oz spaghetti"),
        ))
        output = report_semantic([], [{
            "food_text": "Baked Sweet Potatoes-Master", "quantity_text": "1 serving",
        }])
        for index, text in enumerate((
            "I ate 1 serving of baked sweet potato, 2 servings",
            "I ate 1 serving of baked sweet potato, actually three",
        )):
            with self.subTest(text=text):
                self._assert_complete_rejected(
                    plan, text, output, f"unplanned-competing-{index}",
                )

    def test_corrected_unplanned_draft_keeps_final_literal_after_reopen(self) -> None:
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "4 oz spaghetti"),
        ))
        persisted = self.state.save_meal_plan(plan, plan_id="unplanned-draft-literal")
        food = "Baked Sweet Potatoes-Master"
        first = self._conversation_report(
            persisted, "I ate 1 serving of baked sweet potato",
            report_semantic([], [{"food_text": food, "quantity_text": "1 serving"}])
            .model_copy(update={"report_scope": "partial"}),
            "unplanned-draft-first",
        )
        self.assertEqual(first.draft.unplanned_items[0].official_servings, Decimal("1"))
        text = "I ate 1 serving of baked sweet potato, actually 2 servings"
        corrected = self._conversation_report(
            persisted, text,
            report_semantic([], [{
                "food_text": food, "quantity_text": "1 serving", "is_correction": True,
            }]).model_copy(update={"report_scope": "partial"}),
            "unplanned-draft-corrected",
        )
        self.assertEqual(corrected.draft.unplanned_items[0].official_servings, Decimal("2"))
        self.assertEqual(corrected.draft.unplanned_items[0].quantity_text, "2 servings")
        self.assertEqual(corrected.draft.unplanned_items[0].fact_revision,
                         first.draft.unplanned_items[0].fact_revision + 1)
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        reopened = self.state.load_meal_report_draft(corrected.draft.draft_id)
        self.assertEqual(reopened.unplanned_items[0].official_servings, Decimal("2"))
        self.assertEqual(reopened.unplanned_items[0].quantity_text, "2 servings")
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())

    def test_complete_corrected_scoop_needs_a_frozen_binding(self) -> None:
        text = "I ate 1 scoop, no, 2 scoops"
        output = semantic(scope="complete", planned=[eaten(
            "item_1", text, relation="modified", quantity_text="1 scoop",
        )])
        self._assert_complete_rejected(
            MealPlan(serving_fixtures.DAY, 2, (
                PlannedMealItem(self.spaghetti, Decimal("1"), "about 2 scoops"),
            )), text, output, "unbound-corrected-scoop",
        )

    def test_complete_no_comma_correction_uses_frozen_binding(self) -> None:
        plan = self._scoop_plan()
        text = "I ate 1 scoop, no, 2 scoops"
        output = semantic(scope="complete", planned=[eaten(
            "item_1", text, relation="modified", quantity_text="1 scoop",
        )])
        persisted = self.state.save_meal_plan(plan, plan_id="bound-no-correction")
        result = self._conversation_report(persisted, text, output, "bound-no-correction-guid")
        self.assertEqual(result.outcome, "applied")
        self.assertGreater(
            self.state.get_daily_intake(serving_fixtures.DAY)[0].official_servings,
            Decimal("2"),
        )

    def test_complete_attestation_cannot_ignore_later_correction(self) -> None:
        text = "I ate all the sweet potato, actually half"
        output = semantic(scope="complete", planned=[eaten("item_1", text)])
        self._assert_complete_rejected(
            self._rendered_plan(), text, output, "corrected-attestation",
        )

    def test_complete_fraction_cannot_ignore_later_correction(self) -> None:
        text = "I ate half of what you recommended, actually a quarter"
        output = semantic(scope="complete", planned=[eaten(
            "item_1", text, relation="fraction_of_recommended",
            quantity_text="half of what you recommended",
        )])
        self._assert_complete_rejected(
            self._rendered_plan(), text, output, "corrected-fraction",
        )

    def test_corrected_draft_reopens_with_final_literal_and_replays_once(self) -> None:
        plan = self._rendered_plan()
        persisted = self.state.save_meal_plan(plan, plan_id="corrected-draft")
        text = "I ate 2 quarter pieces, actually 3 quarter pieces"
        output = semantic(scope="partial", planned=[eaten(
            "item_1", text, relation="modified", quantity_text="2 quarter pieces",
        )])
        first = self._conversation_report(persisted, text, output, "corrected-draft-first")
        self.assertEqual(first.outcome, "clarification_required")
        self.assertEqual(first.draft.planned_items[0].official_servings, Decimal("0.75"))
        self.assertEqual(first.draft.planned_items[0].original_quantity_text,
                         "3 quarter pieces")
        draft_id = first.draft.draft_id
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        reopened = self.state.load_meal_report_draft(draft_id)
        self.assertEqual(reopened.planned_items[0].official_servings, Decimal("0.75"))
        self.assertEqual(reopened.planned_items[0].original_quantity_text,
                         "3 quarter pieces")
        loaded = self.state.load_meal_plan("corrected-draft")
        finish = "that's all"
        finish_output = semantic(scope="complete", planned=[])
        result = self._conversation_report(
            loaded, finish, finish_output, "corrected-draft-final",
        )
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY)[0].official_servings,
                         Decimal("0.75"))
        replay = self._conversation_report(
            loaded, finish, finish_output, "corrected-draft-final",
        )
        self.assertEqual(replay.outcome, "replayed")
        self.assertEqual(len(self.state.get_daily_intake(serving_fixtures.DAY)), 1)

    def test_calibrated_render_does_not_invoke_semantic_renderer(self) -> None:
        class _AuthorityChangingRenderer:
            def render(self, request):
                del request
                raise AssertionError("calibrated rendering must stay in Python")

        plan = self._rendered_plan(semantic_renderer=_AuthorityChangingRenderer())
        self.assertEqual(
            plan.items[0].presentation_binding.calibration.calibration_version,
            1,
        )

    def test_complete_report_cannot_apply_model_invented_quarter_pieces(self) -> None:
        persisted = self.state.save_meal_plan(self._rendered_plan(), plan_id="invented-plan")
        text = "I ate sweet potato"
        result = self._conversation_report(
            persisted, text,
            semantic(scope="complete", planned=[eaten(
                "item_1", "sweet potato", relation="modified",
                quantity_text="2 quarter pieces",
            )]),
            "invented-guid",
        )
        self.assertEqual(result.outcome, "clarification_required")
        assert result.draft is not None
        self.assertEqual(result.draft.planned_items[0].quantity_status, "unresolved")
        self.assertIsNone(result.draft.planned_items[0].original_quantity_text)
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())
        self.assertEqual(
            self.catalog._connection.execute("SELECT count(*) FROM meal_report_applications").fetchone()[0],
            0,
        )

    def test_changed_amount_and_unit_fail_and_preserve_literal_quantity(self) -> None:
        base = self._rounded_registry().calibrations[0]
        scoop = replace(
            base,
            presentation_unit_singular="scoop",
            presentation_unit_plural="scoops",
            presentation_aliases=("scoop", "scoops"),
        )
        scoop_plan = self._rendered_plan(registry=PresentationCalibrationRegistry((scoop,)))
        altered = self._semantic_report(
            scoop_plan, "I ate 1 scoop",
            semantic(planned=[eaten(
                "item_1", "1 scoop", relation="modified", quantity_text="2 scoops",
            )]),
        )
        self.assertEqual(altered.eaten_items, ())
        self.assertEqual(altered.clarification_items[0].original_user_phrase, "1 scoop")
        normalized = self._semantic_report(
            scoop_plan, "I ate a scoop",
            semantic(planned=[eaten(
                "item_1", "a scoop", relation="modified", quantity_text="1 scoop",
            )]),
        )
        self.assertEqual(
            normalized.eaten_items[0].official_servings,
            Decimal("1.11111111111111111111111111111111111"),
        )

        potato_plan = self._rendered_plan()
        substituted = self._semantic_report(
            potato_plan, "I ate 2 scoops",
            semantic(planned=[eaten(
                "item_1", "2 scoops", relation="modified",
                quantity_text="2 quarter pieces",
            )]),
        )
        self.assertEqual(substituted.eaten_items, ())
        self.assertEqual(substituted.clarification_items[0].original_user_phrase, "2 scoops")

    def test_model_cannot_strip_a_stated_unit_into_an_official_count(self) -> None:
        potato = self._rendered_plan()
        stripped = self._semantic_report(
            potato, "I ate 2 quarter pieces",
            semantic(planned=[eaten(
                "item_1", "2 quarter pieces", relation="modified", quantity_text="2",
            )]),
        )
        self.assertEqual(stripped.eaten_items, ())
        self.assertEqual(stripped.clarification_items[0].reason, "reported_quantity_not_grounded")

        unplanned = self._semantic_report(
            potato, "I ate 2 oz of spaghetti",
            report_semantic([], [{"food_text": "Corn Spaghetti", "quantity_text": "2"}]),
        )
        self.assertIsNone(unplanned.unplanned_items[0].official_servings)
        self.assertEqual(unplanned.clarification_items[0].reason, "unplanned_food_unresolved")

        reassigned = self._semantic_report(
            potato, "I ate 2 oz of spaghetti and sweet potato",
            report_semantic([], [{"food_text": "Baked Sweet Potatoes-Master", "quantity_text": "2 oz"}]),
        )
        self.assertIsNone(reassigned.unplanned_items[0].official_servings)
        self.assertEqual(reassigned.clarification_items[0].reason, "unplanned_food_unresolved")

    def test_number_word_grounding_and_literal_correction(self) -> None:
        plan = self._rendered_plan()
        word = self._semantic_report(
            plan, "I ate two quarter pieces",
            semantic(planned=[eaten(
                "item_1", "two quarter pieces", relation="modified",
                quantity_text="2 quarter pieces",
            )]),
        )
        self.assertEqual(word.eaten_items[0].official_servings, Decimal("0.5"))
        good = self._semantic_report(
            plan, "actually 3 quarter pieces",
            semantic(planned=[{
                **eaten("item_1", "actually 3 quarter pieces", relation="modified", quantity_text="3 quarter pieces"),
                "is_correction": True,
            }]),
        )
        self.assertEqual(good.eaten_items[0].official_servings, Decimal("0.75"))
        changed = self._semantic_report(
            plan, "actually 3 quarter pieces",
            semantic(planned=[{
                **eaten("item_1", "actually 3 quarter pieces", relation="modified", quantity_text="2 quarter pieces"),
                "is_correction": True,
            }]),
        )
        self.assertEqual(changed.eaten_items, ())
        self.assertEqual(changed.clarification_items[0].original_user_phrase, "3 quarter pieces")

    def test_wrong_item_complete_report_creates_no_intake(self) -> None:
        potato = self._rendered_plan().items[0]
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "about 4 oz of corn spaghetti"),
            potato,
        ))
        persisted = self.state.save_meal_plan(plan, plan_id="wrong-item-plan")
        text = "I ate 1 official serving of sweet potato"
        result = self._conversation_report(
            persisted, text,
            semantic(scope="complete", planned=[eaten(
                "item_1", text, relation="modified", quantity_text="1 official serving",
            )]),
            "wrong-item-guid",
        )
        self.assertEqual(result.outcome, "clarification_required")
        assert result.draft is not None
        self.assertEqual(result.draft.planned_items[0].quantity_status, "unresolved")
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())
        quarter = self._semantic_report(
            plan, "I ate 2 quarter pieces of sweet potato",
            semantic(scope="complete", planned=[eaten(
                "item_1", "2 quarter pieces of sweet potato", relation="modified",
                quantity_text="2 quarter pieces",
            )]),
        )
        self.assertEqual(quarter.eaten_items, ())
        self.assertEqual(quarter.clarification_items[0].reason, "reported_quantity_not_grounded")

    def test_repeated_quantity_selects_the_food_in_its_clause(self) -> None:
        potato = self._rendered_plan().items[0]
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "about 4 oz of corn spaghetti"),
            potato,
        ))
        text = "I ate 1 serving spaghetti and 1 serving sweet potato"
        report = self._semantic_report(
            plan, text,
            semantic(planned=[eaten(
                "item_2", "1 serving sweet potato", relation="modified",
                quantity_text="1 serving",
            )]),
        )
        self.assertEqual(report.eaten_items[0].official_servings, Decimal("1"))
        self.assertIs(report.eaten_items[0].plan_item, potato)

    def test_pronoun_needs_unique_durable_quantity_referent(self) -> None:
        potato = self._rendered_plan().items[0]
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "about 4 oz of corn spaghetti"),
            potato,
        ))
        text = "I ate 2 quarter pieces of those"
        output = semantic(planned=[eaten(
            "item_2", text, relation="modified", quantity_text="2 quarter pieces",
        )])
        reconciler = MealReportReconciler(
            ScriptedInterpreter({text: output}),
            LocalFDFoodResolver(self.catalog),
            NaturalPortionInterpreter(),
        )
        ambiguous = reconciler.reconcile_semantic(plan, output, text)
        self.assertEqual(ambiguous.eaten_items, ())
        grounded = reconciler.reconcile_semantic(
            plan, output, text, quantity_reference_item_id="item_2",
        )
        self.assertEqual(grounded.eaten_items[0].official_servings, Decimal("0.5"))

    def test_draft_quantity_reuse_requires_explicit_reference(self) -> None:
        potato = self._rendered_plan().items[0]
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "about 4 oz of corn spaghetti"),
            potato,
        ))
        persisted = self.state.save_meal_plan(plan, plan_id="reuse-plan")
        first = self._conversation_report(
            persisted, "I ate 2 quarter pieces of sweet potato",
            semantic(scope="partial", planned=[eaten(
                "item_2", "2 quarter pieces of sweet potato", relation="modified",
                quantity_text="2 quarter pieces",
            )]),
            "reuse-first",
        )
        assert first.draft is not None
        self.assertEqual(first.draft.planned_items[0].official_servings, Decimal("0.5"))
        invented = self._conversation_report(
            persisted, "I ate spaghetti",
            semantic(scope="partial", planned=[eaten(
                "item_1", "spaghetti", relation="same_as_draft_item",
                comparison_plan_item_id="item_2",
            )]),
            "reuse-invented",
        )
        assert invented.draft is not None
        facts = {item.plan_item_id: item for item in invented.draft.planned_items}
        self.assertIsNone(facts["item_1"].official_servings)
        valid = self._conversation_report(
            persisted, "I ate spaghetti, same amount as before",
            semantic(scope="partial", planned=[eaten(
                "item_1", "same amount as before", relation="same_as_draft_item",
                comparison_plan_item_id="item_2",
            )]),
            "reuse-valid",
        )
        assert valid.draft is not None
        facts = {item.plan_item_id: item for item in valid.draft.planned_items}
        self.assertEqual(facts["item_1"].official_servings, Decimal("0.5"))
        self.assertEqual(facts["item_2"].official_servings, Decimal("0.5"))

    def test_unplanned_fabricated_quantity_remains_unresolved(self) -> None:
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "about 4 oz of corn spaghetti"),
        ))
        text = "I ate sweet potato"
        report = self._semantic_report(
            plan, text,
            report_semantic([], [{
                "food_text": "Baked Sweet Potatoes-Master",
                "quantity_text": "2 servings",
            }]),
        )
        self.assertEqual(report.eaten_items, ())
        self.assertIsNone(report.unplanned_items[0].official_servings)
        self.assertIsNone(report.unplanned_items[0].quantity_text)
        self.assertEqual(report.unplanned_items[0].resolved_food.source_identifier,
                         self.potato.source_identifier)

    def test_complete_report_articles_do_not_turn_a_lot_into_one(self) -> None:
        for index, (quantity, scoop) in enumerate((
            ("1 serving", False), ("1 piece", False), ("1 scoop", True),
            ("1", False),
        )):
            with self.subTest(quantity=quantity):
                text = "I ate a lot of sweet potato"
                output = semantic(scope="complete", planned=[eaten(
                    "item_1", text, relation="modified", quantity_text=quantity,
                )])
                self._assert_complete_rejected(
                    self._scoop_plan() if scoop else self._rendered_plan(),
                    text, output, f"article-{index}",
                )

    def test_complete_countable_article_normalizes_only_with_its_unit(self) -> None:
        plan = self._scoop_plan()
        text = "I ate a scoop of sweet potato"
        result = self._semantic_report(plan, text, semantic(scope="complete", planned=[eaten(
            "item_1", text, relation="modified", quantity_text="1 scoop",
        )]))
        self.assertEqual(result.eaten_items[0].official_servings,
                         Decimal("1.11111111111111111111111111111111111"))
        bare = self._semantic_report(plan, text, semantic(scope="complete", planned=[eaten(
            "item_1", text, relation="modified", quantity_text="1",
        )]))
        self.assertEqual(bare.eaten_items, ())

    def test_complete_report_cannot_strip_amount_modifiers(self) -> None:
        for index, modifier in enumerate((
            "less than", "more than", "almost", "about", "at least", "at most",
        )):
            with self.subTest(modifier=modifier):
                text = f"I ate {modifier} 1 serving of sweet potato"
                output = semantic(scope="complete", planned=[eaten(
                    "item_1", text, relation="modified", quantity_text="1 serving",
                )])
                self._assert_complete_rejected(
                    self._rendered_plan(), text, output, f"modifier-{index}",
                )

    def test_complete_report_cannot_borrow_egg_count_for_potato(self) -> None:
        for index, text in enumerate((
            "I ate 2 eggs and sweet potato",
            "I ate sweet potato and 2 eggs",
        )):
            with self.subTest(text=text):
                output = semantic(scope="complete", planned=[eaten(
                    "item_1", text, relation="modified", quantity_text="2",
                )])
                self._assert_complete_rejected(
                    self._rendered_plan(), text, output, f"egg-locality-{index}",
                )

    def test_bare_egg_count_cannot_select_an_egg_bake(self) -> None:
        spaghetti = component(75146, "Corn Spaghetti")
        spaghetti.update(recipePortionSize="4", recipePortionSizeUnit="Ounce")
        potato = component(62889, "Baked Sweet Potatoes-Master")
        potato.update(recipePortionSize="1", recipePortionSizeUnit="Each")
        bake = component(62891, "Sweet Potato Egg Bake")
        bake.update(recipePortionSize="1", recipePortionSizeUnit="Each")
        egg = component(62892, "Egg")
        egg.update(recipePortionSize="1", recipePortionSizeUnit="Each")
        self.catalog.synchronize_fd_refresh(
            mapped(menu_day(serving_fixtures.DAY, 2, "Lunch", (
                (spaghetti, "Pasta"), (potato, "Homestyle"),
                (bake, "Homestyle"), (egg, "Homestyle"),
            ))),
            requested_start=serving_fixtures.DAY,
            requested_end=serving_fixtures.DAY,
            observed_at=serving_fixtures.OBSERVED_AT,
        )
        resolved = LocalFDFoodResolver(self.catalog).resolve(FoodResolutionRequest(
            "Sweet Potato Egg Bake", serving_fixtures.DAY, meal=2,
        ))
        self.assertIsInstance(resolved, ResolvedFood)
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(resolved, Decimal("1"), "1 egg bake"),
        ))
        text = "I ate 2 eggs"
        self._assert_complete_rejected(
            plan, text, semantic(scope="complete", planned=[eaten(
                "item_1", text, relation="modified", quantity_text="2",
            )]), "egg-bake-wrong-item",
        )
        resolved_egg = LocalFDFoodResolver(self.catalog).resolve(FoodResolutionRequest(
            "Egg", serving_fixtures.DAY, meal=2,
        ))
        self.assertIsInstance(resolved_egg, ResolvedFood)
        egg_plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(resolved_egg, Decimal("1"), "1 egg"),
        ))
        persisted = self.state.save_meal_plan(egg_plan, plan_id="literal-an-egg")
        egg_text = "I ate an egg"
        result = self._conversation_report(
            persisted, egg_text, semantic(scope="complete", planned=[eaten(
                "item_1", egg_text, relation="modified", quantity_text="1",
            )]), "literal-an-egg-guid", chat_guid="literal-an-egg-chat",
        )
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY)[0].official_servings,
                         Decimal("1"))

    def test_complete_report_keeps_two_local_food_amounts_separate(self) -> None:
        potato = self._rendered_plan().items[0]
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "4 oz spaghetti"), potato,
        ))
        text = "I ate 1 official serving of spaghetti and 2 quarter pieces of sweet potato"
        output = semantic(scope="complete", planned=[
            eaten("item_1", "1 official serving of spaghetti", relation="modified",
                  quantity_text="1 official serving"),
            eaten("item_2", "2 quarter pieces of sweet potato", relation="modified",
                  quantity_text="2 quarter pieces"),
        ])
        persisted = self.state.save_meal_plan(plan, plan_id="two-local-foods")
        result = self._conversation_report(persisted, text, output, "two-local-foods-guid")
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(
            [(entry.plan_item_id, entry.official_servings)
             for entry in self.state.get_daily_intake(serving_fixtures.DAY)],
            [("item_1", Decimal("1")), ("item_2", Decimal("0.5"))],
        )

    def test_complete_unplanned_weak_or_wrong_identity_never_applies(self) -> None:
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "4 oz spaghetti"),
        ))
        for index, text in enumerate((
            "I ate 1 serving of master salad",
            "I ate 1 serving of potato",
        )):
            with self.subTest(text=text):
                case_id = f"unplanned-identity-{index}"
                persisted = self.state.save_meal_plan(plan, plan_id=case_id)
                result = self._conversation_report(
                    persisted, text, report_semantic([], [{
                        "food_text": "Baked Sweet Potatoes-Master", "quantity_text": "1 serving",
                    }]), case_id + "-guid", chat_guid=case_id + "-chat",
                )
                self.assertEqual(result.outcome, "clarification_required")
                self.assertEqual(result.draft.unplanned_items[0].food_text,
                                 "master salad" if index == 0 else "potato")
                self.assertIsNone(result.draft.unplanned_items[0].food)
                self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())
                self.assertEqual(self.catalog._connection.execute(
                    "SELECT count(*) FROM meal_report_applications"
                ).fetchone()[0], 0)

    def test_complete_unplanned_ambiguous_food_name_never_applies(self) -> None:
        spaghetti = component(75146, "Corn Spaghetti")
        spaghetti.update(recipePortionSize="4", recipePortionSizeUnit="Ounce")
        potato = component(62889, "Baked Sweet Potatoes-Master")
        potato.update(recipePortionSize="1", recipePortionSizeUnit="Each")
        salad = component(62890, "Sweet Potato Salad")
        salad.update(recipePortionSize="1", recipePortionSizeUnit="Each")
        self.catalog.synchronize_fd_refresh(
            mapped(menu_day(serving_fixtures.DAY, 2, "Lunch", (
                (spaghetti, "Pasta"), (potato, "Homestyle"), (salad, "Salad Bar"),
            ))),
            requested_start=serving_fixtures.DAY,
            requested_end=serving_fixtures.DAY,
            observed_at=serving_fixtures.OBSERVED_AT,
        )
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "4 oz spaghetti"),
        ))
        text = "I ate 1 serving of sweet potato"
        self._assert_complete_rejected(
            plan, text,
            report_semantic([], [{
                "food_text": "Baked Sweet Potatoes-Master",
                "quantity_text": "1 serving",
            }]),
            "unplanned-ambiguous-identity",
        )

    def test_complete_draft_reuse_requires_durable_referent_and_correct_target(self) -> None:
        potato = self._rendered_plan().items[0]
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "4 oz spaghetti"), potato,
        ))
        for index, text in enumerate((
            "I ate spaghetti, same amount as my friend",
            "I ate spaghetti, same amount as John",
            "I ate spaghetti, same amount as yesterday",
            "I ate spaghetti, but sweet potato same amount as before",
        )):
            with self.subTest(text=text):
                case_id = f"reuse-adversarial-{index}"
                persisted = self.state.save_meal_plan(plan, plan_id=case_id)
                first = self._conversation_report(
                    persisted, "I ate 2 quarter pieces of sweet potato",
                    semantic(scope="partial", planned=[eaten(
                        "item_2", "2 quarter pieces of sweet potato",
                        relation="modified", quantity_text="2 quarter pieces",
                    )]),
                    case_id + "-first", chat_guid=case_id + "-chat",
                )
                self.assertEqual(first.draft.planned_items[0].official_servings, Decimal("0.5"))
                second = self._conversation_report(
                    persisted, text,
                    semantic(scope="complete", planned=[eaten(
                        "item_1", text, relation="same_as_draft_item",
                        comparison_plan_item_id="item_2",
                    )]),
                    case_id + "-second", chat_guid=case_id + "-chat",
                )
                self.assertEqual(second.outcome, "clarification_required")
                self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())
                self.assertEqual(self.catalog._connection.execute(
                    "SELECT count(*) FROM meal_report_applications"
                ).fetchone()[0], 0)

    def test_complete_draft_reuse_with_one_source_and_named_target_applies(self) -> None:
        potato = self._rendered_plan().items[0]
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "4 oz spaghetti"), potato,
        ))
        persisted = self.state.save_meal_plan(plan, plan_id="reuse-valid-complete")
        self._conversation_report(
            persisted, "I ate 2 quarter pieces of sweet potato",
            semantic(scope="partial", planned=[eaten(
                "item_2", "2 quarter pieces of sweet potato", relation="modified",
                quantity_text="2 quarter pieces",
            )]), "reuse-valid-first",
        )
        text = "I ate spaghetti, same amount as before"
        result = self._conversation_report(
            persisted, text,
            semantic(scope="complete", planned=[eaten(
                "item_1", text, relation="same_as_draft_item",
                comparison_plan_item_id="item_2",
            )]), "reuse-valid-second",
        )
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(
            [(entry.plan_item_id, entry.official_servings)
             for entry in self.state.get_daily_intake(serving_fixtures.DAY)],
            [("item_1", Decimal("0.5")), ("item_2", Decimal("0.5"))],
        )

    def test_draft_reuse_does_not_hide_later_unaccounted_clause(self) -> None:
        persisted = self.state.save_meal_plan(
            self._two_food_plan(), plan_id="reuse-unaccounted-clause",
        )
        first = "I ate 2 quarter pieces of sweet potato"
        self._conversation_report(
            persisted, first,
            semantic(scope="partial", planned=[eaten(
                "item_2", first, relation="modified", quantity_text="2 quarter pieces",
            )]),
            "reuse-unaccounted-first",
        )
        text = "I ate spaghetti, same amount as before. Actually I finished the rest."
        result = self._conversation_report(
            persisted, text,
            semantic(scope="complete", planned=[eaten(
                "item_1", text, relation="same_as_draft_item",
                comparison_plan_item_id="item_2",
            )]),
            "reuse-unaccounted-second",
        )
        self.assertEqual(result.outcome, "clarification_required")
        facts = {fact.plan_item_id: fact for fact in result.draft.planned_items}
        self.assertEqual(facts["item_1"].quantity_status, "unresolved")
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())
        self.assertEqual(self.catalog._connection.execute(
            "SELECT count(*) FROM meal_report_applications"
        ).fetchone()[0], 0)

    def test_rejected_cross_item_correction_preserves_durable_fact_after_reopen(self) -> None:
        potato = self._rendered_plan().items[0]
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "4 oz spaghetti"), potato,
        ))
        persisted = self.state.save_meal_plan(plan, plan_id="correction-grounding")
        first = self._conversation_report(
            persisted, "I ate 2 quarter pieces of sweet potato",
            semantic(scope="partial", planned=[eaten(
                "item_2", "2 quarter pieces of sweet potato", relation="modified",
                quantity_text="2 quarter pieces",
            )]), "correction-grounding-first",
        )
        prior = first.draft.planned_items[0]
        text = "actually 2 servings of spaghetti"
        rejected = self._conversation_report(
            persisted, text,
            semantic(scope="complete", planned=[{
                **eaten("item_2", text, relation="modified", quantity_text="2 servings"),
                "is_correction": True,
            }]), "correction-grounding-rejected",
        )
        self.assertEqual(rejected.outcome, "clarification_required")
        self.assertEqual(rejected.draft.planned_items[0], prior)
        self.assertIn("reported_quantity_not_grounded",
                      [item.reason for item in rejected.draft.clarifications])
        draft_id = rejected.draft.draft_id
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        reopened = self.state.load_meal_report_draft(draft_id)
        self.assertEqual(reopened.planned_items[0], prior)
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())
        self.assertEqual(self.catalog._connection.execute(
            "SELECT count(*) FROM meal_report_applications"
        ).fetchone()[0], 0)
        loaded_plan = self.state.load_meal_plan("correction-grounding")
        still_pending = self._conversation_report(
            loaded_plan, "that's all", semantic(scope="complete", planned=[]),
            "correction-grounding-premature-complete",
        )
        self.assertEqual(still_pending.outcome, "clarification_required")
        self.assertEqual(still_pending.draft.planned_items[0], prior)
        self.assertIn("reported_quantity_not_grounded",
                      [item.reason for item in still_pending.draft.clarifications])
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())
        valid_text = "actually 3 quarter pieces of sweet potato"
        corrected = self._conversation_report(
            loaded_plan, valid_text,
            semantic(scope="partial", planned=[{
                **eaten("item_2", valid_text, relation="modified",
                        quantity_text="3 quarter pieces"),
                "is_correction": True,
            }]), "correction-grounding-valid",
        )
        self.assertEqual(corrected.draft.planned_items[0].official_servings, Decimal("0.75"))
        self.assertEqual(corrected.draft.planned_items[0].fact_revision, prior.fact_revision + 1)
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())

    def test_quantityless_unplanned_correction_keeps_resolved_fact_pending(self) -> None:
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "4 oz spaghetti"),
        ))
        persisted = self.state.save_meal_plan(plan, plan_id="unplanned-correction")
        food_name = "Baked Sweet Potatoes-Master"
        first_text = "I ate 1 serving of baked sweet potato"
        first = self._conversation_report(
            persisted, first_text,
            report_semantic([], [{"food_text": food_name, "quantity_text": "1 serving"}])
            .model_copy(update={"report_scope": "partial"}),
            "unplanned-correction-first",
        )
        prior = first.draft.unplanned_items[0]
        self.assertEqual(prior.official_servings, Decimal("1"))
        rejected_text = "actually baked sweet potato"
        rejected = self._conversation_report(
            persisted, rejected_text,
            report_semantic([], [{
                "food_text": food_name, "quantity_text": None, "is_correction": True,
            }]), "unplanned-correction-rejected",
        )
        self.assertEqual(rejected.outcome, "clarification_required")
        self.assertEqual(rejected.draft.unplanned_items[0], prior)
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())
        pending = self._conversation_report(
            persisted, "that's all", report_semantic([], []),
            "unplanned-correction-premature-complete",
        )
        self.assertEqual(pending.outcome, "clarification_required")
        self.assertEqual(pending.draft.unplanned_items[0], prior)
        valid_text = "actually 2 servings of baked sweet potato"
        valid = self._conversation_report(
            persisted, valid_text,
            report_semantic([], [{
                "food_text": food_name, "quantity_text": "2 servings",
                "is_correction": True,
            }]).model_copy(update={"report_scope": "partial"}),
            "unplanned-correction-valid",
        )
        self.assertEqual(valid.draft.unplanned_items[0].official_servings, Decimal("2"))
        self.assertEqual(valid.draft.unplanned_items[0].fact_revision, prior.fact_revision + 1)
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())

    def test_unambiguous_comma_apposition_uses_the_named_food(self) -> None:
        potato = self._rendered_plan().items[0]
        plan = MealPlan(serving_fixtures.DAY, 2, (
            PlannedMealItem(self.spaghetti, Decimal("1"), "4 oz spaghetti"), potato,
        ))
        text = "I ate the sweet potato, 2 quarter pieces"
        result = self._semantic_report(plan, text, semantic(planned=[eaten(
            "item_2", text, relation="modified", quantity_text="2 quarter pieces",
        )]))
        self.assertEqual(result.eaten_items[0].official_servings, Decimal("0.5"))

    def test_mixed_bound_and_unbound_report_survives_restart(self) -> None:
        potato_item = self._rendered_plan(official_servings=Decimal("1")).items[0]
        plan = MealPlan(
            serving_fixtures.DAY,
            2,
            (
                PlannedMealItem(
                    self.spaghetti,
                    Decimal("1.75"),
                    "about 2 scoops",
                ),
                potato_item,
            ),
        )
        persisted = self.state.save_meal_plan(plan, plan_id="mixed-plan")
        text = "I ate 1 scoop spaghetti and 2 quarter pieces sweet potato"
        interpreter = ScriptedInterpreter(
            {
                text: semantic(
                    scope="partial",
                    planned=[
                        eaten(
                            "item_1",
                            "1 scoop spaghetti",
                            relation="modified",
                            quantity_text="1 scoop",
                        ),
                        eaten(
                            "item_2",
                            "2 quarter pieces sweet potato",
                            relation="modified",
                            quantity_text="2 quarter pieces",
                        ),
                    ],
                )
            }
        )
        result = MealReportConversationOrchestrator(
            self.state,
            MealReportReconciler(
                interpreter,
                LocalFDFoodResolver(self.catalog),
                NaturalPortionInterpreter(),
            ),
        ).process(
            chat_guid="mixed-chat",
            user_text=text,
            source_event_id="mixed-guid",
            context_resolver=_PlanResolver(persisted),
        )
        assert result.draft is not None
        facts = {item.plan_item_id: item for item in result.draft.planned_items}
        self.assertEqual(facts["item_1"].quantity_status, "unresolved")
        self.assertEqual(facts["item_1"].original_quantity_text, "1 scoop")
        self.assertEqual(facts["item_2"].quantity_status, "resolved")
        self.assertEqual(facts["item_2"].official_servings, Decimal("0.5"))
        self.assertIn("1 scoop", result.message)
        self.assertNotIn("Baked Sweet Potatoes", result.message)
        self.assertEqual(self.state.get_daily_intake(serving_fixtures.DAY), ())

        draft_id = result.draft.draft_id
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        reopened = self.state.load_meal_report_draft(draft_id)
        assert reopened is not None
        reopened_facts = {item.plan_item_id: item for item in reopened.planned_items}
        self.assertEqual(reopened_facts["item_1"].original_quantity_text, "1 scoop")
        self.assertEqual(reopened_facts["item_2"].official_servings, Decimal("0.5"))
        loaded = self.state.load_meal_plan("mixed-plan")
        self.assertEqual(
            loaded.plan.items[1].presentation_binding,
            potato_item.presentation_binding,
        )


if __name__ == "__main__":
    unittest.main()
