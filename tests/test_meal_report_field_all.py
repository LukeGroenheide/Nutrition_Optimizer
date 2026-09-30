"""Regression coverage for plan-relative full amounts with a composite exception."""

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nutrition_optimizer.durable_state import DurableMealState
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import FoodResolutionRequest, LocalFDFoodResolver, ResolvedFood
from nutrition_optimizer.meal_report import MealPlan, MealReportReconciler, PlannedMealItem
from nutrition_optimizer.meal_report_conversation import MealReportConversationOrchestrator
from nutrition_optimizer.meal_report_structured import MEAL_REPORT_SYSTEM_PROMPT, MealReportSemanticResult
from nutrition_optimizer.portion_interpretation import NaturalPortionInterpreter
from nutrition_optimizer.presentation_binding import PresentationKind, freeze_presentation
from tests.test_food_resolution import mapped, menu_day
from tests.test_meal_report_conversation import ScriptedInterpreter, food
from tests.test_meal_report_conversation import skipped


DAY = date(2026, 9, 28)
CHAT = "field-breakfast"


def report(text, items, *, unresolved=(), additional=()):
    return MealReportSemanticResult.model_validate({
        "intent": "meal_report", "report_scope": "complete", "planned_items": items,
        "additional_foods": list(additional), "unresolved_statements": list(unresolved),
    })


def eaten(item_id, reference, *, quantity=None, correction=False):
    return {
        "plan_item_id": item_id, "reference_text": reference, "action": "eaten",
        "quantity_relation": "modified" if quantity else "as_recommended",
        "quantity_text": quantity, "comparison_plan_item_id": None,
        "is_correction": correction,
    }


class BreakfastFieldReportTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.catalog = OfficialNutritionCatalog(Path(directory.name) / "state.sqlite3")
        self.addCleanup(self.catalog.close)
        self.catalog.synchronize_fd_refresh(mapped(menu_day(DAY, 1, "Breakfast", (
            (food(901, "Roasted Red Potatoes", serving_quantity="6", serving_unit="Ounce"), "Grill"),
            (food(902, "Scrambled Eggs", serving_quantity="4", serving_unit="Ounce"), "Grill"),
            (food(903, "Biscuits and Gravy", serving_quantity="10", serving_unit="Ounce"), "Global Bowls"),
        ))), requested_start=DAY, requested_end=DAY,
            observed_at=datetime(2026, 9, 28, 12, tzinfo=timezone.utc))
        self.state = DurableMealState(self.catalog)
        resolver = LocalFDFoodResolver(self.catalog)
        def item(name, amount, display, kind=None):
            resolved = resolver.resolve(FoodResolutionRequest(name, DAY, meal=1))
            self.assertIsInstance(resolved, ResolvedFood)
            binding = None if kind is None else freeze_presentation(resolved, Decimal(amount), display, kind)
            return PlannedMealItem(resolved, Decimal(amount), display, presentation_binding=binding)
        self.plan = MealPlan(DAY, 1, (
            item("Roasted Red Potatoes", "1", "about 6 oz"),
            item("Scrambled Eggs", "2", "about 2 baseball-sized portions", PresentationKind.DESCRIPTIVE_ONLY),
            item("Biscuits and Gravy", ".75", "about 7½ oz total"),
        ))
        self.resolver = resolver

    def reconcile(self, text, items, *, unresolved=(), additional=()):
        return MealReportReconciler(
            ScriptedInterpreter({}), self.resolver, NaturalPortionInterpreter(),
        ).reconcile_semantic(self.plan, report(text, items, unresolved=unresolved,
                                               additional=additional), text)

    def conversation(self, responses):
        reconciler = MealReportReconciler(
            ScriptedInterpreter(responses), self.resolver, NaturalPortionInterpreter(),
        )
        return MealReportConversationOrchestrator(self.state, reconciler)

    def process(self, conversation, text, event):
        class CurrentBreakfast:
            def __init__(self, state):
                self.state = state
            def resolve(self, source_event_id, *, meal_slot=None):
                return self.state.load_active_meal_plan(DAY, 1)
        return conversation.process(
            chat_guid=CHAT, user_text=text, source_event_id=event,
            context_resolver=CurrentBreakfast(self.state),
        )

    def test_named_all_uses_exact_plan_amounts_even_with_descriptive_eggs(self):
        text = "I had all potatoes and all eggs"
        result = self.reconcile(text, [eaten("item_1", "all potatoes"), eaten("item_2", "all eggs")])
        self.assertEqual([(x.plan_item.display_name, x.official_servings, x.quantity_source)
                          for x in result.eaten_items], [
            ("Roasted Red Potatoes", Decimal("1"), "planned_quantity"),
            ("Scrambled Eggs", Decimal("2"), "planned_quantity"),
        ])
        self.assertEqual(result.clarification_items, ())
        eggs = self.reconcile("I had all eggs", [eaten("item_2", "all eggs")])
        self.assertEqual(eggs.eaten_items[0].official_servings, Decimal("2"))
        self.assertEqual(eggs.eaten_items[0].quantity_source, "planned_quantity")

    def test_bare_all_of_it_uses_selected_active_plan(self):
        self.state.save_meal_plan(self.plan)
        result = self.process(self.conversation({}), "I had all of it", "whole-active-plan")
        self.assertEqual(result.outcome, "applied")
        self.assertEqual([entry.official_servings for entry in self.state.get_daily_intake(DAY)],
                         [Decimal("1"), Decimal("2"), Decimal(".75")])

    def test_bare_recommended_amount_uses_selected_active_plan(self):
        self.state.save_meal_plan(self.plan)
        result = self.process(self.conversation({}), "the recommended amount", "recommended-active-plan")
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(len(self.state.get_daily_intake(DAY)), 3)

    def test_one_baseball_portion_stays_unresolved(self):
        text = "I had 1 baseball-sized portion of eggs"
        result = self.reconcile(text, [eaten("item_2", "1 baseball-sized portion of eggs", quantity="1 baseball-sized portion")])
        self.assertEqual(result.eaten_items, ())
        self.assertEqual(result.unspecified_items, self.plan.items)
        self.assertTrue(result.clarification_items)

    def test_partial_biscuits_preserves_both_full_items_in_draft(self):
        text = "I had all potatoes, all eggs, and only 4 oz of biscuits"
        responses = {text: report(text, [
            eaten("item_1", "all potatoes"), eaten("item_2", "all eggs"),
            eaten("item_3", "4 oz of biscuits", quantity="4 oz"),
        ])}
        self.state.save_meal_plan(self.plan)
        result = self.process(self.conversation(responses), text, "partial-biscuits")
        self.assertEqual(result.outcome, "clarification_required")
        facts = {x.plan_item_id: x for x in result.draft.planned_items}
        self.assertEqual({key: facts[key].official_servings for key in ("item_1", "item_2")},
                         {"item_1": Decimal("1"), "item_2": Decimal("2")})
        self.assertEqual(facts["item_3"].quantity_status, "unresolved")
        self.assertIsNone(facts["item_3"].official_servings)
        self.assertNotIn("Which recommended item", result.message)
        self.assertIn("combined menu item", result.message)
        self.assertEqual(self.state.get_daily_intake(DAY), ())

    def test_whole_plan_with_biscuit_exception_and_final_clarification(self):
        initial = "I had everything you recommended. I had 4 oz of biscuits not gravy"
        followup = "Actually I had what you recommended"
        responses = {
            initial: report(initial, [
                eaten("item_1", "everything you recommended"),
                eaten("item_2", "everything you recommended"),
                eaten("item_3", "4 oz of biscuits not gravy", quantity="4 oz"),
            ]),
            followup: report(followup, [eaten("item_3", "what you recommended", correction=True)]),
        }
        self.state.save_meal_plan(self.plan)
        conversation = self.conversation(responses)
        first = self.process(conversation, initial, "whole-exception")
        self.assertEqual(first.outcome, "clarification_required")
        self.assertEqual({x.plan_item_id: x.official_servings for x in first.draft.planned_items
                          if x.quantity_status == "resolved"},
                         {"item_1": Decimal("1"), "item_2": Decimal("2")})
        self.assertEqual(self.state.get_daily_intake(DAY), ())
        self.assertIsNone(self.state.load_shake(DAY, "breakfast_shake"))
        final = self.process(conversation, followup, "whole-final")
        self.assertEqual(final.outcome, "applied")
        self.assertEqual(len(self.state.get_daily_intake(DAY)), 3)
        self.assertEqual(self.catalog._connection.execute(
            "SELECT COUNT(*) FROM meal_report_applications").fetchone()[0], 1)
        self.assertTrue(self.state.claim_shake_reminder(final.persisted_plan.plan_id, CHAT))
        self.assertFalse(self.state.claim_shake_reminder(final.persisted_plan.plan_id, CHAT))

    def test_all_of_it_with_one_biscuit_has_only_component_question(self):
        text = "I had all of it but I only had 1 biscuit and no gravy"
        items = [
            eaten("item_1", "all of it"), eaten("item_2", "all of it"),
            eaten("item_3", "1 biscuit and no gravy", quantity="1 biscuit"),
        ]
        result = self.reconcile(text, items)
        self.assertEqual([x.plan_item for x in result.eaten_items], list(self.plan.items[:2]))
        self.assertEqual(len(result.clarification_items), 1)
        self.assertEqual(result.clarification_items[0].reason, "composite_component_unresolved")
        self.assertEqual(result.clarification_items[0].plan_item, self.plan.items[2])
        self.state.save_meal_plan(self.plan)
        conversation_result = self.process(self.conversation({text: report(text, items)}),
                                           text, "one-biscuit")
        self.assertEqual(conversation_result.outcome, "clarification_required")
        self.assertNotIn("Which recommended item", conversation_result.message)
        self.assertNotIn("How much Roasted", conversation_result.message)
        self.assertNotIn("How much Scrambled", conversation_result.message)
        self.assertIn("is a meal report with an exception", MEAL_REPORT_SYSTEM_PROMPT)

    def test_unresolved_biscuit_statement_still_identifies_plan_item(self):
        text = "I had all potatoes, all eggs, and only 4 oz of biscuits"
        only_biscuit = {"reference_text": "4 oz of biscuits", "reason": "ambiguous_reference"}
        result = self.reconcile(text, [eaten("item_1", "all potatoes"), eaten("item_2", "all eggs")],
            unresolved=(only_biscuit,))
        self.assertEqual(len(result.eaten_items), 2)
        self.assertEqual(result.clarification_items[0].plan_item, self.plan.items[2])
        followup = "That's all"
        self.state.save_meal_plan(self.plan)
        conversation = self.conversation({
            text: report(text, [eaten("item_1", "all potatoes"), eaten("item_2", "all eggs")],
                         unresolved=(only_biscuit,)),
            followup: report(followup, []),
        })
        first = self.process(conversation, text, "unresolved-component")
        second = self.process(conversation, followup, "still-unresolved")
        self.assertEqual(first.outcome, "clarification_required")
        self.assertEqual(second.outcome, "clarification_required")
        self.assertIn("biscuit-only", second.message)
        self.assertNotIn("Which recommended item", second.message)
        self.assertEqual(self.state.get_daily_intake(DAY), ())

    def test_biscuit_component_as_additional_food_keeps_targeted_question(self):
        text = "I had all potatoes, all eggs, and only 4 oz of biscuits"
        result = self.reconcile(text, [eaten("item_1", "all potatoes"), eaten("item_2", "all eggs")],
            additional=({"food_text": "biscuits", "quantity_text": "4 oz", "is_correction": False},))
        self.assertEqual(len(result.eaten_items), 2)
        self.assertEqual(result.unplanned_items, ())
        self.assertEqual(result.clarification_items[0].plan_item, self.plan.items[2])

    def _pending_biscuit_answer(self, answer):
        opening = "I had all potatoes, all eggs, and only 4 oz of biscuits"
        self.state.save_meal_plan(self.plan)
        conversation = self.conversation({opening: report(opening, [
            eaten("item_1", "all potatoes"), eaten("item_2", "all eggs"),
            eaten("item_3", "4 oz of biscuits", quantity="4 oz"),
        ])})
        first = self.process(conversation, opening, "pending-opening")
        self.assertEqual(first.outcome, "clarification_required")
        return self.process(conversation, answer, "pending-answer")

    def test_pending_zero_ounces_resolves_as_skip(self):
        result = self._pending_biscuit_answer("0 oz of biscuits and gravy")
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(len(self.state.get_daily_intake(DAY)), 2)
        self.assertEqual(self.catalog._connection.execute(
            "SELECT COUNT(*) FROM meal_report_applications").fetchone()[0], 1)

    def test_pending_bare_zero_resolves_as_skip(self):
        self.assertEqual(self._pending_biscuit_answer("0").outcome, "applied")

    def test_pending_none_resolves_as_skip(self):
        self.assertEqual(self._pending_biscuit_answer("none").outcome, "applied")

    def test_pending_skipped_it_resolves_as_skip(self):
        self.assertEqual(self._pending_biscuit_answer("I skipped it").outcome, "applied")

    def test_pending_recommended_amount_resolves_from_plan(self):
        result = self._pending_biscuit_answer("the recommended amount")
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(len(self.state.get_daily_intake(DAY)), 3)

    def test_pending_all_of_it_resolves_from_plan(self):
        result = self._pending_biscuit_answer("all of it")
        self.assertEqual(result.outcome, "applied")
        self.assertEqual(len(self.state.get_daily_intake(DAY)), 3)

    def test_everything_except_named_item_keeps_unaffected_amounts(self):
        text = "I ate everything you recommended except the biscuits"
        result = self.reconcile(text, [
            eaten("item_1", "everything you recommended"),
            eaten("item_2", "everything you recommended"),
            skipped("item_3", "the biscuits"),
        ])
        self.assertEqual([item.official_servings for item in result.eaten_items],
                         [Decimal("1"), Decimal("2")])
        self.assertEqual(len(result.skipped_items), 1)
        self.assertEqual(result.clarification_items, ())


if __name__ == "__main__":
    unittest.main()
