"""Focused schema-v11 coverage for granular, plan-scoped report facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nutrition_optimizer.application_clock import NutritionApplicationClock
from nutrition_optimizer.durable_state import DurableMealState
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import FoodResolutionRequest, LocalFDFoodResolver, ResolvedFood
from nutrition_optimizer.meal_report import MealPlan, MealReportReconciler, PlannedMealItem
from nutrition_optimizer.meal_report_conversation import MealReportConversationOrchestrator
from nutrition_optimizer.meal_report_routing import MealReportRoutingSemanticResult
from nutrition_optimizer.meal_report_structured import MealReportSemanticResult
from nutrition_optimizer.messaging.application import ApplicationMessagingConfig
from nutrition_optimizer.messaging.meal_workflow import (
    ActiveMealPlanContextResolver,
    ConversationalMealReportMessageHandler,
)
from nutrition_optimizer.messaging.webhook import IncomingMessage
from nutrition_optimizer.portion_interpretation import NaturalPortionInterpreter
from nutrition_optimizer.phelps_service_calendar import DEFAULT_PHELPS_SERVICE_CALENDAR
from tests.test_food_resolution import component, mapped, menu_day


DAY = date(2026, 9, 14)
OBSERVED_AT = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)
CHAT = "facts-chat"


def fd_food(
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


def route(slot: str | None) -> MealReportRoutingSemanticResult:
    return MealReportRoutingSemanticResult.model_validate(
        {"intent": "report_or_clarification", "explicit_meal_slot": slot}
    )


def planned(
    item_id: str,
    reference: str,
    *,
    relation: str = "as_recommended",
    quantity: str | None = None,
    correction: bool = False,
) -> dict[str, object]:
    return {
        "plan_item_id": item_id,
        "reference_text": reference,
        "action": "eaten",
        "quantity_relation": relation,
        "quantity_text": quantity,
        "comparison_plan_item_id": None,
        "is_correction": correction,
    }


def semantic(
    *,
    items: list[dict[str, object]] | None = None,
    scope: str = "partial",
    intent: str = "meal_report",
) -> MealReportSemanticResult:
    return MealReportSemanticResult.model_validate(
        {
            "intent": intent,
            "report_scope": scope,
            "planned_items": items or [],
            "additional_foods": [],
            "unresolved_statements": [],
        }
    )


@dataclass
class Interpreter:
    routes: dict[str, MealReportRoutingSemanticResult] = field(default_factory=dict)
    reports: dict[str, MealReportSemanticResult] = field(default_factory=dict)

    def interpret_report_routing(self, text: str):
        return self.routes[text]

    def interpret(self, plan: MealPlan, text: str):
        del plan
        return self.reports[text]

    def interpret_with_context(self, plan: MealPlan, text: str, context):
        del context
        return self.interpret(plan, text)


@dataclass
class Sender:
    calls: list[tuple[str, str]] = field(default_factory=list)

    def send_text(self, chat_guid: str, text: str) -> object:
        self.calls.append((chat_guid, text))
        return object()


class GranularMealReportFactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "nutrition.sqlite3"
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self._close)
        self.catalog.synchronize_fd_refresh(
            mapped(
                menu_day(
                    DAY,
                    1,
                    "Breakfast",
                    ((fd_food(101, "Eggs", serving_quantity="1", serving_unit="Each"), "Grill"),),
                ),
                menu_day(
                    DAY,
                    2,
                    "Lunch",
                    (
                        (fd_food(201, "Corn Spaghetti", serving_quantity="7", serving_unit="Ounce"), "Pasta"),
                        (fd_food(202, "Meatball Sub", serving_quantity="1", serving_unit="Each"), "Deli"),
                        (fd_food(203, "Plant Based Gochujang Meatballs", serving_quantity="4", serving_unit="Ounce"), "Pasta"),
                    ),
                ),
            ),
            requested_start=DAY,
            requested_end=DAY,
            observed_at=OBSERVED_AT,
        )
        self.state = DurableMealState(self.catalog)
        self.interpreter = Interpreter()

    def _close(self) -> None:
        if getattr(self, "catalog", None) is not None:
            self.catalog.close()

    def _food(self, name: str, meal: int) -> ResolvedFood:
        result = LocalFDFoodResolver(self.catalog).resolve(
            FoodResolutionRequest(name, DAY, meal=meal)
        )
        assert isinstance(result, ResolvedFood)
        return result

    def _breakfast(self) -> MealPlan:
        return MealPlan(
            DAY,
            1,
            (PlannedMealItem(self._food("Eggs", 1), Decimal("1"), "1 serving"),),
        )

    def _lunch(self) -> MealPlan:
        return MealPlan(
            DAY,
            2,
            (
                PlannedMealItem(self._food("Corn Spaghetti", 2), Decimal("1"), "about 2 scoops"),
                PlannedMealItem(self._food("Meatball Sub", 2), Decimal("1"), "1 sandwich"),
                PlannedMealItem(
                    self._food("Plant Based Gochujang Meatballs", 2),
                    Decimal("2"),
                    "about 8 oz",
                ),
            ),
        )

    def _runtime(self):
        reconciler = MealReportReconciler(
            self.interpreter,
            LocalFDFoodResolver(self.catalog),
            NaturalPortionInterpreter(),
        )
        conversation = MealReportConversationOrchestrator(self.state, reconciler)
        resolver = ActiveMealPlanContextResolver(
            self.state,
            service_date_provider=lambda: DAY,
        )
        return conversation, resolver

    def _process(self, text: str, guid: str):
        conversation, resolver = self._runtime()
        return conversation.process(
            chat_guid=CHAT,
            user_text=text,
            source_event_id=guid,
            context_resolver=resolver,
        )

    def _open_breakfast(self):
        breakfast = self.state.save_meal_plan(
            self._breakfast(), meal_slot="breakfast", plan_id="breakfast-plan"
        )
        text = "for breakfast I ate all of the eggs"
        self.interpreter.routes[text] = route("breakfast")
        self.interpreter.reports[text] = semantic(items=[planned("item_1", "eggs")])
        result = self._process(text, "breakfast-start")
        return breakfast, result.draft

    def _open_field_lunch(self, *, guid: str = "lunch-start"):
        lunch = self.state.save_meal_plan(
            self._lunch(), meal_slot="lunch", plan_id="lunch-plan"
        )
        text = "for lunch I ate 1 scoop of spaghetti, a sub, and 2 meatballs"
        self.interpreter.routes[text] = route("lunch")
        self.interpreter.reports[text] = semantic(
            items=[
                planned("item_1", "1 scoop of spaghetti", relation="modified", quantity="1 scoop"),
                planned("item_2", "a sub", relation="modified", quantity="a"),
                planned(
                    "item_3",
                    "2 plant based meatballs",
                    relation="modified",
                    quantity="2 plant based meatballs",
                ),
            ]
        )
        result = self._process(text, guid)
        return lunch, result.draft, result.message

    def test_field_shape_persists_partial_axes_and_prompts_only_unresolved_quantities(self) -> None:
        _, draft, prompt = self._open_field_lunch()
        assert draft is not None
        facts = {item.plan_item_id: item for item in draft.planned_items}

        self.assertEqual(facts["item_1"].quantity_status, "unresolved")
        self.assertEqual(facts["item_1"].original_quantity_text, "1 scoop")
        self.assertIsNone(facts["item_1"].official_servings)
        self.assertEqual(facts["item_2"].quantity_status, "resolved")
        self.assertEqual(facts["item_2"].official_servings, Decimal("1"))
        self.assertEqual(facts["item_3"].quantity_status, "unresolved")
        self.assertEqual(facts["item_3"].original_quantity_text, "2 meatballs")
        self.assertIn("1 scoop", prompt)
        self.assertIn("2 meatballs", prompt)
        self.assertNotIn("Meatball Sub", prompt)
        self.assertEqual(self.state.get_daily_intake(DAY), ())

    def test_field_partial_survives_reopen_and_unrelated_clarification(self) -> None:
        _, draft, _ = self._open_field_lunch()
        assert draft is not None
        draft_id = draft.draft_id
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        reopened = self.state.load_meal_report_draft(draft_id)
        assert reopened is not None
        spaghetti_before = reopened.planned_items[0]

        text = "for lunch, 8 oz of the plant meatballs"
        self.interpreter.routes[text] = route("lunch")
        self.interpreter.reports[text] = semantic(
            intent="clarification_answer",
            items=[planned("item_3", "8 oz", relation="modified", quantity="8 oz")],
        )
        result = self._process(text, "meatball-answer")
        assert result.draft is not None
        facts = {item.plan_item_id: item for item in result.draft.planned_items}
        self.assertEqual(facts["item_1"], spaghetti_before)
        self.assertEqual(facts["item_3"].quantity_status, "resolved")
        self.assertIn("1 scoop", result.message)
        self.assertNotIn("Plant Based Gochujang Meatballs", result.message)
        self.assertEqual(self.state.get_daily_intake(DAY), ())

    def test_breakfast_and_lunch_drafts_coexist_and_route_independently(self) -> None:
        breakfast, breakfast_draft = self._open_breakfast()
        _, lunch_draft, _ = self._open_field_lunch()
        assert breakfast_draft is not None and lunch_draft is not None
        drafts = self.state.list_active_meal_report_drafts(CHAT)
        self.assertEqual({draft.plan_id for draft in drafts}, {"breakfast-plan", "lunch-plan"})

        text = "for breakfast that's all"
        self.interpreter.routes[text] = route("breakfast")
        self.interpreter.reports[text] = semantic(
            scope="complete", intent="clarification_answer"
        )
        result = self._process(text, "breakfast-finish")
        self.assertEqual(result.persisted_plan.plan_id, breakfast.plan_id)
        self.assertEqual(result.outcome, "applied")
        still_open = self.state.load_active_meal_report_draft(CHAT, plan_id="lunch-plan")
        self.assertEqual(still_open, lunch_draft)

    def test_multiple_open_drafts_require_scope_and_mutate_neither(self) -> None:
        self._open_breakfast()
        self._open_field_lunch()
        before = self.state.list_active_meal_report_drafts(CHAT)
        text = "2 servings"
        self.interpreter.routes[text] = route(None)
        sender = Sender()
        conversation, resolver = self._runtime()
        handler = ConversationalMealReportMessageHandler(
            sender,
            ApplicationMessagingConfig(expected_chat_guid=CHAT),
            conversation,
            resolver,
        )

        self.assertTrue(handler.handle(IncomingMessage("ambiguous", text, None, CHAT, False)))
        self.assertIn("Which meal are you reporting", sender.calls[0][1])
        self.assertEqual(self.state.list_active_meal_report_drafts(CHAT), before)

    def test_duplicate_pre_route_ambiguity_replays_original_slot_set(self) -> None:
        self.state.save_meal_plan(
            self._breakfast(), meal_slot="breakfast", plan_id="ambiguity-breakfast"
        )
        self.state.save_meal_plan(
            self._lunch(), meal_slot="lunch", plan_id="ambiguity-lunch"
        )
        text = "I ate chicken"
        self.interpreter.routes[text] = route(None)
        sender = Sender()
        conversation, resolver = self._runtime()
        handler = ConversationalMealReportMessageHandler(
            sender,
            ApplicationMessagingConfig(expected_chat_guid=CHAT),
            conversation,
            resolver,
        )

        self.assertTrue(handler.handle(IncomingMessage("ambiguous-guid", text, None, CHAT, False)))
        original_reply = sender.calls[-1][1]
        self.assertEqual(original_reply, "Which meal are you reporting: breakfast or lunch?")
        event = self.state.load_meal_report_message_event("ambiguous-guid")
        assert event is not None
        self.assertEqual(event.outcome_type, "ambiguous_meal_context")
        self.assertIsNone(event.plan_id)

        self.state.save_meal_plan(
            self._lunch(), meal_slot="dinner", plan_id="ambiguity-dinner"
        )
        del self.interpreter.routes[text]
        self.assertTrue(handler.handle(IncomingMessage("ambiguous-guid", text, None, CHAT, False)))
        self.assertEqual(sender.calls[-1][1], original_reply)
        self.assertNotIn("dinner", sender.calls[-1][1])
        self.assertEqual(self.state.get_daily_intake(DAY), ())

    def test_duplicate_pre_route_unavailable_slot_replays_after_delivery(self) -> None:
        self.state.save_meal_plan(
            self._breakfast(), meal_slot="breakfast", plan_id="unavailable-breakfast"
        )
        self.state.save_meal_plan(
            self._lunch(), meal_slot="lunch", plan_id="unavailable-lunch"
        )
        text = "For dinner I ate chicken"
        self.interpreter.routes[text] = route("dinner")
        sender = Sender()
        conversation, resolver = self._runtime()
        handler = ConversationalMealReportMessageHandler(
            sender,
            ApplicationMessagingConfig(expected_chat_guid=CHAT),
            conversation,
            resolver,
        )

        self.assertTrue(handler.handle(IncomingMessage("unavailable-guid", text, None, CHAT, False)))
        original_reply = sender.calls[-1][1]
        event = self.state.load_meal_report_message_event("unavailable-guid")
        assert event is not None
        self.assertEqual(event.outcome_type, "unavailable_meal_slot")
        self.assertIsNone(event.plan_id)

        self.state.save_meal_plan(
            self._lunch(), meal_slot="dinner", plan_id="now-available-dinner"
        )
        del self.interpreter.routes[text]
        self.assertTrue(handler.handle(IncomingMessage("unavailable-guid", text, None, CHAT, False)))
        self.assertEqual(sender.calls[-1][1], original_reply)
        self.assertEqual(self.state.get_daily_intake(DAY), ())

    def test_duplicate_no_context_replays_after_a_plan_becomes_reportable(self) -> None:
        text = "I ate chicken"
        self.interpreter.routes[text] = route(None)
        sender = Sender()
        conversation, resolver = self._runtime()
        handler = ConversationalMealReportMessageHandler(
            sender,
            ApplicationMessagingConfig(expected_chat_guid=CHAT),
            conversation,
            resolver,
        )

        self.assertTrue(handler.handle(IncomingMessage("no-context-guid", text, None, CHAT, False)))
        original_reply = sender.calls[-1][1]
        event = self.state.load_meal_report_message_event("no-context-guid")
        assert event is not None
        self.assertEqual(event.outcome_type, "no_reportable_context")
        self.assertIsNone(event.plan_id)

        self.state.save_meal_plan(
            self._breakfast(), meal_slot="breakfast", plan_id="later-breakfast"
        )
        del self.interpreter.routes[text]
        self.assertTrue(handler.handle(IncomingMessage("no-context-guid", text, None, CHAT, False)))
        self.assertEqual(sender.calls[-1][1], original_reply)
        self.assertEqual(self.state.get_daily_intake(DAY), ())

    def test_resolved_fact_survives_omission_and_equivalent_repeat(self) -> None:
        self._open_breakfast()
        before = self.state.load_active_meal_report_draft(CHAT, plan_id="breakfast-plan")
        assert before is not None
        text = "for breakfast, anything else"
        self.interpreter.routes[text] = route("breakfast")
        self.interpreter.reports[text] = semantic(items=[])
        omitted = self._process(text, "omit-eggs").draft
        assert omitted is not None
        self.assertEqual(omitted.planned_items[0], before.planned_items[0])

        text = "for breakfast I ate all of the eggs"
        self.interpreter.routes[text] = route("breakfast")
        self.interpreter.reports[text] = semantic(items=[planned("item_1", "eggs")])
        repeated = self._process(text, "repeat-eggs").draft
        assert repeated is not None
        self.assertEqual(repeated.planned_items[0], before.planned_items[0])
        self.assertEqual(
            [item.reason for item in repeated.clarifications],
            ["completion_required"],
        )

    def test_explicit_resolved_and_unresolved_corrections_replace_effective_quantity(self) -> None:
        self._open_breakfast()
        text = "for breakfast, actually 2 servings"
        self.interpreter.routes[text] = route("breakfast")
        self.interpreter.reports[text] = semantic(
            intent="clarification_answer",
            items=[
                planned(
                    "item_1",
                    "actually 2 servings",
                    relation="modified",
                    quantity="2 servings",
                    correction=True,
                )
            ],
        )
        corrected = self._process(text, "correct-resolved").draft
        assert corrected is not None
        fact = corrected.planned_items[0]
        self.assertEqual(fact.quantity_status, "resolved")
        self.assertEqual(fact.official_servings, Decimal("2"))
        self.assertEqual(fact.fact_revision, 1)

        text = "for breakfast, actually 2 scoops"
        self.interpreter.routes[text] = route("breakfast")
        self.interpreter.reports[text] = semantic(
            intent="clarification_answer",
            items=[
                planned(
                    "item_1",
                    "actually 2 scoops",
                    relation="modified",
                    quantity="2 scoops",
                    correction=True,
                )
            ],
        )
        unresolved = self._process(text, "correct-unresolved").draft
        assert unresolved is not None
        fact = unresolved.planned_items[0]
        self.assertEqual(fact.quantity_status, "unresolved")
        self.assertEqual(fact.original_quantity_text, "2 scoops")
        self.assertIsNone(fact.official_servings)
        self.assertEqual(fact.fact_revision, 2)
        history = self.catalog._connection.execute(
            "SELECT fact_revision, official_servings FROM meal_report_draft_planned_item_history "
            "WHERE draft_id = ? ORDER BY fact_revision",
            (unresolved.draft_id,),
        ).fetchall()
        self.assertEqual([(row[0], row[1]) for row in history], [(0, "1"), (1, "2")])
        self.assertEqual(self.state.get_daily_intake(DAY), ())

    def test_correction_is_plan_scoped_and_survives_reopen(self) -> None:
        breakfast, _ = self._open_breakfast()
        _, lunch_draft, _ = self._open_field_lunch()
        assert lunch_draft is not None
        text = "for breakfast, actually 2 servings"
        self.interpreter.routes[text] = route("breakfast")
        self.interpreter.reports[text] = semantic(
            items=[planned("item_1", "2 servings", relation="modified", quantity="2 servings", correction=True)]
        )
        corrected = self._process(text, "scoped-correction").draft
        assert corrected is not None
        self.assertEqual(corrected.plan_id, breakfast.plan_id)
        self.assertEqual(
            self.state.load_active_meal_report_draft(CHAT, plan_id="lunch-plan"),
            lunch_draft,
        )
        draft_id = corrected.draft_id
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        reopened = self.state.load_meal_report_draft(draft_id)
        assert reopened is not None
        self.assertEqual(reopened.planned_items[0].official_servings, Decimal("2"))
        self.assertEqual(reopened.planned_items[0].fact_revision, 1)

    def test_fully_resolved_draft_commits_atomically_once(self) -> None:
        self._open_breakfast()
        text = "for breakfast that's all"
        self.interpreter.routes[text] = route("breakfast")
        self.interpreter.reports[text] = semantic(scope="complete", intent="clarification_answer")
        applied = self._process(text, "atomic-finish")
        replayed = self._process(text, "atomic-finish")
        self.assertEqual(applied.outcome, "applied")
        self.assertEqual(replayed.outcome, "replayed")
        self.assertEqual(len(self.state.get_daily_intake(DAY)), 1)
        completed = self.state.load_meal_report_draft(applied.draft.draft_id)  # type: ignore[union-attr]
        self.assertEqual(completed.status, "completed")  # type: ignore[union-attr]

    def test_old_plan_draft_is_never_reattached_to_same_slot_replacement(self) -> None:
        _, old_draft = self._open_breakfast()
        assert old_draft is not None
        replacement = self.state.save_meal_plan(
            self._breakfast(), meal_slot="breakfast", plan_id="breakfast-v2"
        )
        text = "for breakfast I ate all of the eggs"
        self.interpreter.routes[text] = route("breakfast")
        self.interpreter.reports[text] = semantic(items=[planned("item_1", "eggs")])
        result = self._process(text, "replacement-report")
        self.assertEqual(result.persisted_plan.plan_id, replacement.plan_id)
        historical = self.state.load_meal_report_draft(old_draft.draft_id)
        self.assertEqual(historical.plan_id, "breakfast-plan")  # type: ignore[union-attr]

    def test_detroit_rollover_keeps_yesterdays_draft_historical(self) -> None:
        before_midnight = datetime(2026, 9, 15, 3, 59, 59, tzinfo=timezone.utc)
        at_midnight = datetime(2026, 9, 15, 4, 0, 0, tzinfo=timezone.utc)
        persisted = self.state.save_scheduled_meal_plan(
            self._breakfast(),
            meal_slot="breakfast",
            plan_id="rollover-breakfast",
            delivery_token="rollover-delivery",
            created_at=OBSERVED_AT,
        )
        dispatch = self.state.load_scheduled_recommendation_dispatch(
            DAY, 1, meal_slot="breakfast"
        )
        assert dispatch is not None
        self.assertTrue(
            self.state.claim_scheduled_recommendation_delivery(
                dispatch, started_at=OBSERVED_AT
            )
        )
        self.state.mark_scheduled_recommendation_delivered(
            dispatch, delivered_at=OBSERVED_AT
        )

        initial_text = "for breakfast I ate all of the eggs"
        self.interpreter.routes[initial_text] = route("breakfast")
        self.interpreter.reports[initial_text] = semantic(
            items=[planned("item_1", "eggs")]
        )
        before_clock = NutritionApplicationClock(lambda: before_midnight)
        reconciler = MealReportReconciler(
            self.interpreter,
            LocalFDFoodResolver(self.catalog),
            NaturalPortionInterpreter(),
        )
        conversation = MealReportConversationOrchestrator(
            self.state,
            reconciler,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=before_clock,
        )
        resolver = ActiveMealPlanContextResolver(
            self.state,
            clock=before_clock,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
        )
        opened = conversation.process(
            chat_guid=CHAT,
            user_text=initial_text,
            source_event_id="rollover-open",
            context_resolver=resolver,
        )
        assert opened.draft is not None
        draft_id = opened.draft.draft_id

        next_text = "2 eggs"
        self.interpreter.routes[next_text] = route(None)
        midnight_clock = NutritionApplicationClock(lambda: at_midnight)
        after_rollover = MealReportConversationOrchestrator(
            self.state,
            reconciler,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=midnight_clock,
        ).process(
            chat_guid=CHAT,
            user_text=next_text,
            source_event_id="rollover-after",
            context_resolver=ActiveMealPlanContextResolver(
                self.state,
                clock=midnight_clock,
                service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            ),
        )

        self.assertEqual(after_rollover.outcome, "draft_cancelled")
        historical = self.state.load_meal_report_draft(draft_id)
        assert historical is not None
        self.assertEqual(historical.status, "cancelled")
        self.assertEqual(historical.plan_id, persisted.plan_id)
        self.assertEqual(self.state.get_daily_intake(DAY), ())

    def test_final_transaction_rejects_draft_when_clock_crosses_midnight(self) -> None:
        before_midnight = datetime(2026, 9, 15, 3, 59, 50, tzinfo=timezone.utc)
        after_midnight = datetime(2026, 9, 15, 4, 0, 5, tzinfo=timezone.utc)
        persisted = self.state.save_scheduled_meal_plan(
            self._breakfast(),
            meal_slot="breakfast",
            plan_id="commit-race-breakfast",
            delivery_token="commit-race-delivery",
            created_at=OBSERVED_AT,
        )
        dispatch = self.state.load_scheduled_recommendation_dispatch(
            DAY, 1, meal_slot="breakfast"
        )
        assert dispatch is not None
        self.state.claim_scheduled_recommendation_delivery(dispatch, started_at=OBSERVED_AT)
        self.state.mark_scheduled_recommendation_delivered(dispatch, delivered_at=OBSERVED_AT)

        initial = "for breakfast I ate all of the eggs"
        self.interpreter.routes[initial] = route("breakfast")
        self.interpreter.reports[initial] = semantic(items=[planned("item_1", "eggs")])
        before_clock = NutritionApplicationClock(lambda: before_midnight)
        reconciler = MealReportReconciler(
            self.interpreter,
            LocalFDFoodResolver(self.catalog),
            NaturalPortionInterpreter(),
        )
        opened = MealReportConversationOrchestrator(
            self.state,
            reconciler,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=before_clock,
        ).process(
            chat_guid=CHAT,
            user_text=initial,
            source_event_id="commit-race-open",
            context_resolver=ActiveMealPlanContextResolver(
                self.state,
                clock=before_clock,
                service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            ),
        )
        assert opened.draft is not None
        draft_id = opened.draft.draft_id

        final = "for breakfast that's all"
        self.interpreter.routes[final] = route(None)
        self.interpreter.reports[final] = semantic(
            scope="complete", intent="clarification_answer"
        )
        instants = iter((before_midnight, before_midnight, after_midnight))
        crossing_clock = NutritionApplicationClock(lambda: next(instants))
        result = MealReportConversationOrchestrator(
            self.state,
            reconciler,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=crossing_clock,
        ).process(
            chat_guid=CHAT,
            user_text=final,
            source_event_id="commit-race-final",
            context_resolver=ActiveMealPlanContextResolver(
                self.state,
                clock=crossing_clock,
                service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            ),
        )

        self.assertEqual(result.outcome, "draft_cancelled")
        self.assertEqual(self.state.get_daily_intake(DAY), ())
        self.assertIsNone(self.state.load_applied_meal_report("commit-race-final"))
        historical = self.state.load_meal_report_draft(draft_id)
        assert historical is not None
        self.assertEqual(historical.status, "cancelled")
        self.assertEqual(historical.plan_id, persisted.plan_id)
        self.assertIsNone(self.state.load_active_meal_report_draft(CHAT))
        event = self.state.load_meal_report_message_event("commit-race-final")
        assert event is not None
        self.assertEqual(event.outcome_type, "draft_cancelled")


if __name__ == "__main__":
    unittest.main()
