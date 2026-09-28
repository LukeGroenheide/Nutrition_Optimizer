"""Focused coverage for pre-reconciliation product-slot report routing."""

from __future__ import annotations

from dataclasses import dataclass, field
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
from nutrition_optimizer.meal_report_routing import MealReportRoutingSemanticResult
from nutrition_optimizer.meal_report_structured import MealReportSemanticResult
from nutrition_optimizer.messaging.application import ApplicationMessagingConfig
from nutrition_optimizer.messaging.meal_workflow import (
    ActiveMealPlanContextResolver,
    AmbiguousMealPlanContextError,
    ConversationalMealReportMessageHandler,
)
from nutrition_optimizer.messaging.webhook import IncomingMessage
from nutrition_optimizer.portion_interpretation import NaturalPortionInterpreter
from tests.test_food_resolution import component, mapped, menu_day


DAY = date(2026, 8, 30)
OBSERVED_AT = datetime(2026, 8, 30, 12, tzinfo=timezone.utc)
CHAT_GUID = "meal-routing-chat"


def route(slot: str | None, *, report: bool = True) -> MealReportRoutingSemanticResult:
    return MealReportRoutingSemanticResult.model_validate(
        {
            "intent": "report_or_clarification" if report else "other_or_unclear",
            "explicit_meal_slot": slot,
        }
    )


def eaten_semantic(*, scope: str = "complete") -> MealReportSemanticResult:
    return MealReportSemanticResult.model_validate(
        {
            "report_scope": scope,
            "planned_items": [
                {
                    "plan_item_id": "item_1",
                    "reference_text": "the food",
                    "action": "eaten",
                    "quantity_relation": "as_recommended",
                    "quantity_text": None,
                    "comparison_plan_item_id": None,
                }
            ],
            "additional_foods": [],
            "unresolved_statements": [],
        }
    )


def unresolved_semantic() -> MealReportSemanticResult:
    return MealReportSemanticResult.model_validate(
        {
            "report_scope": "partial",
            "planned_items": [],
            "additional_foods": [],
            "unresolved_statements": [
                {
                    "reference_text": "some food",
                    "reason": "ambiguous_reference",
                }
            ],
        }
    )


@dataclass
class RoutingInterpreter:
    routes: dict[str, MealReportRoutingSemanticResult]
    reports: dict[str, MealReportSemanticResult]
    routing_calls: list[str] = field(default_factory=list)
    report_calls: list[tuple[MealPlan, str]] = field(default_factory=list)

    def interpret_report_routing(self, user_text: str) -> MealReportRoutingSemanticResult:
        self.routing_calls.append(user_text)
        return self.routes[user_text]

    def interpret(self, plan: MealPlan, user_text: str) -> MealReportSemanticResult:
        self.report_calls.append((plan, user_text))
        return self.reports[user_text]

    def interpret_with_context(
        self,
        plan: MealPlan,
        user_text: str,
        conversation_context,
    ) -> MealReportSemanticResult:
        del conversation_context
        return self.interpret(plan, user_text)


@dataclass
class RecordingSender:
    calls: list[tuple[str, str]] = field(default_factory=list)

    def send_text(self, chat_guid: str, text: str) -> object:
        self.calls.append((chat_guid, text))
        return object()


class MealReportRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog = OfficialNutritionCatalog(
            Path(self.directory.name) / "nutrition.sqlite3"
        )
        self.addCleanup(self.catalog.close)
        self.catalog.synchronize_fd_refresh(
            mapped(
                menu_day(DAY, 1, "Breakfast", ((component(101, "Chicken"), "Grill"),)),
                menu_day(DAY, 2, "Lunch", ((component(201, "Chicken"), "Grill"),)),
                menu_day(DAY, 3, "Dinner", ((component(301, "Chicken"), "Grill"),)),
            ),
            requested_start=DAY,
            requested_end=DAY,
            observed_at=OBSERVED_AT,
        )
        self.state = DurableMealState(self.catalog)
        self.resolver = ActiveMealPlanContextResolver(
            self.state,
            service_date_provider=lambda: DAY,
        )

    def _plan(self, slot: str) -> MealPlan:
        provider_meal = {"breakfast": 1, "lunch": 2, "brunch": 2, "dinner": 3}[slot]
        resolved = LocalFDFoodResolver(self.catalog).resolve(
            FoodResolutionRequest("Chicken", DAY, meal=provider_meal)
        )
        assert isinstance(resolved, ResolvedFood)
        return MealPlan(
            DAY,
            provider_meal,
            (PlannedMealItem(resolved, Decimal("1"), "one serving"),),
        )

    def _save(self, *slots: str):
        return {
            slot: self.state.save_meal_plan(
                self._plan(slot),
                meal_slot=slot,
                plan_id=f"{slot}-plan",
            )
            for slot in slots
        }

    def _conversation(self, interpreter: RoutingInterpreter):
        reconciler = MealReportReconciler(
            interpreter,
            LocalFDFoodResolver(self.catalog),
            NaturalPortionInterpreter(),
        )
        return MealReportConversationOrchestrator(self.state, reconciler)

    def _process(
        self,
        conversation: MealReportConversationOrchestrator,
        text: str,
        guid: str,
    ):
        return conversation.process(
            chat_guid=CHAT_GUID,
            user_text=text,
            source_event_id=guid,
            context_resolver=self.resolver,
        )

    def _handler(self, interpreter: RoutingInterpreter, sender: RecordingSender):
        return ConversationalMealReportMessageHandler(
            sender,
            ApplicationMessagingConfig(expected_chat_guid=CHAT_GUID),
            self._conversation(interpreter),
            self.resolver,
        )

    def _message(self, text: str, guid: str) -> IncomingMessage:
        return IncomingMessage(guid, text, None, CHAT_GUID, False)

    def _application_count(self) -> int:
        return int(
            self.catalog._connection.execute(
                "SELECT COUNT(*) FROM meal_report_applications"
            ).fetchone()[0]
        )

    def _assert_explicit_slot_routes_before_food_interpretation(self, slot: str) -> None:
        plans = self._save("breakfast", "lunch", "dinner")
        text = f"for {slot} I ate all of the chicken"
        interpreter = RoutingInterpreter(
            {text: route(slot)},
            {text: eaten_semantic()},
        )

        result = self._process(
            self._conversation(interpreter), text, f"explicit-{slot}"
        )

        self.assertEqual(result.persisted_plan.plan_id, plans[slot].plan_id)
        self.assertEqual(interpreter.report_calls[0][0], plans[slot].plan)
        for other_slot, plan in plans.items():
            expected = "applied" if other_slot == slot else "active"
            self.assertEqual(self.state.load_meal_plan(plan.plan_id).status, expected)  # type: ignore[union-attr]

    def test_explicit_breakfast_selects_before_food_interpretation(self) -> None:
        self._assert_explicit_slot_routes_before_food_interpretation("breakfast")

    def test_explicit_lunch_selects_before_food_interpretation(self) -> None:
        self._assert_explicit_slot_routes_before_food_interpretation("lunch")

    def test_explicit_dinner_selects_before_food_interpretation(self) -> None:
        self._assert_explicit_slot_routes_before_food_interpretation("dinner")

    def test_lunch_and_brunch_with_the_same_provider_period_route_by_product_slot(self) -> None:
        plans = self._save("lunch", "brunch")
        self.assertEqual(plans["lunch"].plan.meal, plans["brunch"].plan.meal)
        with self.assertRaises(AmbiguousMealPlanContextError):
            self.resolver.resolve("fd-period-alone")

        for slot in ("lunch", "brunch"):
            selected = self.resolver.resolve(f"explicit-{slot}", meal_slot=slot)
            self.assertEqual(selected.plan_id, plans[slot].plan_id)

        text = "for brunch I ate all of the chicken"
        interpreter = RoutingInterpreter({text: route("brunch")}, {text: eaten_semantic()})
        result = self._process(self._conversation(interpreter), text, "brunch-report")
        self.assertEqual(result.persisted_plan.plan_id, plans["brunch"].plan_id)
        self.assertEqual(self.state.load_meal_plan(plans["lunch"].plan_id).status, "active")  # type: ignore[union-attr]

    def test_unscoped_multiple_contexts_ask_from_real_slots_without_mutation(self) -> None:
        self._save("breakfast", "lunch")
        text = "I ate all of the chicken"
        interpreter = RoutingInterpreter({text: route(None)}, {text: eaten_semantic()})
        sender = RecordingSender()

        self.assertTrue(
            self._handler(interpreter, sender).handle(
                self._message(text, "ambiguous-unscoped")
            )
        )

        self.assertEqual(sender.calls, [(CHAT_GUID, "Which meal are you reporting: breakfast or lunch?")])
        self.assertEqual(interpreter.report_calls, [])
        self.assertIsNone(self.state.load_active_meal_report_draft(CHAT_GUID))
        self.assertEqual(self._application_count(), 0)

    def test_one_unscoped_context_retains_convenient_reporting(self) -> None:
        plans = self._save("lunch")
        text = "I ate all of the chicken"
        interpreter = RoutingInterpreter({text: route(None)}, {text: eaten_semantic()})

        result = self._process(self._conversation(interpreter), text, "one-unscoped")

        self.assertEqual(result.persisted_plan.plan_id, plans["lunch"].plan_id)
        self.assertEqual(result.outcome, "applied")

    def _open_breakfast_draft(self, interpreter: RoutingInterpreter):
        plans = self._save("breakfast")
        text = "for breakfast I ate some chicken"
        interpreter.routes[text] = route("breakfast")
        interpreter.reports[text] = unresolved_semantic()
        result = self._process(self._conversation(interpreter), text, "draft-start")
        self.assertEqual(result.outcome, "clarification_required")
        return plans["breakfast"], result.draft

    def test_unscoped_clarification_continues_exact_draft_after_lunch_arrives(self) -> None:
        interpreter = RoutingInterpreter({}, {})
        breakfast, _ = self._open_breakfast_draft(interpreter)
        lunch = self._save("lunch")["lunch"]
        text = "2 eggs"
        interpreter.routes[text] = route(None)
        interpreter.reports[text] = eaten_semantic(scope="partial")

        result = self._process(self._conversation(interpreter), text, "draft-answer")

        self.assertEqual(result.persisted_plan.plan_id, breakfast.plan_id)
        self.assertEqual(result.outcome, "clarification_required")
        self.assertEqual(self.state.load_meal_plan(lunch.plan_id).status, "active")  # type: ignore[union-attr]

    def test_explicit_same_slot_continues_exact_draft(self) -> None:
        interpreter = RoutingInterpreter({}, {})
        breakfast, _ = self._open_breakfast_draft(interpreter)
        self._save("lunch")
        text = "for breakfast, 2 eggs"
        interpreter.routes[text] = route("breakfast")
        interpreter.reports[text] = eaten_semantic(scope="partial")

        result = self._process(self._conversation(interpreter), text, "same-slot-answer")

        self.assertEqual(result.persisted_plan.plan_id, breakfast.plan_id)
        self.assertEqual(result.outcome, "clarification_required")

    def test_explicit_different_slot_uses_an_independent_context(self) -> None:
        interpreter = RoutingInterpreter({}, {})
        _, opened = self._open_breakfast_draft(interpreter)
        self._save("lunch")
        assert opened is not None
        before = self.state.load_meal_report_draft(opened.draft_id)
        report_call_count = len(interpreter.report_calls)
        text = "for lunch I ate all of the chicken"
        interpreter.routes[text] = route("lunch")
        interpreter.reports[text] = eaten_semantic()

        result = self._process(self._conversation(interpreter), text, "different-slot")

        self.assertEqual(result.outcome, "applied")
        self.assertEqual(result.persisted_plan.meal_slot, "lunch")
        self.assertEqual(self.state.load_meal_report_draft(opened.draft_id), before)
        self.assertEqual(len(interpreter.report_calls), report_call_count + 1)
        self.assertEqual(self._application_count(), 1)

    def test_explicit_unavailable_slot_does_not_fall_back(self) -> None:
        self._save("breakfast", "lunch")
        text = "for dinner I ate chicken"
        interpreter = RoutingInterpreter({text: route("dinner")}, {text: eaten_semantic()})
        sender = RecordingSender()

        self.assertTrue(
            self._handler(interpreter, sender).handle(
                self._message(text, "unavailable-dinner")
            )
        )

        self.assertEqual(
            sender.calls,
            [(CHAT_GUID, "I don't have a reportable dinner recommendation for today. I haven't logged anything.")],
        )
        self.assertEqual(interpreter.report_calls, [])
        self.assertIsNone(self.state.load_active_meal_report_draft(CHAT_GUID))
        self.assertEqual(self._application_count(), 0)

    def test_positive_meal_request_is_not_report_scope(self) -> None:
        plans = self._save("lunch")
        text = "I want chicken for lunch"
        semantic = MealReportSemanticResult.model_validate(
            {
                "intent": "meal_request",
                "report_scope": "unknown",
                "meal_request_mode": "targeted",
                "requested_food_texts": ["chicken"],
                "requested_meal": "lunch",
                "planned_items": [],
                "additional_foods": [],
                "unresolved_statements": [],
            }
        )
        interpreter = RoutingInterpreter(
            {text: route(None, report=False)},
            {text: semantic},
        )

        result = self._process(self._conversation(interpreter), text, "positive-request")

        self.assertEqual(result.outcome, "meal_request")
        self.assertEqual(result.persisted_plan.plan_id, plans["lunch"].plan_id)
        self.assertIsNone(self.state.load_active_meal_report_draft(CHAT_GUID))
        self.assertEqual(self._application_count(), 0)


if __name__ == "__main__":
    unittest.main()
