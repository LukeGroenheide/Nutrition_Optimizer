"""Offline regression coverage for user-initiated one-meal requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nutrition_optimizer.application import MealRecommendationOrchestrator
from nutrition_optimizer.application_clock import (
    NUTRITION_APPLICATION_TIMEZONE,
    NutritionApplicationClock,
)
from nutrition_optimizer.durable_state import DurableMealState
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import FoodResolutionRequest, LocalFDFoodResolver, ResolvedFood
from nutrition_optimizer.meal_recommendation_replacement import (
    MealRecommendationReplacementDelivery,
    MealRecommendationReplacementPreparer,
)
from nutrition_optimizer.meal_report import MealPlan, MealReportReconciler, PlannedMealItem
from nutrition_optimizer.meal_report_conversation import MealReportConversationOrchestrator
from nutrition_optimizer.meal_report_structured import MealReportSemanticResult
from nutrition_optimizer.meal_request_structured import MealRequestSemanticResult
from nutrition_optimizer.meal_user_request import (
    ImmediateMealRequestDelivery,
    ImmediateMealRequestDeliveryError,
    ImmediateMealRequestPreparer,
    NoActiveMealRequestProcessor,
)
from nutrition_optimizer.messaging.application import ApplicationMessagingConfig
from nutrition_optimizer.messaging.meal_workflow import (
    ActiveMealPlanContextResolver,
    ConversationalMealReportMessageHandler,
)
from nutrition_optimizer.messaging.webhook import IncomingMessage
from nutrition_optimizer.nutrition import DailyLedger, DailyMinimums, DailyTargets
from nutrition_optimizer.portion_interpretation import NaturalPortionInterpreter
from nutrition_optimizer.phelps_service_calendar import DEFAULT_PHELPS_SERVICE_CALENDAR
from nutrition_optimizer.recommendation_scheduler import RecommendationScheduler
from tests.test_food_resolution import component, mapped, menu_day


DAY = date(2026, 9, 14)  # Monday, with the default continuous Phelps window.
CHAT_GUID = "chat-user-request-test"
OBSERVED_AT = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)


def targets() -> DailyTargets:
    return DailyTargets(
        calories_kcal=Decimal("1000"),
        protein_g=Decimal("100"),
        carbohydrates_g=Decimal("100"),
        fat_g=Decimal("50"),
        minimums=DailyMinimums(Decimal("10")),
    )


def food(component_id: int, name: str, *, protein: str = "12") -> dict[str, object]:
    value = component(component_id, name, protein=protein)
    value.update(
        {
            "recipePortionSize": "1",
            "recipePortionSizeUnit": "Each",
            "calories": "100",
            "protein": protein,
            "carbohydrates": "10",
            "fat": "4",
            "dietaryFiber": "2",
            "dietaryFiberUOM": "g",
        }
    )
    return value


def active_request_semantic(
    *,
    mode: str,
    requested_foods: tuple[str, ...] = (),
) -> MealReportSemanticResult:
    return MealReportSemanticResult.model_validate(
        {
            "intent": "meal_request",
            "report_scope": "unknown",
            "location_plan_item_id": None,
            "location_food_text": None,
            "replacement_mode": None,
            "replacement_plan_item_ids": [],
            "meal_request_mode": mode,
            "requested_food_texts": list(requested_foods),
            "requested_meal": "dinner",
            "planned_items": [],
            "additional_foods": [],
            "unresolved_statements": [],
        }
    )


def no_active_request_semantic(
    *,
    mode: str,
    requested_foods: tuple[str, ...] = (),
    meal: str | None = "dinner",
) -> MealRequestSemanticResult:
    return MealRequestSemanticResult.model_validate(
        {
            "intent": "meal_request",
            "request_mode": mode,
            "requested_food_texts": list(requested_foods),
            "requested_meal": meal,
        }
    )


@dataclass
class ActiveRequestInterpreter:
    responses: dict[str, MealReportSemanticResult]

    def interpret(self, plan: MealPlan, user_text: str) -> MealReportSemanticResult:
        del plan
        return self.responses[user_text]

    def interpret_with_context(
        self,
        plan: MealPlan,
        user_text: str,
        conversation_context,
    ) -> MealReportSemanticResult:
        del plan, conversation_context
        return self.responses[user_text]


@dataclass
class NoActiveRequestInterpreter:
    responses: dict[str, MealRequestSemanticResult]
    calls: list[str] = field(default_factory=list)

    def interpret_meal_request(self, user_text: str) -> MealRequestSemanticResult:
        self.calls.append(user_text)
        return self.responses[user_text]


@dataclass
class Sender:
    failures: int = 0
    calls: list[tuple[str, str, str | None]] = field(default_factory=list)

    def send_text(
        self,
        chat_guid: str,
        message: str,
        *,
        temp_guid: str | None = None,
    ) -> object:
        self.calls.append((chat_guid, message, temp_guid))
        if self.failures:
            self.failures -= 1
            raise RuntimeError("known test delivery failure")
        return object()


class UserMealRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "nutrition.sqlite3"
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self.catalog.close)
        self._synchronize_menu()
        self.state = DurableMealState(self.catalog)
        self.resolver = LocalFDFoodResolver(self.catalog)

    def _synchronize_menu(
        self,
        *,
        duplicate_pizza: bool = False,
        include_pizza: bool = True,
    ) -> None:
        dinner: list[tuple[dict[str, object], str]] = [
            (food(2, "Chicken", protein="25"), "Grill"),
            (food(3, "Rice", protein="5"), "Grill"),
            (food(4, "Vegetables", protein="4"), "Greens"),
            (food(5, "Fries", protein="3"), "Grill"),
        ]
        if include_pizza:
            dinner.insert(0, (food(1, "Pizza", protein="8"), "Oven"))
        if duplicate_pizza:
            dinner.append((food(6, "Pizza", protein="9"), "Trattoria"))
        self.catalog.synchronize_fd_refresh(
            mapped(menu_day(DAY, 3, "Dinner", tuple(dinner))),
            requested_start=DAY,
            requested_end=DAY,
            observed_at=OBSERVED_AT,
        )

    def _resolved(self, name: str) -> ResolvedFood:
        result = self.resolver.resolve(FoodResolutionRequest(name, DAY, meal=3))
        assert isinstance(result, ResolvedFood)
        return result

    def _save_active_dinner_plan(self):
        return self.state.save_meal_plan(
            MealPlan(
                DAY,
                3,
                (
                    PlannedMealItem(self._resolved("Chicken"), Decimal("1"), "one chicken portion"),
                    PlannedMealItem(self._resolved("Rice"), Decimal("1"), "one rice serving"),
                    PlannedMealItem(self._resolved("Vegetables"), Decimal("1"), "one vegetable serving"),
                ),
            )
        )

    def _active_conversation(
        self,
        responses: dict[str, MealReportSemanticResult],
    ) -> MealReportConversationOrchestrator:
        return MealReportConversationOrchestrator(
            self.state,
            MealReportReconciler(
                ActiveRequestInterpreter(responses),
                self.resolver,
                NaturalPortionInterpreter(),
            ),
            replacement_preparer=MealRecommendationReplacementPreparer(
                self.state,
                MealRecommendationOrchestrator(self.catalog),
                targets,
            ),
        )

    @staticmethod
    def _clock(hour: int, minute: int = 0) -> NutritionApplicationClock:
        local = datetime.combine(DAY, time(hour, minute), tzinfo=NUTRITION_APPLICATION_TIMEZONE)
        return NutritionApplicationClock(lambda: local.astimezone(timezone.utc))

    def _no_active_processor(
        self,
        clock: NutritionApplicationClock,
        responses: dict[str, MealRequestSemanticResult],
    ) -> tuple[NoActiveMealRequestProcessor, NoActiveRequestInterpreter]:
        interpreter = NoActiveRequestInterpreter(responses)
        orchestrator = MealRecommendationOrchestrator(
            self.catalog,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=clock,
        )
        processor = NoActiveMealRequestProcessor(
            self.state,
            interpreter,
            self.resolver,
            ImmediateMealRequestPreparer(self.state, orchestrator, targets),
            clock=clock,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
        )
        return processor, interpreter

    def _application_and_intake_counts(self) -> tuple[int, int]:
        connection = self.catalog._connection
        assert connection is not None
        return (
            int(connection.execute("SELECT COUNT(*) FROM meal_report_applications").fetchone()[0]),
            int(connection.execute("SELECT COUNT(*) FROM accepted_intake_entries").fetchone()[0]),
        )

    def test_active_targeted_request_resolves_current_food_and_never_records_intake(self) -> None:
        prior = self._save_active_dinner_plan()
        text = "Can I get pizza instead?"
        result = self._active_conversation(
            {text: active_request_semantic(mode="targeted", requested_foods=("Pizza",))}
        ).process(
            chat_guid=CHAT_GUID,
            user_text=text,
            source_event_id="active-pizza-guid",
            context_resolver=ActiveMealPlanContextResolver(
                self.state,
                service_date_provider=lambda: DAY,
            ),
        )

        dispatch = result.replacement_dispatch
        assert dispatch is not None
        self.assertEqual(result.outcome, "meal_request")
        self.assertEqual(dispatch.request_kind, "meal_request")
        self.assertEqual(
            tuple(item.display_name for item in dispatch.persisted_plan.plan.items).count("Pizza"),
            1,
        )
        self.assertEqual(dispatch.requested_foods[0].source_value, self._resolved("Pizza").source_identifier.value)
        self.assertEqual(self.state.load_meal_plan(prior.plan_id).status, "superseded")
        self.assertEqual(self._application_and_intake_counts(), (0, 0))
        self.assertEqual(self.state.get_daily_intake(DAY), ())

    def test_active_request_can_require_multiple_authoritative_foods(self) -> None:
        self._save_active_dinner_plan()
        text = "I want pizza and fries tonight"
        result = self._active_conversation(
            {
                text: active_request_semantic(
                    mode="targeted",
                    requested_foods=("Pizza", "Fries"),
                )
            }
        ).process(
            chat_guid=CHAT_GUID,
            user_text=text,
            source_event_id="active-pizza-fries-guid",
            context_resolver=ActiveMealPlanContextResolver(
                self.state,
                service_date_provider=lambda: DAY,
            ),
        )

        dispatch = result.replacement_dispatch
        assert dispatch is not None
        names = {item.display_name for item in dispatch.persisted_plan.plan.items}
        self.assertTrue({"Pizza", "Fries"} <= names)
        self.assertEqual(
            tuple(food.food_text for food in dispatch.requested_foods),
            ("Pizza", "Fries"),
        )
        self.assertEqual(self._application_and_intake_counts(), (0, 0))

    def test_active_request_does_not_retain_items_removed_from_the_current_menu(self) -> None:
        self._save_active_dinner_plan()
        self.catalog.synchronize_fd_refresh(
            mapped(menu_day(DAY, 3, "Dinner", (
                (food(1, "Pizza", protein="8"), "Oven"),
                (food(4, "Vegetables", protein="4"), "Greens"),
            ))),
            requested_start=DAY, requested_end=DAY, observed_at=OBSERVED_AT,
        )
        text = "I want pizza"
        result = self._active_conversation({
            text: active_request_semantic(mode="targeted", requested_foods=("Pizza",)),
        }).process(
            chat_guid=CHAT_GUID, user_text=text, source_event_id="refreshed-active-guid",
            context_resolver=ActiveMealPlanContextResolver(self.state, service_date_provider=lambda: DAY),
        )
        assert result.replacement_dispatch is not None
        plan = result.replacement_dispatch.persisted_plan.plan
        self.assertEqual({item.display_name for item in plan.items}, {"Pizza", "Vegetables"})
        self.assertEqual(self._application_and_intake_counts(), (0, 0))

    def test_explicit_different_meal_cannot_silently_replace_the_active_context(self) -> None:
        prior = self._save_active_dinner_plan()
        text = "Give me pizza for lunch"
        result = self._active_conversation({
            text: active_request_semantic(mode="targeted", requested_foods=("Pizza",)).model_copy(
                update={"requested_meal": "lunch"}
            ),
        }).process(
            chat_guid=CHAT_GUID, user_text=text, source_event_id="different-context-guid",
            context_resolver=ActiveMealPlanContextResolver(self.state, service_date_provider=lambda: DAY),
        )
        self.assertIsNone(result.replacement_dispatch)
        self.assertIn("active recommendation is for dinner", result.message)
        self.assertEqual(self.state.load_meal_plan(prior.plan_id).status, "active")

    def test_active_whole_meal_request_uses_whole_replacement_without_a_negative_preference(self) -> None:
        prior = self._save_active_dinner_plan()
        text = "Build me a different dinner"
        result = self._active_conversation(
            {text: active_request_semantic(mode="whole_meal")}
        ).process(
            chat_guid=CHAT_GUID,
            user_text=text,
            source_event_id="active-whole-guid",
            context_resolver=ActiveMealPlanContextResolver(
                self.state,
                service_date_provider=lambda: DAY,
            ),
        )

        dispatch = result.replacement_dispatch
        assert dispatch is not None
        self.assertEqual(dispatch.request_kind, "meal_request")
        self.assertTrue(dispatch.whole_meal)
        self.assertEqual(dispatch.rejected_plan_item_ids, ())
        self.assertTrue(
            {item.display_name for item in dispatch.persisted_plan.plan.items}.isdisjoint(
                {"Chicken", "Rice", "Vegetables"}
            )
        )
        self.assertEqual(self.state.load_meal_plan(prior.plan_id).status, "superseded")
        self.assertEqual(self._application_and_intake_counts(), (0, 0))

    def test_wholly_different_meal_can_still_require_a_named_food(self) -> None:
        self._save_active_dinner_plan()
        text = "Build a different dinner with pizza"
        result = self._active_conversation({
            text: active_request_semantic(mode="whole_meal", requested_foods=("Pizza",)),
        }).process(
            chat_guid=CHAT_GUID, user_text=text, source_event_id="whole-positive-guid",
            context_resolver=ActiveMealPlanContextResolver(self.state, service_date_provider=lambda: DAY),
        )
        dispatch = result.replacement_dispatch
        assert dispatch is not None
        self.assertTrue(dispatch.whole_meal)
        self.assertEqual(tuple(item.food_text for item in dispatch.requested_foods), ("Pizza",))
        names = {item.display_name for item in dispatch.persisted_plan.plan.items}
        self.assertIn("Pizza", names)
        self.assertTrue(names.isdisjoint({"Chicken", "Rice", "Vegetables"}))
        self.assertEqual(self._application_and_intake_counts(), (0, 0))

    def test_no_active_ambiguous_and_unavailable_requests_do_not_make_a_plan(self) -> None:
        self._synchronize_menu(duplicate_pizza=True)
        clock = self._clock(15)
        ambiguous_text = "Make dinner with pizza"
        unavailable_text = "Make dinner with sushi"
        processor, _ = self._no_active_processor(
            clock,
            {
                ambiguous_text: no_active_request_semantic(
                    mode="targeted", requested_foods=("Pizza",)
                ),
                unavailable_text: no_active_request_semantic(
                    mode="targeted", requested_foods=("Sushi",)
                ),
            },
        )

        ambiguous = processor.process(
            chat_guid=CHAT_GUID,
            user_text=ambiguous_text,
            source_event_id="ambiguous-pizza-guid",
        )
        unavailable = processor.process(
            chat_guid=CHAT_GUID,
            user_text=unavailable_text,
            source_event_id="unavailable-sushi-guid",
        )
        assert ambiguous is not None
        assert unavailable is not None
        self.assertEqual(ambiguous.outcome, "reply")
        self.assertIn("more than one current menu item", ambiguous.reply_text)
        self.assertIn("Pizza at Oven", ambiguous.reply_text)
        self.assertIn("Pizza at Trattoria", ambiguous.reply_text)
        self.assertEqual(unavailable.outcome, "reply")
        self.assertIn("isn't available", unavailable.reply_text)
        self.assertIsNone(self.state.load_pending_meal_request(DAY, 3, CHAT_GUID))
        self.assertEqual(self._application_and_intake_counts(), (0, 0))
        self.assertEqual(self.state.get_daily_intake(DAY), ())

    def test_upcoming_pending_request_is_durable_idempotent_and_scheduler_honors_it(self) -> None:
        request_text = "I want pizza for dinner"
        clock = self._clock(15)
        processor, interpreter = self._no_active_processor(
            clock,
            {
                request_text: no_active_request_semantic(
                    mode="targeted", requested_foods=("Pizza",)
                )
            },
        )

        first = processor.process(
            chat_guid=CHAT_GUID,
            user_text=request_text,
            source_event_id="pending-pizza-guid",
        )
        second = processor.process(
            chat_guid=CHAT_GUID,
            user_text=request_text,
            source_event_id="pending-pizza-guid",
        )
        assert first is not None
        assert second is not None
        self.assertEqual(first.outcome, "pending")
        self.assertEqual(second.outcome, "pending")
        self.assertEqual(interpreter.calls, [request_text])
        pending = self.state.load_pending_meal_request(DAY, 3, CHAT_GUID)
        assert pending is not None
        self.assertEqual(pending.source_event_id, "pending-pizza-guid")
        self.assertEqual(tuple(food.food_text for food in pending.requested_foods), ("Pizza",))
        self.assertIsNone(self.state.load_pending_meal_request(DAY, 2, CHAT_GUID))
        self.assertIsNone(
            self.state.load_pending_meal_request(DAY + timedelta(days=1), 3, CHAT_GUID)
        )
        self.assertEqual(self._application_and_intake_counts(), (0, 0))
        self.assertEqual(self.state.get_daily_intake(DAY), ())

        # Reopen the database before dispatch to cover process-restart durability.
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self.catalog.close)
        self.state = DurableMealState(self.catalog)
        self.resolver = LocalFDFoodResolver(self.catalog)
        restored = self.state.load_pending_meal_request_for_source_event("pending-pizza-guid")
        assert restored is not None
        self.assertEqual(restored.status, "pending")

        dispatch_clock = self._clock(17)
        sender = Sender()
        scheduler = RecommendationScheduler(
            self.state,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=dispatch_clock,
            recommendation_preparer=MealRecommendationOrchestrator(
                self.catalog,
                service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
                clock=dispatch_clock,
            ),
            target_loader=targets,
            outbound_sender=sender,
            chat_guid=CHAT_GUID,
        )
        run = scheduler.run_once()
        dinner = next(item for item in run.decisions if item.opportunity.meal_id == 3)
        self.assertEqual(dinner.outcome, "dispatched")
        scheduled = self.state.load_scheduled_recommendation_dispatch(DAY, 3)
        assert scheduled is not None
        self.assertIn("Pizza", tuple(item.display_name for item in scheduled.persisted_plan.plan.items))
        consumed = self.state.load_pending_meal_request_for_source_event("pending-pizza-guid")
        assert consumed is not None
        self.assertEqual(consumed.status, "consumed")
        self.assertEqual(consumed.consumed_plan_id, scheduled.plan_id)
        self.assertEqual(len(sender.calls), 1)
        self.assertEqual(self._application_and_intake_counts(), (0, 0))
        self.assertEqual(self.state.get_daily_intake(DAY), ())

        # The original GUID remains a request replay after scheduler consumption.
        replay_processor, replay_interpreter = self._no_active_processor(
            dispatch_clock,
            {request_text: no_active_request_semantic(mode="targeted", requested_foods=("Pizza",))},
        )
        replayed = replay_processor.process(
            chat_guid=CHAT_GUID,
            user_text=request_text,
            source_event_id="pending-pizza-guid",
        )
        assert replayed is not None
        self.assertEqual(replayed.outcome, "pending")
        self.assertEqual(replay_interpreter.calls, [])
        self.assertEqual(
            int(self.catalog._connection.execute("SELECT COUNT(*) FROM meal_plans").fetchone()[0]),
            1,
        )

    def test_scheduler_never_silently_drops_a_pending_food_removed_by_refresh(self) -> None:
        request_text = "I want pizza for dinner"
        processor, _ = self._no_active_processor(
            self._clock(15),
            {
                request_text: no_active_request_semantic(
                    mode="targeted", requested_foods=("Pizza",)
                )
            },
        )
        result = processor.process(
            chat_guid=CHAT_GUID,
            user_text=request_text,
            source_event_id="pending-removed-pizza-guid",
        )
        assert result is not None
        self.assertEqual(result.outcome, "pending")

        # A later local refresh is authoritative. The scheduler must not match
        # the old request to another name or drop the positive constraint.
        self._synchronize_menu(include_pizza=False)
        dispatch_clock = self._clock(17)
        sender = Sender()
        run = RecommendationScheduler(
            self.state,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=dispatch_clock,
            recommendation_preparer=MealRecommendationOrchestrator(
                self.catalog,
                service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
                clock=dispatch_clock,
            ),
            target_loader=targets,
            outbound_sender=sender,
            chat_guid=CHAT_GUID,
        ).run_once()

        dinner = next(item for item in run.decisions if item.opportunity.meal_id == 3)
        self.assertEqual(dinner.outcome, "unavailable")
        self.assertEqual(dinner.diagnostic_reason, "requested_food_unavailable")
        self.assertIsNone(self.state.load_scheduled_recommendation_dispatch(DAY, 3))
        pending = self.state.load_pending_meal_request_for_source_event(
            "pending-removed-pizza-guid"
        )
        assert pending is not None
        self.assertEqual(pending.status, "pending")
        self.assertEqual(sender.calls, [])
        self.assertEqual(self._application_and_intake_counts(), (0, 0))

    def test_current_eligible_request_creates_one_immediate_plan_and_delivery_is_idempotent(self) -> None:
        request_text = "Make dinner with pizza"
        clock = self._clock(17)
        processor, interpreter = self._no_active_processor(
            clock,
            {
                request_text: no_active_request_semantic(
                    mode="targeted", requested_foods=("Pizza",)
                )
            },
        )
        result = processor.process(
            chat_guid=CHAT_GUID,
            user_text=request_text,
            source_event_id="immediate-pizza-guid",
        )
        assert result is not None
        dispatch = result.immediate_dispatch
        assert dispatch is not None
        self.assertEqual(result.outcome, "immediate")
        self.assertEqual(dispatch.status, "pending_delivery")
        self.assertIn("Pizza", tuple(item.display_name for item in dispatch.persisted_plan.plan.items))
        self.assertFalse(self.state.is_active_meal_plan_reportable(
            dispatch.persisted_plan, DEFAULT_PHELPS_SERVICE_CALENDAR, evaluated_at=clock.now(),
        ))
        self.assertEqual(self._application_and_intake_counts(), (0, 0))
        self.assertEqual(self.state.get_daily_intake(DAY), ())

        sender = Sender()
        delivery = ImmediateMealRequestDelivery(self.state, sender)
        delivered = delivery.deliver(dispatch)
        self.assertEqual(delivered.status, "delivered")
        self.assertTrue(self.state.is_active_meal_plan_reportable(
            delivered.persisted_plan, DEFAULT_PHELPS_SERVICE_CALENDAR, evaluated_at=clock.now(),
        ))
        self.assertEqual(len(sender.calls), 1)
        replayed = processor.process(
            chat_guid=CHAT_GUID,
            user_text=request_text,
            source_event_id="immediate-pizza-guid",
        )
        assert replayed is not None
        assert replayed.immediate_dispatch is not None
        self.assertEqual(replayed.immediate_dispatch.status, "delivered")
        delivery.deliver(replayed.immediate_dispatch)
        self.assertEqual(len(sender.calls), 1)
        self.assertEqual(interpreter.calls, [request_text])

    def test_unambiguous_current_window_can_make_an_immediate_general_recommendation(self) -> None:
        request_text = "What should I eat?"
        clock = self._clock(17)
        processor, _ = self._no_active_processor(
            clock,
            {
                request_text: no_active_request_semantic(
                    mode="whole_meal",
                    requested_foods=(),
                    meal=None,
                )
            },
        )

        result = processor.process(
            chat_guid=CHAT_GUID,
            user_text=request_text,
            source_event_id="immediate-general-guid",
        )
        assert result is not None
        assert result.immediate_dispatch is not None
        self.assertEqual(result.outcome, "immediate")
        self.assertEqual(result.immediate_dispatch.persisted_plan.plan.meal, 3)
        self.assertTrue(result.immediate_dispatch.persisted_plan.plan.items)
        self.assertEqual(result.immediate_dispatch.requested_foods, ())
        self.assertEqual(self._application_and_intake_counts(), (0, 0))

    def test_immediate_known_send_failure_retries_the_same_plan_and_delivery_token(self) -> None:
        request_text = "Make dinner with pizza"
        processor, _ = self._no_active_processor(
            self._clock(17),
            {
                request_text: no_active_request_semantic(
                    mode="targeted", requested_foods=("Pizza",)
                )
            },
        )
        result = processor.process(
            chat_guid=CHAT_GUID,
            user_text=request_text,
            source_event_id="immediate-retry-guid",
        )
        assert result is not None
        assert result.immediate_dispatch is not None
        sender = Sender(failures=1)
        delivery = ImmediateMealRequestDelivery(
            self.state,
            sender,
            is_known_delivery_failure=lambda exc: isinstance(exc, RuntimeError),
        )

        with self.assertRaises(ImmediateMealRequestDeliveryError):
            delivery.deliver(result.immediate_dispatch)
        pending = self.state.load_immediate_meal_request_dispatch("immediate-retry-guid")
        assert pending is not None
        self.assertEqual(pending.status, "pending_delivery")
        delivered = delivery.deliver(pending)
        self.assertEqual(delivered.status, "delivered")
        delivery.deliver(delivered)
        self.assertEqual(len(sender.calls), 2)
        self.assertEqual(sender.calls[0][2], sender.calls[1][2])
        self.assertEqual(sender.calls[0][2], delivered.delivery_token)
        self.assertEqual(
            int(self.catalog._connection.execute("SELECT COUNT(*) FROM meal_plans").fetchone()[0]),
            1,
        )

    def test_immediate_recommendation_can_be_confirmed_through_normal_meal_reporting(self) -> None:
        processor, _ = self._no_active_processor(self._clock(17), {
            "Dinner with pizza": no_active_request_semantic(mode="targeted", requested_foods=("Pizza",)),
        })
        result = processor.process(
            chat_guid=CHAT_GUID, user_text="Dinner with pizza", source_event_id="request-then-report-guid",
        )
        assert result is not None and result.immediate_dispatch is not None
        dispatch = ImmediateMealRequestDelivery(self.state, Sender()).deliver(result.immediate_dispatch)
        self.assertEqual(self._application_and_intake_counts(), (0, 0))
        plan = dispatch.persisted_plan.plan
        semantic = MealReportSemanticResult.model_validate({
            "intent": "meal_report", "report_scope": "complete",
            "planned_items": [{
                "plan_item_id": plan.item_id(item), "reference_text": item.display_name,
                "action": "eaten", "quantity_relation": "as_recommended", "quantity_text": None,
            } for item in plan.items],
            "additional_foods": [], "unresolved_statements": [],
        })
        report = self._active_conversation({"I ate everything you recommended": semantic}).process(
            chat_guid=CHAT_GUID, user_text="I ate everything you recommended", source_event_id="confirmed-request-report-guid",
            context_resolver=ActiveMealPlanContextResolver(self.state, service_date_provider=lambda: DAY),
        )
        self.assertEqual(report.outcome, "applied")
        self.assertEqual(self._application_and_intake_counts(), (1, len(plan.items)))
        self.assertEqual(self.state.load_meal_plan(dispatch.plan_id).status, "applied")

    def test_inbound_no_active_route_prepares_and_sends_current_eligible_request_once(self) -> None:
        request_text = "Give me dinner with pizza"
        clock = self._clock(17)
        processor, interpreter = self._no_active_processor(
            clock,
            {
                request_text: no_active_request_semantic(
                    mode="targeted", requested_foods=("Pizza",)
                )
            },
        )
        sender = Sender()
        conversation = MealReportConversationOrchestrator(
            self.state,
            MealReportReconciler(
                ActiveRequestInterpreter({}),
                self.resolver,
                NaturalPortionInterpreter(),
            ),
        )
        handler = ConversationalMealReportMessageHandler(
            sender,
            ApplicationMessagingConfig(expected_chat_guid=CHAT_GUID),
            conversation,
            ActiveMealPlanContextResolver(
                self.state,
                clock=clock,
                service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            ),
            no_active_meal_request_processor=processor,
            immediate_meal_request_delivery=ImmediateMealRequestDelivery(self.state, sender),
        )
        message = IncomingMessage("inbound-pizza-guid", request_text, "sender", CHAT_GUID, False)

        self.assertTrue(handler.handle(message))
        self.assertTrue(handler.handle(message))
        dispatch = self.state.load_immediate_meal_request_dispatch("inbound-pizza-guid")
        assert dispatch is not None
        self.assertEqual(dispatch.status, "delivered")
        self.assertEqual(len(sender.calls), 1)
        self.assertEqual(interpreter.calls, [request_text])
        self.assertEqual(self._application_and_intake_counts(), (0, 0))

    def test_delivered_immediate_replacement_survives_a_different_scheduled_slot(self) -> None:
        self.catalog.synchronize_fd_refresh(
            mapped(
                menu_day(DAY, 2, "Lunch", (
                    (food(1, "Pizza", protein="8"), "Oven"),
                    (food(5, "Fries", protein="3"), "Grill"),
                )),
                menu_day(DAY, 3, "Dinner", ((food(2, "Chicken", protein="25"), "Grill"),)),
            ),
            requested_start=DAY, requested_end=DAY, observed_at=OBSERVED_AT,
        )
        lunch_clock = self._clock(12)
        processor, _ = self._no_active_processor(lunch_clock, {
            "Give me lunch": no_active_request_semantic(mode="whole_meal", meal="lunch"),
        })
        initial = processor.process(
            chat_guid=CHAT_GUID, user_text="Give me lunch", source_event_id="immediate-lunch-guid",
        )
        assert initial is not None and initial.immediate_dispatch is not None
        sender = Sender()
        ImmediateMealRequestDelivery(self.state, sender).deliver(initial.immediate_dispatch)
        pizza = self.resolver.resolve(FoodResolutionRequest("Pizza", DAY, meal=2))
        assert isinstance(pizza, ResolvedFood)
        replacement = MealRecommendationReplacementPreparer(
            self.state, MealRecommendationOrchestrator(self.catalog), targets,
        ).prepare_requested(
            prior_plan_id=initial.immediate_dispatch.plan_id, requested_foods=(pizza,),
            whole_meal=False, source_event_id="immediate-lunch-replacement-guid", chat_guid=CHAT_GUID,
        )
        delivery = MealRecommendationReplacementDelivery(self.state, sender)
        delivered = delivery.deliver(replacement)
        self.assertTrue(self.state.is_active_meal_plan_reportable(
            delivered.persisted_plan, DEFAULT_PHELPS_SERVICE_CALENDAR, evaluated_at=lunch_clock.now(),
        ))
        self.assertTrue(self.state.is_active_meal_plan_reportable(
            delivered.persisted_plan, DEFAULT_PHELPS_SERVICE_CALENDAR, evaluated_at=self._clock(23).now(),
        ))
        dinner_clock = self._clock(17)
        run = RecommendationScheduler(
            self.state, clock=dinner_clock, service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            recommendation_preparer=MealRecommendationOrchestrator(
                self.catalog, clock=dinner_clock, service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            ),
            target_loader=targets, outbound_sender=sender, chat_guid=CHAT_GUID,
        ).run_once()
        self.assertEqual(next(x for x in run.decisions if x.opportunity.meal_id == 3).outcome, "dispatched")
        self.assertEqual(self.state.load_meal_plan(replacement.plan_id).status, "active")
        self.assertTrue(self.state.is_active_meal_plan_reportable(
            delivered.persisted_plan,
            DEFAULT_PHELPS_SERVICE_CALENDAR,
            evaluated_at=dinner_clock.now(),
        ))
        self.assertEqual(
            {plan.meal_slot for plan in self.state.list_reportable_active_meal_plans(
                DAY,
                DEFAULT_PHELPS_SERVICE_CALENDAR,
                evaluated_at=dinner_clock.now(),
            )},
            {"lunch", "dinner"},
        )
        self.assertEqual(self._application_and_intake_counts(), (0, 0))

    def test_new_immediate_request_supersedes_an_older_pending_request_in_its_scope(self) -> None:
        processor, _ = self._no_active_processor(self._clock(15), {
            "Dinner with pizza": no_active_request_semantic(mode="targeted", requested_foods=("Pizza",)),
        })
        processor.process(chat_guid=CHAT_GUID, user_text="Dinner with pizza", source_event_id="older-pending-guid")
        processor, _ = self._no_active_processor(self._clock(17), {
            "Dinner with chicken": no_active_request_semantic(mode="targeted", requested_foods=("Chicken",)),
        })
        result = processor.process(chat_guid=CHAT_GUID, user_text="Dinner with chicken", source_event_id="new-immediate-guid")
        assert result is not None and result.immediate_dispatch is not None
        self.assertIsNone(self.state.load_pending_meal_request(DAY, 3, CHAT_GUID))
        self.assertEqual(self.state.load_pending_meal_request_for_source_event("older-pending-guid").status, "superseded")
        self.assertEqual(self._application_and_intake_counts(), (0, 0))


if __name__ == "__main__":
    unittest.main()
