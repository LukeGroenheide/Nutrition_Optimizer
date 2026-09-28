"""BlueBubbles adapter coverage for the persisted meal workflow."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
from http.client import HTTPConnection
import json
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from threading import Barrier, Event, Lock, Thread, get_ident
import unittest
from unittest.mock import patch

from nutrition_optimizer.application_clock import NutritionApplicationClock
from nutrition_optimizer.application import (
    MealRecommendationOrchestrator,
    MealReportOrchestrator,
    ProductionNutritionConfigurationError,
    load_production_daily_targets,
)
from nutrition_optimizer.durable_state import DurableMealState
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import (
    FoodResolutionRequest,
    FoodSemanticDecision,
    LocalFDFoodResolver,
    ResolvedFood,
)
from nutrition_optimizer.meal_report import MealPlan, MealReportReconciler, PlannedMealItem
from nutrition_optimizer.meal_report_structured import MealReportSemanticResult
from nutrition_optimizer.messaging.application import ApplicationMessagingConfig
from nutrition_optimizer.messaging.meal_workflow import (
    AMBIGUOUS_ACTIVE_MEAL_PLAN_REPLY_TEXT,
    NO_ACTIVE_MEAL_PLAN_REPLY_TEXT,
    ActiveMealPlanContextResolver,
    ManualMealRecommendationDelivery,
    MessageScopedMealReportMessageHandler,
    MealPlanDeliveryError,
    MealReportMessageHandler,
    MealWorkflowMessageError,
    open_production_meal_report_runtime,
    send_manual_recommendation,
)
from nutrition_optimizer.messaging.polling import BlueBubblesPollingWorker, RowIDCursorStore
from nutrition_optimizer.messaging.webhook import (
    IncomingMessage,
    RecentMessageGuids,
    create_webhook_server,
    deliver_incoming_message,
)
from nutrition_optimizer.nutrition import DailyMinimums, DailyTargets
from nutrition_optimizer.portion_interpretation import NaturalPortionInterpreter
from nutrition_optimizer.recommendation_rendering import format_meal_message
from tests.test_food_resolution import component, mapped, menu_day


DAY = date(2026, 8, 25)
OBSERVED_AT = datetime(2026, 8, 25, 12, tzinfo=timezone.utc)
PRODUCTION_CATALOG_PATH = Path(__file__).resolve().parents[1] / "data/state/nutrition.sqlite3"


def target_values() -> DailyTargets:
    return DailyTargets(
        calories_kcal=Decimal("3200"),
        protein_g=Decimal("125"),
        carbohydrates_g=Decimal("450"),
        fat_g=Decimal("100"),
        minimums=DailyMinimums(Decimal("38")),
    )


def production_target_environment() -> dict[str, str]:
    return {
        "NUTRITION_OPTIMIZER_DAILY_CALORIES_KCAL": "3200",
        "NUTRITION_OPTIMIZER_DAILY_PROTEIN_G": "125",
        "NUTRITION_OPTIMIZER_DAILY_CARBOHYDRATES_G": "450",
        "NUTRITION_OPTIMIZER_DAILY_FAT_G": "100",
        "NUTRITION_OPTIMIZER_DAILY_DIETARY_FIBER_MINIMUM_G": "38",
    }


def food(
    component_id: int,
    name: str,
    *,
    calories: str = "180",
    protein: str = "18",
    carbohydrates: str = "20",
    fat: str = "6",
    fiber: str = "3",
    serving_quantity: str = "1",
    serving_unit: str = "Each",
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


def semantic(
    planned: list[dict[str, object]] | None = None,
    *,
    unresolved: list[dict[str, object]] | None = None,
) -> MealReportSemanticResult:
    planned_items = [
        (
            item
            if "quantity_relation" in item
            else {
                **item,
                "quantity_relation": (
                    None
                    if item.get("action") == "skipped"
                    else "as_recommended"
                    if item.get("quantity_text") is None
                    else "modified"
                ),
            }
        )
        for item in (planned or [])
    ]
    return MealReportSemanticResult.model_validate(
        {
            "planned_items": planned_items,
            "additional_foods": [],
            "unresolved_statements": unresolved or [],
        }
    )


@dataclass
class FakeMealReportSemanticInterpreter:
    result: object
    calls: list[tuple[MealPlan, str]] = field(default_factory=list)
    availability_checks: int = 0

    def interpret(self, plan: MealPlan, user_text: str) -> object:
        self.calls.append((plan, user_text))
        return self.result

    def ensure_executable_available(self) -> None:
        self.availability_checks += 1


class FakeFoodSemanticMatcher:
    def __init__(self) -> None:
        self.availability_checks = 0
        self.calls: list[object] = []

    def ensure_executable_available(self) -> None:
        self.availability_checks += 1

    def decide(self, request, candidates):
        self.calls.append((request, candidates))
        return FoodSemanticDecision("no_match", None, "test-only no match")


class AllPlannedItemsEatenInterpreter:
    """A deterministic fake for the isolated real-menu integration path."""

    def __init__(self) -> None:
        self.calls: list[tuple[MealPlan, str]] = []

    def interpret(self, plan: MealPlan, user_text: str) -> MealReportSemanticResult:
        self.calls.append((plan, user_text))
        return semantic(
            [
                {
                    "plan_item_id": plan.item_id(item),
                    "reference_text": item.display_name,
                    "action": "eaten",
                    "quantity_text": None,
                }
                for item in plan.items
            ]
        )


class FakeOutboundSender:
    def __init__(self, *, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.calls: list[tuple[str, str]] = []

    def send_text(self, chat_guid: str, message: str) -> object:
        self.calls.append((chat_guid, message))
        if self.failure is not None:
            raise self.failure
        return object()


class ThreadSafeOutboundSender:
    """Record test-only replies safely when dispatch uses several threads."""

    def __init__(self, *, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.calls: list[tuple[str, str]] = []
        self.sent = Event()
        self._lock = Lock()

    def send_text(self, chat_guid: str, message: str) -> object:
        with self._lock:
            self.calls.append((chat_guid, message))
            failure = self.failure
            self.sent.set()
        if failure is not None:
            raise failure
        return object()


class TrackingCatalog(OfficialNutritionCatalog):
    """Expose only test lifecycle facts for a temporary catalog connection."""

    def __init__(self, path: str | Path) -> None:
        self.created_thread_id = get_ident()
        self.closed_thread_id: int | None = None
        self.close_calls = 0
        super().__init__(path)

    def close(self) -> None:
        self.close_calls += 1
        self.closed_thread_id = get_ident()
        super().close()


class TrackingCatalogFactory:
    """Create temporary tracking catalogs without retaining their connections."""

    def __init__(self) -> None:
        self.catalogs: list[TrackingCatalog] = []
        self._lock = Lock()

    def __call__(self, path: str | Path) -> TrackingCatalog:
        catalog = TrackingCatalog(path)
        with self._lock:
            self.catalogs.append(catalog)
        return catalog


class CoordinatedMealReportSemanticInterpreter:
    """Hold two HTTP callbacks together long enough to prove distinct threads."""

    def __init__(self, result: MealReportSemanticResult) -> None:
        self.result = result
        self.calls: list[tuple[MealPlan, str]] = []
        self._barrier = Barrier(2)
        self._lock = Lock()

    def interpret(self, plan: MealPlan, user_text: str) -> MealReportSemanticResult:
        with self._lock:
            self.calls.append((plan, user_text))
        self._barrier.wait(timeout=5)
        return self.result


class FakeQueryClient:
    def __init__(self, *payloads: object) -> None:
        self.payloads = deque(payloads)

    def query_messages(self, after_rowid: int, *, limit: int, sort: str) -> object:
        del after_rowid, limit, sort
        return self.payloads.popleft()


class CountingRecommendationPreparer:
    def __init__(self, delegate: MealRecommendationOrchestrator) -> None:
        self.delegate = delegate
        self.calls: list[tuple[date, str | int, DailyTargets]] = []

    def prepare(
        self,
        service_date: date,
        meal: str | int,
        targets: DailyTargets,
    ):
        self.calls.append((service_date, meal, targets))
        return self.delegate.prepare(service_date, meal, targets)


def incoming_message(
    *,
    message_guid: str = "bluebubbles-guid-1",
    text: str | None = "I ate everything except the potatoes.",
    is_from_me: bool = False,
    chat_guid: str | None = "expected-chat",
    sender_address: str | None = "student@example.test",
) -> IncomingMessage:
    return IncomingMessage(
        message_guid=message_guid,
        text=text,
        sender_address=sender_address,
        chat_guid=chat_guid,
        is_from_me=is_from_me,
    )


def query_payload(rowid: int, guid: str, text: str) -> dict[str, object]:
    return {
        "data": [
            {
                "originalROWID": rowid,
                "guid": guid,
                "text": text,
                "isFromMe": False,
                "handle": {"address": "student@example.test"},
                "chats": [{"guid": "expected-chat"}],
            }
        ]
    }


class MealWorkflowTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog = OfficialNutritionCatalog(Path(self.directory.name) / "nutrition.sqlite3")
        self.addCleanup(self.catalog.close)
        self.catalog.synchronize_fd_refresh(
            mapped(
                menu_day(
                    DAY,
                    1,
                    "Breakfast",
                    (
                        (food(1, "Eggs"), "Grill"),
                        (food(2, "Potatoes"), "Grill"),
                        (food(3, "Broccoli"), "Vegetables"),
                    ),
                ),
                menu_day(
                    DAY,
                    2,
                    "Lunch",
                    (
                        (food(4, "Chicken", calories="240", protein="28"), "Grill"),
                        (food(5, "Rice", carbohydrates="42", serving_unit="Cup"), "Sides"),
                        (food(6, "Salad"), "Vegetables"),
                    ),
                ),
            ),
            requested_start=DAY,
            requested_end=DAY,
            observed_at=OBSERVED_AT,
        )
        self.state = DurableMealState(self.catalog)
        self.config = ApplicationMessagingConfig(
            expected_chat_guid="expected-chat",
            expected_sender_address="student@example.test",
        )

    def resolved(self, name: str, meal: int = 1) -> ResolvedFood:
        result = LocalFDFoodResolver(self.catalog).resolve(
            FoodResolutionRequest(name, DAY, meal=meal)
        )
        assert isinstance(result, ResolvedFood)
        return result

    def breakfast_plan(self) -> MealPlan:
        return MealPlan(
            DAY,
            1,
            (
                PlannedMealItem(self.resolved("Eggs"), Decimal("1"), "two eggs"),
                PlannedMealItem(self.resolved("Potatoes"), Decimal("1"), "some potatoes"),
                PlannedMealItem(self.resolved("Broccoli"), Decimal("1"), "some broccoli"),
            ),
        )

    def lunch_plan(self) -> MealPlan:
        return MealPlan(
            DAY,
            2,
            (PlannedMealItem(self.resolved("Chicken", 2), Decimal("1"), "one chicken"),),
        )

    def report_handler(
        self,
        result: object,
        sender: FakeOutboundSender,
    ) -> tuple[MealReportMessageHandler, FakeMealReportSemanticInterpreter]:
        interpreter = FakeMealReportSemanticInterpreter(result)
        reconciler = MealReportReconciler(
            interpreter,
            LocalFDFoodResolver(self.catalog),
            NaturalPortionInterpreter(),
        )
        handler = MealReportMessageHandler(
            sender,
            self.config,
            MealReportOrchestrator(self.catalog, reconciler),
            ActiveMealPlanContextResolver(self.state, service_date_provider=lambda: DAY),
        )
        return handler, interpreter

    def application_count(self) -> int:
        return int(
            self.catalog._connection.execute(
                "SELECT COUNT(*) FROM meal_report_applications"
            ).fetchone()[0]
        )

    def test_manual_preparation_sends_exact_application_message_and_keeps_active_plan(self) -> None:
        sender = FakeOutboundSender()
        preparer = CountingRecommendationPreparer(MealRecommendationOrchestrator(self.catalog))
        delivery = ManualMealRecommendationDelivery(
            sender,
            "expected-chat",
            preparer,
            self.state,
        )

        prepared = send_manual_recommendation(
            delivery,
            DAY,
            "breakfast",
            target_loader=target_values,
        )

        self.assertEqual(preparer.calls, [(DAY, "breakfast", target_values())])
        self.assertEqual(sender.calls, [("expected-chat", prepared.message)])
        self.assertEqual(
            self.state.load_active_meal_plan(DAY, "Breakfast"),
            prepared.persisted_plan,
        )

    def test_send_failure_never_prepares_again_and_explicit_resend_uses_same_active_plan(self) -> None:
        sender = FakeOutboundSender(failure=RuntimeError("transport unavailable"))
        preparer = CountingRecommendationPreparer(MealRecommendationOrchestrator(self.catalog))
        delivery = ManualMealRecommendationDelivery(
            sender,
            "expected-chat",
            preparer,
            self.state,
        )

        with self.assertRaises(MealPlanDeliveryError) as raised:
            send_manual_recommendation(
                delivery,
                DAY,
                "breakfast",
                target_loader=target_values,
            )

        plan_id = raised.exception.plan_id
        persisted = self.state.load_meal_plan(plan_id)
        assert persisted is not None
        self.assertEqual(persisted.status, "active")
        self.assertEqual(len(preparer.calls), 1)
        self.assertEqual(len(sender.calls), 1)

        sender.failure = None
        resent = delivery.resend_active_plan(plan_id)

        self.assertEqual(resent, persisted)
        self.assertEqual(len(preparer.calls), 1)
        self.assertEqual(
            sender.calls[-1],
            ("expected-chat", format_meal_message(persisted.plan)),
        )
        self.assertEqual(sender.calls[0], sender.calls[-1])
        self.assertEqual(
            int(self.catalog._connection.execute("SELECT COUNT(*) FROM meal_plans").fetchone()[0]),
            1,
        )

    def test_manual_preparation_uses_the_existing_strict_target_loader(self) -> None:
        sender = FakeOutboundSender()
        preparer = CountingRecommendationPreparer(MealRecommendationOrchestrator(self.catalog))
        delivery = ManualMealRecommendationDelivery(
            sender,
            "expected-chat",
            preparer,
            self.state,
        )

        with self.assertRaises(ProductionNutritionConfigurationError):
            send_manual_recommendation(
                delivery,
                DAY,
                "breakfast",
                target_loader=lambda: load_production_daily_targets({}),
            )

        self.assertEqual(preparer.calls, [])
        self.assertEqual(sender.calls, [])

    def test_happy_path_uses_guid_as_source_event_id_and_advances_cursor_after_reply(self) -> None:
        plan = self.state.save_meal_plan(self.breakfast_plan(), plan_id="breakfast-active")
        sender = FakeOutboundSender()
        handler, interpreter = self.report_handler(
            semantic(
                [
                    {
                        "plan_item_id": "item_1",
                        "reference_text": "everything",
                        "action": "eaten",
                        "quantity_text": None,
                    },
                    {
                        "plan_item_id": "item_2",
                        "reference_text": "the potatoes",
                        "action": "skipped",
                        "quantity_text": None,
                    },
                    {
                        "plan_item_id": "item_3",
                        "reference_text": "everything",
                        "action": "eaten",
                        "quantity_text": None,
                    },
                ]
            ),
            sender,
        )
        payload = query_payload(42, "bluebubbles-guid-1", "I ate everything except the potatoes.")

        with TemporaryDirectory() as cursor_directory:
            cursor_store = RowIDCursorStore(Path(cursor_directory) / "cursor.json")
            cursor_store.initialize(41)
            recent = RecentMessageGuids()
            worker = BlueBubblesPollingWorker(
                FakeQueryClient(payload),
                handler,
                recent_message_guids=recent,
                cursor_store=cursor_store,
            )
            self.assertEqual(worker.poll_once(), 1)
            self.assertEqual(worker.cursor, 42)
            self.assertEqual(len(recent), 1)
            self.assertFalse(
                deliver_incoming_message(
                    incoming_message(),
                    handler,
                    recent,
                )
            )

        self.assertEqual(interpreter.calls, [(plan.plan, "I ate everything except the potatoes.")])
        self.assertEqual(self.application_count(), 1)
        intake = self.state.get_daily_intake(DAY)
        self.assertEqual([entry.source_event_id for entry in intake], ["bluebubbles-guid-1"] * 2)
        self.assertEqual([entry.record.name for entry in intake], ["Eggs", "Broccoli"])
        self.assertIsNone(self.state.load_active_meal_plan(DAY, "breakfast"))
        self.assertEqual(
            sender.calls,
            [
                (
                    "expected-chat",
                    "I understood that as:\n- Eggs\n- Broccoli\n\nSkipped:\n- Potatoes",
                )
            ],
        )

    def test_durable_guid_replay_is_safe_after_the_recent_guid_cache_is_lost(self) -> None:
        self.state.save_meal_plan(self.breakfast_plan())
        sender = FakeOutboundSender()
        handler, interpreter = self.report_handler(
            semantic(
                [
                    {
                        "plan_item_id": "item_1",
                        "reference_text": "eggs",
                        "action": "eaten",
                        "quantity_text": None,
                    }
                ]
            ),
            sender,
        )
        message = incoming_message()

        self.assertTrue(handler.handle(message))
        self.assertTrue(handler.handle(message))

        self.assertEqual(self.application_count(), 1)
        self.assertEqual(len(self.state.get_daily_intake(DAY)), 1)
        self.assertEqual(len(interpreter.calls), 1)
        self.assertEqual(
            sender.calls[-1],
            ("expected-chat", "Got it — that meal report was already logged."),
        )

    def test_application_success_then_confirmation_failure_replays_without_duplicate_intake(self) -> None:
        self.state.save_meal_plan(self.breakfast_plan())
        sender = FakeOutboundSender(failure=RuntimeError("transport unavailable"))
        handler, interpreter = self.report_handler(
            semantic(
                [
                    {
                        "plan_item_id": "item_1",
                        "reference_text": "eggs",
                        "action": "eaten",
                        "quantity_text": None,
                    }
                ]
            ),
            sender,
        )
        payload = query_payload(42, "bluebubbles-guid-1", "I ate everything except the potatoes.")

        with TemporaryDirectory() as cursor_directory:
            cursor_store = RowIDCursorStore(Path(cursor_directory) / "cursor.json")
            cursor_store.initialize(41)
            recent = RecentMessageGuids()
            worker = BlueBubblesPollingWorker(
                FakeQueryClient(payload, payload),
                handler,
                recent_message_guids=recent,
                cursor_store=cursor_store,
            )
            with self.assertRaises(MealWorkflowMessageError):
                worker.poll_once()
            self.assertEqual(worker.cursor, 41)
            self.assertEqual(len(recent), 0)
            self.assertEqual(self.application_count(), 1)
            self.assertEqual(len(self.state.get_daily_intake(DAY)), 1)

            sender.failure = None
            self.assertEqual(worker.poll_once(), 1)
            self.assertEqual(worker.cursor, 42)
            self.assertEqual(len(recent), 1)

        self.assertEqual(self.application_count(), 1)
        self.assertEqual(len(self.state.get_daily_intake(DAY)), 1)
        self.assertEqual(len(interpreter.calls), 1)
        completed = self.state.load_meal_plan_for_source_event("bluebubbles-guid-1")
        assert completed is not None
        self.assertEqual(completed.status, "applied")
        self.assertEqual(
            sender.calls[-1],
            ("expected-chat", "Got it — that meal report was already logged."),
        )

    def test_clarification_sends_application_text_without_persisting_intake(self) -> None:
        saved = self.state.save_meal_plan(self.breakfast_plan())
        sender = FakeOutboundSender()
        handler, interpreter = self.report_handler(
            semantic(
                unresolved=[
                    {
                        "reference_text": "some of it",
                        "reason": "ambiguous_reference",
                    }
                ]
            ),
            sender,
        )

        self.assertTrue(handler.handle(incoming_message(text="I ate some of it.")))

        self.assertEqual(self.application_count(), 0)
        self.assertEqual(self.state.get_daily_intake(DAY), ())
        self.assertEqual(self.state.load_active_meal_plan(DAY, "breakfast"), saved)
        self.assertEqual(len(interpreter.calls), 1)
        self.assertTrue(sender.calls[0][1].startswith("I need a little clarification before I log that:"))

    def test_zero_or_multiple_active_contexts_fail_closed_without_reconciliation(self) -> None:
        no_plan_sender = FakeOutboundSender()
        no_plan_handler, no_plan_interpreter = self.report_handler(semantic(), no_plan_sender)

        self.assertTrue(no_plan_handler.handle(incoming_message()))
        self.assertEqual(no_plan_sender.calls, [("expected-chat", NO_ACTIVE_MEAL_PLAN_REPLY_TEXT)])
        self.assertEqual(no_plan_interpreter.calls, [])
        self.assertEqual(self.application_count(), 0)

        self.state.save_meal_plan(self.breakfast_plan())
        self.state.save_meal_plan(self.lunch_plan())
        ambiguous_sender = FakeOutboundSender()
        ambiguous_handler, ambiguous_interpreter = self.report_handler(semantic(), ambiguous_sender)

        self.assertTrue(ambiguous_handler.handle(incoming_message(message_guid="ambiguous-guid")))
        self.assertEqual(
            ambiguous_sender.calls,
            [("expected-chat", "Which meal are you reporting: breakfast or lunch?")],
        )
        self.assertEqual(ambiguous_interpreter.calls, [])
        self.assertEqual(
            [plan.plan.meal for plan in self.state.list_active_meal_plans(DAY)],
            [1, 2],
        )
        self.assertEqual(self.application_count(), 0)

    def test_active_plan_from_another_service_date_is_not_a_context_fallback(self) -> None:
        self.state.save_meal_plan(self.breakfast_plan())
        sender = FakeOutboundSender()
        interpreter = FakeMealReportSemanticInterpreter(semantic())
        handler = MealReportMessageHandler(
            sender,
            self.config,
            MealReportOrchestrator(
                self.catalog,
                MealReportReconciler(
                    interpreter,
                    LocalFDFoodResolver(self.catalog),
                    NaturalPortionInterpreter(),
                ),
            ),
            ActiveMealPlanContextResolver(
                self.state,
                service_date_provider=lambda: DAY + timedelta(days=1),
            ),
        )

        self.assertTrue(handler.handle(incoming_message(message_guid="next-day-guid")))

        self.assertEqual(sender.calls, [("expected-chat", NO_ACTIVE_MEAL_PLAN_REPLY_TEXT)])
        self.assertEqual(interpreter.calls, [])
        self.assertEqual(self.application_count(), 0)

    def test_filtering_rejects_self_unrelated_non_text_and_guidless_messages(self) -> None:
        self.state.save_meal_plan(self.breakfast_plan())
        sender = FakeOutboundSender()
        handler, interpreter = self.report_handler(semantic(), sender)

        ignored = (
            incoming_message(is_from_me=True),
            incoming_message(chat_guid="other-chat"),
            incoming_message(sender_address="other@example.test"),
            incoming_message(text="   "),
            incoming_message(message_guid=""),
        )
        for message in ignored:
            with self.subTest(message=message):
                self.assertFalse(handler.handle(message))

        self.assertEqual(sender.calls, [])
        self.assertEqual(interpreter.calls, [])
        self.assertEqual(self.application_count(), 0)


class MessageScopedMealReportRuntimeTests(unittest.TestCase):
    """Regression coverage for threaded inbound SQLite ownership."""

    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog_path = Path(self.directory.name) / "nutrition.sqlite3"
        self.config = ApplicationMessagingConfig(
            expected_chat_guid="expected-chat",
            expected_sender_address="student@example.test",
        )
        self._seed_active_breakfast_plan()

    def _seed_active_breakfast_plan(self) -> None:
        with OfficialNutritionCatalog(self.catalog_path) as catalog:
            catalog.synchronize_fd_refresh(
                mapped(
                    menu_day(
                        DAY,
                        1,
                        "Breakfast",
                        ((food(1, "Eggs"), "Grill"),),
                    )
                ),
                requested_start=DAY,
                requested_end=DAY,
                observed_at=OBSERVED_AT,
            )
            resolved = LocalFDFoodResolver(catalog).resolve(
                FoodResolutionRequest("Eggs", DAY, meal=1)
            )
            assert isinstance(resolved, ResolvedFood)
            plan = MealPlan(
                DAY,
                1,
                (PlannedMealItem(resolved, Decimal("1"), "two eggs"),),
            )
            self.persisted_plan = DurableMealState(catalog).save_meal_plan(
                plan,
                plan_id="threaded-breakfast-plan",
            )

    def _handler(
        self,
        sender: ThreadSafeOutboundSender,
        interpreter: object,
        catalog_factory: TrackingCatalogFactory,
    ) -> MessageScopedMealReportMessageHandler:
        return MessageScopedMealReportMessageHandler(
            sender,
            self.config,
            interpreter,  # type: ignore[arg-type]
            catalog_path=self.catalog_path,
            service_date_provider=lambda: DAY,
            catalog_factory=catalog_factory,
        )

    def _application_and_intake_counts(self) -> tuple[int, int]:
        with OfficialNutritionCatalog(self.catalog_path) as catalog:
            connection = catalog._connection
            return (
                int(connection.execute("SELECT COUNT(*) FROM meal_report_applications").fetchone()[0]),
                int(connection.execute("SELECT COUNT(*) FROM accepted_intake_entries").fetchone()[0]),
            )

    def _active_plan(self):
        with OfficialNutritionCatalog(self.catalog_path) as catalog:
            return DurableMealState(catalog).load_active_meal_plan(DAY, "breakfast")

    def _assert_closed_in_creator_threads(self, factory: TrackingCatalogFactory) -> None:
        self.assertTrue(factory.catalogs)
        for catalog in factory.catalogs:
            self.assertIsNone(catalog._connection)
            self.assertEqual(catalog.close_calls, 1)
            self.assertEqual(catalog.closed_thread_id, catalog.created_thread_id)

    @staticmethod
    def _clarification_interpreter() -> FakeMealReportSemanticInterpreter:
        return FakeMealReportSemanticInterpreter(
            semantic(
                unresolved=[
                    {
                        "reference_text": "unclear test quantity",
                        "reason": "ambiguous_reference",
                    }
                ]
            )
        )

    @staticmethod
    def _accepted_interpreter() -> FakeMealReportSemanticInterpreter:
        return FakeMealReportSemanticInterpreter(
            semantic(
                [
                    {
                        "plan_item_id": "item_1",
                        "reference_text": "eggs",
                        "action": "eaten",
                        "quantity_text": None,
                    }
                ]
            )
        )

    def test_runtime_factory_uses_detroit_service_date_in_polling_thread(self) -> None:
        factory = TrackingCatalogFactory()
        sender = ThreadSafeOutboundSender()
        interpreter = self._clarification_interpreter()
        food_matcher = FakeFoodSemanticMatcher()
        environment = {
            "NUTRITION_OPTIMIZER_BLUEBUBBLES_CHAT_GUID": "expected-chat",
            "NUTRITION_OPTIMIZER_BLUEBUBBLES_SENDER_ADDRESS": "student@example.test",
        }

        with patch(
            "nutrition_optimizer.messaging.meal_workflow.OpenClawMealReportInterpreter.from_env",
            return_value=interpreter,
        ), patch(
            "nutrition_optimizer.messaging.meal_workflow.OpenClawFoodSemanticMatcher.from_env",
            return_value=food_matcher,
        ):
            runtime = open_production_meal_report_runtime(
                sender,
                environ=environment,
                catalog_path=self.catalog_path,
                clock=NutritionApplicationClock(
                    lambda: datetime(2026, 8, 25, 12, 45, tzinfo=timezone.utc)
                ),
                catalog_factory=factory,
            )

        # Runtime construction itself retains no SQLite connection.  Under the
        # old production composition this factory call created the one catalog
        # on this test's main thread before the polling thread ran.  The
        # injected 12:45 UTC instant is 08:45 Detroit on the seeded Aug. 25
        # service date, proving the production-style scoped path carries the
        # application clock and normal Phelps eligibility all the way to
        # active-context resolution.
        self.assertEqual(factory.catalogs, [])
        self.assertEqual(interpreter.availability_checks, 1)
        self.assertEqual(food_matcher.availability_checks, 1)
        self.assertIs(runtime.handler.food_semantic_matcher, food_matcher)

        with TemporaryDirectory() as cursor_directory:
            cursor_store = RowIDCursorStore(Path(cursor_directory) / "cursor.json")
            cursor_store.initialize(41)
            worker = BlueBubblesPollingWorker(
                FakeQueryClient(query_payload(42, "polling-thread-guid", "I ate what you recommended")),
                runtime.handler,
                cursor_store=cursor_store,
                interval=60,
            )
            worker.start()
            try:
                self.assertTrue(sender.sent.wait(timeout=5))
            finally:
                worker.stop(timeout=5)
            self.assertEqual(worker.cursor, 42)

        runtime.close()
        self.assertEqual(len(factory.catalogs), 1)
        self._assert_closed_in_creator_threads(factory)
        self.assertNotEqual(factory.catalogs[0].created_thread_id, get_ident())
        self.assertEqual(len(interpreter.calls), 1)
        self.assertEqual(self._application_and_intake_counts(), (0, 0))
        self.assertEqual(self._active_plan(), self.persisted_plan)

    def test_webhook_request_threads_use_distinct_closed_catalogs(self) -> None:
        factory = TrackingCatalogFactory()
        sender = ThreadSafeOutboundSender()
        interpreter = CoordinatedMealReportSemanticInterpreter(
            self._clarification_interpreter().result
        )
        handler = self._handler(sender, interpreter, factory)
        server = create_webhook_server(
            host="127.0.0.1",
            port=0,
            message_handler=handler,
        )
        server_thread = Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        start_posts = Event()
        statuses: list[int] = []
        failures: list[BaseException] = []
        result_lock = Lock()

        def post_message(guid: str) -> None:
            start_posts.wait(timeout=5)
            connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
            try:
                connection.request(
                    "POST",
                    "/",
                    body=json.dumps(
                        {
                            "type": "new-message",
                            "data": {
                                "guid": guid,
                                "text": "I ate what you recommended",
                                "isFromMe": False,
                                "handle": {"address": "student@example.test"},
                                "chats": [{"guid": "expected-chat"}],
                            },
                        }
                    ),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                response.read()
                with result_lock:
                    statuses.append(response.status)
            except BaseException as exc:
                with result_lock:
                    failures.append(exc)
            finally:
                connection.close()

        clients = [
            Thread(target=post_message, args=("webhook-thread-guid-1",)),
            Thread(target=post_message, args=("webhook-thread-guid-2",)),
        ]
        try:
            for client in clients:
                client.start()
            start_posts.set()
            for client in clients:
                client.join(timeout=10)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)

        self.assertFalse(failures)
        self.assertEqual(statuses, [200, 200])
        self.assertEqual(len(factory.catalogs), 2)
        self.assertEqual(len({catalog.created_thread_id for catalog in factory.catalogs}), 2)
        self._assert_closed_in_creator_threads(factory)
        self.assertTrue(all(catalog.created_thread_id != get_ident() for catalog in factory.catalogs))
        self.assertEqual(len(interpreter.calls), 2)
        self.assertEqual(len(sender.calls), 2)
        self.assertEqual(self._application_and_intake_counts(), (0, 0))
        self.assertEqual(self._active_plan(), self.persisted_plan)

    def test_poll_retry_replays_once_and_closes_each_scoped_catalog(self) -> None:
        factory = TrackingCatalogFactory()
        sender = ThreadSafeOutboundSender(failure=RuntimeError("test transport failure"))
        interpreter = self._accepted_interpreter()
        handler = self._handler(sender, interpreter, factory)
        payload = query_payload(42, "retry-threaded-guid", "I ate what you recommended")

        with TemporaryDirectory() as cursor_directory:
            cursor_store = RowIDCursorStore(Path(cursor_directory) / "cursor.json")
            cursor_store.initialize(41)
            worker = BlueBubblesPollingWorker(
                FakeQueryClient(payload, payload),
                handler,
                cursor_store=cursor_store,
            )
            with self.assertRaises(MealWorkflowMessageError):
                worker.poll_once()
            self.assertEqual(worker.cursor, 41)
            self.assertEqual(self._application_and_intake_counts(), (1, 1))
            self.assertEqual(len(factory.catalogs), 1)
            self._assert_closed_in_creator_threads(factory)

            sender.failure = None
            self.assertEqual(worker.poll_once(), 1)
            self.assertEqual(worker.cursor, 42)

        self.assertEqual(self._application_and_intake_counts(), (1, 1))
        self.assertEqual(len(interpreter.calls), 1)
        self.assertEqual(len(factory.catalogs), 2)
        self._assert_closed_in_creator_threads(factory)

    def test_concurrent_same_guid_dispatch_persists_one_intake(self) -> None:
        factory = TrackingCatalogFactory()
        sender = ThreadSafeOutboundSender()
        interpreter = self._accepted_interpreter()
        handler = self._handler(sender, interpreter, factory)
        cache = RecentMessageGuids()
        message = incoming_message(message_guid="concurrent-threaded-guid", text="I ate what you recommended")
        start_dispatch = Event()
        results: list[bool] = []
        failures: list[BaseException] = []
        result_lock = Lock()

        def dispatch() -> None:
            start_dispatch.wait(timeout=5)
            try:
                delivered = deliver_incoming_message(message, handler, cache)
                with result_lock:
                    results.append(delivered)
            except BaseException as exc:
                with result_lock:
                    failures.append(exc)

        workers = [Thread(target=dispatch), Thread(target=dispatch)]
        for worker in workers:
            worker.start()
        start_dispatch.set()
        for worker in workers:
            worker.join(timeout=10)

        self.assertFalse(failures)
        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(self._application_and_intake_counts(), (1, 1))
        self.assertEqual(len(interpreter.calls), 1)
        self.assertEqual(len(sender.calls), 2)
        self.assertEqual([message for _, message in sender.calls].count("Drink a shake"), 1)
        self.assertEqual(len(factory.catalogs), 1)
        self._assert_closed_in_creator_threads(factory)


@unittest.skipUnless(PRODUCTION_CATALOG_PATH.is_file(), "requires the local Phelps SQLite catalog")
class IsolatedPhelpsBlueBubblesIntegrationTests(unittest.TestCase):
    """Exercise the deployed catalog only through an isolated filesystem copy."""

    def test_real_phelps_breakfast_to_adaptive_lunch_never_mutates_production_catalog(self) -> None:
        before_hash = sha256(PRODUCTION_CATALOG_PATH.read_bytes()).hexdigest()
        with TemporaryDirectory() as directory:
            working_path = Path(directory) / "nutrition.sqlite3"
            baseline_path = Path(directory) / "baseline.sqlite3"
            copy2(PRODUCTION_CATALOG_PATH, working_path)
            copy2(PRODUCTION_CATALOG_PATH, baseline_path)

            with OfficialNutritionCatalog(working_path) as catalog, OfficialNutritionCatalog(
                baseline_path
            ) as baseline_catalog:
                self._clear_application_state(catalog)
                self._clear_application_state(baseline_catalog)
                service_date = self._date_with_real_breakfast_and_lunch(catalog)
                targets = load_production_daily_targets(production_target_environment())

                sender = FakeOutboundSender()
                state = DurableMealState(catalog)
                breakfast_delivery = ManualMealRecommendationDelivery(
                    sender,
                    "expected-chat",
                    MealRecommendationOrchestrator(catalog),
                    state,
                )
                breakfast = breakfast_delivery.prepare_and_send(
                    service_date,
                    "breakfast",
                    targets,
                )
                self.assertEqual(sender.calls, [("expected-chat", breakfast.message)])
                self.assertEqual(
                    state.load_active_meal_plan(service_date, "breakfast"),
                    breakfast.persisted_plan,
                )

                report_interpreter = AllPlannedItemsEatenInterpreter()
                report_handler = MealReportMessageHandler(
                    sender,
                    ApplicationMessagingConfig(expected_chat_guid="expected-chat"),
                    MealReportOrchestrator(
                        catalog,
                        MealReportReconciler(
                            report_interpreter,
                            LocalFDFoodResolver(catalog),
                            NaturalPortionInterpreter(),
                        ),
                    ),
                    ActiveMealPlanContextResolver(
                        state,
                        service_date_provider=lambda: service_date,
                    ),
                )
                self.assertTrue(
                    report_handler.handle(
                        IncomingMessage(
                            message_guid="phelps-breakfast-bluebubbles-guid",
                            text="I ate everything from breakfast.",
                            sender_address="student@example.test",
                            chat_guid="expected-chat",
                            is_from_me=False,
                        )
                    )
                )
                accepted_breakfast = state.get_daily_intake(service_date)
                self.assertGreater(len(accepted_breakfast), 0)
                self.assertEqual(
                    [entry.source_event_id for entry in accepted_breakfast],
                    ["phelps-breakfast-bluebubbles-guid"] * len(accepted_breakfast),
                )
                self.assertEqual(len(sender.calls), 2)

                adaptive_lunch = MealRecommendationOrchestrator(catalog).prepare(
                    service_date,
                    "lunch",
                    targets,
                )
                baseline_lunch = MealRecommendationOrchestrator(baseline_catalog).prepare(
                    service_date,
                    "lunch",
                    targets,
                )

                self.assertEqual(
                    adaptive_lunch.current_ledger,
                    state.load_daily_ledger(service_date),
                )
                self.assertGreater(len(adaptive_lunch.current_ledger.entries), 0)
                self.assertEqual(baseline_lunch.current_ledger.entries, ())

        self.assertEqual(
            sha256(PRODUCTION_CATALOG_PATH.read_bytes()).hexdigest(),
            before_hash,
        )

    @staticmethod
    def _clear_application_state(catalog: OfficialNutritionCatalog) -> None:
        with catalog._connection:
            catalog._connection.execute("DELETE FROM meal_report_message_events")
            catalog._connection.execute("DELETE FROM meal_report_draft_clarifications")
            catalog._connection.execute("DELETE FROM meal_report_draft_unplanned_items")
            catalog._connection.execute("DELETE FROM meal_report_draft_planned_items")
            catalog._connection.execute("DELETE FROM meal_report_drafts")
            catalog._connection.execute("DELETE FROM meal_recommendation_replacement_rejections")
            catalog._connection.execute("DELETE FROM meal_recommendation_replacement_requested_foods")
            catalog._connection.execute("DELETE FROM meal_recommendation_replacements")
            catalog._connection.execute("DELETE FROM immediate_meal_request_requested_foods")
            catalog._connection.execute("DELETE FROM immediate_meal_request_dispatches")
            catalog._connection.execute("DELETE FROM pending_meal_request_foods")
            catalog._connection.execute("DELETE FROM pending_meal_requests")
            catalog._connection.execute("DELETE FROM scheduled_recommendation_dispatches")
            catalog._connection.execute("DELETE FROM accepted_intake_entries")
            catalog._connection.execute("DELETE FROM meal_report_applications")
            catalog._connection.execute("DELETE FROM meal_plan_items")
            catalog._connection.execute("DELETE FROM meal_plans")

    @staticmethod
    def _date_with_real_breakfast_and_lunch(catalog: OfficialNutritionCatalog) -> date:
        row = catalog._connection.execute(
            """
            SELECT breakfast.service_date
            FROM fd_menu_occurrences AS breakfast
            WHERE lower(breakfast.meal_period_name) = 'breakfast'
              AND EXISTS (
                  SELECT 1 FROM fd_menu_occurrences AS lunch
                  WHERE lunch.service_date = breakfast.service_date
                    AND lower(lunch.meal_period_name) = 'lunch'
              )
            ORDER BY breakfast.service_date
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            raise unittest.SkipTest("local Phelps catalog has no date with breakfast and lunch")
        return date.fromisoformat(str(row[0]))


if __name__ == "__main__":
    unittest.main()
