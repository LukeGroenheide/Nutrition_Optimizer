"""Offline regression coverage for pre-meal deterministic replacements."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nutrition_optimizer.application import (
    MealRecommendationOrchestrator,
    MealRecommendationReplacementUnavailableError,
)
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
from nutrition_optimizer.meal_optimizer import LocalMealOptimizer
from nutrition_optimizer.meal_request import RequestedMealFood, requested_meal_food_from_resolved
from nutrition_optimizer.meal_report import MealPlan, MealReportReconciler, PlannedMealItem
from nutrition_optimizer.meal_report_conversation import MealReportConversationOrchestrator
from nutrition_optimizer.meal_report_structured import MealReportSemanticResult
from nutrition_optimizer.messaging.application import ApplicationMessagingConfig
from nutrition_optimizer.messaging.meal_workflow import (
    ActiveMealPlanContextResolver,
    ConversationalMealReportMessageHandler,
    MealWorkflowMessageError,
)
from nutrition_optimizer.messaging.webhook import IncomingMessage
from nutrition_optimizer.nutrition import DailyLedger, DailyMinimums, DailyTargets
from nutrition_optimizer.portion_interpretation import NaturalPortionInterpreter
from nutrition_optimizer.phelps_service_calendar import DEFAULT_PHELPS_SERVICE_CALENDAR
from nutrition_optimizer.recommendation_scheduler import (
    RecommendationDispatchSchedule,
    RecommendationScheduler,
    ScheduledMealOpportunity,
)
from tests.test_food_resolution import component, mapped, menu_day


DAY = date(2026, 9, 4)
OBSERVED_AT = datetime(2026, 9, 4, 16, tzinfo=timezone.utc)
CHAT_GUID = "chat-replacement-test"


def targets() -> DailyTargets:
    return DailyTargets(
        calories_kcal=Decimal("1000"),
        protein_g=Decimal("100"),
        carbohydrates_g=Decimal("100"),
        fat_g=Decimal("50"),
        minimums=DailyMinimums(Decimal("10")),
    )


def food(component_id: int, name: str, *, protein: str = "10") -> dict[str, object]:
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


def replacement_semantic(
    *,
    mode: str,
    item_ids: tuple[str, ...] = (),
) -> MealReportSemanticResult:
    return MealReportSemanticResult.model_validate(
        {
            "intent": "replacement_request",
            "report_scope": "unknown",
            "location_plan_item_id": None,
            "location_food_text": None,
            "replacement_mode": mode,
            "replacement_plan_item_ids": list(item_ids),
            "planned_items": [],
            "additional_foods": [],
            "unresolved_statements": [],
        }
    )


def meal_request_semantic(
    *,
    mode: str,
    food_texts: tuple[str, ...] = (),
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
            "requested_food_texts": list(food_texts),
            "requested_meal": None,
            "planned_items": [],
            "additional_foods": [],
            "unresolved_statements": [],
        }
    )


def partial_report_semantic() -> MealReportSemanticResult:
    return MealReportSemanticResult.model_validate(
        {
            "intent": "meal_report",
            "report_scope": "partial",
            "location_plan_item_id": None,
            "location_food_text": None,
            "replacement_mode": None,
            "replacement_plan_item_ids": [],
            "planned_items": [
                {
                    "plan_item_id": "item_1",
                    "reference_text": "the salad",
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


@dataclass
class ScriptedInterpreter:
    responses: dict[str, MealReportSemanticResult]
    calls: list[str] = field(default_factory=list)

    def interpret(self, plan: MealPlan, user_text: str) -> MealReportSemanticResult:
        del plan
        self.calls.append(user_text)
        return self.responses[user_text]

    def interpret_with_context(
        self,
        plan: MealPlan,
        user_text: str,
        conversation_context,
    ) -> MealReportSemanticResult:
        del plan, conversation_context
        self.calls.append(user_text)
        return self.responses[user_text]


@dataclass
class StableSender:
    failures: int = 0
    unknown_failure: bool = False
    calls: list[tuple[str, str, str | None]] = field(default_factory=list)

    def send_text(
        self,
        chat_guid: str,
        message: str,
        *,
        temp_guid: str | None = None,
    ) -> object:
        self.calls.append((chat_guid, message, temp_guid))
        if self.unknown_failure:
            raise ConnectionError("test unknown transport outcome")
        if self.failures:
            self.failures -= 1
            raise RuntimeError("test known transport failure")
        return object()


class MealRecommendationReplacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "nutrition.sqlite3"
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self.catalog.close)
        self._synchronize_menu(include_replacements=True)
        self.state = DurableMealState(self.catalog)
        self.resolver = LocalFDFoodResolver(self.catalog)
        self.old_plan = self._old_plan()
        self.old_persisted = self.state.save_meal_plan(self.old_plan)

    def _synchronize_menu(self, *, include_replacements: bool) -> None:
        lunch_items: list[tuple[dict[str, object], str]] = [
            (food(1, "Salad"), "Greens"),
            (food(2, "Pork"), "Homestyle"),
            (food(3, "Rice"), "Grill"),
        ]
        if include_replacements:
            lunch_items.extend(
                [
                    (food(4, "Chicken", protein="20"), "Grill"),
                    (food(5, "Beans", protein="15"), "Greens"),
                ]
            )
        self.catalog.synchronize_fd_refresh(
            mapped(
                menu_day(DAY, 2, "Lunch", tuple(lunch_items)),
                menu_day(
                    DAY,
                    3,
                    "Dinner",
                    ((food(10, "Dinner Chicken", protein="20"), "Grill"),),
                ),
            ),
            requested_start=DAY,
            requested_end=DAY,
            observed_at=OBSERVED_AT,
        )

    def _resolved(self, name: str, *, meal: int = 2) -> ResolvedFood:
        resolved = self.resolver.resolve(FoodResolutionRequest(name, DAY, meal=meal))
        assert isinstance(resolved, ResolvedFood)
        return resolved

    def _old_plan(self) -> MealPlan:
        return MealPlan(
            DAY,
            2,
            (
                PlannedMealItem(self._resolved("Salad"), Decimal("1"), "one scoop"),
                PlannedMealItem(self._resolved("Pork"), Decimal("1"), "one portion"),
                PlannedMealItem(self._resolved("Rice"), Decimal("1"), "one scoop"),
            ),
        )

    def _preparer(self) -> MealRecommendationReplacementPreparer:
        return MealRecommendationReplacementPreparer(
            self.state,
            MealRecommendationOrchestrator(self.catalog),
            targets,
        )

    def _conversation(
        self,
        responses: dict[str, MealReportSemanticResult],
    ) -> MealReportConversationOrchestrator:
        reconciler = MealReportReconciler(
            ScriptedInterpreter(responses),
            LocalFDFoodResolver(self.catalog),
            NaturalPortionInterpreter(),
        )
        return MealReportConversationOrchestrator(
            self.state,
            reconciler,
            replacement_preparer=self._preparer(),
        )

    def _resolver(self) -> ActiveMealPlanContextResolver:
        return ActiveMealPlanContextResolver(
            self.state,
            service_date_provider=lambda: DAY,
        )

    def _replace(
        self,
        *,
        source_event_id: str = "replacement-guid",
        mode: str = "targeted",
        item_ids: tuple[str, ...] = ("item_2",),
    ):
        text = "replace request"
        result = self._conversation(
            {text: replacement_semantic(mode=mode, item_ids=item_ids)}
        ).process(
            chat_guid=CHAT_GUID,
            user_text=text,
            source_event_id=source_event_id,
            context_resolver=self._resolver(),
        )
        assert result.replacement_dispatch is not None
        return result

    def _identity_names(self, plan: MealPlan) -> set[str]:
        return {item.display_name for item in plan.items}

    def _counts(self) -> tuple[int, int]:
        connection = self.catalog._connection
        assert connection is not None
        return (
            int(connection.execute("SELECT count(*) FROM meal_report_applications").fetchone()[0]),
            int(connection.execute("SELECT count(*) FROM accepted_intake_entries").fetchone()[0]),
        )

    def test_targeted_replacement_preserves_non_rejected_items_and_excludes_pork(self) -> None:
        result = self._replace()
        dispatch = result.replacement_dispatch
        assert dispatch is not None

        self.assertEqual(dispatch.prior_plan.plan_id, self.old_persisted.plan_id)
        self.assertEqual(dispatch.persisted_plan.status, "active")
        names = self._identity_names(dispatch.persisted_plan.plan)
        self.assertIn("Salad", names)
        self.assertIn("Rice", names)
        self.assertNotIn("Pork", names)
        self.assertTrue({"Chicken", "Beans"} & names)
        self.assertEqual(self.state.load_meal_plan(self.old_persisted.plan_id).status, "superseded")
        self.assertEqual(self._counts(), (0, 0))
        self.assertEqual(
            self._resolver().resolve("later-meal-report").plan_id,
            dispatch.plan_id,
        )

    def test_positive_food_request_reuses_replacement_delivery_and_retains_authoritative_food(self) -> None:
        text = "I want chicken"
        result = self._conversation(
            {text: meal_request_semantic(mode="targeted", food_texts=("Chicken",))}
        ).process(
            chat_guid=CHAT_GUID,
            user_text=text,
            source_event_id="positive-chicken-guid",
            context_resolver=self._resolver(),
        )

        self.assertEqual(result.outcome, "meal_request")
        dispatch = result.replacement_dispatch
        assert dispatch is not None
        self.assertEqual(dispatch.request_kind, "meal_request")
        self.assertEqual(dispatch.rejected_plan_item_ids, ())
        self.assertEqual(len(dispatch.requested_foods), 1)
        self.assertEqual(dispatch.requested_foods[0].food_text, "Chicken")
        self.assertIn("Chicken", self._identity_names(dispatch.persisted_plan.plan))
        self.assertEqual(self.state.load_meal_plan(self.old_persisted.plan_id).status, "superseded")
        self.assertEqual(self._counts(), (0, 0))

    def test_positive_food_request_uses_optimizer_quantity_and_fills_remaining_meal(self) -> None:
        required = requested_meal_food_from_resolved(self._resolved("Chicken"))
        optimizer = LocalMealOptimizer(self.catalog)
        first = optimizer.optimize_meal(
            DAY,
            2,
            targets(),
            DailyLedger(),
            required_foods=(required,),
        )
        second = optimizer.optimize_meal(
            DAY,
            2,
            targets(),
            DailyLedger(),
            required_foods=(required,),
        )

        self.assertTrue(first.is_recommendation)
        self.assertGreaterEqual(len(first.items), 2)
        self.assertIn("Chicken", tuple(item.record.name for item in first.items))
        self.assertEqual(
            tuple(
                (item.occurrence.occurrence_id, item.official_servings)
                for item in first.items
            ),
            tuple(
                (item.occurrence.occurrence_id, item.official_servings)
                for item in second.items
            ),
        )
        self.assertEqual(first.projected_meal_nutrition, second.projected_meal_nutrition)
        self.assertEqual(first.projected_daily_total, first.projected_meal_nutrition)
        chicken_item = next(item for item in first.items if item.record.name == "Chicken")
        self.assertEqual(chicken_item.physical_quantity.amount, Decimal("1"))
        self.assertEqual(self._counts(), (0, 0))

    def test_unavailable_positive_constraint_is_deterministically_rejected(self) -> None:
        unavailable = RequestedMealFood("Pizza", "recipe", "missing", "missing-signature")
        result = LocalMealOptimizer(self.catalog).optimize_meal(
            DAY,
            2,
            targets(),
            DailyLedger(),
            required_foods=(unavailable,),
        )
        self.assertFalse(result.is_recommendation)
        self.assertEqual(result.diagnostics.outcome, "required_food_unavailable")

        text = "I want pizza"
        interaction = self._conversation(
            {text: meal_request_semantic(mode="targeted", food_texts=("Pizza",))}
        ).process(
            chat_guid=CHAT_GUID,
            user_text=text,
            source_event_id="positive-unavailable-guid",
            context_resolver=self._resolver(),
        )
        self.assertEqual(interaction.outcome, "meal_request")
        self.assertIn("isn't available", interaction.message)
        self.assertIsNone(
            self.state.load_meal_recommendation_replacement_dispatch("positive-unavailable-guid")
        )
        self.assertEqual(self.state.load_meal_plan(self.old_persisted.plan_id).status, "active")

    def test_whole_meal_request_uses_existing_whole_replacement_semantics(self) -> None:
        text = "Build me a different lunch"
        result = self._conversation(
            {text: meal_request_semantic(mode="whole_meal")}
        ).process(
            chat_guid=CHAT_GUID,
            user_text=text,
            source_event_id="whole-meal-request-guid",
            context_resolver=self._resolver(),
        )
        dispatch = result.replacement_dispatch
        assert dispatch is not None
        self.assertEqual(result.outcome, "meal_request")
        self.assertEqual(dispatch.request_kind, "meal_request")
        self.assertTrue(dispatch.whole_meal)
        self.assertEqual(dispatch.rejected_plan_item_ids, ())
        self.assertTrue(
            self._identity_names(dispatch.persisted_plan.plan).isdisjoint(
                {"Salad", "Pork", "Rice"}
            )
        )
        self.assertEqual(self._counts(), (0, 0))

    def test_multiple_and_whole_meal_rejections_use_authoritative_plan_ids(self) -> None:
        multiple = self._replace(item_ids=("item_2", "item_3"))
        assert multiple.replacement_dispatch is not None
        self.assertNotIn("Pork", self._identity_names(multiple.persisted_plan.plan))
        self.assertNotIn("Rice", self._identity_names(multiple.persisted_plan.plan))

        # Fresh state makes whole-meal behavior independent of the prior case.
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        # The previous replacement intentionally changed this fixture; use a
        # new isolated catalog for the whole-meal assertion instead.
        self.catalog.close()
        isolated = TemporaryDirectory()
        self.addCleanup(isolated.cleanup)
        self.path = Path(isolated.name) / "whole.sqlite3"
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self.catalog.close)
        self._synchronize_menu(include_replacements=True)
        self.state = DurableMealState(self.catalog)
        self.resolver = LocalFDFoodResolver(self.catalog)
        self.old_plan = self._old_plan()
        self.old_persisted = self.state.save_meal_plan(self.old_plan)
        whole = self._replace(source_event_id="whole-guid", mode="whole_meal", item_ids=())
        assert whole.replacement_dispatch is not None
        self.assertTrue(
            self._identity_names(whole.persisted_plan.plan).isdisjoint(
                {"Salad", "Pork", "Rice"}
            )
        )

    def test_ambiguous_request_and_no_valid_candidate_leave_original_plan_unchanged(self) -> None:
        text = "replace that"
        ambiguous = self._conversation(
            {text: replacement_semantic(mode="ambiguous")}
        ).process(
            chat_guid=CHAT_GUID,
            user_text=text,
            source_event_id="ambiguous-guid",
            context_resolver=self._resolver(),
        )
        self.assertIn("Which recommended item", ambiguous.message)
        self.assertEqual(self.state.load_meal_plan(self.old_persisted.plan_id).status, "active")
        self.assertIsNone(
            self.state.load_meal_recommendation_replacement_dispatch("ambiguous-guid")
        )

        # A catalog containing only the original authoritative foods cannot
        # satisfy the replacement plan's distinct-food invariant.
        no_candidate_dir = TemporaryDirectory()
        self.addCleanup(no_candidate_dir.cleanup)
        with OfficialNutritionCatalog(Path(no_candidate_dir.name) / "no-candidate.sqlite3") as catalog:
            self.catalog = catalog
            self._synchronize_menu(include_replacements=False)
            state = DurableMealState(catalog)
            resolver = LocalFDFoodResolver(catalog)
            old = MealPlan(
                DAY,
                2,
                tuple(
                    PlannedMealItem(
                        resolver.resolve(FoodResolutionRequest(name, DAY, meal=2)),
                        Decimal("1"),
                        "one portion",
                    )
                    for name in ("Salad", "Pork", "Rice")
                ),
            )
            persisted = state.save_meal_plan(old)
            with self.assertRaises(MealRecommendationReplacementUnavailableError):
                MealRecommendationOrchestrator(catalog).prepare_replacement(
                    persisted,
                    ("item_2",),
                    whole_meal=False,
                    targets=targets(),
                )
            self.assertEqual(state.load_meal_plan(persisted.plan_id).status, "active")

    def test_handler_replays_a_known_delivery_failure_with_one_persisted_plan(self) -> None:
        text = "replace pork"
        conversation = self._conversation(
            {text: replacement_semantic(mode="targeted", item_ids=("item_2",))}
        )
        sender = StableSender(failures=1)
        delivery = MealRecommendationReplacementDelivery(
            self.state,
            sender,
            is_known_delivery_failure=lambda exc: isinstance(exc, RuntimeError),
        )
        handler = ConversationalMealReportMessageHandler(
            sender,
            ApplicationMessagingConfig(expected_chat_guid=CHAT_GUID),
            conversation,
            self._resolver(),
            replacement_delivery=delivery,
        )
        message = IncomingMessage("replace-send-guid", text, "sender", CHAT_GUID, False)

        with self.assertRaises(MealWorkflowMessageError):
            handler.handle(message)
        first = self.state.load_meal_recommendation_replacement_dispatch("replace-send-guid")
        assert first is not None
        self.assertEqual(first.status, "pending_delivery")
        self.assertEqual(len(sender.calls), 1)

        self.assertTrue(handler.handle(message))
        delivered = self.state.load_meal_recommendation_replacement_dispatch("replace-send-guid")
        assert delivered is not None
        self.assertEqual(delivered.status, "delivered")
        self.assertEqual(delivered.plan_id, first.plan_id)
        self.assertEqual(len(sender.calls), 2)
        self.assertEqual(sender.calls[0][2], sender.calls[1][2])
        self.assertTrue(handler.handle(message))
        self.assertEqual(len(sender.calls), 2)
        self.assertEqual(self._counts(), (0, 0))

    def test_unknown_delivery_outcome_never_automatically_sends_again(self) -> None:
        text = "replace pork"
        sender = StableSender(unknown_failure=True)
        handler = ConversationalMealReportMessageHandler(
            sender,
            ApplicationMessagingConfig(expected_chat_guid=CHAT_GUID),
            self._conversation(
                {text: replacement_semantic(mode="targeted", item_ids=("item_2",))}
            ),
            self._resolver(),
            replacement_delivery=MealRecommendationReplacementDelivery(
                self.state,
                sender,
                is_known_delivery_failure=lambda exc: False,
            ),
        )
        message = IncomingMessage("unknown-send-guid", text, "sender", CHAT_GUID, False)
        with self.assertRaises(MealWorkflowMessageError):
            handler.handle(message)
        with self.assertRaises(MealWorkflowMessageError):
            handler.handle(message)
        dispatch = self.state.load_meal_recommendation_replacement_dispatch("unknown-send-guid")
        assert dispatch is not None
        self.assertEqual(dispatch.status, "sending")
        self.assertEqual(len(sender.calls), 1)

    def test_open_report_draft_blocks_replacement_without_cross_plan_mutation(self) -> None:
        partial_text = "I had the salad"
        replace_text = "replace pork"
        conversation = self._conversation(
            {
                partial_text: partial_report_semantic(),
                replace_text: replacement_semantic(mode="targeted", item_ids=("item_2",)),
            }
        )
        first = conversation.process(
            chat_guid=CHAT_GUID,
            user_text=partial_text,
            source_event_id="draft-guid",
            context_resolver=self._resolver(),
        )
        self.assertEqual(first.outcome, "clarification_required")
        blocked = conversation.process(
            chat_guid=CHAT_GUID,
            user_text=replace_text,
            source_event_id="replace-while-draft-guid",
            context_resolver=self._resolver(),
        )
        self.assertIn("finish or cancel", blocked.message)
        self.assertEqual(self.state.load_meal_plan(self.old_persisted.plan_id).status, "active")
        assert first.draft is not None
        persisted_draft = self.state.load_meal_report_draft(first.draft.draft_id)
        assert persisted_draft is not None
        self.assertEqual(persisted_draft.plan_id, self.old_persisted.plan_id)
        self.assertIsNone(
            self.state.load_meal_recommendation_replacement_dispatch(
                "replace-while-draft-guid"
            )
        )
        self.assertEqual(self._counts(), (0, 0))

    def test_rejection_is_not_a_permanent_global_preference(self) -> None:
        self._replace(source_event_id="one-time-rejection-guid")
        pork_occurrence = next(
            occurrence
            for occurrence in self.catalog.list_current_meal_occurrences(DAY, meal=2)
            if occurrence.nutrition_record.name == "Pork"
        )

        class PorkOnlyCurrentMenu:
            def list_current_meal_occurrences(self, service_date, meal=None):
                self_seen.append((service_date, meal))
                return (pork_occurrence,)

        self_seen: list[tuple[date, int | str | None]] = []
        future_operation = LocalMealOptimizer(PorkOnlyCurrentMenu()).optimize_meal(
            DAY,
            2,
            targets(),
            DailyLedger(),
        )
        self.assertEqual(self_seen, [(DAY, 2)])
        self.assertEqual(
            tuple(item.record.name for item in future_operation.items),
            ("Pork",),
        )

    def test_scheduler_origin_replacement_survives_a_later_different_slot(self) -> None:
        # Replace the fixture's manual plan with a delivered scheduler plan.
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        self.resolver = LocalFDFoodResolver(self.catalog)
        # The manual fixture is still active after reopening; supersede it only
        # in this isolated test before reserving the scheduler-owned context.
        self.state.save_meal_plan(self.old_plan)
        manual = self.state.load_active_meal_plan(DAY, 2)
        assert manual is not None
        self.catalog._connection.execute(
            "UPDATE meal_plans SET status = 'superseded' WHERE plan_id = ?", (manual.plan_id,)
        )
        self.catalog._connection.commit()
        scheduled_plan = self.state.save_scheduled_meal_plan(self.old_plan)
        scheduled = self.state.load_scheduled_recommendation_dispatch(DAY, 2)
        assert scheduled is not None
        self.assertTrue(self.state.claim_scheduled_recommendation_delivery(scheduled))
        self.state.mark_scheduled_recommendation_delivered(scheduled)
        self.old_persisted = scheduled_plan

        replacement = self._replace(source_event_id="scheduled-replace-guid")
        assert replacement.replacement_dispatch is not None
        self.assertTrue(
            self.state.claim_replacement_recommendation_delivery(
                replacement.replacement_dispatch
            )
        )
        replacement_dispatch = self.state.mark_replacement_recommendation_delivered(
            replacement.replacement_dispatch
        )
        self.assertEqual(
            self.state.load_scheduled_recommendation_dispatch(DAY, 2).status,
            "delivered",
        )
        self.assertEqual(
            replacement.replacement_dispatch.scheduled_origin_plan_id,
            scheduled_plan.plan_id,
        )
        self.assertEqual(
            self.state.load_meal_plan(scheduled_plan.plan_id).status,
            "superseded",
        )
        self.assertEqual(
            self.state.load_active_meal_plan(DAY, 2, meal_slot="lunch").plan_id,
            replacement_dispatch.plan_id,
        )
        self.assertTrue(
            self.state.is_active_meal_plan_reportable(
                replacement_dispatch.persisted_plan,
                DEFAULT_PHELPS_SERVICE_CALENDAR,
                evaluated_at=datetime(
                    2026, 9, 4, 12, tzinfo=NUTRITION_APPLICATION_TIMEZONE
                ),
            )
        )

        class FailingSchedulerPreparer:
            calls = 0

            def prepare_scheduled(self, service_date, meal, supplied_targets, *, meal_slot):
                del service_date, meal, supplied_targets
                self.calls += 1
                raise AssertionError("delivered scheduler work must not be prepared again")

        weekly = {weekday: () for weekday in range(7)}
        scheduler_preparer = FailingSchedulerPreparer()
        scheduler_sender = StableSender()
        scheduler = RecommendationScheduler(
            self.state,
            schedule=RecommendationDispatchSchedule(
                weekly,
                date_overrides={DAY: (ScheduledMealOpportunity(2, time(12), "lunch"),)},
            ),
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=NutritionApplicationClock(
                lambda: datetime(2026, 9, 4, 12, tzinfo=NUTRITION_APPLICATION_TIMEZONE)
            ),
            recommendation_preparer=scheduler_preparer,
            target_loader=targets,
            outbound_sender=scheduler_sender,
            chat_guid=CHAT_GUID,
        )
        rerun = scheduler.run_once()
        self.assertEqual(rerun.decisions[0].outcome, "already_delivered")
        self.assertEqual(scheduler_preparer.calls, 0)
        self.assertEqual(scheduler_sender.calls, [])
        self.assertEqual(
            self.state.load_meal_plan(replacement_dispatch.plan_id).status,
            "active",
        )

        dinner_plan = MealPlan(
            DAY,
            3,
            (PlannedMealItem(self._resolved("Dinner Chicken", meal=3), Decimal("1"), "one portion"),),
        )
        self.state.save_scheduled_meal_plan(dinner_plan)
        dinner_dispatch = self.state.load_scheduled_recommendation_dispatch(DAY, 3)
        assert dinner_dispatch is not None
        self.assertTrue(self.state.claim_scheduled_recommendation_delivery(dinner_dispatch))
        self.state.mark_scheduled_recommendation_delivered(dinner_dispatch)
        self.assertEqual(
            self.state.load_meal_plan(replacement.persisted_plan.plan_id).status,
            "active",
        )
        self.assertEqual(
            {
                plan.meal_slot
                for plan in self.state.list_reportable_active_meal_plans(
                    DAY,
                    DEFAULT_PHELPS_SERVICE_CALENDAR,
                    evaluated_at=datetime(
                        2026, 9, 4, 17, tzinfo=NUTRITION_APPLICATION_TIMEZONE
                    ),
                )
            },
            {"lunch", "dinner"},
        )


if __name__ == "__main__":
    unittest.main()
