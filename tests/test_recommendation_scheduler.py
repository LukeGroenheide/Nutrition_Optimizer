"""Offline coverage for durable automatic recommendation dispatch."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier, Thread
import unittest
from unittest.mock import patch

from nutrition_optimizer.application import MealRecommendationOrchestrator
from nutrition_optimizer.application_clock import (
    NUTRITION_APPLICATION_TIMEZONE,
    NutritionApplicationClock,
)
from nutrition_optimizer.durable_state import (
    DraftClarification,
    DurableMealState,
    MealReportApplicationError,
)
from nutrition_optimizer.fdmealplanner import CATALOG_SCHEMA_VERSION, OfficialNutritionCatalog
from nutrition_optimizer.nutrition import DailyMinimums, DailyTargets
from nutrition_optimizer.phelps_service_calendar import DEFAULT_PHELPS_SERVICE_CALENDAR
from nutrition_optimizer.recommendation_scheduler import (
    RecommendationDispatchSchedule,
    RecommendationScheduler,
    ScheduledMealOpportunity,
)
from nutrition_optimizer.messaging.meal_workflow import (
    ActiveMealPlanContextResolver,
    AmbiguousMealPlanContextError,
)
from tests.test_food_resolution import component, mapped, menu_day


MONDAY = date(2026, 8, 31)
SATURDAY = date(2026, 8, 29)
SUNDAY = date(2026, 8, 30)


def targets() -> DailyTargets:
    return DailyTargets(
        calories_kcal=Decimal("1000"),
        protein_g=Decimal("100"),
        carbohydrates_g=Decimal("100"),
        fat_g=Decimal("50"),
        minimums=DailyMinimums(Decimal("10")),
    )


def food(component_id: int, name: str) -> dict[str, object]:
    value = component(component_id, name, protein="20")
    value.update(
        {
            "recipePortionSize": "1",
            "recipePortionSizeUnit": "Each",
            "calories": "100",
            "protein": "20",
            "carbohydrates": "10",
            "fat": "5",
            "dietaryFiber": "2",
            "dietaryFiberUOM": "g",
        }
    )
    return value


def local_instant(service_date: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(
        service_date,
        time(hour, minute),
        tzinfo=NUTRITION_APPLICATION_TIMEZONE,
    )


class FakeSender:
    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.calls: list[tuple[str, str, str | None]] = []

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
            raise RuntimeError("known test send failure")
        return object()


class FailingPreparer:
    def prepare_scheduled(self, service_date, meal, supplied_targets, *, meal_slot):
        raise RuntimeError("forced preparation failure")


class RecommendationSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "state" / "nutrition.sqlite3"
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self.catalog.close)
        self.state = DurableMealState(self.catalog)

    def sync(self, service_date: date, *meal_ids: int) -> None:
        names = {1: "Breakfast", 2: "Lunch", 3: "Dinner"}
        self.catalog.synchronize_fd_refresh(
            mapped(
                *(
                    menu_day(
                        service_date,
                        meal_id,
                        names[meal_id],
                        ((food(1000 + meal_id, f"{names[meal_id]} Food"), "Grill"),),
                    )
                    for meal_id in meal_ids
                )
            ),
            requested_start=service_date,
            requested_end=service_date,
            observed_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        )

    def plan_count(self) -> int:
        return int(self.catalog._connection.execute("SELECT COUNT(*) FROM meal_plans").fetchone()[0])

    def scheduler_at(
        self,
        instant: datetime,
        sender: FakeSender,
        *,
        schedule: RecommendationDispatchSchedule | None = None,
        preparer: object | None = None,
        target_loader=None,
        is_known_delivery_failure=None,
    ) -> RecommendationScheduler:
        clock = NutritionApplicationClock(lambda: instant.astimezone(timezone.utc))
        actual_preparer = preparer or MealRecommendationOrchestrator(
            self.catalog,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=clock,
        )
        return RecommendationScheduler(
            self.state,
            schedule=schedule or RecommendationDispatchSchedule(),
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=clock,
            recommendation_preparer=actual_preparer,  # type: ignore[arg-type]
            target_loader=target_loader or targets,
            outbound_sender=sender,
            chat_guid="chat-guid",
            is_known_delivery_failure=is_known_delivery_failure,
        )

    @staticmethod
    def result_for(run, meal_id: int):
        return next(result for result in run.decisions if result.opportunity.meal_id == meal_id)

    def test_product_activation_boundaries_are_centralized_and_exact(self) -> None:
        schedule = RecommendationDispatchSchedule()

        expected_times = {
            MONDAY: {1: time(7, 30), 2: time(11, 30), 3: time(17)},
            SATURDAY: {1: time(7, 30), 2: time(10, 30), 3: time(17)},
            SUNDAY: {2: time(10, 30), 3: time(17)},
        }
        for service_date, meal_times in expected_times.items():
            opportunities = schedule.opportunities_for(service_date)
            self.assertEqual(
                {opportunity.meal_id: opportunity.activation_time for opportunity in opportunities},
                meal_times,
            )
            for meal_id, activation in meal_times.items():
                opportunity = next(
                    opportunity
                    for opportunity in opportunities
                    if opportunity.meal_id == meal_id
                )
                before = (datetime.combine(service_date, activation) - timedelta(minutes=1)).time()
                before_instant = local_instant(service_date, before.hour, before.minute)
                activation_instant = local_instant(service_date, activation.hour, activation.minute)
                self.assertEqual(
                    schedule.timing_at(service_date, opportunity, before_instant),
                    "not_due",
                )
                self.assertEqual(
                    schedule.timing_at(service_date, opportunity, activation_instant),
                    "due",
                )

        self.assertEqual([item.meal_id for item in schedule.opportunities_for(SUNDAY)], [2, 3])

    def test_weekday_breakfast_due_persists_one_plan_and_sends_once(self) -> None:
        self.sync(MONDAY, 1)
        sender = FakeSender()

        run = self.scheduler_at(local_instant(MONDAY, 7, 30), sender).run_once()

        breakfast = self.result_for(run, 1)
        self.assertEqual(breakfast.outcome, "dispatched")
        self.assertEqual(len(sender.calls), 1)
        self.assertEqual(self.plan_count(), 1)
        dispatch = self.state.load_scheduled_recommendation_dispatch(MONDAY, 1)
        assert dispatch is not None
        self.assertEqual(dispatch.status, "delivered")
        self.assertEqual(dispatch.persisted_plan.status, "active")
        self.assertEqual(sender.calls[0][2], dispatch.delivery_token)

    def test_too_early_does_not_prepare_even_when_phelps_is_open(self) -> None:
        self.sync(MONDAY, 1)
        sender = FakeSender()

        run = self.scheduler_at(local_instant(MONDAY, 7, 29), sender).run_once()

        self.assertEqual(self.result_for(run, 1).outcome, "not_due")
        self.assertEqual(sender.calls, [])
        self.assertEqual(self.plan_count(), 0)

    def test_within_sixty_minute_catch_up_window_dispatches_once(self) -> None:
        self.sync(MONDAY, 1)
        sender = FakeSender()

        run = self.scheduler_at(local_instant(MONDAY, 7, 50), sender).run_once()

        self.assertEqual(self.result_for(run, 1).outcome, "dispatched")
        self.assertEqual(len(sender.calls), 1)

    def test_after_catch_up_window_does_not_prepare_or_send(self) -> None:
        self.sync(MONDAY, 1)
        sender = FakeSender()

        run = self.scheduler_at(local_instant(MONDAY, 8, 31), sender).run_once()

        self.assertEqual(self.result_for(run, 1).outcome, "too_late")
        self.assertEqual(sender.calls, [])
        self.assertEqual(self.plan_count(), 0)

    def test_repeated_successful_invocation_never_replaces_or_resends(self) -> None:
        self.sync(MONDAY, 1)
        sender = FakeSender()
        scheduler = self.scheduler_at(local_instant(MONDAY, 7, 35), sender)

        first = scheduler.run_once()
        second = scheduler.run_once()

        self.assertEqual(self.result_for(first, 1).outcome, "dispatched")
        self.assertEqual(self.result_for(second, 1).outcome, "already_delivered")
        self.assertEqual(len(sender.calls), 1)
        self.assertEqual(self.plan_count(), 1)

    def test_restart_against_same_database_does_not_duplicate_delivery(self) -> None:
        self.sync(MONDAY, 1)
        first_sender = FakeSender()
        first = self.scheduler_at(local_instant(MONDAY, 7, 30), first_sender).run_once()
        first_plan_id = self.result_for(first, 1).plan_id
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self.catalog.close)
        self.state = DurableMealState(self.catalog)
        second_sender = FakeSender()

        second = self.scheduler_at(local_instant(MONDAY, 7, 50), second_sender).run_once()

        self.assertEqual(self.result_for(second, 1).outcome, "already_delivered")
        self.assertEqual(self.result_for(second, 1).plan_id, first_plan_id)
        self.assertEqual(second_sender.calls, [])
        self.assertEqual(self.plan_count(), 1)

    def test_concurrent_scheduler_processes_create_one_plan_and_one_send(self) -> None:
        self.sync(MONDAY, 1)
        self.catalog.close()
        sender = FakeSender()
        barrier = Barrier(2)
        results: list[object] = []
        failures: list[BaseException] = []

        def invoke() -> None:
            try:
                with OfficialNutritionCatalog(self.path) as catalog:
                    state = DurableMealState(catalog)
                    clock = NutritionApplicationClock(
                        lambda: local_instant(MONDAY, 7, 30).astimezone(timezone.utc)
                    )
                    scheduler = RecommendationScheduler(
                        state,
                        clock=clock,
                        service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
                        recommendation_preparer=MealRecommendationOrchestrator(
                            catalog,
                            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
                            clock=clock,
                        ),
                        target_loader=targets,
                        outbound_sender=sender,
                        chat_guid="chat-guid",
                    )
                    barrier.wait(timeout=5)
                    results.append(scheduler.run_once())
            except BaseException as exc:  # keep thread failures visible below
                failures.append(exc)

        threads = [Thread(target=invoke), Thread(target=invoke)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(failures, [])
        self.assertEqual(len(results), 2)
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self.catalog.close)
        self.state = DurableMealState(self.catalog)
        self.assertEqual(self.plan_count(), 1)
        self.assertEqual(len(sender.calls), 1)

    def test_known_send_failure_retries_the_same_persisted_plan_and_message(self) -> None:
        self.sync(MONDAY, 1)
        sender = FakeSender(failures=1)

        first = self.scheduler_at(local_instant(MONDAY, 7, 30), sender).run_once()
        dispatch_after_failure = self.state.load_scheduled_recommendation_dispatch(MONDAY, 1)
        assert dispatch_after_failure is not None
        self.assertTrue(all(item.presentation_binding is not None
                            for item in dispatch_after_failure.persisted_plan.plan.items))
        # Retrying must not ask any renderer/current calibration to regenerate it.
        with patch('nutrition_optimizer.recommendation_rendering.render_meal_recommendation',
                   side_effect=AssertionError('retry must use frozen plan')):
            second = self.scheduler_at(local_instant(MONDAY, 7, 40), sender).run_once()
        final_dispatch = self.state.load_scheduled_recommendation_dispatch(MONDAY, 1)
        assert final_dispatch is not None

        self.assertEqual(self.result_for(first, 1).outcome, "delivery_failed")
        self.assertEqual(dispatch_after_failure.status, "pending_delivery")
        self.assertEqual(self.result_for(second, 1).outcome, "dispatched")
        self.assertEqual(final_dispatch.status, "delivered")
        self.assertEqual(self.plan_count(), 1)
        self.assertEqual(len(sender.calls), 2)
        self.assertEqual(sender.calls[0][1:], sender.calls[1][1:])
        self.assertEqual(sender.calls[0][2], final_dispatch.delivery_token)

    def test_repeated_delivery_failures_do_not_create_duplicate_plans(self) -> None:
        self.sync(MONDAY, 1)
        sender = FakeSender(failures=2)
        scheduler = self.scheduler_at(local_instant(MONDAY, 7, 30), sender)

        first = scheduler.run_once()
        second = scheduler.run_once()

        self.assertEqual(self.result_for(first, 1).outcome, "delivery_failed")
        self.assertEqual(self.result_for(second, 1).outcome, "delivery_failed")
        self.assertEqual(self.plan_count(), 1)
        dispatch = self.state.load_scheduled_recommendation_dispatch(MONDAY, 1)
        assert dispatch is not None
        self.assertEqual(dispatch.status, "pending_delivery")
        self.assertEqual(len(sender.calls), 2)

    def test_expired_unsent_dispatch_retires_its_orphan_active_plan(self) -> None:
        self.sync(MONDAY, 1)
        sender = FakeSender(failures=1)

        first = self.scheduler_at(local_instant(MONDAY, 7, 30), sender).run_once()
        plan_id = self.result_for(first, 1).plan_id
        late = self.scheduler_at(local_instant(MONDAY, 8, 31), sender).run_once()

        self.assertEqual(self.result_for(first, 1).outcome, "delivery_failed")
        self.assertEqual(self.result_for(late, 1).outcome, "expired")
        assert plan_id is not None
        dispatch = self.state.load_scheduled_recommendation_dispatch(MONDAY, 1)
        retired = self.state.load_meal_plan(plan_id)
        assert dispatch is not None
        assert retired is not None
        self.assertEqual(dispatch.status, "expired")
        self.assertEqual(retired.status, "superseded")
        self.assertIsNone(self.state.load_active_meal_plan(MONDAY, 1))
        self.assertEqual(len(sender.calls), 1)

    def test_uncertain_post_send_state_never_automatically_sends_twice(self) -> None:
        self.sync(MONDAY, 1)
        sender = FakeSender()
        scheduler = self.scheduler_at(local_instant(MONDAY, 7, 30), sender)

        with patch.object(
            self.state,
            "mark_scheduled_recommendation_delivered",
            side_effect=MealReportApplicationError("forced durable acknowledgement failure"),
        ):
            first = scheduler.run_once()
        second = scheduler.run_once()

        self.assertEqual(self.result_for(first, 1).outcome, "delivery_state_unknown")
        self.assertEqual(self.result_for(second, 1).outcome, "delivery_in_progress")
        self.assertTrue(first.has_operational_failure)
        self.assertEqual(len(sender.calls), 1)
        dispatch = self.state.load_scheduled_recommendation_dispatch(MONDAY, 1)
        assert dispatch is not None
        self.assertEqual(dispatch.status, "sending")

    def test_ambiguous_transport_error_is_held_for_operator_not_resent(self) -> None:
        self.sync(MONDAY, 1)
        sender = FakeSender(failures=1)
        scheduler = self.scheduler_at(
            local_instant(MONDAY, 7, 30),
            sender,
            is_known_delivery_failure=lambda exc: False,
        )

        first = scheduler.run_once()
        second = scheduler.run_once()

        self.assertEqual(self.result_for(first, 1).outcome, "delivery_state_unknown")
        self.assertEqual(self.result_for(second, 1).outcome, "delivery_in_progress")
        self.assertEqual(len(sender.calls), 1)

    def test_manual_active_plan_conflict_is_never_replaced_or_sent(self) -> None:
        self.sync(MONDAY, 1)
        sender = FakeSender()
        clock = NutritionApplicationClock(lambda: local_instant(MONDAY, 7, 30).astimezone(timezone.utc))
        manual = MealRecommendationOrchestrator(
            self.catalog,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=clock,
        ).prepare(MONDAY, 1, targets())

        run = self.scheduler_at(local_instant(MONDAY, 7, 30), sender).run_once()

        self.assertEqual(self.result_for(run, 1).outcome, "manual_plan_conflict")
        self.assertEqual(self.result_for(run, 1).plan_id, manual.plan_id)
        self.assertEqual(self.plan_count(), 1)
        self.assertEqual(sender.calls, [])
        self.assertIsNone(self.state.load_scheduled_recommendation_dispatch(MONDAY, 1))

    def test_sunday_omits_breakfast_and_dispatches_brunch_then_dinner(self) -> None:
        self.sync(SUNDAY, 2, 3)
        sender = FakeSender()

        brunch = self.scheduler_at(local_instant(SUNDAY, 10, 30), sender).run_once()
        dinner = self.scheduler_at(local_instant(SUNDAY, 17), sender).run_once()

        self.assertEqual([item.opportunity.meal_id for item in brunch.decisions], [2, 3])
        self.assertEqual(self.result_for(brunch, 2).outcome, "dispatched")
        self.assertEqual(self.result_for(dinner, 3).outcome, "dispatched")
        self.assertEqual(len(sender.calls), 2)
        self.assertEqual(self.plan_count(), 2)
        brunch_dispatch = self.state.load_scheduled_recommendation_dispatch(
            SUNDAY, 2, meal_slot="brunch"
        )
        assert brunch_dispatch is not None
        self.assertEqual(brunch_dispatch.meal_slot, "brunch")
        self.assertEqual(brunch_dispatch.persisted_plan.plan.meal, 2)
        reportable = self.state.list_reportable_active_meal_plans(
            SUNDAY,
            DEFAULT_PHELPS_SERVICE_CALENDAR,
            evaluated_at=local_instant(SUNDAY, 17),
        )
        self.assertEqual({plan.meal_slot for plan in reportable}, {"brunch", "dinner"})

    def test_saturday_explicit_meal_windows_allow_all_three_configured_opportunities(self) -> None:
        self.sync(SATURDAY, 1, 2, 3)
        sender = FakeSender()

        breakfast = self.scheduler_at(local_instant(SATURDAY, 7, 30), sender).run_once()
        lunch = self.scheduler_at(local_instant(SATURDAY, 10, 30), sender).run_once()
        dinner = self.scheduler_at(local_instant(SATURDAY, 17), sender).run_once()

        self.assertEqual(self.result_for(breakfast, 1).outcome, "dispatched")
        self.assertEqual(self.result_for(lunch, 2).outcome, "dispatched")
        self.assertEqual(self.result_for(dinner, 3).outcome, "dispatched")
        self.assertEqual(len(sender.calls), 3)

    def test_later_scheduled_delivery_keeps_earlier_slot_reportable(self) -> None:
        self.sync(MONDAY, 1, 2)
        sender = FakeSender()

        breakfast = self.scheduler_at(local_instant(MONDAY, 7, 30), sender).run_once()
        lunch = self.scheduler_at(local_instant(MONDAY, 11, 30), sender).run_once()

        breakfast_plan_id = self.result_for(breakfast, 1).plan_id
        lunch_plan_id = self.result_for(lunch, 2).plan_id
        assert breakfast_plan_id is not None
        assert lunch_plan_id is not None
        active_breakfast = self.state.load_meal_plan(breakfast_plan_id)
        active_lunch = self.state.load_meal_plan(lunch_plan_id)
        assert active_breakfast is not None
        assert active_lunch is not None
        self.assertEqual(active_breakfast.status, "active")
        self.assertEqual(active_lunch.status, "active")
        reportable = self.state.list_reportable_active_meal_plans(
            MONDAY,
            DEFAULT_PHELPS_SERVICE_CALENDAR,
            evaluated_at=local_instant(MONDAY, 11, 30),
        )
        self.assertEqual({plan.meal_slot for plan in reportable}, {"breakfast", "lunch"})

        resolver = ActiveMealPlanContextResolver(
            self.state,
            clock=NutritionApplicationClock(
                lambda: local_instant(MONDAY, 11, 30).astimezone(timezone.utc)
            ),
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
        )
        with self.assertRaises(AmbiguousMealPlanContextError):
            resolver.resolve("ambiguous-breakfast-or-lunch-report")

    def test_breakfast_lunch_and_dinner_remain_independently_reportable(self) -> None:
        self.sync(MONDAY, 1, 2, 3)
        sender = FakeSender()

        for hour, minute in ((7, 30), (11, 30), (17, 0)):
            self.scheduler_at(local_instant(MONDAY, hour, minute), sender).run_once()

        reportable = self.state.list_reportable_active_meal_plans(
            MONDAY,
            DEFAULT_PHELPS_SERVICE_CALENDAR,
            evaluated_at=local_instant(MONDAY, 17),
        )
        self.assertEqual(
            {plan.meal_slot for plan in reportable},
            {"breakfast", "lunch", "dinner"},
        )
        self.assertTrue(all(plan.status == "active" for plan in reportable))

    def test_lunch_delivery_does_not_mutate_an_open_breakfast_draft(self) -> None:
        self.sync(MONDAY, 1, 2)
        sender = FakeSender()
        breakfast_result = self.scheduler_at(
            local_instant(MONDAY, 7, 30), sender
        ).run_once()
        breakfast_plan_id = self.result_for(breakfast_result, 1).plan_id
        assert breakfast_plan_id is not None
        breakfast = self.state.load_meal_plan(breakfast_plan_id)
        assert breakfast is not None
        draft, _ = self.state.save_meal_report_draft(
            chat_guid="chat-guid",
            persisted_plan=breakfast,
            current_draft=None,
            planned_items=(),
            unplanned_items=(),
            clarifications=(DraftClarification("completion_required"),),
            last_prompt="Anything else?",
            source_event_id="breakfast-draft-guid",
            intent="meal_report",
            processed_at=local_instant(MONDAY, 8).astimezone(timezone.utc),
        )

        self.scheduler_at(local_instant(MONDAY, 11, 30), sender).run_once()

        reloaded = self.state.load_meal_report_draft(draft.draft_id)
        assert reloaded is not None
        self.assertEqual(reloaded, draft)
        self.assertEqual(reloaded.plan_id, breakfast_plan_id)
        self.assertEqual(self.state.load_meal_plan(breakfast_plan_id).status, "active")
        self.assertEqual(
            {
                plan.meal_slot
                for plan in self.state.list_reportable_active_meal_plans(
                    MONDAY,
                    DEFAULT_PHELPS_SERVICE_CALENDAR,
                    evaluated_at=local_instant(MONDAY, 11, 30),
                )
            },
            {"breakfast", "lunch"},
        )

    def test_delivered_weekday_dinner_remains_reportable_without_resend_after_close(self) -> None:
        self.sync(MONDAY, 3)
        sender = FakeSender()

        delivered = self.scheduler_at(local_instant(MONDAY, 17), sender).run_once()
        dinner_plan_id = self.result_for(delivered, 3).plan_id
        assert dinner_plan_id is not None
        after_close_runs = tuple(
            self.scheduler_at(local_instant(MONDAY, hour, minute), sender).run_once()
            for hour, minute in ((20, 1), (20, 6), (20, 11))
        )
        status = self.scheduler_at(local_instant(MONDAY, 20, 11), sender).status()

        self.assertEqual(len(sender.calls), 1)
        for run in after_close_runs:
            self.assertEqual(run.stale_retirement.retired_plan_ids, ())
            self.assertEqual(self.result_for(run, 3).outcome, "already_delivered")
        current = self.state.load_meal_plan(dinner_plan_id)
        dispatch = self.state.load_scheduled_recommendation_dispatch(MONDAY, 3)
        dinner_status = next(item for item in status.items if item.opportunity.meal_id == 3)
        assert current is not None
        assert dispatch is not None
        self.assertEqual(current.status, "active")
        self.assertEqual(dispatch.status, "delivered")
        self.assertFalse(dinner_status.phelps_eligible)
        self.assertTrue(dinner_status.reportable)
        self.assertEqual(dinner_status.dispatch_status, "delivered")
        self.assertEqual(dinner_status.active_plan_id, dinner_plan_id)

    def test_failed_later_scheduled_delivery_keeps_earlier_delivered_plan_reportable(self) -> None:
        self.sync(MONDAY, 1, 2)
        breakfast_sender = FakeSender()
        breakfast = self.scheduler_at(local_instant(MONDAY, 7, 30), breakfast_sender).run_once()
        breakfast_plan_id = self.result_for(breakfast, 1).plan_id
        assert breakfast_plan_id is not None
        lunch_sender = FakeSender(failures=1)

        lunch = self.scheduler_at(local_instant(MONDAY, 11, 30), lunch_sender).run_once()
        lunch_dispatch = self.state.load_scheduled_recommendation_dispatch(MONDAY, 2)
        resolver = ActiveMealPlanContextResolver(
            self.state,
            clock=NutritionApplicationClock(
                lambda: local_instant(MONDAY, 11, 30).astimezone(timezone.utc)
            ),
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
        )

        self.assertEqual(self.result_for(lunch, 2).outcome, "delivery_failed")
        assert lunch_dispatch is not None
        self.assertEqual(lunch_dispatch.status, "pending_delivery")
        breakfast_plan = self.state.load_meal_plan(breakfast_plan_id)
        assert breakfast_plan is not None
        self.assertEqual(breakfast_plan.status, "active")
        self.assertEqual(resolver.resolve("lunch-delivery-failed-report").plan_id, breakfast_plan_id)

    def test_unknown_later_delivery_keeps_earlier_delivered_plan_reportable(self) -> None:
        self.sync(MONDAY, 1, 2)
        breakfast_sender = FakeSender()
        breakfast = self.scheduler_at(local_instant(MONDAY, 7, 30), breakfast_sender).run_once()
        breakfast_plan_id = self.result_for(breakfast, 1).plan_id
        assert breakfast_plan_id is not None
        lunch_sender = FakeSender(failures=1)

        lunch = self.scheduler_at(
            local_instant(MONDAY, 11, 30),
            lunch_sender,
            is_known_delivery_failure=lambda exc: False,
        ).run_once()
        lunch_dispatch = self.state.load_scheduled_recommendation_dispatch(MONDAY, 2)
        resolver = ActiveMealPlanContextResolver(
            self.state,
            clock=NutritionApplicationClock(
                lambda: local_instant(MONDAY, 11, 30).astimezone(timezone.utc)
            ),
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
        )

        self.assertEqual(self.result_for(lunch, 2).outcome, "delivery_state_unknown")
        assert lunch_dispatch is not None
        self.assertEqual(lunch_dispatch.status, "sending")
        self.assertEqual(resolver.resolve("lunch-delivery-unknown-report").plan_id, breakfast_plan_id)

    def test_due_but_phelps_closed_never_creates_a_plan(self) -> None:
        self.sync(MONDAY, 1)
        late = RecommendationDispatchSchedule(
            date_overrides={
                MONDAY: (ScheduledMealOpportunity(1, time(20, 5), "breakfast"),)
            }
        )
        sender = FakeSender()

        run = self.scheduler_at(local_instant(MONDAY, 20, 5), sender, schedule=late).run_once()

        self.assertEqual(run.decisions[0].outcome, "unavailable")
        self.assertEqual(run.decisions[0].diagnostic_reason, "phelps_service_unavailable")
        self.assertEqual(self.plan_count(), 0)
        self.assertEqual(sender.calls, [])

    def test_missing_menu_fails_safely_without_plan_or_send(self) -> None:
        sender = FakeSender()

        run = self.scheduler_at(local_instant(MONDAY, 7, 30), sender).run_once()

        breakfast = self.result_for(run, 1)
        self.assertEqual(breakfast.outcome, "unavailable")
        self.assertEqual(breakfast.diagnostic_reason, "menu_coverage_missing")
        self.assertEqual(self.plan_count(), 0)
        self.assertEqual(sender.calls, [])

    def test_target_loading_and_preparation_failures_never_create_delivery_state(self) -> None:
        self.sync(MONDAY, 1)
        target_failure = self.scheduler_at(
            local_instant(MONDAY, 7, 30),
            FakeSender(),
            target_loader=lambda: (_ for _ in ()).throw(RuntimeError("targets unavailable")),
        ).run_once()
        self.assertEqual(self.result_for(target_failure, 1).outcome, "target_loading_failed")
        self.assertEqual(self.plan_count(), 0)

        preparation_failure = self.scheduler_at(
            local_instant(MONDAY, 7, 30),
            FakeSender(),
            preparer=FailingPreparer(),
        ).run_once()
        self.assertEqual(self.result_for(preparation_failure, 1).outcome, "preparation_failed")
        self.assertEqual(self.plan_count(), 0)
        self.assertIsNone(self.state.load_scheduled_recommendation_dispatch(MONDAY, 1))

    def test_scheduler_retires_prior_date_active_plan_before_current_dispatch(self) -> None:
        previous = MONDAY.replace(day=30)
        self.sync(previous, 1)
        self.sync(MONDAY, 1)
        old_clock = NutritionApplicationClock(
            lambda: local_instant(MONDAY, 7, 30).astimezone(timezone.utc)
        )
        old_plan = MealRecommendationOrchestrator(
            self.catalog,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=old_clock,
        ).prepare(previous, 1, targets())
        sender = FakeSender()

        run = self.scheduler_at(local_instant(MONDAY, 7, 30), sender).run_once()

        self.assertIn(old_plan.plan_id, run.stale_retirement.retired_plan_ids)
        retired = self.state.load_meal_plan(old_plan.plan_id)
        assert retired is not None
        self.assertEqual(retired.status, "superseded")
        self.assertEqual(self.result_for(run, 1).outcome, "dispatched")

    def test_status_is_read_only_and_reports_detroit_service_date(self) -> None:
        self.sync(MONDAY, 1)
        # 2026-09-01 UTC is still Aug. 31 in Detroit.  The host timezone is
        # irrelevant because the injected clock starts with an absolute UTC instant.
        utc_boundary = datetime(2026, 9, 1, 0, 15, tzinfo=timezone.utc)
        sender = FakeSender()
        scheduler = self.scheduler_at(utc_boundary, sender)
        before = self.plan_count()

        status = scheduler.status()

        self.assertEqual(status.local_now.date(), MONDAY)
        self.assertEqual(status.local_now.tzname(), "EDT")
        self.assertEqual(self.plan_count(), before)
        self.assertEqual(sender.calls, [])
        breakfast = next(item for item in status.items if item.opportunity.meal_id == 1)
        self.assertEqual(breakfast.timing, "too_late")
        self.assertIsNone(breakfast.dispatch_status)

    def test_schedule_uses_zoneinfo_for_summer_and_winter_not_host_offsets(self) -> None:
        schedule = RecommendationDispatchSchedule()
        summer = datetime(2026, 8, 31, 12, tzinfo=timezone.utc)
        winter = datetime(2026, 1, 5, 13, tzinfo=timezone.utc)

        self.assertEqual(NutritionApplicationClock(lambda: summer).now().tzname(), "EDT")
        self.assertEqual(NutritionApplicationClock(lambda: winter).now().tzname(), "EST")
        self.assertEqual(
            schedule.timing_at(
                MONDAY,
                schedule.opportunities_for(MONDAY)[0],
                local_instant(MONDAY, 7, 30),
            ),
            "due",
        )

    def test_schema_v5_migration_adds_dispatch_table_without_losing_plan_history(self) -> None:
        self.sync(MONDAY, 1)
        clock = NutritionApplicationClock(lambda: local_instant(MONDAY, 7, 30).astimezone(timezone.utc))
        existing = MealRecommendationOrchestrator(
            self.catalog,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=clock,
        ).prepare(MONDAY, 1, targets())
        self.catalog.close()
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("DROP TABLE scheduled_recommendation_dispatches")
            connection.execute("PRAGMA user_version = 5")
            connection.commit()
        finally:
            connection.close()

        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self.catalog.close)
        self.state = DurableMealState(self.catalog)

        self.assertEqual(
            self.catalog._connection.execute("PRAGMA user_version").fetchone()[0],
            CATALOG_SCHEMA_VERSION,
        )
        tables = {
            row[0]
            for row in self.catalog._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        self.assertIn("scheduled_recommendation_dispatches", tables)
        restored = self.state.load_meal_plan(existing.plan_id)
        self.assertEqual(restored, existing.persisted_plan)

    def test_read_only_catalog_refuses_a_v5_schema_without_migrating_it(self) -> None:
        self.catalog.close()
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("DROP TABLE scheduled_recommendation_dispatches")
            connection.execute("PRAGMA user_version = 5")
            connection.commit()
        finally:
            connection.close()

        with self.assertRaisesRegex(Exception, "without migration"):
            OfficialNutritionCatalog(self.path, read_only=True)

        connection = sqlite3.connect(self.path)
        try:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 5)
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            self.assertNotIn("scheduled_recommendation_dispatches", tables)
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
