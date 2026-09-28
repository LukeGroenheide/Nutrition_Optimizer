"""Deterministic Phelps availability and stale-plan lifecycle coverage."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory
from contextlib import redirect_stdout
from unittest.mock import patch

from nutrition_optimizer.application import (
    MealRecommendationOrchestrator,
    MealRecommendationServiceUnavailableError,
    MealReportOrchestrator,
)
from nutrition_optimizer.application_clock import NutritionApplicationClock
from nutrition_optimizer.durable_state import DurableMealState, MealReportApplicationError
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import FoodResolutionRequest, LocalFDFoodResolver, ResolvedFood
from nutrition_optimizer.meal_report import (
    MealPlan,
    MealReportReconciler,
    PlannedMealItem,
    ProposedEatenMealItem,
    ReconciledMealReport,
)
from nutrition_optimizer.messaging.meal_workflow import (
    ActiveMealPlanContextResolver,
    ManualMealRecommendationDelivery,
    MealPlanResendError,
    NoActiveMealPlanContextError,
    main as meal_workflow_main,
)
from nutrition_optimizer.meal_report_structured import (
    MealReportSemanticResult,
    ReportedMealItem,
)
from nutrition_optimizer.nutrition import DailyMinimums, DailyTargets
from nutrition_optimizer.portion_interpretation import NaturalPortionInterpreter
from nutrition_optimizer.phelps_service_calendar import DEFAULT_PHELPS_SERVICE_CALENDAR
from tests.test_food_resolution import component, mapped, menu_day


SATURDAY = date(2026, 8, 29)
SUNDAY = date(2026, 8, 30)
MONDAY = date(2026, 8, 31)


def targets() -> DailyTargets:
    return DailyTargets(
        calories_kcal=Decimal("2000"),
        protein_g=Decimal("100"),
        carbohydrates_g=Decimal("250"),
        fat_g=Decimal("70"),
        minimums=DailyMinimums(Decimal("25")),
    )


class NeverInterpretMealReport:
    """Assert that durable replay does not invoke semantics a second time."""

    def __init__(self) -> None:
        self.calls = 0

    def interpret(self, plan: MealPlan, user_text: str) -> object:
        self.calls += 1
        raise AssertionError("semantic interpretation must not run for durable replay")


class FullyEatenMealReport:
    """Deterministic semantic fixture for one accepted after-close report."""

    def interpret(self, plan: MealPlan, user_text: str) -> MealReportSemanticResult:
        return MealReportSemanticResult(
            planned_items=tuple(
                ReportedMealItem(
                    plan_item_id=plan.item_id(item),
                    reference_text=item.display_name,
                    action="eaten",
                    quantity_relation="as_recommended",
                    quantity_text=None,
                )
                for item in plan.items
            ),
            additional_foods=(),
            unresolved_statements=(),
        )


class NeverSendOutbound:
    """Assert an after-close manual resend is rejected before transport."""

    def send_text(self, chat_guid: str, message: str) -> object:
        raise AssertionError("after-close resend must not reach transport")


class PhelpsServiceCalendarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog_path = Path(self.directory.name) / "nutrition.sqlite3"
        self.catalog = OfficialNutritionCatalog(self.catalog_path)
        self.addCleanup(self.catalog.close)
        self.state = DurableMealState(self.catalog)
        self._next_component_id = 1000

    @staticmethod
    def instant(
        year: int,
        month: int,
        day: int,
        hour: int,
        minute: int = 0,
    ) -> datetime:
        return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)

    @staticmethod
    def clock_at(instant: datetime) -> NutritionApplicationClock:
        return NutritionApplicationClock(lambda: instant)

    def seed_plan(
        self,
        service_date: date,
        meal_id: int,
        *,
        plan_id: str,
        name: str | None = None,
        scheduled: bool = False,
        delivered_at: datetime | None = None,
    ):
        food_name = name or f"Food {plan_id}"
        component_id = self._next_component_id
        self._next_component_id += 1
        self.catalog.synchronize_fd_refresh(
            mapped(
                menu_day(
                    service_date,
                    meal_id,
                    {1: "Breakfast", 2: "Lunch", 3: "Dinner"}[meal_id],
                    ((component(component_id, food_name), "Test Station"),),
                )
            ),
            requested_start=service_date,
            requested_end=service_date,
            observed_at=self.instant(2026, 8, 25, 12),
        )
        resolved = LocalFDFoodResolver(self.catalog).resolve(
            FoodResolutionRequest(food_name, service_date, meal=meal_id)
        )
        assert isinstance(resolved, ResolvedFood)
        meal_plan = MealPlan(
            service_date,
            meal_id,
            (PlannedMealItem(resolved, Decimal("1"), "one serving"),),
        )
        if not scheduled:
            return self.state.save_meal_plan(meal_plan, plan_id=plan_id)
        if delivered_at is None:
            raise ValueError("scheduled test plans require delivered_at")
        persisted = self.state.save_scheduled_meal_plan(
            meal_plan,
            plan_id=plan_id,
            delivery_token=f"{plan_id}-delivery-token",
        )
        dispatch = self.state.load_scheduled_recommendation_dispatch(service_date, meal_id)
        assert dispatch is not None
        self.assertTrue(
            self.state.claim_scheduled_recommendation_delivery(
                dispatch,
                started_at=delivered_at,
            )
        )
        delivered = self.state.mark_scheduled_recommendation_delivered(
            dispatch,
            delivered_at=delivered_at,
        )
        self.assertEqual(delivered.plan_id, persisted.plan_id)
        return delivered.persisted_plan

    @staticmethod
    def fully_eaten_report(plan: MealPlan) -> ReconciledMealReport:
        item = plan.items[0]
        return ReconciledMealReport(
            plan,
            (
                ProposedEatenMealItem(
                    item,
                    item.recommended_official_servings,
                    "planned_quantity",
                    item.food.nutrition_record.name,
                ),
            ),
            (),
            (),
            (),
            (),
        )

    def test_previous_date_plan_is_retired_transactionally_and_history_is_preserved(self) -> None:
        plan = self.seed_plan(SUNDAY, 3, plan_id="sunday-dinner")

        retirement = self.state.retire_stale_active_meal_plans(
            DEFAULT_PHELPS_SERVICE_CALENDAR,
            evaluated_at=self.instant(2026, 8, 31, 12),
        )

        self.assertEqual(retirement.retired_plan_ids, (plan.plan_id,))
        historical = self.state.load_meal_plan(plan.plan_id)
        assert historical is not None
        self.assertEqual(historical.status, "superseded")
        self.assertEqual(historical.plan, plan.plan)
        self.assertEqual(
            self.catalog._connection.execute(
                "SELECT COUNT(*) FROM meal_plan_items WHERE plan_id = ?",
                (plan.plan_id,),
            ).fetchone()[0],
            1,
        )

    def test_current_weekday_plan_remains_active_while_phelps_is_open(self) -> None:
        plan = self.seed_plan(MONDAY, 3, plan_id="monday-dinner")

        retirement = self.state.retire_stale_active_meal_plans(
            DEFAULT_PHELPS_SERVICE_CALENDAR,
            evaluated_at=self.instant(2026, 8, 31, 18),  # 14:00 EDT
        )

        self.assertEqual(retirement.retired_plan_ids, ())
        loaded = self.state.load_meal_plan(plan.plan_id)
        assert loaded is not None
        self.assertEqual(loaded.status, "active")

    def test_current_sunday_dinner_is_retired_after_its_authoritative_window(self) -> None:
        plan = self.seed_plan(SUNDAY, 3, plan_id="sunday-after-close")

        retirement = self.state.retire_stale_active_meal_plans(
            DEFAULT_PHELPS_SERVICE_CALENDAR,
            evaluated_at=self.instant(2026, 8, 30, 23, 1),  # 19:01 EDT
        )

        self.assertEqual(retirement.retired_plan_ids, (plan.plan_id,))
        loaded = self.state.load_meal_plan(plan.plan_id)
        assert loaded is not None
        self.assertEqual(loaded.status, "superseded")

    def test_delivered_weekday_dinner_survives_close_and_resolves_for_reporting(self) -> None:
        dinner = self.seed_plan(
            MONDAY,
            3,
            plan_id="scheduled-weekday-dinner",
            scheduled=True,
            delivered_at=self.instant(2026, 8, 31, 21, 30),  # 17:30 EDT
        )
        after_close = self.instant(2026, 9, 1, 0, 1)  # 20:01 EDT

        retirement = self.state.retire_stale_active_meal_plans(
            DEFAULT_PHELPS_SERVICE_CALENDAR,
            evaluated_at=after_close,
        )
        resolver = ActiveMealPlanContextResolver(
            self.state,
            clock=self.clock_at(self.instant(2026, 9, 1, 0, 41)),  # 20:41 EDT
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
        )

        self.assertEqual(retirement.retired_plan_ids, ())
        self.assertEqual(
            self.state.list_stale_active_meal_plans(
                DEFAULT_PHELPS_SERVICE_CALENDAR,
                evaluated_at=after_close,
            ),
            (),
        )
        current = self.state.load_meal_plan(dinner.plan_id)
        assert current is not None
        self.assertEqual(current.status, "active")
        self.assertEqual(resolver.resolve("after-close-dinner-report").plan_id, dinner.plan_id)

    def test_after_close_delivered_scheduled_dinner_applies_and_replays_after_rollover(self) -> None:
        dinner = self.seed_plan(
            MONDAY,
            3,
            plan_id="scheduled-after-close-application",
            scheduled=True,
            delivered_at=self.instant(2026, 8, 31, 21, 30),
        )
        after_close_clock = self.clock_at(self.instant(2026, 9, 1, 0, 41))
        reconciler = MealReportReconciler(
            FullyEatenMealReport(),
            LocalFDFoodResolver(self.catalog),
            NaturalPortionInterpreter(),
        )
        orchestrator = MealReportOrchestrator(
            self.catalog,
            reconciler,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=after_close_clock,
        )

        applied = orchestrator.process(
            MONDAY,
            3,
            "I ate what you recommended.",
            "after-close-dinner-event",
        )
        replay_after_rollover = MealReportOrchestrator(
            self.catalog,
            reconciler,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=self.clock_at(self.instant(2026, 9, 1, 12)),  # 08:00 EDT Tuesday
        ).process(
            MONDAY,
            3,
            "same durable event",
            "after-close-dinner-event",
        )

        self.assertEqual(applied.outcome, "applied")
        self.assertEqual(applied.persisted_plan.plan_id, dinner.plan_id)
        self.assertEqual(replay_after_rollover.outcome, "replayed")
        self.assertTrue(replay_after_rollover.already_applied)
        completed = self.state.load_meal_plan(dinner.plan_id)
        assert completed is not None
        self.assertEqual(completed.status, "applied")

    def test_after_close_delivered_scheduled_plan_cannot_be_manually_resent(self) -> None:
        dinner = self.seed_plan(
            MONDAY,
            3,
            plan_id="scheduled-after-close-no-resend",
            scheduled=True,
            delivered_at=self.instant(2026, 8, 31, 21, 30),
        )
        delivery = ManualMealRecommendationDelivery(
            NeverSendOutbound(),
            "chat-guid",
            MealRecommendationOrchestrator(self.catalog),
            self.state,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=self.clock_at(self.instant(2026, 9, 1, 0, 41)),
        )

        with self.assertRaisesRegex(MealPlanResendError, "not eligible"):
            delivery.resend_active_plan(dinner.plan_id)

        current = self.state.load_meal_plan(dinner.plan_id)
        assert current is not None
        self.assertEqual(current.status, "active")

    def test_delivered_sunday_brunch_remains_reportable_after_dinner_delivery(self) -> None:
        brunch = self.seed_plan(
            SUNDAY,
            2,
            plan_id="scheduled-sunday-brunch",
            scheduled=True,
            delivered_at=self.instant(2026, 8, 30, 15),  # 11:00 EDT
        )
        after_brunch = self.instant(2026, 8, 30, 17, 31)  # 13:31 EDT

        retirement = self.state.retire_stale_active_meal_plans(
            DEFAULT_PHELPS_SERVICE_CALENDAR,
            evaluated_at=after_brunch,
        )
        before_dinner = self.state.load_meal_plan(brunch.plan_id)
        assert before_dinner is not None
        self.assertEqual(retirement.retired_plan_ids, ())
        self.assertEqual(before_dinner.status, "active")

        dinner = self.seed_plan(
            SUNDAY,
            3,
            plan_id="scheduled-sunday-dinner",
            scheduled=True,
            delivered_at=self.instant(2026, 8, 30, 21, 30),  # 17:30 EDT
        )

        active_brunch = self.state.load_meal_plan(brunch.plan_id)
        active_dinner = self.state.load_meal_plan(dinner.plan_id)
        assert active_brunch is not None
        assert active_dinner is not None
        self.assertEqual(active_brunch.status, "active")
        self.assertEqual(active_dinner.status, "active")
        self.assertEqual(
            {
                plan.meal_slot
                for plan in self.state.list_reportable_active_meal_plans(
                    SUNDAY,
                    DEFAULT_PHELPS_SERVICE_CALENDAR,
                    evaluated_at=self.instant(2026, 8, 30, 21, 30),
                )
            },
            {"brunch", "dinner"},
        )

    def test_delivered_saturday_breakfast_remains_reportable_after_window_close(self) -> None:
        breakfast = self.seed_plan(
            SATURDAY,
            1,
            plan_id="scheduled-saturday-breakfast",
            scheduled=True,
            delivered_at=self.instant(2026, 8, 29, 12),  # 08:00 EDT
        )
        after_breakfast = self.instant(2026, 8, 29, 13, 31)  # 09:31 EDT

        self.assertFalse(
            DEFAULT_PHELPS_SERVICE_CALENDAR.is_meal_context_eligible_at(
                SATURDAY, 1, after_breakfast
            )
        )
        self.assertTrue(
            self.state.is_active_meal_plan_reportable(
                breakfast,
                DEFAULT_PHELPS_SERVICE_CALENDAR,
                evaluated_at=after_breakfast,
            )
        )
        self.assertEqual(
            self.state.retire_stale_active_meal_plans(
                DEFAULT_PHELPS_SERVICE_CALENDAR,
                evaluated_at=after_breakfast,
            ).retired_plan_ids,
            (),
        )

    def test_reportability_ends_at_exact_detroit_midnight_without_deleting_history(self) -> None:
        dinner = self.seed_plan(
            SUNDAY,
            3,
            plan_id="scheduled-midnight-boundary",
            scheduled=True,
            delivered_at=self.instant(2026, 8, 30, 21, 30),
        )
        before_midnight = datetime(2026, 8, 31, 3, 59, 59, tzinfo=timezone.utc)
        at_midnight = datetime(2026, 8, 31, 4, 0, 0, tzinfo=timezone.utc)

        self.assertTrue(
            self.state.is_active_meal_plan_reportable(
                dinner,
                DEFAULT_PHELPS_SERVICE_CALENDAR,
                evaluated_at=before_midnight,
            )
        )
        self.assertFalse(
            self.state.is_active_meal_plan_reportable(
                dinner,
                DEFAULT_PHELPS_SERVICE_CALENDAR,
                evaluated_at=at_midnight,
            )
        )
        historical = self.state.load_meal_plan(dinner.plan_id)
        assert historical is not None
        self.assertEqual(historical.status, "active")

    def test_applied_slot_stays_authoritative_and_is_not_reportable_again(self) -> None:
        breakfast = self.seed_plan(
            MONDAY,
            1,
            plan_id="scheduled-applied-breakfast",
            scheduled=True,
            delivered_at=self.instant(2026, 8, 31, 12),
        )
        report = self.fully_eaten_report(breakfast.plan)
        first = self.state.apply_reconciled_meal_report(
            breakfast,
            report,
            source_event_id="accepted-breakfast-guid",
            recorded_at=self.instant(2026, 8, 31, 12, 15),
        )
        lunch = self.seed_plan(
            MONDAY,
            2,
            plan_id="scheduled-after-applied-lunch",
            scheduled=True,
            delivered_at=self.instant(2026, 8, 31, 16),
        )

        replay = self.state.apply_reconciled_meal_report(
            breakfast,
            report,
            source_event_id="accepted-breakfast-guid",
            recorded_at=self.instant(2026, 8, 31, 16, 5),
        )
        self.assertFalse(first.already_applied)
        self.assertTrue(replay.already_applied)
        self.assertEqual(
            self.catalog._connection.execute(
                "SELECT COUNT(*) FROM meal_report_applications"
            ).fetchone()[0],
            1,
        )
        self.assertEqual(
            self.catalog._connection.execute(
                "SELECT COUNT(*) FROM accepted_intake_entries"
            ).fetchone()[0],
            1,
        )
        self.assertEqual(
            tuple(
                plan.plan_id
                for plan in self.state.list_reportable_active_meal_plans(
                    MONDAY,
                    DEFAULT_PHELPS_SERVICE_CALENDAR,
                    evaluated_at=self.instant(2026, 8, 31, 16, 5),
                )
            ),
            (lunch.plan_id,),
        )

    def test_delivered_sunday_dinner_survives_facility_close_but_retires_on_rollover(self) -> None:
        dinner = self.seed_plan(
            SUNDAY,
            3,
            plan_id="scheduled-sunday-after-close",
            scheduled=True,
            delivered_at=self.instant(2026, 8, 30, 21, 30),
        )

        after_close = self.state.retire_stale_active_meal_plans(
            DEFAULT_PHELPS_SERVICE_CALENDAR,
            evaluated_at=self.instant(2026, 8, 30, 23, 1),  # 19:01 EDT
        )
        after_rollover = self.state.retire_stale_active_meal_plans(
            DEFAULT_PHELPS_SERVICE_CALENDAR,
            evaluated_at=self.instant(2026, 8, 31, 12),  # 08:00 EDT Monday
        )

        self.assertEqual(after_close.retired_plan_ids, ())
        self.assertEqual(after_rollover.retired_plan_ids, (dinner.plan_id,))
        retired = self.state.load_meal_plan(dinner.plan_id)
        assert retired is not None
        self.assertEqual(retired.status, "superseded")

    def test_retirement_is_idempotent(self) -> None:
        plan = self.seed_plan(SUNDAY, 3, plan_id="idempotent-sunday")
        evaluated_at = self.instant(2026, 8, 31, 12)

        first = self.state.retire_stale_active_meal_plans(
            DEFAULT_PHELPS_SERVICE_CALENDAR,
            evaluated_at=evaluated_at,
        )
        second = self.state.retire_stale_active_meal_plans(
            DEFAULT_PHELPS_SERVICE_CALENDAR,
            evaluated_at=evaluated_at,
        )

        self.assertEqual(first.retired_plan_ids, (plan.plan_id,))
        self.assertEqual(second.retired_plan_ids, ())

    def test_retiring_multiple_stale_plans_rolls_back_as_one_transaction(self) -> None:
        first = self.seed_plan(SATURDAY, 1, plan_id="rollback-first")
        second = self.seed_plan(SUNDAY, 3, plan_id="rollback-second")
        self.catalog._connection.execute(
            """
            CREATE TRIGGER reject_second_stale_retirement
            BEFORE UPDATE OF status ON meal_plans
            WHEN OLD.plan_id = 'rollback-second' AND NEW.status = 'superseded'
            BEGIN
                SELECT RAISE(ABORT, 'test retirement failure');
            END
            """
        )
        self.catalog._connection.commit()

        with self.assertRaisesRegex(MealReportApplicationError, "retirement failed"):
            self.state.retire_stale_active_meal_plans(
                DEFAULT_PHELPS_SERVICE_CALENDAR,
                evaluated_at=self.instant(2026, 8, 31, 12),
            )

        for plan in (first, second):
            with self.subTest(plan_id=plan.plan_id):
                loaded = self.state.load_meal_plan(plan.plan_id)
                assert loaded is not None
                self.assertEqual(loaded.status, "active")

    def test_recommendation_preparation_rejects_before_open_and_after_close(self) -> None:
        self.seed_plan(MONDAY, 1, plan_id="existing-breakfast", name="Breakfast Food")
        before_open = MealRecommendationOrchestrator(
            self.catalog,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=self.clock_at(self.instant(2026, 8, 31, 11)),  # 07:00 EDT
        )

        with self.assertRaises(MealRecommendationServiceUnavailableError):
            before_open.prepare(MONDAY, 1, targets())

        self.assertEqual(
            self.catalog._connection.execute("SELECT COUNT(*) FROM meal_plans").fetchone()[0],
            1,
        )
        after_close = MealRecommendationOrchestrator(
            self.catalog,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=self.clock_at(self.instant(2026, 9, 1, 1)),  # 21:00 EDT Monday
        )

        with self.assertRaises(MealRecommendationServiceUnavailableError):
            after_close.prepare(MONDAY, 1, targets())

        self.assertEqual(
            self.catalog._connection.execute("SELECT COUNT(*) FROM meal_plans").fetchone()[0],
            1,
        )

    def test_continuous_weekday_hours_do_not_invent_meal_transitions(self) -> None:
        inside_continuous_dining = self.instant(2026, 8, 31, 18)  # 14:00 EDT Monday

        availability = DEFAULT_PHELPS_SERVICE_CALENDAR.availability_at(
            inside_continuous_dining
        )
        self.assertTrue(availability.is_open)
        self.assertEqual([window.meal_id for window in availability.windows], [None])
        for meal_id in (1, 2, 3):
            with self.subTest(meal_id=meal_id):
                self.assertTrue(
                    DEFAULT_PHELPS_SERVICE_CALENDAR.is_meal_context_eligible_at(
                        MONDAY,
                        meal_id,
                        inside_continuous_dining,
                    )
                )
                self.assertFalse(
                    DEFAULT_PHELPS_SERVICE_CALENDAR.has_plan_become_stale(
                        MONDAY,
                        meal_id,
                        inside_continuous_dining,
                    )
                )

    def test_saturday_uses_explicit_published_fd_meal_windows(self) -> None:
        breakfast = self.instant(2026, 8, 29, 12)  # 08:00 EDT
        lunch = self.instant(2026, 8, 29, 15)  # 11:00 EDT
        dinner = self.instant(2026, 8, 29, 21, 30)  # 17:30 EDT

        self.assertTrue(
            DEFAULT_PHELPS_SERVICE_CALENDAR.is_meal_context_eligible_at(
                SATURDAY, 1, breakfast
            )
        )
        self.assertFalse(
            DEFAULT_PHELPS_SERVICE_CALENDAR.is_meal_context_eligible_at(
                SATURDAY, 2, breakfast
            )
        )
        self.assertTrue(
            DEFAULT_PHELPS_SERVICE_CALENDAR.is_meal_context_eligible_at(
                SATURDAY, 2, lunch
            )
        )
        self.assertTrue(
            DEFAULT_PHELPS_SERVICE_CALENDAR.is_meal_context_eligible_at(
                SATURDAY, 3, dinner
            )
        )
        self.assertTrue(
            DEFAULT_PHELPS_SERVICE_CALENDAR.has_plan_become_stale(
                SATURDAY,
                1,
                self.instant(2026, 8, 29, 14, 30),  # 10:30 EDT
            )
        )

    def test_sunday_brunch_uses_current_fd_lunch_period_mapping(self) -> None:
        brunch = self.instant(2026, 8, 30, 15)  # 11:00 EDT
        dinner = self.instant(2026, 8, 30, 21, 30)  # 17:30 EDT

        self.assertTrue(
            DEFAULT_PHELPS_SERVICE_CALENDAR.is_meal_context_eligible_at(SUNDAY, 2, brunch)
        )
        self.assertFalse(
            DEFAULT_PHELPS_SERVICE_CALENDAR.is_meal_context_eligible_at(SUNDAY, 1, brunch)
        )
        self.assertFalse(
            DEFAULT_PHELPS_SERVICE_CALENDAR.is_meal_context_eligible_at(SUNDAY, 3, brunch)
        )
        self.assertTrue(
            DEFAULT_PHELPS_SERVICE_CALENDAR.is_meal_context_eligible_at(SUNDAY, 3, dinner)
        )

    def test_replay_lookup_happens_before_stale_retirement(self) -> None:
        applied_plan = self.seed_plan(SUNDAY, 3, plan_id="applied-sunday-dinner")
        self.state.apply_reconciled_meal_report(
            applied_plan,
            self.fully_eaten_report(applied_plan.plan),
            source_event_id="historical-applied-event",
        )
        untouched_active = self.seed_plan(SUNDAY, 2, plan_id="other-sunday-lunch")
        resolver = ActiveMealPlanContextResolver(
            self.state,
            clock=self.clock_at(self.instant(2026, 8, 31, 12)),
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
        )

        resolved = resolver.resolve("historical-applied-event")

        self.assertEqual(resolved.plan_id, applied_plan.plan_id)
        active_after_replay = self.state.load_meal_plan(untouched_active.plan_id)
        assert active_after_replay is not None
        self.assertEqual(active_after_replay.status, "active")

    def test_orchestrator_replay_is_safe_after_rollover_without_semantic_work(self) -> None:
        applied_plan = self.seed_plan(SUNDAY, 3, plan_id="orchestrator-applied")
        self.state.apply_reconciled_meal_report(
            applied_plan,
            self.fully_eaten_report(applied_plan.plan),
            source_event_id="orchestrator-historical-event",
        )
        untouched_active = self.seed_plan(SUNDAY, 2, plan_id="orchestrator-other-active")
        interpreter = NeverInterpretMealReport()
        orchestrator = MealReportOrchestrator(
            self.catalog,
            MealReportReconciler(
                interpreter,
                LocalFDFoodResolver(self.catalog),
                NaturalPortionInterpreter(),
            ),
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            clock=self.clock_at(self.instant(2026, 8, 31, 12)),
        )

        result = orchestrator.process(
            SUNDAY,
            3,
            "ignored durable replay text",
            "orchestrator-historical-event",
        )

        self.assertEqual(result.outcome, "replayed")
        self.assertEqual(interpreter.calls, 0)
        active_after_replay = self.state.load_meal_plan(untouched_active.plan_id)
        assert active_after_replay is not None
        self.assertEqual(active_after_replay.status, "active")

    def test_new_event_never_attaches_to_a_stale_historical_plan(self) -> None:
        stale_plan = self.seed_plan(SUNDAY, 3, plan_id="new-event-stale-plan")
        resolver = ActiveMealPlanContextResolver(
            self.state,
            clock=self.clock_at(self.instant(2026, 8, 31, 12)),
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
        )

        with self.assertRaises(NoActiveMealPlanContextError):
            resolver.resolve("new-event-after-rollover")

        retired = self.state.load_meal_plan(stale_plan.plan_id)
        assert retired is not None
        self.assertEqual(retired.status, "superseded")

    def test_dst_uses_zoneinfo_for_summer_and_winter_service_windows(self) -> None:
        summer = DEFAULT_PHELPS_SERVICE_CALENDAR.availability_at(
            self.instant(2026, 7, 15, 12)
        )
        winter = DEFAULT_PHELPS_SERVICE_CALENDAR.availability_at(
            self.instant(2026, 1, 15, 13)
        )

        self.assertEqual(summer.local_now.tzname(), "EDT")
        self.assertTrue(summer.is_open)
        self.assertEqual(winter.local_now.tzname(), "EST")
        self.assertTrue(winter.is_open)

    def test_service_status_is_read_only_and_reports_stale_active_plan(self) -> None:
        plan = self.seed_plan(SUNDAY, 3, plan_id="diagnostic-stale-plan")
        output = StringIO()
        clock = self.clock_at(self.instant(2026, 8, 31, 12))

        with patch(
            "nutrition_optimizer.messaging.meal_workflow.DEFAULT_NUTRITION_APPLICATION_CLOCK",
            clock,
        ), redirect_stdout(output):
            result = meal_workflow_main(
                ["service-status", "--catalog-path", str(self.catalog_path)]
            )

        self.assertEqual(result, 0)
        self.assertIn("Detroit now: 2026-08-31T08:00:00-04:00", output.getvalue())
        self.assertIn(plan.plan_id, output.getvalue())
        loaded = self.state.load_meal_plan(plan.plan_id)
        assert loaded is not None
        self.assertEqual(loaded.status, "active")

    def test_explicit_retirement_command_only_changes_stale_plan_lifecycle(self) -> None:
        plan = self.seed_plan(SUNDAY, 3, plan_id="command-stale-plan")
        output = StringIO()
        clock = self.clock_at(self.instant(2026, 8, 31, 12))

        with patch(
            "nutrition_optimizer.messaging.meal_workflow.DEFAULT_NUTRITION_APPLICATION_CLOCK",
            clock,
        ), redirect_stdout(output):
            result = meal_workflow_main(
                ["retire-stale-plans", "--catalog-path", str(self.catalog_path)]
            )

        self.assertEqual(result, 0)
        self.assertIn("count=1", output.getvalue())
        loaded = self.state.load_meal_plan(plan.plan_id)
        assert loaded is not None
        self.assertEqual(loaded.status, "superseded")
        self.assertEqual(
            self.catalog._connection.execute(
                "SELECT COUNT(*) FROM meal_report_applications"
            ).fetchone()[0],
            0,
        )
        self.assertEqual(
            self.catalog._connection.execute(
                "SELECT COUNT(*) FROM accepted_intake_entries"
            ).fetchone()[0],
            0,
        )


if __name__ == "__main__":
    unittest.main()
