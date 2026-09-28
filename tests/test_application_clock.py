"""Deterministic coverage for Nutrition Optimizer's Detroit service-date clock."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from nutrition_optimizer.application_clock import (
    DEFAULT_NUTRITION_APPLICATION_CLOCK,
    NUTRITION_APPLICATION_TIMEZONE_NAME,
    NutritionApplicationClock,
)
from nutrition_optimizer.durable_state import DurableMealState
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import (
    FoodResolutionRequest,
    LocalFDFoodResolver,
    ResolvedFood,
)
from nutrition_optimizer.meal_report import (
    MealPlan,
    PlannedMealItem,
    ProposedEatenMealItem,
    ReconciledMealReport,
)
from nutrition_optimizer.messaging.meal_workflow import (
    ActiveMealPlanContextResolver,
    AmbiguousMealPlanContextError,
    NoActiveMealPlanContextError,
)
from tests.test_food_resolution import component, mapped, menu_day


SERVICE_DATE = date(2026, 8, 30)
NEXT_SERVICE_DATE = date(2026, 8, 31)
UTC_MIDNIGHT_BOUNDARY = datetime(2026, 8, 31, 0, 45, tzinfo=timezone.utc)


class NutritionApplicationClockTests(unittest.TestCase):
    def clock_at(self, instant: datetime) -> NutritionApplicationClock:
        return NutritionApplicationClock(lambda: instant)

    def test_utc_midnight_uses_the_prior_detroit_service_date(self) -> None:
        local = self.clock_at(UTC_MIDNIGHT_BOUNDARY).now()

        self.assertEqual(local.isoformat(), "2026-08-30T20:45:00-04:00")
        self.assertEqual(local.tzname(), "EDT")
        self.assertEqual(self.clock_at(UTC_MIDNIGHT_BOUNDARY).service_date(), SERVICE_DATE)

    def test_later_utc_instant_uses_august_31_in_detroit(self) -> None:
        instant = datetime(2026, 8, 31, 5, 45, tzinfo=timezone.utc)

        self.assertEqual(self.clock_at(instant).service_date(), NEXT_SERVICE_DATE)

    def test_dst_summer_uses_edt_without_a_hardcoded_offset_in_production(self) -> None:
        local = self.clock_at(datetime(2026, 7, 15, 12, tzinfo=timezone.utc)).now()

        self.assertEqual(local.tzname(), "EDT")
        self.assertEqual(local.utcoffset(), timedelta(hours=-4))

    def test_dst_winter_uses_est_without_a_hardcoded_offset_in_production(self) -> None:
        local = self.clock_at(datetime(2026, 1, 15, 12, tzinfo=timezone.utc)).now()

        self.assertEqual(local.tzname(), "EST")
        self.assertEqual(local.utcoffset(), timedelta(hours=-5))

    def test_same_absolute_instant_has_the_same_service_date_from_any_source_timezone(self) -> None:
        instant = UTC_MIDNIGHT_BOUNDARY
        source_in_another_timezone = instant.astimezone(ZoneInfo("Pacific/Auckland"))

        self.assertEqual(
            self.clock_at(instant).service_date(),
            self.clock_at(source_in_another_timezone).service_date(),
        )

    def test_default_clock_is_explicitly_detroit_and_rejects_naive_instants(self) -> None:
        self.assertEqual(
            DEFAULT_NUTRITION_APPLICATION_CLOCK.now().tzinfo.key,
            NUTRITION_APPLICATION_TIMEZONE_NAME,
        )
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            self.clock_at(datetime(2026, 8, 31, 0, 45)).service_date()


class ActiveMealPlanContextClockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog = OfficialNutritionCatalog(Path(self.directory.name) / "nutrition.sqlite3")
        self.addCleanup(self.catalog.close)
        observed_at = datetime(2026, 8, 30, 12, tzinfo=timezone.utc)
        self.catalog.synchronize_fd_refresh(
            mapped(
                menu_day(
                    SERVICE_DATE,
                    1,
                    "Breakfast",
                    ((component(1, "Breakfast Eggs"), "Grill"),),
                ),
                menu_day(
                    SERVICE_DATE,
                    3,
                    "Dinner",
                    ((component(2, "Dinner Chicken"), "Grill"),),
                ),
            ),
            requested_start=SERVICE_DATE,
            requested_end=SERVICE_DATE,
            observed_at=observed_at,
        )
        self.state = DurableMealState(self.catalog)
        dinner_food = self._food("Dinner Chicken", "Dinner")
        self.dinner_plan = self.state.save_meal_plan(
            MealPlan(
                SERVICE_DATE,
                "Dinner",
                (PlannedMealItem(dinner_food, Decimal("1"), "one chicken"),),
            ),
            plan_id="detroit-midnight-dinner",
        )

    def _food(self, name: str, meal: str) -> ResolvedFood:
        result = LocalFDFoodResolver(self.catalog).resolve(
            FoodResolutionRequest(name, SERVICE_DATE, meal=meal)
        )
        assert isinstance(result, ResolvedFood)
        return result

    def _fully_eaten_report(self) -> ReconciledMealReport:
        plan = self.dinner_plan.plan
        item = plan.items[0]
        return ReconciledMealReport(
            plan,
            (
                ProposedEatenMealItem(
                    item,
                    item.recommended_official_servings,
                    "planned_quantity",
                    "dinner chicken",
                ),
            ),
            (),
            (),
            (),
            (),
        )

    def test_utc_midnight_resolves_the_active_detroit_dinner_plan(self) -> None:
        resolver = ActiveMealPlanContextResolver(
            self.state,
            clock=NutritionApplicationClock(lambda: UTC_MIDNIGHT_BOUNDARY),
        )

        self.assertEqual(resolver.resolve("utc-midnight-event"), self.dinner_plan)

    def test_default_resolver_uses_the_detroit_application_clock(self) -> None:
        clock = NutritionApplicationClock(lambda: UTC_MIDNIGHT_BOUNDARY)
        with patch(
            "nutrition_optimizer.messaging.meal_workflow.DEFAULT_NUTRITION_APPLICATION_CLOCK",
            clock,
        ):
            resolver = ActiveMealPlanContextResolver(self.state)

        self.assertEqual(resolver.resolve("default-clock-event"), self.dinner_plan)

    def test_later_boundary_has_no_august_31_plan(self) -> None:
        resolver = ActiveMealPlanContextResolver(
            self.state,
            clock=NutritionApplicationClock(
                lambda: datetime(2026, 8, 31, 5, 45, tzinfo=timezone.utc)
            ),
        )

        with self.assertRaises(NoActiveMealPlanContextError):
            resolver.resolve("later-boundary-event")

    def test_explicit_service_date_provider_remains_authoritative(self) -> None:
        resolver = ActiveMealPlanContextResolver(
            self.state,
            service_date_provider=lambda: SERVICE_DATE,
        )

        self.assertEqual(resolver.resolve("explicit-service-date-event"), self.dinner_plan)

    def test_existing_ambiguity_behavior_is_unchanged(self) -> None:
        breakfast_food = self._food("Breakfast Eggs", "Breakfast")
        self.state.save_meal_plan(
            MealPlan(
                SERVICE_DATE,
                "Breakfast",
                (PlannedMealItem(breakfast_food, Decimal("1"), "two eggs"),),
            ),
            plan_id="detroit-midnight-breakfast",
        )
        resolver = ActiveMealPlanContextResolver(
            self.state,
            clock=NutritionApplicationClock(lambda: UTC_MIDNIGHT_BOUNDARY),
        )

        with self.assertRaises(AmbiguousMealPlanContextError):
            resolver.resolve("ambiguous-detroit-event")

    def test_clock_and_explicit_provider_cannot_be_combined(self) -> None:
        with self.assertRaisesRegex(ValueError, "service_date_provider or clock"):
            ActiveMealPlanContextResolver(
                self.state,
                service_date_provider=lambda: SERVICE_DATE,
                clock=NutritionApplicationClock(lambda: UTC_MIDNIGHT_BOUNDARY),
            )

    def test_replay_is_resolved_before_a_later_current_service_date(self) -> None:
        self.state.apply_reconciled_meal_report(
            self.dinner_plan,
            self._fully_eaten_report(),
            source_event_id="august-30-applied-event",
        )
        resolver = ActiveMealPlanContextResolver(
            self.state,
            clock=NutritionApplicationClock(
                lambda: datetime(2026, 8, 31, 5, 45, tzinfo=timezone.utc)
            ),
        )

        replayed = resolver.resolve("august-30-applied-event")

        self.assertEqual(replayed.plan_id, self.dinner_plan.plan_id)
        self.assertEqual(replayed.status, "applied")

    def test_default_audit_timestamps_remain_aware_utc(self) -> None:
        applied = self.state.apply_reconciled_meal_report(
            self.dinner_plan,
            self._fully_eaten_report(),
            source_event_id="utc-audit-event",
        )

        self.assertIs(self.dinner_plan.created_at.tzinfo, timezone.utc)
        self.assertIs(applied.accepted_intake_entries[0].recorded_at.tzinfo, timezone.utc)
        row = self.catalog._connection.execute(
            "SELECT applied_at FROM meal_report_applications WHERE source_event_id = ?",
            ("utc-audit-event",),
        ).fetchone()
        assert row is not None
        self.assertIs(datetime.fromisoformat(str(row[0])).tzinfo, timezone.utc)


if __name__ == "__main__":
    unittest.main()
