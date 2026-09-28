"""Offline coverage for rolling FD menu-cache maintenance."""

from __future__ import annotations

from contextlib import redirect_stdout
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
from io import StringIO
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest

from nutrition_optimizer.application import MealRecommendationOrchestrator
from nutrition_optimizer.application_clock import NutritionApplicationClock
from nutrition_optimizer.durable_state import DurableMealState
from nutrition_optimizer.fdmealplanner import FDMealsPayload, OfficialNutritionCatalog, map_meals_payload
from nutrition_optimizer.fdmealplanner.client import FDMealPlannerError
from nutrition_optimizer.fdmealplanner.menu_cache import (
    DEFAULT_MENU_CACHE_HORIZON_DAYS,
    expected_menu_opportunities,
    main as menu_cache_main,
    menu_cache_date_range,
    menu_coverage_status,
    refresh_menu_cache,
)
from nutrition_optimizer.fdmealplanner.refresh import FDRefreshValidationError
from nutrition_optimizer.nutrition import DailyMinimums, DailyTargets
from nutrition_optimizer.phelps_service_calendar import DEFAULT_PHELPS_SERVICE_CALENDAR
from nutrition_optimizer.recommendation_scheduler import RecommendationScheduler


MONDAY = date(2026, 9, 7)
SATURDAY = date(2026, 9, 12)
SUNDAY = date(2026, 9, 13)
LIVE_START = date(2026, 9, 9)
LIVE_END = date(2026, 9, 15)
CURRENT_LIVE_START = date(2026, 9, 13)
CURRENT_LIVE_END = date(2026, 9, 19)
OBSERVED_NULL_BREAKFAST_STATION_DATES = (
    date(2026, 9, 10),
    date(2026, 9, 11),
    date(2026, 9, 12),
    date(2026, 9, 14),
    date(2026, 9, 15),
)
T1 = datetime(2026, 9, 7, 10, tzinfo=timezone.utc)
T2 = datetime(2026, 9, 7, 11, tzinfo=timezone.utc)


def component(component_id: int, name: str, *, malformed: bool = False) -> dict[str, object]:
    return {
        "componentId": component_id,
        "componentTypeId": 181,
        "englishAlternateName": name,
        "componentName": f"{name} master",
        "recipePortionSize": "not-a-serving" if malformed else "1",
        "recipePortionSizeUnit": "Each",
        "calories": "300",
        "caloriesUOM": "kcal",
        "protein": "20",
        "proteinUOM": "g",
        "carbohydrates": "30",
        "carbohydratesUOM": "g",
        "fat": "10",
        "fatUOM": "g",
        "sodium": "100",
        "sodiumUOM": "mg",
        "dietaryFiber": "2",
        "dietaryFiberUOM": "g",
        "isShowOnMenu": "1",
        "isFoodBar": "0",
    }


def menu_day(
    service_date: date,
    meal_id: int,
    recipes: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "menuForDate": service_date.isoformat(),
        "strMenuForDate": service_date.isoformat(),
        "menuId": 0,
        "mealPeriodId": meal_id,
        "mealPeriodName": {1: "Breakfast", 2: "Lunch", 3: "Dinner"}[meal_id],
        "conceptData": [
            {"rowId": "row", "conceptId": 48, "conceptName": "AMERICAN GRILLE"}
        ],
        "allMenuRecipes": [{**recipe, "rowId": "row"} for recipe in recipes],
    }


class FakeRollingFDClient:
    """Fixture client that proves the maintenance path has no live dependency."""

    def __init__(
        self,
        *,
        omitted: set[tuple[date, int]] | None = None,
        empty: set[tuple[date, int]] | None = None,
        malformed: set[tuple[date, int]] | None = None,
        null_station_placeholders: set[tuple[date, int]] | None = None,
        malformed_station_rows: set[tuple[date, int]] | None = None,
        null_recipe_containers: set[tuple[date, int]] | None = None,
        unserved_meal_placeholders: set[tuple[date, int]] | None = None,
        stationful_unserved_meal_placeholders: set[tuple[date, int]] | None = None,
        schema_changed_recipe_rows: set[tuple[date, int]] | None = None,
        expected_only: bool = False,
    ) -> None:
        self.omitted = omitted or set()
        self.empty = empty or set()
        self.malformed = malformed or set()
        self.null_station_placeholders = null_station_placeholders or set()
        self.malformed_station_rows = malformed_station_rows or set()
        self.null_recipe_containers = null_recipe_containers or set()
        self.unserved_meal_placeholders = unserved_meal_placeholders or set()
        self.stationful_unserved_meal_placeholders = (
            stationful_unserved_meal_placeholders or set()
        )
        self.schema_changed_recipe_rows = schema_changed_recipe_rows or set()
        self.expected_only = expected_only
        self.calls: list[dict[str, object]] = []

    def fetch_phelps_month(self, **kwargs: object) -> FDMealsPayload:
        self.calls.append(kwargs)
        start_date = kwargs["start_date"]
        end_date = kwargs["end_date"]
        meal_id = int(kwargs["meal_period_id"])
        assert isinstance(start_date, date)
        assert isinstance(end_date, date)
        expected = expected_menu_opportunities(start_date, end_date)
        rows: list[dict[str, object]] = []
        current = start_date
        while current <= end_date:
            key = (current, meal_id)
            if key in self.omitted or (
                self.expected_only
                and meal_id not in {opportunity.meal_id for opportunity in expected[current]}
            ):
                if key in self.stationful_unserved_meal_placeholders:
                    placeholder = menu_day(current, meal_id, [])
                    # Sanitized exact Sep. 13 live form: recipe data is absent,
                    # while the response retains structurally valid station metadata.
                    placeholder["conceptData"] = [
                        {"rowId": "american", "conceptId": 1, "conceptName": "AMERICAN GRILLE"},
                        {"rowId": "global", "conceptId": 2, "conceptName": "Global Bowls"},
                        {"rowId": "zone", "conceptId": 3, "conceptName": "ZONE"},
                    ]
                    placeholder["allMenuRecipes"] = None
                    rows.append(placeholder)
                elif key in self.unserved_meal_placeholders:
                    placeholder = menu_day(current, meal_id, [])
                    # Exact live FD form for Sunday fd:1: an explicitly absent
                    # station container and no recipe container at all.
                    placeholder["conceptData"] = None
                    placeholder["allMenuRecipes"] = None
                    rows.append(placeholder)
                current = date.fromordinal(current.toordinal() + 1)
                continue
            recipes: list[dict[str, object]]
            if key in self.empty:
                recipes = []
            else:
                recipes = [
                    component(
                        current.toordinal() * 10 + meal_id,
                        f"FD {current.isoformat()} {meal_id}",
                        malformed=key in self.malformed,
                    )
                ]
            day = menu_day(current, meal_id, recipes)
            if key in self.null_station_placeholders:
                # Exact live FD form: recipe rows remain complete while the
                # optional station container is explicit JSON null.
                day["conceptData"] = None
            if key in self.malformed_station_rows:
                # This is not the harmless null-container form: a malformed
                # station row accompanies otherwise usable recipe data.
                day["conceptData"] = [None]
            if key in self.null_recipe_containers:
                day["allMenuRecipes"] = None
            if key in self.schema_changed_recipe_rows:
                recipes_value = day["allMenuRecipes"]
                assert isinstance(recipes_value, list)
                recipes_value.append(["Potential Entrée", current.toordinal() * 10 + meal_id])
            rows.append(day)
            current = date.fromordinal(current.toordinal() + 1)
        return FDMealsPayload.from_payload({"result": rows})


class FailingFDClient:
    def __init__(self) -> None:
        self.calls = 0

    def fetch_phelps_month(self, **kwargs: object) -> FDMealsPayload:
        del kwargs
        self.calls += 1
        raise FDMealPlannerError("upstream unavailable")


def observed_null_station_fd_client() -> FakeRollingFDClient:
    """Return the sanitized Sep. 9--15 fd:1 payload shape seen live."""

    return FakeRollingFDClient(
        expected_only=True,
        null_station_placeholders={
            (service_date, 1) for service_date in OBSERVED_NULL_BREAKFAST_STATION_DATES
        },
        unserved_meal_placeholders={(SUNDAY, 1)},
    )


def observed_stationful_null_recipe_fd_client() -> FakeRollingFDClient:
    """Return the sanitized Sep. 13--19 payload form observed live."""

    return FakeRollingFDClient(
        expected_only=True,
        stationful_unserved_meal_placeholders={(CURRENT_LIVE_START, 1)},
    )


class FakeSender:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    def send_text(self, chat_guid: str, message: str, *, temp_guid: str | None = None) -> None:
        self.calls.append((chat_guid, message, temp_guid))


def targets() -> DailyTargets:
    return DailyTargets(
        calories_kcal=Decimal("2400"),
        protein_g=Decimal("120"),
        carbohydrates_g=Decimal("300"),
        fat_g=Decimal("80"),
        minimums=DailyMinimums(Decimal("25")),
    )


class MenuCacheMaintenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "state" / "nutrition.sqlite3"

    def seed_current(self, service_date: date, meal_ids: tuple[int, ...]) -> None:
        payload = {
            "result": [
                menu_day(
                    service_date,
                    meal_id,
                    [component(900000 + meal_id, f"Seed {meal_id}")],
                )
                for meal_id in meal_ids
            ]
        }
        with OfficialNutritionCatalog(self.path) as catalog:
            catalog.synchronize_fd_refresh(
                map_meals_payload(payload, tenant_id=7, retrieved_at=T1).results,
                requested_start=service_date,
                requested_end=service_date,
                observed_at=T1,
            )

    def digest(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def table_count(self, table: str) -> int:
        with OfficialNutritionCatalog(self.path, read_only=True) as catalog:
            return int(catalog._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def test_monday_breakfast_gap_is_read_only_and_clearly_missing(self) -> None:
        self.seed_current(MONDAY, (2, 3))
        before = self.digest()

        status = menu_coverage_status(catalog_path=self.path, start_date=MONDAY, days=1)

        self.assertEqual(before, self.digest())
        by_meal = {item.meal_id: item for item in status.items}
        self.assertEqual(by_meal[1].coverage, "missing")
        self.assertTrue(by_meal[1].expected)
        self.assertEqual(by_meal[1].occurrence_count, 0)
        self.assertEqual(by_meal[2].coverage, "present")
        self.assertEqual(by_meal[3].coverage, "present")

    def test_successful_refresh_populates_breakfast_and_targets_default_horizon(self) -> None:
        client = FakeRollingFDClient()

        result = refresh_menu_cache(
            catalog_path=self.path,
            start_date=MONDAY,
            client=client,
            observed_at=T1,
        )

        self.assertEqual(result.refresh.requested_start, MONDAY)
        self.assertEqual(
            result.refresh.requested_end,
            MONDAY + timedelta(days=DEFAULT_MENU_CACHE_HORIZON_DAYS - 1),
        )
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(
            all(call["start_date"] == MONDAY for call in client.calls)
        )
        self.assertTrue(
            all(
                call["end_date"]
                == MONDAY + timedelta(days=DEFAULT_MENU_CACHE_HORIZON_DAYS - 1)
                for call in client.calls
            )
        )
        monday_breakfast = next(
            item
            for item in result.coverage.items
            if item.service_date == MONDAY and item.meal_id == 1
        )
        self.assertEqual(monday_breakfast.coverage, "present")
        self.assertGreater(monday_breakfast.occurrence_count, 0)
        self.assertFalse(result.coverage.missing_expected)

    def test_observed_null_station_containers_map_published_breakfast(self) -> None:
        result = refresh_menu_cache(
            catalog_path=self.path,
            start_date=LIVE_START,
            days=7,
            client=observed_null_station_fd_client(),
            observed_at=T1,
        )

        self.assertEqual(result.refresh.requested_end, LIVE_END)
        by_key = {(item.service_date, item.meal_id): item for item in result.coverage.items}
        self.assertEqual(by_key[(LIVE_START, 1)].coverage, "present")
        for service_date in OBSERVED_NULL_BREAKFAST_STATION_DATES:
            self.assertEqual(by_key[(service_date, 1)].coverage, "present")
            self.assertGreater(by_key[(service_date, 1)].occurrence_count, 0)
        self.assertEqual(by_key[(SUNDAY, 1)].coverage, "not_expected")
        self.assertEqual(by_key[(SUNDAY, 1)].occurrence_count, 0)
        self.assertFalse(result.coverage.missing_expected)

    def test_repeated_identical_refresh_is_idempotent(self) -> None:
        first = refresh_menu_cache(
            catalog_path=self.path,
            start_date=LIVE_START,
            days=7,
            client=observed_null_station_fd_client(),
            observed_at=T1,
        )
        second = refresh_menu_cache(
            catalog_path=self.path,
            start_date=LIVE_START,
            days=7,
            client=observed_null_station_fd_client(),
            observed_at=T2,
        )

        self.assertGreater(first.refresh.occurrence_additions, 0)
        self.assertEqual(second.refresh.catalog_snapshots_inserted, 0)
        self.assertEqual(second.refresh.catalog_new_versions, 0)
        self.assertEqual(
            (
                second.refresh.occurrence_additions,
                second.refresh.occurrence_changes,
                second.refresh.occurrence_removals,
            ),
            (0, 0, 0),
        )

    def test_sunday_breakfast_is_not_expected_and_is_not_missing(self) -> None:
        result = refresh_menu_cache(
            catalog_path=self.path,
            start_date=SUNDAY,
            days=1,
            client=FakeRollingFDClient(expected_only=True),
            observed_at=T1,
        )

        by_meal = {item.meal_id: item for item in result.coverage.items}
        self.assertEqual(by_meal[1].meal_context, "breakfast")
        self.assertFalse(by_meal[1].expected)
        self.assertEqual(by_meal[1].occurrence_count, 0)
        self.assertEqual(by_meal[1].coverage, "not_expected")
        self.assertEqual(by_meal[2].meal_context, "brunch")
        self.assertEqual(by_meal[2].coverage, "present")
        self.assertEqual(by_meal[3].coverage, "present")
        self.assertFalse(result.coverage.missing_expected)

    def test_observed_empty_sunday_breakfast_placeholder_is_not_expected(self) -> None:
        result = refresh_menu_cache(
            catalog_path=self.path,
            start_date=SUNDAY,
            days=1,
            client=FakeRollingFDClient(
                expected_only=True,
                unserved_meal_placeholders={(SUNDAY, 1)},
            ),
            observed_at=T1,
        )

        by_meal = {item.meal_id: item for item in result.coverage.items}
        self.assertFalse(by_meal[1].expected)
        self.assertEqual(by_meal[1].coverage, "not_expected")
        self.assertEqual(by_meal[1].occurrence_count, 0)
        self.assertEqual(by_meal[2].coverage, "present")
        self.assertEqual(by_meal[3].coverage, "present")

    def test_observed_stationful_null_sunday_breakfast_does_not_poison_current_week(self) -> None:
        first = refresh_menu_cache(
            catalog_path=self.path,
            start_date=CURRENT_LIVE_START,
            days=7,
            client=observed_stationful_null_recipe_fd_client(),
            observed_at=T1,
        )

        by_key = {(item.service_date, item.meal_id): item for item in first.coverage.items}
        sunday_breakfast = by_key[(CURRENT_LIVE_START, 1)]
        self.assertFalse(sunday_breakfast.expected)
        self.assertEqual(sunday_breakfast.coverage, "not_expected")
        self.assertEqual(sunday_breakfast.occurrence_count, 0)
        for meal_id in (1, 2, 3):
            saturday = by_key[(CURRENT_LIVE_END, meal_id)]
            self.assertTrue(saturday.expected)
            self.assertEqual(saturday.coverage, "present")
            self.assertGreater(saturday.occurrence_count, 0)
        self.assertFalse(first.coverage.missing_expected)

        second = refresh_menu_cache(
            catalog_path=self.path,
            start_date=CURRENT_LIVE_START,
            days=7,
            client=observed_stationful_null_recipe_fd_client(),
            observed_at=T2,
        )
        self.assertEqual(second.refresh.catalog_snapshots_inserted, 0)
        self.assertEqual(second.refresh.catalog_new_versions, 0)
        self.assertEqual(
            (
                second.refresh.occurrence_additions,
                second.refresh.occurrence_changes,
                second.refresh.occurrence_removals,
            ),
            (0, 0, 0),
        )

    def test_nonexpected_null_recipes_do_not_hide_malformed_station_rows(self) -> None:
        with self.assertRaisesRegex(FDRefreshValidationError, "malformed station rows"):
            refresh_menu_cache(
                catalog_path=self.path,
                start_date=SUNDAY,
                days=1,
                client=FakeRollingFDClient(
                    malformed_station_rows={(SUNDAY, 1)},
                    null_recipe_containers={(SUNDAY, 1)},
                ),
                observed_at=T1,
            )

    def test_schema_changed_recipe_row_with_meaningful_data_still_fails_closed(self) -> None:
        self.seed_current(MONDAY, (1, 2, 3))
        before = self.digest()

        with self.assertRaisesRegex(FDRefreshValidationError, "malformed recipe rows"):
            refresh_menu_cache(
                catalog_path=self.path,
                start_date=MONDAY,
                days=1,
                client=FakeRollingFDClient(
                    schema_changed_recipe_rows={(MONDAY, 1)},
                ),
                observed_at=T2,
            )

        self.assertEqual(self.digest(), before)

    def test_saturday_and_sunday_use_existing_scheduler_opportunities(self) -> None:
        schedule = expected_menu_opportunities(SATURDAY, SUNDAY)

        self.assertEqual(
            [(opportunity.meal_id, opportunity.label) for opportunity in schedule[SATURDAY]],
            [(1, "breakfast"), (2, "lunch"), (3, "dinner")],
        )
        self.assertEqual(
            [(opportunity.meal_id, opportunity.label) for opportunity in schedule[SUNDAY]],
            [(2, "brunch"), (3, "dinner")],
        )

    def test_partial_one_date_response_preserves_preexisting_cache(self) -> None:
        self.seed_current(MONDAY, (1, 2, 3))
        before = self.digest()

        with self.assertRaisesRegex(FDRefreshValidationError, "incomplete"):
            refresh_menu_cache(
                catalog_path=self.path,
                start_date=MONDAY,
                client=FakeRollingFDClient(omitted={(MONDAY, 1)}),
                observed_at=T2,
            )

        self.assertEqual(before, self.digest())
        status = menu_coverage_status(catalog_path=self.path, start_date=MONDAY, days=1)
        self.assertEqual(
            next(item for item in status.items if item.meal_id == 1).coverage,
            "present",
        )

    def test_malformed_visible_recipe_preserves_preexisting_cache(self) -> None:
        self.seed_current(MONDAY, (1, 2, 3))
        before = self.digest()

        with self.assertRaisesRegex(FDRefreshValidationError, "unsafe visible recipe"):
            refresh_menu_cache(
                catalog_path=self.path,
                start_date=MONDAY,
                client=FakeRollingFDClient(malformed={(MONDAY, 1)}),
                observed_at=T2,
            )

        self.assertEqual(before, self.digest())

    def test_malformed_station_row_with_recipe_data_preserves_preexisting_cache(self) -> None:
        self.seed_current(MONDAY, (1, 2, 3))
        before = self.digest()

        with self.assertRaisesRegex(FDRefreshValidationError, "malformed station rows"):
            refresh_menu_cache(
                catalog_path=self.path,
                start_date=MONDAY,
                client=FakeRollingFDClient(malformed_station_rows={(MONDAY, 1)}),
                observed_at=T2,
            )

        self.assertEqual(before, self.digest())

    def test_null_recipe_container_for_expected_breakfast_preserves_preexisting_cache(self) -> None:
        self.seed_current(MONDAY, (1, 2, 3))
        before = self.digest()

        with self.assertRaisesRegex(FDRefreshValidationError, "malformed recipe rows"):
            refresh_menu_cache(
                catalog_path=self.path,
                start_date=MONDAY,
                client=FakeRollingFDClient(
                    null_station_placeholders={(MONDAY, 1)},
                    null_recipe_containers={(MONDAY, 1)},
                ),
                observed_at=T2,
            )

        self.assertEqual(before, self.digest())

    def test_empty_expected_response_never_destructively_clears_cache(self) -> None:
        self.seed_current(MONDAY, (1, 2, 3))
        before = self.digest()

        with self.assertRaisesRegex(FDRefreshValidationError, "incomplete"):
            refresh_menu_cache(
                catalog_path=self.path,
                start_date=MONDAY,
                client=FakeRollingFDClient(empty={(MONDAY, 1)}),
                observed_at=T2,
            )

        self.assertEqual(before, self.digest())

    def test_network_failure_preserves_preexisting_cache(self) -> None:
        self.seed_current(MONDAY, (1, 2, 3))
        before = self.digest()
        client = FailingFDClient()

        with self.assertRaises(FDMealPlannerError):
            refresh_menu_cache(
                catalog_path=self.path,
                start_date=MONDAY,
                client=client,
                observed_at=T2,
            )

        self.assertEqual(client.calls, 1)
        self.assertEqual(before, self.digest())

    def test_refresh_does_not_create_plan_dispatch_application_or_intake(self) -> None:
        refresh_menu_cache(
            catalog_path=self.path,
            start_date=MONDAY,
            client=FakeRollingFDClient(),
            observed_at=T1,
        )

        self.assertEqual(self.table_count("meal_plans"), 0)
        self.assertEqual(self.table_count("scheduled_recommendation_dispatches"), 0)
        self.assertEqual(self.table_count("meal_report_applications"), 0)
        self.assertEqual(self.table_count("accepted_intake_entries"), 0)

    def test_scheduler_uses_refreshed_local_catalog_without_fd_network_access(self) -> None:
        client = FakeRollingFDClient()
        refresh_menu_cache(
            catalog_path=self.path,
            start_date=MONDAY,
            client=client,
            observed_at=T1,
        )
        calls_after_refresh = len(client.calls)
        sender = FakeSender()
        local_now = datetime(2026, 9, 7, 11, 30, tzinfo=timezone.utc)  # 07:30 EDT
        clock = NutritionApplicationClock(lambda: local_now)

        with OfficialNutritionCatalog(self.path) as catalog:
            scheduler = RecommendationScheduler(
                DurableMealState(catalog),
                clock=clock,
                service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
                recommendation_preparer=MealRecommendationOrchestrator(
                    catalog,
                    service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
                    clock=clock,
                ),
                target_loader=targets,
                outbound_sender=sender,
                chat_guid="test-chat",
            )
            result = scheduler.run_once()

        breakfast = next(item for item in result.decisions if item.opportunity.meal_id == 1)
        self.assertEqual(breakfast.outcome, "dispatched")
        self.assertEqual(len(sender.calls), 1)
        self.assertEqual(len(client.calls), calls_after_refresh)

    def test_detroit_service_date_is_independent_of_utc_host_and_dst_changes(self) -> None:
        before_spring_forward = NutritionApplicationClock(
            lambda: datetime(2026, 3, 8, 4, 30, tzinfo=timezone.utc)
        )
        after_spring_forward = NutritionApplicationClock(
            lambda: datetime(2026, 3, 8, 7, 30, tzinfo=timezone.utc)
        )
        before_fall_back = NutritionApplicationClock(
            lambda: datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc)
        )
        after_fall_back = NutritionApplicationClock(
            lambda: datetime(2026, 11, 1, 6, 30, tzinfo=timezone.utc)
        )

        self.assertEqual(menu_cache_date_range(clock=before_spring_forward)[0], date(2026, 3, 7))
        self.assertEqual(menu_cache_date_range(clock=after_spring_forward)[0], date(2026, 3, 8))
        self.assertEqual(menu_cache_date_range(clock=before_fall_back)[0], date(2026, 11, 1))
        self.assertEqual(menu_cache_date_range(clock=after_fall_back)[0], date(2026, 11, 1))

        client = FakeRollingFDClient()
        result = refresh_menu_cache(
            catalog_path=self.path,
            clock=before_spring_forward,
            days=1,
            client=client,
            observed_at=T1,
        )
        self.assertEqual(result.refresh.requested_start, date(2026, 3, 7))
        self.assertTrue(all(call["start_date"] == date(2026, 3, 7) for call in client.calls))

    def test_coverage_cli_is_local_and_reports_a_gap(self) -> None:
        self.seed_current(MONDAY, (2, 3))
        output = StringIO()

        with redirect_stdout(output):
            exit_code = menu_cache_main(
                [
                    "menu-coverage-status",
                    "--catalog-path",
                    str(self.path),
                    "--start-date",
                    MONDAY.isoformat(),
                    "--days",
                    "1",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertIn(f"{MONDAY.isoformat()} fd:1 breakfast: expected; occurrences=0; coverage=missing", output.getvalue())

    def test_menu_refresh_timer_is_detroit_local_persistent_and_verifiable(self) -> None:
        timer_path = Path(__file__).resolve().parents[1] / "systemd" / "nutrition-menu-refresh.timer"
        service_path = timer_path.with_suffix(".service")
        timer_text = timer_path.read_text(encoding="utf-8")
        service_text = service_path.read_text(encoding="utf-8")

        self.assertIn("OnCalendar=*-*-* 06:30:00 America/Detroit", timer_text)
        self.assertIn("OnCalendar=*-*-* 15:30:00 America/Detroit", timer_text)
        self.assertIn("Persistent=true", timer_text)
        self.assertNotIn("RandomizedDelaySec", timer_text)
        self.assertIn("Type=oneshot", service_text)
        self.assertIn("refresh-menu-cache", service_text)
        self.assertNotIn("BlueBubblesClient", service_text)
        self.assertNotIn("EnvironmentFile", service_text)

        systemd_analyze = shutil.which("systemd-analyze")
        if systemd_analyze is None:  # pragma: no cover - deployment hosts provide it
            self.skipTest("systemd-analyze is unavailable")
        verification = subprocess.run(
            [systemd_analyze, "verify", str(service_path), str(timer_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(verification.returncode, 0, verification.stderr)
        spring = subprocess.run(
            [
                systemd_analyze,
                "calendar",
                "--base-time=2026-03-08 00:00:00 UTC",
                "--iterations=2",
                "*-*-* 06:30:00 America/Detroit",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        fall = subprocess.run(
            [
                systemd_analyze,
                "calendar",
                "--base-time=2026-10-31 00:00:00 UTC",
                "--iterations=2",
                "*-*-* 06:30:00 America/Detroit",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(spring.returncode, 0, spring.stderr)
        self.assertEqual(fall.returncode, 0, fall.stderr)
        self.assertIn("Sun 2026-03-08 10:30:00 UTC", spring.stdout)
        self.assertIn("Sun 2026-11-01 11:30:00 UTC", fall.stdout)


if __name__ == "__main__":
    unittest.main()
