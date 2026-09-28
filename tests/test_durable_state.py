"""Offline coverage for durable plans, accepted intake, and exact snapshots."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from nutrition_optimizer.durable_state import (
    ActiveMealPlanInvariantError,
    DurableMealState,
    MealReportApplicationError,
)
from nutrition_optimizer.fdmealplanner import CATALOG_SCHEMA_VERSION, OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import FoodResolutionRequest, LocalFDFoodResolver, ResolvedFood
from nutrition_optimizer.meal_report import (
    ClarificationItem,
    MealPlan,
    PlannedMealItem,
    ProposedEatenMealItem,
    ProposedUnplannedMealItem,
    ReconciledMealReport,
    SkippedMealItem,
)
from nutrition_optimizer.nutrition import DailyTargets
from tests.test_food_resolution import DAY1, T1, T2, component, mapped, menu_day


class DurableMealStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "state" / "nutrition.sqlite3"
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self.catalog.close)
        self._sync(
            (
                (component(1, "Chicken"), "Grill"),
                (component(2, "Broccoli"), "Vegetables"),
                (component(3, "Rice"), "Sides"),
            ),
            observed_at=T1,
        )
        self.state = DurableMealState(self.catalog)

    def _sync(
        self,
        recipes,
        *,
        observed_at,
        service_date=DAY1,
        meal_id=3,
        meal_name="Dinner",
    ) -> None:
        self.catalog.synchronize_fd_refresh(
            mapped(menu_day(service_date, meal_id, meal_name, recipes)),
            requested_start=service_date,
            requested_end=service_date,
            observed_at=observed_at,
        )

    def food(self, name: str) -> ResolvedFood:
        result = LocalFDFoodResolver(self.catalog).resolve(
            FoodResolutionRequest(name, DAY1, meal="Dinner")
        )
        assert isinstance(result, ResolvedFood)
        return result

    def food_for(self, name: str, service_date, meal: str | int) -> ResolvedFood:
        result = LocalFDFoodResolver(self.catalog).resolve(
            FoodResolutionRequest(name, service_date, meal=meal)
        )
        assert isinstance(result, ResolvedFood)
        return result

    def plan(self) -> MealPlan:
        return MealPlan(
            DAY1,
            "Dinner",
            (
                PlannedMealItem(self.food("Chicken"), Decimal("1"), "one chicken serving"),
                PlannedMealItem(self.food("Broccoli"), Decimal("1.5"), "some broccoli"),
            ),
        )

    @staticmethod
    def report(
        plan: MealPlan,
        *,
        eaten: tuple[ProposedEatenMealItem, ...] = (),
        skipped: tuple[SkippedMealItem, ...] = (),
        unplanned: tuple[ProposedUnplannedMealItem, ...] = (),
        clarification: tuple = (),
    ) -> ReconciledMealReport:
        decided = {item.plan_item for item in eaten} | {item.plan_item for item in skipped}
        return ReconciledMealReport(
            plan,
            eaten,
            skipped,
            tuple(item for item in plan.items if item not in decided),
            unplanned,
            clarification,
        )

    def test_schema_v3_migration_preserves_existing_catalog_and_occurrences(self) -> None:
        occurrences_before = self.catalog.list_current_meal_occurrences(DAY1)
        snapshots_before = self.catalog.snapshot_count
        self.catalog.close()
        connection = sqlite3.connect(self.path)
        try:
            for table in ("accepted_intake_entries", "meal_report_applications", "meal_plan_items", "meal_plans"):
                connection.execute(f"DROP TABLE {table}")
            connection.execute("PRAGMA user_version = 2")
            connection.commit()
        finally:
            connection.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        self.assertEqual(self.catalog.snapshot_count, snapshots_before)
        self.assertEqual(self.catalog.list_current_meal_occurrences(DAY1), occurrences_before)
        version = self.catalog._connection.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(version, CATALOG_SCHEMA_VERSION)
        tables = {row[0] for row in self.catalog._connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({"meal_plans", "meal_plan_items", "meal_report_applications", "accepted_intake_entries"} <= tables)

    def test_first_plan_is_active_and_active_lookup_canonicalizes_known_meals(self) -> None:
        saved = self.state.save_meal_plan(self.plan(), plan_id="first-dinner")

        self.assertEqual(self.state.load_active_meal_plan(DAY1, "Dinner"), saved)
        self.assertEqual(self.state.load_active_meal_plan(DAY1, 3), saved)
        self.assertEqual(self.state.load_active_meal_plan(DAY1, " 3 "), saved)
        self.assertIsNone(self.state.load_active_meal_plan(DAY1, "breakfast"))

    def test_replacement_supersedes_only_the_prior_active_plan_in_its_context(self) -> None:
        first = self.state.save_meal_plan(self.plan(), plan_id="old-dinner", created_at=T1)
        second = self.state.save_meal_plan(self.plan(), plan_id="new-dinner", created_at=T2)

        old_history = self.state.load_meal_plan(first.plan_id)
        new_history = self.state.load_meal_plan(second.plan_id)
        assert old_history is not None
        assert new_history is not None
        self.assertEqual(old_history.status, "superseded")
        self.assertEqual(old_history.plan, first.plan)
        self.assertEqual(new_history.status, "active")
        self.assertEqual(self.state.load_active_meal_plan(DAY1, "dinner"), second)

    def test_different_meal_and_date_plans_remain_independently_active(self) -> None:
        next_day = DAY1 + timedelta(days=1)
        dinner = self.state.save_meal_plan(self.plan(), plan_id="dinner-day-one")
        self._sync(
            ((component(4, "Breakfast Eggs"), "Grill"),),
            observed_at=T2,
            meal_id=1,
            meal_name="Breakfast",
        )
        self._sync(
            ((component(5, "Next Day Chicken"), "Grill"),),
            observed_at=T2,
            service_date=next_day,
        )
        breakfast_food = self.food_for("Breakfast Eggs", DAY1, 1)
        breakfast = self.state.save_meal_plan(
            MealPlan(
                DAY1,
                1,
                (PlannedMealItem(breakfast_food, Decimal("1"), "two eggs"),),
            ),
            plan_id="breakfast-day-one",
        )
        next_day_food = self.food_for("Next Day Chicken", next_day, 3)
        tomorrow = self.state.save_meal_plan(
            MealPlan(
                next_day,
                3,
                (PlannedMealItem(next_day_food, Decimal("1"), "one chicken serving"),),
            ),
            plan_id="dinner-day-two",
        )

        self.assertEqual(self.state.load_active_meal_plan(DAY1, 3), dinner)
        self.assertEqual(self.state.load_active_meal_plan(DAY1, "breakfast"), breakfast)
        self.assertEqual(self.state.load_active_meal_plan(next_day, "dinner"), tomorrow)

    def test_active_lookup_fails_closed_when_manual_corruption_has_multiple_rows(self) -> None:
        first = self.state.save_meal_plan(self.plan(), plan_id="old-dinner")
        self.state.save_meal_plan(self.plan(), plan_id="new-dinner")
        connection = self.catalog._connection
        connection.execute("DROP INDEX meal_plans_one_active_slot_idx")
        connection.execute("UPDATE meal_plans SET status = 'active' WHERE plan_id = ?", (first.plan_id,))
        connection.commit()

        with self.assertRaisesRegex(ActiveMealPlanInvariantError, "multiple active"):
            self.state.load_active_meal_plan(DAY1, "dinner")

    def test_failed_replacement_write_keeps_the_prior_plan_active(self) -> None:
        first = self.state.save_meal_plan(self.plan(), plan_id="duplicate-plan-id")

        with self.assertRaisesRegex(MealReportApplicationError, "unable to persist"):
            self.state.save_meal_plan(self.plan(), plan_id="duplicate-plan-id")

        restored = self.state.load_meal_plan(first.plan_id)
        assert restored is not None
        self.assertEqual(restored.status, "active")
        self.assertEqual(self.state.load_active_meal_plan(DAY1, "dinner"), first)

    def test_v4_to_v5_migration_preserves_history_and_deterministically_backfills_lifecycle(self) -> None:
        plan = self.plan()
        applied_plan = self.state.save_meal_plan(plan, plan_id="legacy-applied", created_at=T1)
        applied_report = self.report(
            plan,
            eaten=(
                ProposedEatenMealItem(
                    plan.items[0],
                    Decimal("1"),
                    "planned_quantity",
                    "chicken",
                ),
            ),
        )
        self.state.apply_reconciled_meal_report(
            applied_plan,
            applied_report,
            source_event_id="legacy-applied-event",
            recorded_at=T1,
        )
        active_plan = self.state.save_meal_plan(plan, plan_id="legacy-active", created_at=T2)
        snapshot_count = self.catalog.snapshot_count
        intake_before = self.state.get_daily_intake(DAY1)
        self.catalog.close()

        connection = sqlite3.connect(self.path)
        try:
            plan_rows = tuple(
                connection.execute(
                    """
                    SELECT plan_id, service_date, meal, meal_kind, created_at
                    FROM meal_plans ORDER BY plan_id
                    """
                )
            )
            item_rows = tuple(
                connection.execute(
                    """
                    SELECT plan_id, plan_item_id, item_position, occurrence_id,
                           nutrition_snapshot_id, source_kind, source_value,
                           content_signature, recommended_official_servings,
                           natural_quantity_text, display_food_name
                    FROM meal_plan_items ORDER BY plan_id, item_position
                    """
                )
            )
            application_rows = tuple(
                connection.execute(
                    "SELECT source_event_id, plan_id, applied_at FROM meal_report_applications"
                )
            )
            intake_rows = tuple(
                connection.execute(
                    """
                    SELECT intake_id, source_event_id, service_date, recorded_at, meal,
                           plan_id, plan_item_id, occurrence_id, nutrition_snapshot_id,
                           source_kind, source_value, content_signature,
                           official_servings, quantity_source, original_reference_text,
                           original_quantity_text, item_position
                    FROM accepted_intake_entries
                    """
                )
            )
            connection.execute("PRAGMA foreign_keys = OFF")
            for table in (
                "accepted_intake_entries",
                "meal_report_applications",
                "meal_plan_items",
                "meal_plans",
            ):
                connection.execute(f"DROP TABLE {table}")
            connection.executescript(
                """
                CREATE TABLE meal_plans (
                    plan_id TEXT PRIMARY KEY,
                    service_date TEXT NOT NULL,
                    meal TEXT NOT NULL,
                    meal_kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('active')) DEFAULT 'active'
                );
                CREATE TABLE meal_plan_items (
                    plan_id TEXT NOT NULL,
                    plan_item_id TEXT NOT NULL,
                    item_position INTEGER NOT NULL,
                    occurrence_id INTEGER NOT NULL,
                    nutrition_snapshot_id INTEGER NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_value TEXT NOT NULL,
                    content_signature TEXT NOT NULL,
                    recommended_official_servings TEXT NOT NULL,
                    natural_quantity_text TEXT NOT NULL,
                    display_food_name TEXT,
                    PRIMARY KEY (plan_id, plan_item_id),
                    UNIQUE (plan_id, item_position),
                    FOREIGN KEY (plan_id) REFERENCES meal_plans(plan_id)
                );
                CREATE TABLE meal_report_applications (
                    source_event_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    applied_at TEXT NOT NULL,
                    FOREIGN KEY (plan_id) REFERENCES meal_plans(plan_id)
                );
                CREATE TABLE accepted_intake_entries (
                    intake_id TEXT PRIMARY KEY,
                    source_event_id TEXT NOT NULL,
                    service_date TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    meal TEXT,
                    plan_id TEXT,
                    plan_item_id TEXT,
                    occurrence_id INTEGER NOT NULL,
                    nutrition_snapshot_id INTEGER NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_value TEXT NOT NULL,
                    content_signature TEXT NOT NULL,
                    official_servings TEXT NOT NULL,
                    quantity_source TEXT NOT NULL,
                    original_reference_text TEXT,
                    original_quantity_text TEXT,
                    item_position INTEGER NOT NULL,
                    FOREIGN KEY (source_event_id) REFERENCES meal_report_applications(source_event_id),
                    FOREIGN KEY (plan_id) REFERENCES meal_plans(plan_id)
                );
                """
            )
            connection.executemany(
                """
                INSERT INTO meal_plans
                (plan_id, service_date, meal, meal_kind, created_at, status)
                VALUES (?, ?, ?, ?, ?, 'active')
                """,
                plan_rows,
            )
            connection.executemany(
                """
                INSERT INTO meal_plan_items
                (plan_id, plan_item_id, item_position, occurrence_id,
                 nutrition_snapshot_id, source_kind, source_value,
                 content_signature, recommended_official_servings,
                 natural_quantity_text, display_food_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                item_rows,
            )
            connection.executemany(
                """
                INSERT INTO meal_report_applications (source_event_id, plan_id, applied_at)
                VALUES (?, ?, ?)
                """,
                application_rows,
            )
            connection.executemany(
                """
                INSERT INTO accepted_intake_entries
                (intake_id, source_event_id, service_date, recorded_at, meal, plan_id,
                 plan_item_id, occurrence_id, nutrition_snapshot_id, source_kind,
                 source_value, content_signature, official_servings, quantity_source,
                 original_reference_text, original_quantity_text, item_position)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                intake_rows,
            )
            connection.execute("PRAGMA user_version = 4")
            connection.commit()
        finally:
            connection.close()

        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self.catalog.close)
        self.state = DurableMealState(self.catalog)
        migrated_applied = self.state.load_meal_plan(applied_plan.plan_id)
        migrated_active = self.state.load_meal_plan(active_plan.plan_id)
        assert migrated_applied is not None
        assert migrated_active is not None
        self.assertEqual(
            self.catalog._connection.execute("PRAGMA user_version").fetchone()[0],
            CATALOG_SCHEMA_VERSION,
        )
        self.assertEqual(self.catalog.snapshot_count, snapshot_count)
        self.assertEqual(migrated_applied.status, "applied")
        self.assertEqual(migrated_active.status, "active")
        self.assertEqual(self.state.load_active_meal_plan(DAY1, 3), migrated_active)
        self.assertEqual(self.state.get_daily_intake(DAY1), intake_before)

    def test_plan_round_trip_preserves_decimal_occurrence_and_snapshot_after_restart(self) -> None:
        original = self.plan()
        stored = self.state.save_meal_plan(original, plan_id="breakfast-1", created_at=T1)
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        loaded = self.state.load_meal_plan("breakfast-1")
        assert loaded is not None
        self.assertEqual(loaded.plan.service_date, original.service_date)
        self.assertEqual(loaded.plan.meal, original.meal)
        self.assertEqual(loaded.plan.items[1].recommended_official_servings, Decimal("1.5"))
        self.assertEqual(loaded.plan.items[0].food.occurrence.occurrence_id, original.items[0].food.occurrence.occurrence_id)
        self.assertEqual(loaded.plan.items[0].food.nutrition_snapshot_id, original.items[0].food.nutrition_snapshot_id)
        self.assertEqual(loaded.created_at, stored.created_at)
        reloaded_report = self.report(
            loaded.plan,
            eaten=(
                ProposedEatenMealItem(
                    loaded.plan.items[0],
                    Decimal("1"),
                    "planned_quantity",
                    "the chicken",
                ),
            ),
        )
        applied = self.state.apply_reconciled_meal_report(
            loaded,
            reloaded_report,
            source_event_id="restart-message",
        )
        self.assertEqual(applied.accepted_intake_entries[0].record, original.items[0].record)

    def test_apply_persists_only_eaten_and_preserves_planned_and_explicit_quantities(self) -> None:
        plan = self.plan()
        persisted = self.state.save_meal_plan(plan, plan_id="dinner-1")
        eaten = (
            ProposedEatenMealItem(plan.items[0], Decimal("1"), "planned_quantity", "the chicken"),
            ProposedEatenMealItem(plan.items[1], Decimal("0.5"), "explicit_deterministic", "half broccoli", "high"),
        )
        skipped = (SkippedMealItem(plan.items[1], "skipped broccoli"),)
        # A single report cannot mark one item both eaten and skipped; use its
        # unspecified counterpart for the skip assertion in a separate report.
        result = self.state.apply_reconciled_meal_report(
            persisted,
            self.report(plan, eaten=eaten),
            source_event_id="message-1",
            recorded_at=T2,
        )
        self.assertFalse(result.already_applied)
        self.assertEqual([entry.official_servings for entry in result.accepted_intake_entries], [Decimal("1"), Decimal("0.5")])
        self.assertEqual([entry.quantity_source for entry in result.accepted_intake_entries], ["planned_quantity", "explicit_deterministic"])
        self.assertEqual(len(self.state.get_daily_intake(DAY1)), 2)
        self.assertEqual(skipped[0].plan_item.record.name, "Broccoli")

    def test_skipped_unspecified_and_clarification_never_become_intake(self) -> None:
        plan = self.plan()
        persisted = self.state.save_meal_plan(plan)
        safe_skip = self.report(plan, skipped=(SkippedMealItem(plan.items[0], "skipped chicken"),))
        applied = self.state.apply_reconciled_meal_report(persisted, safe_skip, source_event_id="message-skip")
        self.assertEqual(applied.accepted_intake_entries, ())
        unresolved = ReconciledMealReport(
            plan, (), (), plan.items, (),
            (ClarificationItem("needs_clarification"),),
        )
        with self.assertRaisesRegex(MealReportApplicationError, "requires clarification"):
            self.state.apply_reconciled_meal_report(persisted, unresolved, source_event_id="message-unsafe")
        self.assertEqual(len(self.state.get_daily_intake(DAY1)), 0)

    def test_semantic_quantity_estimate_never_becomes_authoritative_intake(self) -> None:
        plan = self.plan()
        persisted = self.state.save_meal_plan(plan)
        estimated = self.report(
            plan,
            eaten=(
                ProposedEatenMealItem(
                    plan.items[0],
                    Decimal("1"),
                    "explicit_semantic_estimate",
                    "about one serving",
                    "medium",
                ),
            ),
        )

        with self.assertRaisesRegex(MealReportApplicationError, "not authoritative"):
            self.state.apply_reconciled_meal_report(
                persisted,
                estimated,
                source_event_id="semantic-estimate",
            )
        self.assertEqual(self.state.get_daily_intake(DAY1), ())
        self.assertIsNone(self.state.load_applied_meal_report("semantic-estimate"))

    def test_unplanned_resolved_eaten_food_persists_exact_snapshot(self) -> None:
        plan = self.plan()
        persisted = self.state.save_meal_plan(plan)
        rice = self.food("Rice")
        report = self.report(
            plan,
            unplanned=(ProposedUnplannedMealItem("rice", "2 servings", rice, Decimal("2"), "explicit_deterministic", "high"),),
        )
        result = self.state.apply_reconciled_meal_report(persisted, report, source_event_id="message-rice")
        entry = result.accepted_intake_entries[0]
        self.assertEqual(entry.record, rice.nutrition_record)
        self.assertEqual(entry.occurrence.nutrition_snapshot_id, rice.nutrition_snapshot_id)
        self.assertEqual(entry.official_servings, Decimal("2"))
        self.assertIsNone(entry.plan_item_id)

    def test_idempotency_replays_but_a_completed_plan_rejects_a_new_event(self) -> None:
        plan = self.plan()
        persisted = self.state.save_meal_plan(plan)
        report = self.report(plan, eaten=(ProposedEatenMealItem(plan.items[0], Decimal("1"), "planned_quantity", "chicken"),))
        first = self.state.apply_reconciled_meal_report(persisted, report, source_event_id="message-dup")
        repeated = self.state.apply_reconciled_meal_report(persisted, report, source_event_id="message-dup")
        self.assertFalse(first.already_applied)
        self.assertTrue(repeated.already_applied)
        self.assertEqual(len(repeated.accepted_intake_entries), 1)
        with self.assertRaisesRegex(MealReportApplicationError, "no longer active"):
            self.state.apply_reconciled_meal_report(
                persisted,
                report,
                source_event_id="message-new",
            )
        loaded = self.state.load_meal_plan(persisted.plan_id)
        assert loaded is not None
        self.assertEqual(loaded.status, "applied")
        self.assertEqual(len(self.state.get_daily_intake(DAY1)), 1)

    def test_multi_item_application_rolls_back_completely_on_insert_failure(self) -> None:
        plan = self.plan()
        persisted = self.state.save_meal_plan(plan)
        report = self.report(plan, eaten=(
            ProposedEatenMealItem(plan.items[0], Decimal("1"), "planned_quantity", "chicken"),
            ProposedEatenMealItem(plan.items[1], Decimal("1.5"), "planned_quantity", "broccoli"),
        ))
        original = self.state._insert_intake
        call_count = 0

        def fail_second_insert(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise sqlite3.OperationalError("forced")
            return original(*args, **kwargs)

        with patch.object(self.state, "_insert_intake", side_effect=fail_second_insert):
            with self.assertRaisesRegex(MealReportApplicationError, "failed atomically"):
                self.state.apply_reconciled_meal_report(persisted, report, source_event_id="message-atomic")
        self.assertEqual(call_count, 2)
        self.assertEqual(self.state.get_daily_intake(DAY1), ())
        self.assertEqual(self.state._connection().execute("SELECT COUNT(*) FROM meal_report_applications").fetchone()[0], 0)

    def test_old_intake_keeps_old_snapshot_after_catalog_new_version(self) -> None:
        plan = self.plan()
        old_snapshot = plan.items[0].food.nutrition_snapshot_id
        persisted = self.state.save_meal_plan(plan)
        report = self.report(plan, eaten=(ProposedEatenMealItem(plan.items[0], Decimal("1"), "planned_quantity", "chicken"),))
        self.state.apply_reconciled_meal_report(persisted, report, source_event_id="message-old")
        newer_chicken = component(1, "Chicken")
        newer_chicken["calories"] = "250"
        self._sync(((newer_chicken, "Grill"),), observed_at=T2)
        ledger = self.state.load_daily_ledger(DAY1)
        intake = self.state.get_daily_intake(DAY1)[0]
        self.assertEqual(intake.occurrence.nutrition_snapshot_id, old_snapshot)
        self.assertEqual(intake.record.nutrients.calories_kcal, Decimal("100"))
        self.assertEqual(ledger.total_consumed_nutrients.calories_kcal, Decimal("100"))

    def test_historical_intake_hydrates_fiber_from_its_exact_snapshot_extension(self) -> None:
        old_chicken = component(1, "Chicken")
        old_chicken["dietaryFiber"] = "2"
        old_chicken["dietaryFiberUOM"] = "g"
        self._sync(((old_chicken, "Grill"),), observed_at=T2)
        plan = MealPlan(
            DAY1,
            "Dinner",
            (PlannedMealItem(self.food("Chicken"), Decimal("1"), "one chicken serving"),),
        )
        persisted = self.state.save_meal_plan(plan)
        self.state.apply_reconciled_meal_report(
            persisted,
            self.report(plan, eaten=(
                ProposedEatenMealItem(plan.items[0], Decimal("1"), "planned_quantity", "chicken"),
            )),
            source_event_id="fiber-old-version",
        )
        newer_chicken = component(1, "Chicken")
        newer_chicken["dietaryFiber"] = "7"
        newer_chicken["dietaryFiberUOM"] = "g"
        self._sync(((newer_chicken, "Grill"),), observed_at=datetime(2026, 8, 22, tzinfo=timezone.utc))
        intake = self.state.get_daily_intake(DAY1)[0]
        ledger = self.state.load_daily_ledger(DAY1)
        self.assertEqual(intake.record.nutrients.dietary_fiber_g, Decimal("2"))
        self.assertEqual(ledger.total_consumed_nutrients.dietary_fiber_g, Decimal("2"))

    def test_ledger_and_caller_provided_balance_use_existing_deterministic_arithmetic(self) -> None:
        plan = self.plan()
        persisted = self.state.save_meal_plan(plan)
        report = self.report(plan, eaten=(ProposedEatenMealItem(plan.items[0], Decimal("2"), "planned_quantity", "chicken"),))
        self.state.apply_reconciled_meal_report(persisted, report, source_event_id="message-ledger")
        ledger = self.state.load_daily_ledger(DAY1)
        balance = self.state.calculate_daily_balance(
            DAY1,
            DailyTargets(Decimal("300"), Decimal("30"), Decimal("20"), Decimal("10")),
        )
        self.assertEqual(ledger.total_consumed_nutrients.calories_kcal, Decimal("200"))
        self.assertEqual(balance.remaining.calories_kcal, Decimal("100"))

    def test_application_has_no_ai_network_messaging_target_or_recommendation_dependency(self) -> None:
        source = Path(__import__("nutrition_optimizer.durable_state", fromlist=["x"]).__file__).read_text(encoding="utf-8")
        for forbidden in ("OpenClaw", "OpenAI", "FDMealPlannerClient", "BlueBubbles", "DiningBucket"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
