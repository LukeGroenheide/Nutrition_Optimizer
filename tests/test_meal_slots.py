"""Focused domain coverage for product meal-slot identity."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from nutrition_optimizer.durable_state import (
    ActiveMealPlanInvariantError,
    DraftClarification,
    DurableMealState,
    MealReportApplicationError,
)
from nutrition_optimizer.fdmealplanner import (
    CATALOG_SCHEMA_VERSION,
    NutritionCatalogError,
    OfficialNutritionCatalog,
)
from nutrition_optimizer.food_resolution import FoodResolutionRequest, LocalFDFoodResolver, ResolvedFood
from nutrition_optimizer.meal_identity import (
    legacy_meal_slot,
    meal_slot_for_provider_meal,
    require_meal_slot,
)
from nutrition_optimizer.meal_report import (
    MealPlan,
    PlannedMealItem,
    ProposedEatenMealItem,
    ReconciledMealReport,
)
from nutrition_optimizer.recommendation_scheduler import ScheduledMealOpportunity
from nutrition_optimizer.phelps_service_calendar import DEFAULT_PHELPS_SERVICE_CALENDAR
from tests.test_food_resolution import component, mapped, menu_day


SUNDAY = date(2026, 8, 30)
MONDAY = date(2026, 8, 31)
T1 = datetime(2026, 8, 30, 14, tzinfo=timezone.utc)


class MealSlotSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "nutrition.sqlite3"
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self._close_catalog)
        self.catalog.synchronize_fd_refresh(
            mapped(
                menu_day(SUNDAY, 2, "Lunch", ((component(200, "Sunday Food"), "Grill"),)),
                menu_day(MONDAY, 1, "Breakfast", ((component(101, "Eggs"), "Grill"),)),
                menu_day(MONDAY, 2, "Lunch", ((component(201, "Monday Food"), "Grill"),)),
                menu_day(MONDAY, 3, "Dinner", ((component(301, "Chicken"), "Grill"),)),
            ),
            requested_start=SUNDAY,
            requested_end=MONDAY,
            observed_at=T1,
        )
        self.state = DurableMealState(self.catalog)

    def _close_catalog(self) -> None:
        if getattr(self, "catalog", None) is not None:
            self.catalog.close()

    def _food(self, service_date: date, meal: int, name: str) -> ResolvedFood:
        result = LocalFDFoodResolver(self.catalog).resolve(
            FoodResolutionRequest(name, service_date, meal=meal)
        )
        assert isinstance(result, ResolvedFood)
        return result

    def _plan(self, service_date: date, meal: int, name: str) -> MealPlan:
        return MealPlan(
            service_date,
            meal,
            (PlannedMealItem(self._food(service_date, meal, name), Decimal("1"), "one serving"),),
        )

    def test_fresh_schema_is_current_and_reopens_without_migration(self) -> None:
        self.assertEqual(CATALOG_SCHEMA_VERSION, 14)
        self.assertEqual(
            self.catalog._connection.execute("PRAGMA user_version").fetchone()[0],
            CATALOG_SCHEMA_VERSION,
        )
        columns = {
            row["name"] for row in self.catalog._connection.execute("PRAGMA table_info(meal_plans)")
        }
        self.assertIn("meal_slot", columns)
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.assertEqual(
            self.catalog._connection.execute("PRAGMA user_version").fetchone()[0],
            CATALOG_SCHEMA_VERSION,
        )

    def test_scheduled_sunday_brunch_and_normal_slots_persist_provider_identity_separately(self) -> None:
        brunch = self.state.save_scheduled_meal_plan(
            self._plan(SUNDAY, 2, "Sunday Food"),
            meal_slot="brunch",
            plan_id="sunday-brunch",
        )
        self.assertEqual(brunch.meal_slot, "brunch")
        self.assertEqual(brunch.plan.meal, 2)
        brunch_dispatch = self.state.load_scheduled_recommendation_dispatch(
            SUNDAY, 2, meal_slot="brunch"
        )
        assert brunch_dispatch is not None
        self.assertEqual(brunch_dispatch.meal_slot, "brunch")
        self.assertEqual(brunch_dispatch.meal_context, "fd:2")

        for meal, slot, name in (
            (1, "breakfast", "Eggs"),
            (2, "lunch", "Monday Food"),
            (3, "dinner", "Chicken"),
        ):
            saved = self.state.save_meal_plan(
                self._plan(MONDAY, meal, name), meal_slot=slot, plan_id=f"monday-{slot}"
            )
            self.assertEqual(saved.meal_slot, slot)
            self.assertEqual(saved.plan.meal, meal)

    def test_lunch_and_brunch_are_distinct_slots_for_the_same_provider_period(self) -> None:
        plan = self._plan(SUNDAY, 2, "Sunday Food")
        lunch = self.state.save_scheduled_meal_plan(
            plan,
            meal_slot="lunch",
            plan_id="explicit-lunch",
            delivery_token="explicit-lunch-token",
        )
        brunch = self.state.save_scheduled_meal_plan(
            plan,
            meal_slot="brunch",
            plan_id="explicit-brunch",
            delivery_token="explicit-brunch-token",
        )
        for slot in ("lunch", "brunch"):
            dispatch = self.state.load_scheduled_recommendation_dispatch(
                SUNDAY, 2, meal_slot=slot
            )
            assert dispatch is not None
            self.assertTrue(
                self.state.claim_scheduled_recommendation_delivery(
                    dispatch, started_at=T1
                )
            )
            self.state.mark_scheduled_recommendation_delivered(
                dispatch, delivered_at=T1
            )

        self.assertEqual(self.state.load_meal_plan(lunch.plan_id).meal_slot, "lunch")  # type: ignore[union-attr]
        self.assertEqual(self.state.load_meal_plan(brunch.plan_id).meal_slot, "brunch")  # type: ignore[union-attr]
        self.assertEqual(lunch.plan.meal, brunch.plan.meal)
        self.assertNotEqual(lunch.meal_slot, brunch.meal_slot)
        self.assertEqual(
            self.state.load_active_meal_plan(
                SUNDAY, 2, meal_slot="lunch"
            ).plan_id,  # type: ignore[union-attr]
            lunch.plan_id,
        )
        self.assertEqual(
            self.state.load_active_meal_plan(
                SUNDAY, 2, meal_slot="brunch"
            ).plan_id,  # type: ignore[union-attr]
            brunch.plan_id,
        )
        with self.assertRaises(ActiveMealPlanInvariantError):
            self.state.load_active_meal_plan(SUNDAY, 2)
        self.assertEqual(
            {
                saved.meal_slot
                for saved in self.state.list_reportable_active_meal_plans(
                    SUNDAY,
                    DEFAULT_PHELPS_SERVICE_CALENDAR,
                    evaluated_at=T1,
                )
            },
            {"lunch", "brunch"},
        )

    def test_slot_domain_and_legacy_backfill_are_deterministic(self) -> None:
        self.assertEqual(meal_slot_for_provider_meal(1), "breakfast")
        self.assertEqual(meal_slot_for_provider_meal(2), "lunch")
        self.assertEqual(meal_slot_for_provider_meal(3), "dinner")
        self.assertEqual(legacy_meal_slot(SUNDAY, 2), "brunch")
        self.assertEqual(legacy_meal_slot(MONDAY, 2), "lunch")
        self.assertEqual(ScheduledMealOpportunity(2, time(10, 30), "brunch").meal_id, 2)
        with self.assertRaisesRegex(ValueError, "meal_slot"):
            require_meal_slot("late lunch")
        with self.assertRaisesRegex(ValueError, "meal_slot"):
            ScheduledMealOpportunity(2, time(10, 30), "late lunch")  # type: ignore[arg-type]

    def test_invalid_persisted_slot_fails_closed(self) -> None:
        saved = self.state.save_meal_plan(
            self._plan(MONDAY, 3, "Chicken"), meal_slot="dinner", plan_id="invalid-slot"
        )
        connection = self.catalog._connection
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "UPDATE meal_plans SET meal_slot = 'supper' WHERE plan_id = ?", (saved.plan_id,)
        )
        connection.commit()
        connection.execute("PRAGMA ignore_check_constraints = OFF")

        with self.assertRaisesRegex(MealReportApplicationError, "corrupt"):
            self.state.load_meal_plan(saved.plan_id)

    def test_v9_migration_preserves_ids_intake_draft_dispatch_and_foreign_keys(self) -> None:
        brunch = self.state.save_scheduled_meal_plan(
            self._plan(SUNDAY, 2, "Sunday Food"),
            meal_slot="brunch",
            plan_id="legacy-brunch",
            delivery_token="legacy-delivery-token",
        )
        dinner_plan = self._plan(MONDAY, 3, "Chicken")
        dinner = self.state.save_meal_plan(
            dinner_plan, meal_slot="dinner", plan_id="legacy-dinner"
        )
        report = ReconciledMealReport(
            dinner_plan,
            (
                ProposedEatenMealItem(
                    dinner_plan.items[0], Decimal("1"), "planned_quantity", "the chicken"
                ),
            ),
            (),
            (),
            (),
            (),
        )
        self.state.apply_reconciled_meal_report(
            dinner, report, source_event_id="legacy-intake-event", recorded_at=T1
        )
        breakfast = self.state.save_meal_plan(
            self._plan(MONDAY, 1, "Eggs"), meal_slot="breakfast", plan_id="legacy-breakfast"
        )
        draft, _ = self.state.save_meal_report_draft(
            chat_guid="chat-guid",
            persisted_plan=breakfast,
            current_draft=None,
            planned_items=(),
            unplanned_items=(),
            clarifications=(DraftClarification("completion_required"),),
            last_prompt="Anything else?",
            source_event_id="legacy-draft-event",
            intent="meal_report",
            processed_at=T1,
        )
        intake_ids = tuple(
            row[0]
            for row in self.catalog._connection.execute(
                "SELECT intake_id FROM accepted_intake_entries ORDER BY intake_id"
            )
        )
        item_ids = tuple(
            self.catalog._connection.execute(
                "SELECT plan_id, plan_item_id FROM meal_plan_items ORDER BY plan_id, plan_item_id"
            )
        )
        self.catalog.close()
        self._downgrade_to_v9()

        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        self.assertEqual(
            self.catalog._connection.execute("PRAGMA user_version").fetchone()[0],
            CATALOG_SCHEMA_VERSION,
        )
        self.assertEqual(self.state.load_meal_plan(brunch.plan_id).meal_slot, "brunch")  # type: ignore[union-attr]
        self.assertEqual(self.state.load_meal_plan(brunch.plan_id).plan.meal, 2)  # type: ignore[union-attr]
        self.assertEqual(self.state.load_meal_plan(dinner.plan_id).meal_slot, "dinner")  # type: ignore[union-attr]
        self.assertEqual(self.state.load_meal_plan(breakfast.plan_id).meal_slot, "breakfast")  # type: ignore[union-attr]
        self.assertEqual(self.state.load_meal_report_draft(draft.draft_id).draft_id, draft.draft_id)  # type: ignore[union-attr]
        self.assertEqual(
            tuple(
                row[0]
                for row in self.catalog._connection.execute(
                    "SELECT intake_id FROM accepted_intake_entries ORDER BY intake_id"
                )
            ),
            intake_ids,
        )
        self.assertEqual(
            tuple(
                self.catalog._connection.execute(
                    "SELECT plan_id, plan_item_id FROM meal_plan_items ORDER BY plan_id, plan_item_id"
                )
            ),
            item_ids,
        )
        self.assertEqual(
            tuple(
                self.catalog._connection.execute(
                    "SELECT plan_id, meal_context, meal_slot "
                    "FROM scheduled_recommendation_dispatches"
                ).fetchone()
            ),
            (brunch.plan_id, "fd:2", "brunch"),
        )
        self.assertEqual(
            self.catalog._connection.execute(
                "SELECT plan_id FROM meal_report_applications WHERE source_event_id = ?",
                ("legacy-intake-event",),
            ).fetchone()[0],
            dinner.plan_id,
        )
        self.assertEqual(self.catalog._connection.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_v9_migration_failure_rolls_back_schema_and_version(self) -> None:
        saved = self.state.save_meal_plan(
            self._plan(MONDAY, 3, "Chicken"), meal_slot="dinner", plan_id="unclassifiable"
        )
        self.catalog.close()
        self._downgrade_to_v9()
        connection = sqlite3.connect(self.path)
        connection.execute(
            "UPDATE meal_plans SET meal = '99', meal_kind = 'integer', meal_context = 'integer:99' "
            "WHERE plan_id = ?",
            (saved.plan_id,),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(NutritionCatalogError, "deterministic meal slot"):
            OfficialNutritionCatalog(self.path)

        connection = sqlite3.connect(self.path)
        try:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 9)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(meal_plans)")}
            self.assertNotIn("meal_slot", columns)
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
        finally:
            connection.close()

    def _downgrade_to_v9(self) -> None:
        """Rebuild only the three v10 slot-bearing tables in a temporary test DB."""

        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.executescript(
                """
                BEGIN;
                CREATE TABLE meal_plans_v9 (
                    plan_id TEXT PRIMARY KEY CHECK (length(trim(plan_id)) > 0),
                    service_date TEXT NOT NULL CHECK (length(service_date) = 10),
                    meal TEXT NOT NULL CHECK (length(trim(meal)) > 0),
                    meal_kind TEXT NOT NULL CHECK (meal_kind IN ('text', 'integer')),
                    meal_context TEXT NOT NULL CHECK (length(trim(meal_context)) > 0),
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('active', 'applied', 'superseded'))
                        DEFAULT 'active'
                );
                INSERT INTO meal_plans_v9
                SELECT plan_id, service_date, meal, meal_kind, meal_context, created_at, status
                FROM meal_plans;
                DROP TABLE meal_plans;
                ALTER TABLE meal_plans_v9 RENAME TO meal_plans;
                CREATE UNIQUE INDEX meal_plans_one_active_context_idx
                ON meal_plans(service_date, meal_context) WHERE status = 'active';

                CREATE TABLE scheduled_recommendation_dispatches_v9 (
                    service_date TEXT NOT NULL CHECK (length(service_date) = 10),
                    meal_context TEXT NOT NULL CHECK (length(trim(meal_context)) > 0),
                    plan_id TEXT NOT NULL UNIQUE,
                    delivery_token TEXT NOT NULL UNIQUE CHECK (length(trim(delivery_token)) > 0),
                    status TEXT NOT NULL CHECK (status IN (
                        'pending_delivery', 'sending', 'delivered', 'expired'
                    )),
                    created_at TEXT NOT NULL,
                    delivery_started_at TEXT,
                    delivered_at TEXT,
                    expired_at TEXT,
                    PRIMARY KEY (service_date, meal_context),
                    FOREIGN KEY (plan_id) REFERENCES meal_plans(plan_id)
                );
                INSERT INTO scheduled_recommendation_dispatches_v9
                SELECT service_date, meal_context, plan_id, delivery_token, status,
                       created_at, delivery_started_at, delivered_at, expired_at
                FROM scheduled_recommendation_dispatches;
                DROP TABLE scheduled_recommendation_dispatches;
                ALTER TABLE scheduled_recommendation_dispatches_v9
                    RENAME TO scheduled_recommendation_dispatches;
                CREATE INDEX scheduled_recommendation_dispatches_status_idx
                ON scheduled_recommendation_dispatches(status, service_date, meal_context);

                CREATE TABLE pending_meal_requests_v9 (
                    source_event_id TEXT PRIMARY KEY CHECK (length(trim(source_event_id)) > 0),
                    chat_guid TEXT NOT NULL CHECK (length(trim(chat_guid)) > 0),
                    service_date TEXT NOT NULL CHECK (length(service_date) = 10),
                    meal TEXT NOT NULL CHECK (length(trim(meal)) > 0),
                    meal_kind TEXT NOT NULL CHECK (meal_kind IN ('text', 'integer')),
                    meal_context TEXT NOT NULL CHECK (length(trim(meal_context)) > 0),
                    whole_meal INTEGER NOT NULL CHECK (whole_meal IN (0, 1)),
                    status TEXT NOT NULL CHECK (status IN ('pending', 'consumed', 'superseded')),
                    reply_text TEXT NOT NULL CHECK (length(trim(reply_text)) > 0),
                    created_at TEXT NOT NULL,
                    consumed_at TEXT,
                    consumed_plan_id TEXT,
                    FOREIGN KEY (consumed_plan_id) REFERENCES meal_plans(plan_id)
                );
                INSERT INTO pending_meal_requests_v9
                SELECT source_event_id, chat_guid, service_date, meal, meal_kind, meal_context,
                       whole_meal, status, reply_text, created_at, consumed_at, consumed_plan_id
                FROM pending_meal_requests;
                DROP TABLE pending_meal_requests;
                ALTER TABLE pending_meal_requests_v9 RENAME TO pending_meal_requests;
                CREATE UNIQUE INDEX pending_meal_requests_one_open_context_idx
                ON pending_meal_requests(chat_guid, service_date, meal_context)
                WHERE status = 'pending';
                CREATE INDEX pending_meal_requests_scheduler_idx
                ON pending_meal_requests(chat_guid, service_date, meal_context, status);
                PRAGMA user_version = 9;
                COMMIT;
                """
            )
            connection.execute("PRAGMA foreign_keys = ON")
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
