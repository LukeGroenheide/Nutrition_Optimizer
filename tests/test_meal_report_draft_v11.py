"""Transactional schema-v11 migration coverage for report fact state."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from nutrition_optimizer.durable_state import (
    DraftClarification,
    DraftPlannedItem,
    DurableMealState,
    MealReportDraftConflictError,
)
from nutrition_optimizer.fdmealplanner import (
    CATALOG_SCHEMA_VERSION,
    NutritionCatalogError,
    OfficialNutritionCatalog,
)
from nutrition_optimizer.fdmealplanner import catalog as catalog_module
from nutrition_optimizer.food_resolution import FoodResolutionRequest, LocalFDFoodResolver, ResolvedFood
from nutrition_optimizer.meal_report import (
    MealPlan,
    PlannedMealItem,
    ProposedEatenMealItem,
    ReconciledMealReport,
)
from tests.test_food_resolution import component, mapped, menu_day


DAY = date(2026, 9, 14)
NOW = datetime(2026, 9, 14, 16, tzinfo=timezone.utc)


class MealReportDraftV11MigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "nutrition.sqlite3"
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self._close)
        self.catalog.synchronize_fd_refresh(
            mapped(
                menu_day(DAY, 1, "Breakfast", ((component(101, "Eggs"), "Grill"),)),
                menu_day(DAY, 2, "Lunch", ((component(201, "Chicken"), "Grill"),)),
            ),
            requested_start=DAY,
            requested_end=DAY,
            observed_at=NOW,
        )
        self.state = DurableMealState(self.catalog)

    def _close(self) -> None:
        if getattr(self, "catalog", None) is not None:
            self.catalog.close()

    def _plan(self, meal: int, name: str) -> MealPlan:
        food = LocalFDFoodResolver(self.catalog).resolve(
            FoodResolutionRequest(name, DAY, meal=meal)
        )
        assert isinstance(food, ResolvedFood)
        return MealPlan(
            DAY,
            meal,
            (PlannedMealItem(food, Decimal("1"), "one serving"),),
        )

    def _seed_v11_and_downgrade_to_v10(self) -> dict[str, str]:
        breakfast = self.state.save_meal_plan(
            self._plan(1, "Eggs"), meal_slot="breakfast", plan_id="migration-breakfast"
        )
        open_draft, _ = self.state.save_meal_report_draft(
            chat_guid="migration-chat",
            persisted_plan=breakfast,
            current_draft=None,
            planned_items=(
                DraftPlannedItem(
                    "item_1",
                    "eaten",
                    Decimal("1"),
                    "planned_quantity",
                    "eggs",
                    last_source_event_id="open-event",
                ),
            ),
            unplanned_items=(),
            clarifications=(DraftClarification("completion_required"),),
            last_prompt="Anything else?",
            source_event_id="open-event",
            intent="meal_report",
            processed_at=NOW,
        )
        lunch_plan = self._plan(2, "Chicken")
        lunch = self.state.save_meal_plan(
            lunch_plan, meal_slot="lunch", plan_id="migration-lunch"
        )
        completed_draft, _ = self.state.save_meal_report_draft(
            chat_guid="completed-chat",
            persisted_plan=lunch,
            current_draft=None,
            planned_items=(
                DraftPlannedItem(
                    "item_1",
                    "eaten",
                    Decimal("1"),
                    "planned_quantity",
                    "chicken",
                    last_source_event_id="completed-start",
                ),
            ),
            unplanned_items=(),
            clarifications=(DraftClarification("completion_required"),),
            last_prompt="Anything else?",
            source_event_id="completed-start",
            intent="meal_report",
            processed_at=NOW,
        )
        report = ReconciledMealReport(
            lunch_plan,
            (
                ProposedEatenMealItem(
                    lunch_plan.items[0], Decimal("1"), "planned_quantity", "chicken"
                ),
            ),
            (),
            (),
            (),
            (),
        )
        self.state.apply_draft_reconciled_meal_report(
            completed_draft,
            report,
            source_event_id="completed-event",
            reply_text="Logged.",
            intent="clarification_answer",
            recorded_at=NOW,
        )
        intake_id = self.catalog._connection.execute(
            "SELECT intake_id FROM accepted_intake_entries WHERE source_event_id = 'completed-event'"
        ).fetchone()[0]
        self.catalog.close()
        self._downgrade_to_v10()
        return {
            "open_draft": open_draft.draft_id,
            "completed_draft": completed_draft.draft_id,
            "intake_id": str(intake_id),
        }

    def _downgrade_to_v10(self) -> None:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.executescript(
                """
                BEGIN;
                DROP TABLE meal_report_draft_planned_item_history;
                DROP TABLE meal_report_draft_unplanned_item_history;
                DROP INDEX meal_report_drafts_one_open_chat_idx;
                CREATE UNIQUE INDEX meal_report_drafts_one_open_chat_idx
                ON meal_report_drafts(chat_guid)
                WHERE status IN ('draft', 'awaiting_clarification');

                ALTER TABLE meal_report_draft_planned_items RENAME TO planned_v11;
                CREATE TABLE meal_report_draft_planned_items (
                    draft_id TEXT NOT NULL,
                    plan_item_id TEXT NOT NULL CHECK (length(trim(plan_item_id)) > 0),
                    action TEXT NOT NULL CHECK (action IN ('eaten', 'skipped')),
                    official_servings TEXT,
                    quantity_source TEXT CHECK (quantity_source IN (
                        'planned_quantity', 'explicit_deterministic', 'explicit_semantic_estimate'
                    )),
                    original_reference_text TEXT NOT NULL CHECK (length(trim(original_reference_text)) > 0),
                    original_quantity_text TEXT,
                    PRIMARY KEY (draft_id, plan_item_id),
                    FOREIGN KEY (draft_id) REFERENCES meal_report_drafts(draft_id)
                );
                INSERT INTO meal_report_draft_planned_items
                SELECT draft_id, plan_item_id, action, official_servings, quantity_source,
                       original_reference_text, original_quantity_text
                FROM planned_v11;
                DROP TABLE planned_v11;

                ALTER TABLE meal_report_draft_unplanned_items RENAME TO unplanned_v11;
                CREATE TABLE meal_report_draft_unplanned_items (
                    draft_id TEXT NOT NULL,
                    item_position INTEGER NOT NULL CHECK (item_position >= 0),
                    food_text TEXT NOT NULL CHECK (length(trim(food_text)) > 0),
                    quantity_text TEXT,
                    occurrence_id INTEGER NOT NULL,
                    nutrition_snapshot_id INTEGER NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_value TEXT NOT NULL,
                    content_signature TEXT NOT NULL,
                    official_servings TEXT NOT NULL,
                    quantity_source TEXT NOT NULL,
                    confidence TEXT,
                    PRIMARY KEY (draft_id, item_position),
                    FOREIGN KEY (draft_id) REFERENCES meal_report_drafts(draft_id)
                );
                INSERT INTO meal_report_draft_unplanned_items
                SELECT draft_id, item_position, food_text, quantity_text, occurrence_id,
                       nutrition_snapshot_id, source_kind, source_value,
                       content_signature, official_servings, quantity_source, confidence
                FROM unplanned_v11;
                DROP TABLE unplanned_v11;

                ALTER TABLE meal_report_draft_clarifications RENAME TO clarifications_v11;
                CREATE TABLE meal_report_draft_clarifications (
                    draft_id TEXT NOT NULL,
                    clarification_position INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    plan_item_id TEXT,
                    food_text TEXT,
                    PRIMARY KEY (draft_id, clarification_position),
                    FOREIGN KEY (draft_id) REFERENCES meal_report_drafts(draft_id)
                );
                INSERT INTO meal_report_draft_clarifications
                SELECT draft_id, clarification_position, reason, plan_item_id, food_text
                FROM clarifications_v11;
                DROP TABLE clarifications_v11;
                PRAGMA user_version = 10;
                COMMIT;
                """
            )
        finally:
            connection.close()

    def test_fresh_schema_includes_v11_facts_and_reopens(self) -> None:
        self.assertEqual(CATALOG_SCHEMA_VERSION, 14)
        self.assertEqual(
            self.catalog._connection.execute("PRAGMA user_version").fetchone()[0],
            CATALOG_SCHEMA_VERSION,
        )
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.assertEqual(
            self.catalog._connection.execute("PRAGMA user_version").fetchone()[0],
            CATALOG_SCHEMA_VERSION,
        )

    def test_v10_migration_preserves_drafts_intake_applications_and_ids(self) -> None:
        ids = self._seed_v11_and_downgrade_to_v10()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)

        self.assertEqual(
            self.catalog._connection.execute("PRAGMA user_version").fetchone()[0],
            CATALOG_SCHEMA_VERSION,
        )
        open_draft = self.state.load_meal_report_draft(ids["open_draft"])
        completed = self.state.load_meal_report_draft(ids["completed_draft"])
        self.assertEqual(open_draft.status, "awaiting_clarification")  # type: ignore[union-attr]
        self.assertEqual(open_draft.planned_items[0].quantity_status, "resolved")  # type: ignore[union-attr]
        self.assertEqual(completed.status, "completed")  # type: ignore[union-attr]
        self.assertEqual(
            self.catalog._connection.execute(
                "SELECT intake_id FROM accepted_intake_entries WHERE source_event_id = 'completed-event'"
            ).fetchone()[0],
            ids["intake_id"],
        )
        self.assertEqual(
            self.catalog._connection.execute(
                "SELECT plan_id FROM meal_report_applications WHERE source_event_id = 'completed-event'"
            ).fetchone()[0],
            "migration-lunch",
        )
        self.assertEqual(
            self.catalog._connection.execute(
                "SELECT plan_item_id FROM meal_plan_items WHERE plan_id = 'migration-breakfast'"
            ).fetchone()[0],
            "item_1",
        )
        self.assertEqual(self.catalog._connection.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_v11_constraint_allows_distinct_plan_drafts_and_rejects_duplicate_context(self) -> None:
        breakfast = self.state.save_meal_plan(
            self._plan(1, "Eggs"), meal_slot="breakfast", plan_id="constraint-breakfast"
        )
        lunch = self.state.save_meal_plan(
            self._plan(2, "Chicken"), meal_slot="lunch", plan_id="constraint-lunch"
        )
        for plan, event in ((breakfast, "breakfast-event"), (lunch, "lunch-event")):
            self.state.save_meal_report_draft(
                chat_guid="one-chat",
                persisted_plan=plan,
                current_draft=None,
                planned_items=(),
                unplanned_items=(),
                clarifications=(DraftClarification("completion_required"),),
                last_prompt="Anything else?",
                source_event_id=event,
                intent="meal_report",
            )
        self.assertEqual(len(self.state.list_active_meal_report_drafts("one-chat")), 2)
        with self.assertRaises(MealReportDraftConflictError):
            self.state.save_meal_report_draft(
                chat_guid="one-chat",
                persisted_plan=breakfast,
                current_draft=None,
                planned_items=(),
                unplanned_items=(),
                clarifications=(DraftClarification("completion_required"),),
                last_prompt="Anything else?",
                source_event_id="duplicate-plan-event",
                intent="meal_report",
            )

    def test_injected_v11_failure_rolls_back_to_valid_v10(self) -> None:
        self._seed_v11_and_downgrade_to_v10()
        original = catalog_module._migrate_report_draft_facts_v11

        def migrate_then_fail(connection):
            original(connection)
            raise sqlite3.OperationalError("injected migration failure")

        with patch.object(
            catalog_module,
            "_migrate_report_draft_facts_v11",
            side_effect=migrate_then_fail,
        ):
            with self.assertRaisesRegex(NutritionCatalogError, "unable to migrate"):
                OfficialNutritionCatalog(self.path)

        connection = sqlite3.connect(self.path)
        try:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 10)
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(meal_report_draft_planned_items)"
                )
            }
            self.assertNotIn("quantity_status", columns)
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
