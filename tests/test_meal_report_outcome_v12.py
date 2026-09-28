"""Transactional schema-v12 coverage for durable pre-route message outcomes."""

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


DAY = date(2026, 9, 16)
NOW = datetime(2026, 9, 16, 16, tzinfo=timezone.utc)


class MealReportOutcomeV12MigrationTests(unittest.TestCase):
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

    @staticmethod
    def _report(plan: MealPlan, servings: Decimal = Decimal("1")) -> ReconciledMealReport:
        return ReconciledMealReport(
            plan,
            (
                ProposedEatenMealItem(
                    plan.items[0],
                    servings,
                    "explicit_deterministic",
                    f"{servings} serving",
                ),
            ),
            (),
            (),
            (),
            (),
        )

    def _seed_routed_v12_history(self) -> dict[str, str]:
        breakfast = self.state.save_meal_plan(
            self._plan(1, "Eggs"), meal_slot="breakfast", plan_id="v12-breakfast"
        )
        draft, _ = self.state.save_meal_report_draft(
            chat_guid="v12-chat",
            persisted_plan=breakfast,
            current_draft=None,
            planned_items=(
                DraftPlannedItem(
                    "item_1",
                    "eaten",
                    Decimal("1"),
                    "planned_quantity",
                    "eggs",
                    last_source_event_id="draft-open",
                ),
            ),
            unplanned_items=(),
            clarifications=(DraftClarification("completion_required"),),
            last_prompt="Anything else?",
            source_event_id="draft-open",
            intent="meal_report",
            processed_at=NOW,
        )
        corrected, _ = self.state.save_meal_report_draft(
            chat_guid="v12-chat",
            persisted_plan=breakfast,
            current_draft=draft,
            planned_items=(
                DraftPlannedItem(
                    "item_1",
                    "eaten",
                    Decimal("2"),
                    "explicit_deterministic",
                    "actually two servings",
                    original_quantity_text="2 servings",
                    last_source_event_id="draft-correction",
                    fact_revision=1,
                ),
            ),
            unplanned_items=(),
            clarifications=(DraftClarification("completion_required"),),
            last_prompt="Anything else?",
            source_event_id="draft-correction",
            intent="clarification_answer",
            processed_at=NOW,
        )
        lunch_plan = self._plan(2, "Chicken")
        lunch = self.state.save_meal_plan(
            lunch_plan, meal_slot="lunch", plan_id="v12-lunch"
        )
        self.state.apply_reconciled_meal_report(
            lunch,
            self._report(lunch_plan),
            source_event_id="lunch-application",
            chat_guid="v12-chat",
            reply_text="Logged lunch.",
            unavailable_reply_text="Lunch is unavailable.",
            recorded_at=NOW,
        )
        intake_id = str(
            self.catalog._connection.execute(
                "SELECT intake_id FROM accepted_intake_entries "
                "WHERE source_event_id = 'lunch-application'"
            ).fetchone()[0]
        )
        return {
            "draft_id": corrected.draft_id,
            "intake_id": intake_id,
        }

    def _downgrade_to_v11(self) -> None:
        self.catalog.close()
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.executescript(
                """
                BEGIN;
                DROP INDEX meal_report_message_events_draft_idx;
                ALTER TABLE meal_report_message_events RENAME TO message_events_v12;
                CREATE TABLE meal_report_message_events (
                    source_event_id TEXT PRIMARY KEY CHECK (length(trim(source_event_id)) > 0),
                    chat_guid TEXT NOT NULL CHECK (length(trim(chat_guid)) > 0),
                    plan_id TEXT NOT NULL,
                    draft_id TEXT,
                    intent TEXT NOT NULL CHECK (intent IN (
                        'meal_report', 'clarification_answer', 'location_question',
                        'replacement_request', 'meal_request', 'unsupported_or_ambiguous'
                    )),
                    reply_text TEXT NOT NULL CHECK (length(trim(reply_text)) > 0),
                    processed_at TEXT NOT NULL,
                    FOREIGN KEY (plan_id) REFERENCES meal_plans(plan_id),
                    FOREIGN KEY (draft_id) REFERENCES meal_report_drafts(draft_id)
                );
                INSERT INTO meal_report_message_events
                (source_event_id, chat_guid, plan_id, draft_id, intent,
                 reply_text, processed_at)
                SELECT source_event_id, chat_guid, plan_id, draft_id, intent,
                       reply_text, processed_at
                FROM message_events_v12
                WHERE plan_id IS NOT NULL;
                DROP TABLE message_events_v12;
                CREATE INDEX meal_report_message_events_draft_idx
                ON meal_report_message_events(draft_id, processed_at);
                PRAGMA user_version = 11;
                COMMIT;
                """
            )
        finally:
            connection.close()

    def test_fresh_v12_supports_routed_and_plan_free_outcomes_and_reopens(self) -> None:
        self.assertEqual(CATALOG_SCHEMA_VERSION, 14)
        event = self.state.record_pre_route_meal_report_outcome(
            source_event_id="pre-route-guid",
            chat_guid="v12-chat",
            outcome_type="ambiguous_meal_context",
            reply_text="Which meal are you reporting: breakfast or lunch?",
            processed_at=NOW,
        )
        self.assertIsNone(event.plan_id)
        self.assertIsNone(event.draft_id)
        self.assertIsNone(event.application_source_event_id)
        plan = self.state.save_meal_plan(
            self._plan(1, "Eggs"), meal_slot="breakfast", plan_id="routed-plan"
        )
        routed = self.state.record_meal_report_message_event(
            source_event_id="routed-guid",
            chat_guid="v12-chat",
            persisted_plan=plan,
            intent="location_question",
            reply_text="Eggs are at Grill.",
            processed_at=NOW,
        )
        self.assertEqual(routed.plan_id, plan.plan_id)
        self.assertEqual(
            self.catalog._connection.execute("PRAGMA foreign_key_check").fetchall(), []
        )
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        self.assertEqual(
            self.catalog._connection.execute("PRAGMA user_version").fetchone()[0], CATALOG_SCHEMA_VERSION
        )
        self.assertEqual(
            self.state.load_meal_report_message_event("pre-route-guid"), event
        )

    def test_v11_to_v12_preserves_drafts_facts_history_applications_and_events(self) -> None:
        ids = self._seed_routed_v12_history()
        self._downgrade_to_v11()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)

        self.assertEqual(
            self.catalog._connection.execute("PRAGMA user_version").fetchone()[0], CATALOG_SCHEMA_VERSION
        )
        draft = self.state.load_meal_report_draft(ids["draft_id"])
        assert draft is not None
        self.assertEqual(draft.planned_items[0].official_servings, Decimal("2"))
        self.assertEqual(draft.planned_items[0].fact_revision, 1)
        history = self.catalog._connection.execute(
            "SELECT fact_revision, official_servings "
            "FROM meal_report_draft_planned_item_history WHERE draft_id = ?",
            (ids["draft_id"],),
        ).fetchall()
        self.assertEqual([(row[0], row[1]) for row in history], [(0, "1")])
        application = self.state.load_applied_meal_report("lunch-application")
        assert application is not None
        self.assertEqual(application.accepted_intake_entries[0].intake_id, ids["intake_id"])
        final_event = self.state.load_meal_report_message_event("lunch-application")
        assert final_event is not None
        self.assertEqual(final_event.outcome_type, "final_application")
        self.assertEqual(final_event.application_source_event_id, "lunch-application")
        self.assertIsNotNone(self.state.load_meal_report_message_event("draft-open"))
        self.assertIsNotNone(self.state.load_meal_report_message_event("draft-correction"))
        self.assertEqual(
            self.catalog._connection.execute("PRAGMA foreign_key_check").fetchall(), []
        )

    def test_injected_v12_failure_leaves_valid_v11_then_reopens_normally(self) -> None:
        self._seed_routed_v12_history()
        self._downgrade_to_v11()
        original = catalog_module._migrate_message_outcomes_v12

        def migrate_then_fail(connection):
            original(connection)
            raise sqlite3.OperationalError("injected v12 migration failure")

        with patch.object(
            catalog_module,
            "_migrate_message_outcomes_v12",
            side_effect=migrate_then_fail,
        ):
            with self.assertRaisesRegex(NutritionCatalogError, "unable to migrate"):
                OfficialNutritionCatalog(self.path)

        connection = sqlite3.connect(self.path)
        try:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 11)
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(meal_report_message_events)"
                )
            }
            self.assertNotIn("outcome_type", columns)
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
        finally:
            connection.close()

        self.catalog = OfficialNutritionCatalog(self.path)
        self.assertEqual(
            self.catalog._connection.execute("PRAGMA user_version").fetchone()[0], CATALOG_SCHEMA_VERSION
        )
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.assertEqual(
            self.catalog._connection.execute("PRAGMA user_version").fetchone()[0], CATALOG_SCHEMA_VERSION
        )


if __name__ == "__main__":
    unittest.main()
