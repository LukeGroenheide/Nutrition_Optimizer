"""Fixed shake reminders, durable intake, and meal-only recommendation accounting."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest
from unittest.mock import patch

from nutrition_optimizer.durable_state import DurableMealState, DraftClarification
from nutrition_optimizer.fdmealplanner import NutritionCatalogError, OfficialNutritionCatalog
from nutrition_optimizer.fdmealplanner import catalog as catalog_module
from nutrition_optimizer.food_resolution import FoodResolutionRequest, LocalFDFoodResolver, ResolvedFood
from nutrition_optimizer.meal_report import (
    MealPlan, MealReportReconciler, PlannedMealItem, ProposedEatenMealItem,
    ReconciledMealReport,
)
from nutrition_optimizer.meal_report_conversation import (
    MealReportConversationOrchestrator, MealReportConversationResult,
)
from nutrition_optimizer.messaging.application import ApplicationMessagingConfig
from nutrition_optimizer.messaging.meal_workflow import (
    ActiveMealPlanContextResolver, ConversationalMealReportMessageHandler,
    MealWorkflowMessageError,
)
from nutrition_optimizer.messaging.webhook import IncomingMessage
from nutrition_optimizer.nutrition import DailyTargets
from nutrition_optimizer.portion_interpretation import NaturalPortionInterpreter
from tests.test_food_resolution import component, mapped, menu_day
from tests.test_meal_report_conversation import RecordingSender, ScriptedInterpreter


DAY = date(2026, 9, 28)
NEXT_DAY = date(2026, 9, 29)
NOW = datetime(2026, 9, 28, 15, tzinfo=timezone.utc)
CHAT = "shake-test-chat"
TARGETS = DailyTargets(Decimal("3000"), Decimal("150"), Decimal("400"), Decimal("100"))


class ShakeIntakeV14Tests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "nutrition.sqlite3"
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self.catalog.close)
        self.catalog.synchronize_fd_refresh(
            mapped(
                menu_day(DAY, 1, "Breakfast", ((component(101, "Eggs"), "Grill"),)),
                menu_day(DAY, 2, "Lunch", ((component(201, "Chicken"), "Grill"),)),
                menu_day(DAY, 3, "Dinner", ((component(301, "Rice"), "Grill"),)),
                menu_day(NEXT_DAY, 1, "Breakfast", ((component(401, "Oats"), "Grill"),)),
            ),
            requested_start=DAY,
            requested_end=NEXT_DAY,
            observed_at=NOW,
        )
        self.state = DurableMealState(self.catalog)
        self.today = DAY
        self.sender = RecordingSender()
        self.handler = self._handler()

    def _handler(self) -> ConversationalMealReportMessageHandler:
        resolver = LocalFDFoodResolver(self.catalog)
        conversation = MealReportConversationOrchestrator(
            self.state,
            MealReportReconciler(ScriptedInterpreter({}), resolver, NaturalPortionInterpreter()),
        )
        return ConversationalMealReportMessageHandler(
            self.sender,
            ApplicationMessagingConfig(expected_chat_guid=CHAT),
            conversation,
            ActiveMealPlanContextResolver(self.state, service_date_provider=lambda: self.today),
            shake_state=self.state,
        )

    def _message(self, guid: str, text: str) -> IncomingMessage:
        return IncomingMessage(guid, text, "sender", CHAT, False)

    def _applied(self, meal: int, name: str, *, local_date: date = DAY, slot: str | None = None):
        food = LocalFDFoodResolver(self.catalog).resolve(
            FoodResolutionRequest(name, local_date, meal=meal)
        )
        assert isinstance(food, ResolvedFood)
        plan = MealPlan(
            local_date, meal,
            (PlannedMealItem(food, Decimal("1"), "one serving"),),
        )
        saved = self.state.save_meal_plan(plan, meal_slot=slot)
        report = ReconciledMealReport(
            plan,
            (ProposedEatenMealItem(plan.items[0], Decimal("1"), "explicit_deterministic", "one serving"),),
            (), (), (), (),
        )
        application = self.state.apply_reconciled_meal_report(
            saved, report, source_event_id=f"meal-{local_date}-{meal}-{slot}", recorded_at=NOW
        )
        completed = self.state.load_meal_plan(saved.plan_id)
        assert completed is not None
        return completed, application

    def _complete(self, meal: int, name: str, slot: str | None = None) -> None:
        completed, application = self._applied(meal, name, slot=slot)
        result = MealReportConversationResult("applied", completed, "Meal logged.", application)
        with patch.object(self.handler.conversation_orchestrator, "process", return_value=result):
            self.assertTrue(self.handler.handle(self._message(f"report-{meal}-{slot}", "ate it")))

    def test_reminders_only_breakfast_and_dinner_once(self) -> None:
        self._complete(1, "Eggs")
        self.assertEqual([text for _, text in self.sender.calls].count("Drink a shake"), 1)
        self.assertIsNotNone(self.state.load_shake(DAY, "breakfast_shake")["reminder_delivered_at"])
        self._complete(2, "Chicken")
        self.assertEqual([text for _, text in self.sender.calls].count("Drink a shake"), 1)
        self._complete(3, "Rice")
        self.assertEqual([text for _, text in self.sender.calls].count("Drink a shake"), 2)

    def test_natural_confirmation_without_active_meal_logs_actual_only(self) -> None:
        self._complete(1, "Eggs")
        self.assertIsNone(self.state.load_active_meal_plan(DAY, 1))
        meal_only = self.state.load_recommendation_ledger(DAY).total_consumed_nutrients
        self.assertTrue(self.handler.handle(self._message("natural-shake", "I had a shake")))
        self.assertEqual(self.sender.calls[-1][1], "Logged your breakfast shake.")
        self.assertEqual(self.state.load_shake(DAY, "breakfast_shake")["status"], "confirmed")
        actual = self.state.load_actual_daily_nutrients(DAY)
        self.assertEqual(actual.calories_kcal - meal_only.calories_kcal, Decimal("580"))
        self.assertEqual(self.state.load_recommendation_ledger(DAY).total_consumed_nutrients,
                         meal_only)
        self.handler.handle(self._message("natural-shake", "I had a shake"))
        self.assertEqual(self.state.load_actual_daily_nutrients(DAY), actual)

    def test_natural_shake_reply_asks_when_two_are_pending(self) -> None:
        self._complete(1, "Eggs")
        self._complete(3, "Rice")
        self.handler.handle(self._message("ambiguous-natural-shake", "I drank the shake"))
        self.assertEqual(self.sender.calls[-1][1], "Which shake: breakfast or dinner?")
        self.assertEqual(self.state.pending_shakes(DAY, CHAT),
                         ("breakfast_shake", "dinner_shake"))
        self.assertIsNotNone(self.state.load_shake(DAY, "dinner_shake")["reminder_delivered_at"])
        self.assertFalse(self.state.claim_shake_reminder(
            self.state.load_meal_plan_for_source_event("meal-2026-09-28-1-None").plan_id, CHAT
        ))
        self.assertEqual([text for _, text in self.sender.calls].count("Drink a shake"), 2)

    def test_brunch_produces_no_reminder(self) -> None:
        completed, application = self._applied(2, "Chicken", slot="brunch")
        result = MealReportConversationResult("applied", completed, "Meal logged.", application)
        with patch.object(self.handler.conversation_orchestrator, "process", return_value=result):
            self.handler.handle(self._message("brunch-report", "ate it"))
        self.assertNotIn("Drink a shake", [text for _, text in self.sender.calls])

    def test_applied_guid_replay_recovers_unsent_reminder_once(self) -> None:
        completed, _ = self._applied(1, "Eggs")
        replay = MealReportConversationResult("replayed", completed, "Already logged.")
        with patch.object(self.handler.conversation_orchestrator, "process", return_value=replay):
            self.handler.handle(self._message("meal-2026-09-28-1-None", "ate it"))
            self.handler.handle(self._message("meal-2026-09-28-1-None", "ate it"))
        self.assertEqual([text for _, text in self.sender.calls].count("Drink a shake"), 1)

    def test_existing_meal_guid_cannot_resolve_pending_shake(self) -> None:
        completed, _ = self._applied(1, "Eggs")
        self.assertTrue(self.state.claim_shake_reminder(completed.plan_id, CHAT))
        replay = MealReportConversationResult("replayed", completed, "Already logged.")
        with patch.object(self.handler.conversation_orchestrator, "process", return_value=replay):
            self.handler.handle(self._message("meal-2026-09-28-1-None", "done"))
        self.assertEqual(self.state.load_shake(DAY, "breakfast_shake")["status"], "pending")

    def test_uncertain_reminder_send_is_not_duplicated_on_retry(self) -> None:
        completed, application = self._applied(1, "Eggs")
        result = MealReportConversationResult("applied", completed, "Meal logged.", application)
        original_send = self.sender.send_text

        def fail_reminder(chat_guid: str, text: str) -> object:
            if text == "Drink a shake":
                self.sender.calls.append((chat_guid, text))
                raise RuntimeError("uncertain transport result")
            return original_send(chat_guid, text)

        with patch.object(self.handler.conversation_orchestrator, "process", return_value=result):
            with patch.object(self.sender, "send_text", side_effect=fail_reminder):
                with self.assertRaises(MealWorkflowMessageError):
                    self.handler.handle(self._message("first-attempt", "ate it"))
            self.handler.handle(self._message("first-attempt", "ate it"))
        self.assertEqual([text for _, text in self.sender.calls].count("Drink a shake"), 1)
        self.assertIsNone(self.state.load_shake(DAY, "breakfast_shake")["reminder_delivered_at"])

    def test_confirm_skip_replay_reopen_and_recommendation_isolation(self) -> None:
        self._complete(1, "Eggs")
        baseline = self.state.load_recommendation_ledger(DAY).balance_against(TARGETS)
        actual_baseline = self.state.load_actual_daily_nutrients(DAY)
        self.handler.handle(self._message("shake-done", "done"))
        row = self.state.load_shake(DAY, "breakfast_shake")
        self.assertEqual(row["status"], "confirmed")
        self.assertEqual(tuple(row[key] for key in (
            "calories_kcal", "protein_g", "carbohydrates_g", "fat_g"
        )), ("580", "23", "116", "3"))
        actual = self.state.load_actual_daily_nutrients(DAY)
        self.assertEqual(actual.calories_kcal - actual_baseline.calories_kcal, Decimal("580"))
        self.assertEqual(actual.protein_g - actual_baseline.protein_g, Decimal("23"))
        self.assertEqual(actual.carbohydrates_g - actual_baseline.carbohydrates_g, Decimal("116"))
        self.assertEqual(actual.fat_g - actual_baseline.fat_g, Decimal("3"))
        self.assertEqual(self.state.load_recommendation_ledger(DAY).balance_against(TARGETS), baseline)
        self.assertEqual(self.state.calculate_daily_balance(DAY, TARGETS).consumed, actual)
        self.handler.handle(self._message("shake-done", "done"))
        self.assertEqual(self.state.load_actual_daily_nutrients(DAY), actual)
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        self.handler = self._handler()
        self.handler.handle(self._message("shake-done", "done"))
        self.assertEqual(self.state.load_actual_daily_nutrients(DAY), actual)
        self._complete(3, "Rice")
        before_skip = self.state.load_recommendation_ledger(DAY).balance_against(TARGETS)
        self.handler.handle(self._message("shake-skip", "skip"))
        self.assertEqual(self.state.load_shake(DAY, "dinner_shake")["status"], "skipped")
        self.assertEqual(self.state.load_recommendation_ledger(DAY).balance_against(TARGETS), before_skip)
        self.assertEqual(self.state.load_actual_daily_nutrients(DAY).calories_kcal,
                         self.state.load_recommendation_ledger(DAY).total_consumed_nutrients.calories_kcal + 580)

    def test_two_confirmed_shakes_and_ambiguous_bare_reply(self) -> None:
        self._complete(1, "Eggs")
        self._complete(3, "Rice")
        meal_only = self.state.load_recommendation_ledger(DAY).balance_against(TARGETS)
        self.handler.handle(self._message("ambiguous-done", "done"))
        self.assertEqual(self.sender.calls[-1][1], "Which shake: breakfast or dinner?")
        self.assertEqual(self.state.load_actual_daily_nutrients(DAY), meal_only.consumed)
        self.handler.handle(self._message("breakfast-done", "breakfast shake done"))
        self.handler.handle(self._message("dinner-done", "dinner shake done"))
        actual = self.state.load_actual_daily_nutrients(DAY)
        self.assertEqual(actual.calories_kcal - meal_only.consumed.calories_kcal, Decimal("1160"))
        self.assertEqual(actual.protein_g - meal_only.consumed.protein_g, Decimal("46"))
        self.assertEqual(actual.carbohydrates_g - meal_only.consumed.carbohydrates_g, Decimal("232"))
        self.assertEqual(actual.fat_g - meal_only.consumed.fat_g, Decimal("6"))
        self.assertEqual(self.state.load_recommendation_ledger(DAY).balance_against(TARGETS), meal_only)

    def test_skipped_breakfast_does_not_change_later_targets(self) -> None:
        self._complete(1, "Eggs")
        baseline = self.state.load_recommendation_ledger(DAY).balance_against(TARGETS)
        self.handler.handle(self._message("breakfast-skip", "skip"))
        self.assertEqual(self.state.load_shake(DAY, "breakfast_shake")["status"], "skipped")
        self.assertEqual(self.state.load_recommendation_ledger(DAY).balance_against(TARGETS), baseline)
        self.assertEqual(self.state.load_actual_daily_nutrients(DAY), baseline.consumed)

    def test_unresolved_meal_draft_keeps_bare_done(self) -> None:
        self._complete(1, "Eggs")
        saved = self.state.save_meal_plan(
            MealPlan(DAY, 2, (self._meal_item(2, "Chicken"),)), meal_slot="lunch"
        )
        self.state.save_meal_report_draft(
            chat_guid=CHAT, persisted_plan=saved, current_draft=None,
            planned_items=(), unplanned_items=(),
            clarifications=(DraftClarification("completion_required"),),
            last_prompt="Anything else?", source_event_id="draft-open",
            intent="meal_report", processed_at=NOW,
        )
        with patch.object(self.handler.conversation_orchestrator, "process", side_effect=RuntimeError("meal owns reply")):
            with self.assertRaises(Exception):
                self.handler.handle(self._message("bare-done", "done"))
        self.assertEqual(self.state.load_shake(DAY, "breakfast_shake")["status"], "pending")

    def test_unresolved_meal_draft_keeps_natural_shake_reply(self) -> None:
        self._complete(1, "Eggs")
        saved = self.state.save_meal_plan(
            MealPlan(DAY, 2, (self._meal_item(2, "Chicken"),)), meal_slot="lunch"
        )
        self.state.save_meal_report_draft(
            chat_guid=CHAT, persisted_plan=saved, current_draft=None,
            planned_items=(), unplanned_items=(),
            clarifications=(DraftClarification("completion_required"),),
            last_prompt="Anything else?", source_event_id="draft-natural-shake",
            intent="meal_report", processed_at=NOW,
        )
        with patch.object(self.handler.conversation_orchestrator, "process") as process:
            self.assertTrue(self.handler.handle(self._message("natural-with-draft", "I had a shake")))
            process.assert_not_called()
        self.assertEqual(self.sender.calls[-1][1], "Anything else?")
        self.assertEqual(self.state.load_shake(DAY, "breakfast_shake")["status"], "pending")
        self.assertEqual(self.state.list_active_meal_report_drafts(CHAT)[0].unplanned_items, ())

    def test_unresolved_meal_draft_keeps_explicit_shake_reply(self) -> None:
        self._complete(1, "Eggs")
        saved = self.state.save_meal_plan(
            MealPlan(DAY, 2, (self._meal_item(2, "Chicken"),)), meal_slot="lunch"
        )
        self.state.save_meal_report_draft(
            chat_guid=CHAT, persisted_plan=saved, current_draft=None,
            planned_items=(), unplanned_items=(),
            clarifications=(DraftClarification("completion_required"),),
            last_prompt="Anything else?", source_event_id="draft-open-explicit",
            intent="meal_report", processed_at=NOW,
        )
        with patch.object(self.handler.conversation_orchestrator, "process") as process:
            self.assertTrue(self.handler.handle(self._message("explicit-done", "breakfast shake done")))
            process.assert_not_called()
        self.assertEqual(self.sender.calls[-1][1], "Anything else?")
        self.assertEqual(self.state.load_shake(DAY, "breakfast_shake")["status"], "pending")

    def _meal_item(self, meal: int, name: str) -> PlannedMealItem:
        food = LocalFDFoodResolver(self.catalog).resolve(FoodResolutionRequest(name, DAY, meal=meal))
        assert isinstance(food, ResolvedFood)
        return PlannedMealItem(food, Decimal("1"), "one serving")

    def test_detroit_date_scope(self) -> None:
        self._complete(1, "Eggs")
        self.today = NEXT_DAY
        self.assertEqual(self.state.pending_shakes(NEXT_DAY, CHAT), ())
        self.assertEqual(self.state.pending_shakes(DAY, CHAT), ("breakfast_shake",))
        self.assertEqual(self.state.load_actual_daily_nutrients(NEXT_DAY).calories_kcal, Decimal("0"))

    def test_v13_migration_preserves_history_and_reopen(self) -> None:
        self._applied(2, "Chicken")
        self.catalog.close()
        with sqlite3.connect(self.path) as connection:
            before = {}
            tables = [row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' AND name != 'shake_intake'"
            )]
            for table in tables:
                before[table] = connection.execute(f"SELECT * FROM {table}").fetchall()
            connection.execute("DROP TABLE shake_intake")
            connection.execute("PRAGMA user_version = 13")
        self.catalog = OfficialNutritionCatalog(self.path)
        connection = self.catalog._connection
        for table, rows in before.items():
            self.assertEqual(
                [tuple(row) for row in connection.execute(f"SELECT * FROM {table}").fetchall()],
                rows,
            )
        self.assertEqual(connection.execute("SELECT count(*) FROM shake_intake").fetchone()[0], 0)
        self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 14)
        self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
        self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.assertEqual(self.catalog._connection.execute("SELECT count(*) FROM shake_intake").fetchone()[0], 0)
        self.assertEqual(self.catalog._connection.total_changes, 0)
        self.catalog.close()
        with patch.object(catalog_module, "CATALOG_SCHEMA_VERSION", 13):
            with self.assertRaises(NutritionCatalogError):
                OfficialNutritionCatalog(self.path)


if __name__ == "__main__":
    unittest.main()
