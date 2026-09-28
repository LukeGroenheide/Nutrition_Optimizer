"""Deterministic coverage for durable multi-turn meal-report conversations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from nutrition_optimizer.durable_state import DurableMealState, PersistedMealPlan
from nutrition_optimizer.fdmealplanner import CATALOG_SCHEMA_VERSION, OfficialNutritionCatalog
from nutrition_optimizer.fdmealplanner import catalog as catalog_module
from nutrition_optimizer.food_resolution import FoodResolutionRequest, LocalFDFoodResolver, ResolvedFood
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
from nutrition_optimizer.portion_interpretation import NaturalPortionInterpreter
from tests.test_food_resolution import component, mapped, menu_day


DAY = date(2026, 9, 3)
OBSERVED_AT = datetime(2026, 9, 3, 17, tzinfo=timezone.utc)
CHAT_GUID = "chat-conversation-test"


def food(
    component_id: int,
    name: str,
    *,
    serving_quantity: str,
    serving_unit: str,
) -> dict[str, object]:
    value = component(component_id, name)
    value.update(
        {
            "recipePortionSize": serving_quantity,
            "recipePortionSizeUnit": serving_unit,
            "calories": "100",
            "protein": "10",
            "carbohydrates": "10",
            "fat": "4",
            "dietaryFiber": "2",
            "dietaryFiberUOM": "g",
        }
    )
    return value


def semantic(
    *,
    intent: str = "meal_report",
    scope: str = "complete",
    planned: list[dict[str, object]] | None = None,
    unresolved: list[dict[str, object]] | None = None,
    location_plan_item_id: str | None = None,
) -> MealReportSemanticResult:
    return MealReportSemanticResult.model_validate(
        {
            "intent": intent,
            "report_scope": scope,
            "location_plan_item_id": location_plan_item_id,
            "planned_items": planned or [],
            "additional_foods": [],
            "unresolved_statements": unresolved or [],
        }
    )


def eaten(
    plan_item_id: str,
    reference_text: str,
    *,
    relation: str = "as_recommended",
    quantity_text: str | None = None,
    comparison_plan_item_id: str | None = None,
) -> dict[str, object]:
    return {
        "plan_item_id": plan_item_id,
        "reference_text": reference_text,
        "action": "eaten",
        "quantity_relation": relation,
        "quantity_text": quantity_text,
        "comparison_plan_item_id": comparison_plan_item_id,
    }


def skipped(plan_item_id: str, reference_text: str) -> dict[str, object]:
    return {
        "plan_item_id": plan_item_id,
        "reference_text": reference_text,
        "action": "skipped",
        "quantity_relation": None,
        "quantity_text": None,
        "comparison_plan_item_id": None,
    }


@dataclass
class ScriptedInterpreter:
    responses: dict[str, MealReportSemanticResult]
    contexts: list[dict[str, object]] = field(default_factory=list)

    def interpret(self, plan: MealPlan, user_text: str) -> MealReportSemanticResult:
        del plan
        return self.responses[user_text]

    def interpret_with_context(
        self,
        plan: MealPlan,
        user_text: str,
        conversation_context,
    ) -> MealReportSemanticResult:
        del plan
        self.contexts.append(dict(conversation_context))
        return self.responses[user_text]


@dataclass
class FailingInterpreter:
    def interpret(self, plan: MealPlan, user_text: str):
        del plan, user_text
        raise RuntimeError("test semantic failure")

    def interpret_with_context(self, plan: MealPlan, user_text: str, conversation_context):
        del plan, user_text, conversation_context
        raise RuntimeError("test semantic failure")


@dataclass
class CurrentPlanResolver:
    state: DurableMealState
    calls: list[str] = field(default_factory=list)

    def resolve(self, source_event_id: str) -> PersistedMealPlan:
        self.calls.append(source_event_id)
        plan = self.state.load_active_meal_plan(DAY, 2)
        if plan is None:
            raise RuntimeError("test has no active plan")
        return plan


@dataclass
class RecordingSender:
    failure: BaseException | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)

    def send_text(self, chat_guid: str, text: str) -> object:
        self.calls.append((chat_guid, text))
        if self.failure is not None:
            raise self.failure
        return object()


class MealReportConversationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "nutrition.sqlite3"
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(self.catalog.close)
        self.catalog.synchronize_fd_refresh(
            mapped(
                menu_day(
                    DAY,
                    2,
                    "Lunch",
                    (
                        (food(1, "Baked Sweet Potatoes-Master", serving_quantity="1", serving_unit="Each"), "Homestyle"),
                        (food(2, "Herbed Pork Loin", serving_quantity="6", serving_unit="Ounce"), "Homestyle"),
                        (food(3, "Chicken Vesuvio", serving_quantity="1", serving_unit="Each"), "Grill"),
                        (food(4, "Creamy Cauliflower Soup", serving_quantity="1", serving_unit="Cup"), "Soup"),
                    ),
                )
            ),
            requested_start=DAY,
            requested_end=DAY,
            observed_at=OBSERVED_AT,
        )
        self.state = DurableMealState(self.catalog)
        resolver = LocalFDFoodResolver(self.catalog)
        self.plan = MealPlan(
            DAY,
            2,
            (
                PlannedMealItem(self._resolved(resolver, "Baked Sweet Potatoes-Master"), Decimal("2"), "2 baked sweet potatoes"),
                PlannedMealItem(self._resolved(resolver, "Herbed Pork Loin"), Decimal("1"), "about 6 oz"),
                PlannedMealItem(self._resolved(resolver, "Chicken Vesuvio"), Decimal("1"), "1 chicken serving"),
                PlannedMealItem(self._resolved(resolver, "Creamy Cauliflower Soup"), Decimal("1"), "1 cup"),
            ),
        )

    def _resolved(self, resolver: LocalFDFoodResolver, name: str) -> ResolvedFood:
        result = resolver.resolve(FoodResolutionRequest(name, DAY, meal=2))
        assert isinstance(result, ResolvedFood)
        return result

    def _conversation(
        self,
        responses: dict[str, MealReportSemanticResult],
    ) -> tuple[MealReportConversationOrchestrator, ScriptedInterpreter, CurrentPlanResolver]:
        interpreter = ScriptedInterpreter(responses)
        reconciler = MealReportReconciler(
            interpreter,
            LocalFDFoodResolver(self.catalog),
            NaturalPortionInterpreter(),
        )
        return (
            MealReportConversationOrchestrator(self.state, reconciler),
            interpreter,
            CurrentPlanResolver(self.state),
        )

    def _save_plan(self) -> PersistedMealPlan:
        return self.state.save_meal_plan(self.plan)

    def _process(
        self,
        conversation: MealReportConversationOrchestrator,
        resolver: CurrentPlanResolver,
        text: str,
        guid: str,
    ):
        return conversation.process(
            chat_guid=CHAT_GUID,
            user_text=text,
            source_event_id=guid,
            context_resolver=resolver,
        )

    def _application_count(self) -> int:
        return self.catalog._connection.execute(
            "SELECT COUNT(*) FROM meal_report_applications"
        ).fetchone()[0]

    def test_self_contained_complete_report_happy_path_remains_atomic(self) -> None:
        self._save_plan()
        responses = {
            "I had all of it but the soup": semantic(
                planned=[
                    eaten("item_1", "all of it"),
                    eaten("item_2", "all of it"),
                    eaten("item_3", "all of it"),
                    skipped("item_4", "but the soup"),
                ]
            )
        }
        conversation, _, resolver = self._conversation(responses)

        result = self._process(conversation, resolver, "I had all of it but the soup", "happy-1")

        self.assertEqual(result.outcome, "applied")
        self.assertEqual(self._application_count(), 1)
        self.assertEqual(
            [entry.record.name for entry in self.state.get_daily_intake(DAY)],
            ["Baked Sweet Potatoes-Master", "Herbed Pork Loin", "Chicken Vesuvio"],
        )
        persisted = self.state.load_meal_plan(result.persisted_plan.plan_id)
        assert persisted is not None
        self.assertEqual(persisted.status, "applied")
        self.assertIsNone(self.state.load_active_meal_report_draft(CHAT_GUID))

    def test_location_question_uses_authoritative_station_and_creates_no_draft(self) -> None:
        self._save_plan()
        responses = {
            "where is the herbed pork loon located": semantic(
                intent="location_question",
                scope="unknown",
                location_plan_item_id="item_2",
            )
        }
        conversation, _, resolver = self._conversation(responses)

        result = self._process(conversation, resolver, "where is the herbed pork loon located", "location-1")

        self.assertEqual(result.outcome, "location_question")
        self.assertEqual(result.message, "Herbed Pork Loin is at Homestyle.")
        self.assertEqual(self._application_count(), 0)
        self.assertEqual(self.state.get_daily_intake(DAY), ())
        self.assertIsNone(self.state.load_active_meal_report_draft(CHAT_GUID))
        self.assertEqual(self.state.load_active_meal_plan(DAY, 2).status, "active")

    def test_partial_incremental_report_stays_active_until_explicit_completion(self) -> None:
        self._save_plan()
        responses = {
            "I had all of the sweet potato": semantic(
                scope="partial",
                planned=[eaten("item_1", "the sweet potato")],
            ),
            "I had all of the chicken as well": semantic(
                intent="clarification_answer",
                scope="partial",
                planned=[eaten("item_3", "the chicken")],
            ),
            "that's all": semantic(intent="clarification_answer", scope="complete"),
        }
        conversation, interpreter, resolver = self._conversation(responses)

        first = self._process(conversation, resolver, "I had all of the sweet potato", "partial-1")
        self.assertEqual(first.outcome, "clarification_required")
        self.assertIn("Anything else", first.message)
        self.assertEqual(self._application_count(), 0)
        self.assertEqual(self.state.load_active_meal_plan(DAY, 2).status, "active")

        second = self._process(conversation, resolver, "I had all of the chicken as well", "partial-2")
        self.assertEqual(second.outcome, "clarification_required")
        self.assertEqual(self._application_count(), 0)
        self.assertIn("item_1", interpreter.contexts[1]["draft"]["resolved_items"][0].values())

        final = self._process(conversation, resolver, "that's all", "partial-3")
        self.assertEqual(final.outcome, "applied")
        self.assertEqual(self._application_count(), 1)
        self.assertEqual(
            [entry.record.name for entry in self.state.get_daily_intake(DAY)],
            ["Baked Sweet Potatoes-Master", "Chicken Vesuvio"],
        )
        self.assertEqual(self.state.load_active_meal_plan(DAY, 2), None)

    def test_short_fraction_answer_uses_outstanding_plan_item_context(self) -> None:
        self._save_plan()
        responses = {
            "some potato": semantic(
                scope="partial",
                planned=[eaten("item_1", "some potato", relation="modified", quantity_text="some")],
            ),
            "1/4": semantic(
                intent="clarification_answer",
                scope="partial",
                planned=[
                    eaten(
                        "item_1",
                        "1/4",
                        relation="fraction_of_recommended",
                        quantity_text="1/4",
                    )
                ],
            ),
            "that's all": semantic(intent="clarification_answer", scope="complete"),
        }
        conversation, interpreter, resolver = self._conversation(responses)

        initial = self._process(conversation, resolver, "some potato", "fraction-1")
        self.assertIn(
            "I have “some” for Baked Sweet Potatoes-Master",
            initial.message,
        )
        answer = self._process(conversation, resolver, "1/4", "fraction-2")
        self.assertEqual(answer.outcome, "clarification_required")
        self.assertNotIn("How much Baked Sweet Potatoes-Master", answer.message)
        self.assertIn("item_1", str(interpreter.contexts[1]))
        draft = self.state.load_active_meal_report_draft(CHAT_GUID)
        assert draft is not None
        self.assertEqual(draft.planned_items[0].official_servings, Decimal("0.50"))
        self.assertEqual(self._application_count(), 0)

        final = self._process(conversation, resolver, "that's all", "fraction-3")
        self.assertEqual(final.outcome, "applied")
        self.assertEqual(self.state.get_daily_intake(DAY)[0].official_servings, Decimal("0.50"))

    def test_multiple_short_answers_resolve_exact_pork_unit_and_plan_fraction(self) -> None:
        self._save_plan()
        responses = {
            "some potato and some pork": semantic(
                scope="partial",
                planned=[
                    eaten("item_1", "some potato", relation="modified", quantity_text="some"),
                    eaten("item_2", "some pork", relation="modified", quantity_text="some"),
                ],
            ),
            "6oz of the pork and .5 of the recommended potato amount": semantic(
                intent="clarification_answer",
                scope="partial",
                planned=[
                    eaten("item_2", "6oz of the pork", relation="modified", quantity_text="6oz"),
                    eaten("item_1", ".5 of the recommended potato amount", relation="fraction_of_recommended", quantity_text=".5"),
                ],
            ),
        }
        conversation, _, resolver = self._conversation(responses)

        initial = self._process(conversation, resolver, "some potato and some pork", "multi-1")
        self.assertIn("Baked Sweet Potatoes-Master", initial.message)
        self.assertIn("Herbed Pork Loin", initial.message)
        answer = self._process(
            conversation,
            resolver,
            "6oz of the pork and .5 of the recommended potato amount",
            "multi-2",
        )
        self.assertEqual(answer.outcome, "clarification_required")
        self.assertIn("Anything else", answer.message)
        draft = self.state.load_active_meal_report_draft(CHAT_GUID)
        assert draft is not None
        quantities = {item.plan_item_id: item.official_servings for item in draft.planned_items}
        self.assertEqual(quantities, {"item_1": Decimal("1.0"), "item_2": Decimal("1")})

    def test_ambiguous_short_pronoun_stays_unresolved_without_a_guess(self) -> None:
        self._save_plan()
        responses = {
            "some potato and some pork": semantic(
                scope="partial",
                planned=[
                    eaten("item_1", "some potato", relation="modified", quantity_text="some"),
                    eaten("item_2", "some pork", relation="modified", quantity_text="some"),
                ],
            ),
            "half": semantic(
                intent="clarification_answer",
                scope="partial",
                unresolved=[{"reference_text": "half", "reason": "ambiguous_reference"}],
            ),
        }
        conversation, _, resolver = self._conversation(responses)

        self._process(conversation, resolver, "some potato and some pork", "ambiguous-1")
        answer = self._process(conversation, resolver, "half", "ambiguous-2")
        self.assertEqual(answer.outcome, "clarification_required")
        self.assertIn("Baked Sweet Potatoes-Master", answer.message)
        self.assertIn("Herbed Pork Loin", answer.message)
        self.assertEqual(self._application_count(), 0)

    def test_both_can_resolve_two_clear_outstanding_items(self) -> None:
        self._save_plan()
        responses = {
            "some potato and some pork": semantic(
                scope="partial",
                planned=[
                    eaten("item_1", "some potato", relation="modified", quantity_text="some"),
                    eaten("item_2", "some pork", relation="modified", quantity_text="some"),
                ],
            ),
            "I had both amounts you recommended": semantic(
                intent="clarification_answer",
                scope="partial",
                planned=[eaten("item_1", "them both"), eaten("item_2", "them both")],
            ),
        }
        conversation, _, resolver = self._conversation(responses)

        self._process(conversation, resolver, "some potato and some pork", "both-1")
        answer = self._process(conversation, resolver, "I had both amounts you recommended", "both-2")
        self.assertEqual(answer.outcome, "clarification_required")
        self.assertIn("Anything else", answer.message)
        self.assertNotIn("How much", answer.message)

    def test_half_of_it_and_same_amount_use_only_plan_or_draft_quantities(self) -> None:
        self._save_plan()
        responses = {
            "some potato": semantic(
                scope="partial",
                planned=[eaten("item_1", "some potato", relation="modified", quantity_text="some")],
            ),
            "half of it": semantic(
                intent="clarification_answer",
                scope="partial",
                planned=[
                    eaten(
                        "item_1",
                        "half of it",
                        relation="fraction_of_recommended",
                        quantity_text="half of it",
                    )
                ],
            ),
            "the pork, same amount": semantic(
                intent="clarification_answer",
                scope="partial",
                planned=[
                    eaten(
                        "item_2",
                        "same amount",
                        relation="same_as_draft_item",
                        comparison_plan_item_id="item_1",
                    )
                ],
            ),
        }
        conversation, _, resolver = self._conversation(responses)

        self._process(conversation, resolver, "some potato", "same-1")
        self._process(conversation, resolver, "half of it", "same-2")
        answer = self._process(conversation, resolver, "the pork, same amount", "same-3")

        self.assertEqual(answer.outcome, "clarification_required")
        draft = self.state.load_active_meal_report_draft(CHAT_GUID)
        assert draft is not None
        quantities = {item.plan_item_id: item.official_servings for item in draft.planned_items}
        self.assertEqual(quantities, {"item_1": Decimal("1.0"), "item_2": Decimal("1.0")})
        self.assertEqual(self._application_count(), 0)

    def test_unplanned_menu_location_is_validated_by_the_local_resolver(self) -> None:
        self._save_plan()
        responses = {
            "where is the soup": MealReportSemanticResult.model_validate(
                {
                    "intent": "location_question",
                    "report_scope": "unknown",
                    "location_plan_item_id": None,
                    "location_food_text": "Creamy Cauliflower Soup",
                    "planned_items": [],
                    "additional_foods": [],
                    "unresolved_statements": [],
                }
            )
        }
        conversation, _, resolver = self._conversation(responses)

        result = self._process(conversation, resolver, "where is the soup", "location-menu-1")

        self.assertEqual(result.outcome, "location_question")
        self.assertEqual(result.message, "Creamy Cauliflower Soup is at Soup.")
        self.assertEqual(self._application_count(), 0)

    def test_restart_and_duplicate_guid_preserve_draft_without_double_incorporation(self) -> None:
        saved = self._save_plan()
        responses = {
            "I had all of the sweet potato": semantic(
                scope="partial",
                planned=[eaten("item_1", "the sweet potato")],
            ),
            "that's all": semantic(intent="clarification_answer", scope="complete"),
        }
        conversation, _, resolver = self._conversation(responses)
        first = self._process(conversation, resolver, "I had all of the sweet potato", "restart-1")
        assert first.draft is not None
        revision = first.draft.revision
        duplicate = self._process(conversation, resolver, "I had all of the sweet potato", "restart-1")
        self.assertEqual(duplicate.outcome, "replayed")
        loaded = self.state.load_active_meal_report_draft(CHAT_GUID)
        assert loaded is not None
        self.assertEqual(loaded.revision, revision)

        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        conversation, interpreter, resolver = self._conversation(responses)
        final = self._process(conversation, resolver, "that's all", "restart-2")
        self.assertEqual(final.outcome, "applied")
        self.assertEqual(self._application_count(), 1)
        self.assertIn("item_1", str(interpreter.contexts[0]))
        self.assertEqual(final.persisted_plan.plan_id, saved.plan_id)

    def test_replacement_request_never_logs_or_reoptimizes(self) -> None:
        self._save_plan()
        responses = {
            "I don't want the pork": semantic(intent="replacement_request", scope="unknown")
        }
        conversation, _, resolver = self._conversation(responses)

        result = self._process(conversation, resolver, "I don't want the pork", "replace-1")

        self.assertEqual(result.outcome, "replacement_request")
        self.assertIn("replacements aren't available", result.message)
        self.assertEqual(self._application_count(), 0)
        self.assertEqual(self.state.get_daily_intake(DAY), ())
        self.assertEqual(self.state.load_active_meal_plan(DAY, 2).status, "active")

    def test_new_plan_cancels_an_unresolved_draft_without_cross_plan_attachment(self) -> None:
        self._save_plan()
        responses = {
            "I had all of the sweet potato": semantic(
                scope="partial",
                planned=[eaten("item_1", "the sweet potato")],
            ),
            "I had the chicken too": semantic(
                intent="clarification_answer",
                scope="partial",
                planned=[eaten("item_3", "the chicken")],
            ),
        }
        conversation, _, resolver = self._conversation(responses)
        first = self._process(conversation, resolver, "I had all of the sweet potato", "replace-draft-1")
        assert first.draft is not None

        replacement = self.state.save_meal_plan(self.plan)
        self.assertNotEqual(replacement.plan_id, first.persisted_plan.plan_id)
        result = self._process(conversation, resolver, "I had the chicken too", "replace-draft-2")

        self.assertEqual(result.outcome, "draft_cancelled")
        old = self.state.load_meal_report_draft(first.draft.draft_id)
        assert old is not None
        self.assertEqual(old.status, "cancelled")
        self.assertEqual(self._application_count(), 0)
        self.assertEqual(self.state.load_active_meal_plan(DAY, 2).plan_id, replacement.plan_id)

    def test_malformed_or_failed_semantics_fail_closed_without_intake(self) -> None:
        self._save_plan()
        reconciler = MealReportReconciler(
            FailingInterpreter(),
            LocalFDFoodResolver(self.catalog),
            NaturalPortionInterpreter(),
        )
        conversation = MealReportConversationOrchestrator(self.state, reconciler)
        resolver = CurrentPlanResolver(self.state)

        result = self._process(conversation, resolver, "unusable reply", "semantic-failure-1")

        self.assertEqual(result.outcome, "unsupported_or_ambiguous")
        self.assertEqual(self._application_count(), 0)
        self.assertEqual(self.state.get_daily_intake(DAY), ())
        self.assertIsNone(self.state.load_active_meal_report_draft(CHAT_GUID))

    def test_completed_draft_confirmation_failure_replays_without_duplicate_intake(self) -> None:
        self._save_plan()
        responses = {
            "I had all of the sweet potato": semantic(
                scope="partial",
                planned=[eaten("item_1", "the sweet potato")],
            ),
            "that's all": semantic(intent="clarification_answer", scope="complete"),
        }
        conversation, _, resolver = self._conversation(responses)
        sender = RecordingSender()
        handler = ConversationalMealReportMessageHandler(
            sender,
            ApplicationMessagingConfig(expected_chat_guid=CHAT_GUID),
            conversation,
            ActiveMealPlanContextResolver(self.state, service_date_provider=lambda: DAY),
        )
        partial = IncomingMessage("send-partial", "I had all of the sweet potato", "sender", CHAT_GUID, False)
        final = IncomingMessage("send-final", "that's all", "sender", CHAT_GUID, False)
        self.assertTrue(handler.handle(partial))
        sender.failure = RuntimeError("test outbound failure")
        with self.assertRaises(MealWorkflowMessageError):
            handler.handle(final)
        self.assertEqual(self._application_count(), 1)
        self.assertEqual(len(self.state.get_daily_intake(DAY)), 1)

        sender.failure = None
        self.assertTrue(handler.handle(final))
        self.assertEqual(self._application_count(), 1)
        self.assertEqual(len(self.state.get_daily_intake(DAY)), 1)
        self.assertIn("I understood that as", sender.calls[-1][1])

    def test_completed_draft_application_rolls_back_intake_plan_and_draft_together(self) -> None:
        self._save_plan()
        responses = {
            "I had all of the sweet potato": semantic(
                scope="partial",
                planned=[eaten("item_1", "the sweet potato")],
            ),
            "that's all": semantic(intent="clarification_answer", scope="complete"),
        }
        conversation, _, resolver = self._conversation(responses)
        initial = self._process(conversation, resolver, "I had all of the sweet potato", "atomic-1")
        assert initial.draft is not None
        self.catalog._connection.execute(
            """
            CREATE TRIGGER fail_draft_intake
            BEFORE INSERT ON accepted_intake_entries
            BEGIN SELECT RAISE(ABORT, 'test failure'); END
            """
        )
        with self.assertRaises(Exception):
            self._process(conversation, resolver, "that's all", "atomic-2")
        self.assertEqual(self._application_count(), 0)
        self.assertEqual(self.state.get_daily_intake(DAY), ())
        self.assertEqual(self.state.load_active_meal_plan(DAY, 2).status, "active")
        draft = self.state.load_active_meal_report_draft(CHAT_GUID)
        assert draft is not None
        self.assertEqual(draft.status, "awaiting_clarification")
        self.assertIsNone(self.state.load_meal_report_message_event("atomic-2"))

    def test_v6_to_current_migration_adds_conversation_replacement_and_request_state_without_plan_loss(self) -> None:
        saved = self._save_plan()
        self.catalog.close()
        connection = sqlite3.connect(self.path)
        try:
            for table in (
                "immediate_meal_request_requested_foods",
                "immediate_meal_request_dispatches",
                "pending_meal_request_foods",
                "pending_meal_requests",
                "meal_recommendation_replacement_requested_foods",
                "meal_recommendation_replacement_rejections",
                "meal_recommendation_replacements",
                "meal_report_message_events",
                "meal_report_draft_clarifications",
                "meal_report_draft_unplanned_items",
                "meal_report_draft_planned_items",
                "meal_report_drafts",
            ):
                connection.execute(f"DROP TABLE IF EXISTS {table}")
            connection.execute("PRAGMA user_version = 6")
            connection.commit()
        finally:
            connection.close()

        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        version = self.catalog._connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in self.catalog._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        self.assertEqual(version, CATALOG_SCHEMA_VERSION)
        self.assertTrue(
            {
                "meal_report_drafts",
                "meal_report_draft_planned_items",
                "meal_report_draft_unplanned_items",
                "meal_report_draft_clarifications",
                "meal_report_message_events",
                "meal_recommendation_replacements",
                "meal_recommendation_replacement_rejections",
                "meal_recommendation_replacement_requested_foods",
                "pending_meal_requests",
                "pending_meal_request_foods",
                "immediate_meal_request_dispatches",
                "immediate_meal_request_requested_foods",
            }.issubset(tables)
        )
        loaded = self.state.load_meal_plan(saved.plan_id)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.status, "active")

    def test_v8_to_v9_migration_widens_events_and_adds_request_state_without_plan_loss(self) -> None:
        saved = self._save_plan()
        self.catalog.close()
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA foreign_keys = OFF")
            for table in (
                "immediate_meal_request_requested_foods",
                "immediate_meal_request_dispatches",
                "pending_meal_request_foods",
                "pending_meal_requests",
                "meal_recommendation_replacement_requested_foods",
                "meal_recommendation_replacement_rejections",
                "meal_recommendation_replacements",
            ):
                connection.execute(f"DROP TABLE IF EXISTS {table}")
            connection.execute("DROP INDEX IF EXISTS meal_report_message_events_draft_idx")
            connection.execute("DROP TABLE meal_report_message_events")
            connection.execute(
                """
                CREATE TABLE meal_recommendation_replacements (
                    source_event_id TEXT PRIMARY KEY,
                    chat_guid TEXT NOT NULL,
                    prior_plan_id TEXT NOT NULL,
                    replacement_plan_id TEXT NOT NULL UNIQUE,
                    scheduled_origin_plan_id TEXT,
                    whole_meal INTEGER NOT NULL,
                    delivery_token TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    delivery_started_at TEXT,
                    delivered_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE meal_recommendation_replacement_rejections (
                    source_event_id TEXT NOT NULL,
                    plan_item_id TEXT NOT NULL,
                    PRIMARY KEY (source_event_id, plan_item_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE meal_report_message_events (
                    source_event_id TEXT PRIMARY KEY,
                    chat_guid TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    draft_id TEXT,
                    intent TEXT NOT NULL CHECK (intent IN (
                        'meal_report', 'clarification_answer', 'location_question',
                        'replacement_request', 'unsupported_or_ambiguous'
                    )),
                    reply_text TEXT NOT NULL,
                    processed_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO meal_report_message_events
                (source_event_id, chat_guid, plan_id, draft_id, intent, reply_text, processed_at)
                VALUES ('v8-location-event', ?, ?, NULL, 'location_question', 'legacy reply', ?)
                """,
                (CHAT_GUID, saved.plan_id, OBSERVED_AT.isoformat()),
            )
            connection.execute("PRAGMA user_version = 8")
            connection.commit()
        finally:
            connection.close()

        migrate = catalog_module._migrate_meal_request_state_v9

        def interrupted_migration(connection):
            migrate(connection)
            raise sqlite3.OperationalError("simulated migration interruption")

        with patch.object(catalog_module, "_migrate_meal_request_state_v9", interrupted_migration):
            with self.assertRaises(catalog_module.NutritionCatalogError):
                OfficialNutritionCatalog(self.path)
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 8)
            original_columns = {row[1] for row in connection.execute(
                "PRAGMA table_info(meal_recommendation_replacements)"
            )}
            self.assertNotIn("request_kind", original_columns)
            self.assertEqual(connection.execute(
                "SELECT reply_text FROM meal_report_message_events WHERE source_event_id = 'v8-location-event'"
            ).fetchone()[0], "legacy reply")

        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)
        columns = {
            row[1]
            for row in self.catalog._connection.execute(
                "PRAGMA table_info(meal_recommendation_replacements)"
            ).fetchall()
        }
        event_sql = self.catalog._connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'meal_report_message_events'"
        ).fetchone()[0]
        replacement_sql = self.catalog._connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'meal_recommendation_replacements'"
        ).fetchone()[0]
        self.assertEqual(
            self.catalog._connection.execute("PRAGMA user_version").fetchone()[0],
            CATALOG_SCHEMA_VERSION,
        )
        self.assertIn("request_kind", columns)
        self.assertIn("request_kind IN ('rejection', 'meal_request')", replacement_sql)
        self.assertIn("'meal_request'", event_sql)
        self.assertIsNotNone(self.state.load_meal_report_message_event("v8-location-event"))
        loaded = self.state.load_meal_plan(saved.plan_id)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.status, "active")


if __name__ == "__main__":
    unittest.main()
