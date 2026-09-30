"""BlueBubbles transport adapters for persisted meal recommendations and reports.

This module deliberately contains transport coordination only.  It neither
interprets food nor performs nutrition arithmetic: recommendation preparation
and report reconciliation remain in the existing application boundaries.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
import logging
from pathlib import Path
import re
import sys
from typing import Protocol

from ..application_clock import (
    DEFAULT_NUTRITION_APPLICATION_CLOCK,
    NutritionApplicationClock,
)
from ..application import (
    MealRecommendationOrchestrator,
    MealReportNoActivePlanError,
    MealReportProcessingResult,
    PreparedMealRecommendation,
    load_production_daily_targets,
)
from ..durable_state import DurableMealState, PersistedMealPlan
from ..fdmealplanner import DEFAULT_CATALOG_PATH, OfficialNutritionCatalog
from ..food_resolution import FoodSemanticMatcher, LocalFDFoodResolver
from ..meal_identity import MEAL_SLOTS, MealSlot, canonical_meal_id, require_meal_slot
from ..meal_report import MealReportReconciler, MealReportSemanticInterpreter
from ..meal_report_conversation import MealReportConversationOrchestrator
from ..meal_recommendation_replacement import (
    MealRecommendationReplacementDelivery,
    MealRecommendationReplacementDeliveryError,
    MealRecommendationReplacementPreparer,
)
from ..meal_user_request import (
    ImmediateMealRequestDelivery,
    ImmediateMealRequestDeliveryError,
    ImmediateMealRequestPreparer,
    NoActiveMealRequestProcessor,
    NoActiveMealRequestResult,
)
from ..nutrition import DailyTargets
from ..openclaw_food_resolution import OpenClawFoodSemanticMatcher
from ..openclaw_meal_report import OpenClawMealReportInterpreter
from ..openclaw_recommendation_rendering import OpenClawRecommendedPortionRenderer
from ..portion_interpretation import NaturalPortionInterpreter
from ..phelps_service_calendar import (
    DEFAULT_PHELPS_SERVICE_CALENDAR,
    PhelpsServiceCalendar,
)
from ..recommendation_rendering import RecommendedPortionRenderer, format_meal_message
from .application import (
    ApplicationMessagingConfig,
    CHAT_GUID_ENV_VAR,
    OutboundTextSender,
)
from .bluebubbles import BlueBubblesAPIError, BlueBubblesClient
from .webhook import IncomingMessage


__all__ = [
    "AMBIGUOUS_ACTIVE_MEAL_PLAN_REPLY_TEXT",
    "ActiveMealPlanContextResolver",
    "ConversationalMealReportMessageHandler",
    "ManualMealRecommendationDelivery",
    "MessageScopedMealReportMessageHandler",
    "MealPlanDeliveryError",
    "MealPlanResendError",
    "MealRecommendationReplacementDeliveryError",
    "MealReportContextError",
    "MealReportMessageHandler",
    "MealWorkflowMessageError",
    "NO_ACTIVE_MEAL_PLAN_REPLY_TEXT",
    "ProductionMealReportRuntime",
    "open_production_meal_report_runtime",
    "send_manual_recommendation",
]


NO_ACTIVE_MEAL_PLAN_REPLY_TEXT = (
    "I don't have an active meal plan to log right now. "
    "Please ask for a new meal recommendation first."
)


def _shake_reply_intent(text: str) -> tuple[str | None, bool] | None:
    """Recognize a small set of ordinary acknowledgments for a pending shake."""

    normalized = " ".join(text.casefold().strip().rstrip(".!?").split())
    explicit = re.fullmatch(
        r"(?:i (?:had|drank|finished) (?:a |the |my )?)?(breakfast|dinner) shake(?: (done|skip))?",
        normalized,
    )
    if explicit is not None:
        return explicit.group(1) + "_shake", explicit.group(2) != "skip"
    if normalized in {"done", "i had it", "i drank it", "i had a shake", "i had the shake",
                      "i drank a shake", "i drank the shake", "i finished the shake"}:
        return None, True
    if normalized in {"skip", "skipped the shake", "i skipped the shake"}:
        return None, False
    return None


def _shake_acknowledgment(slot: str, confirmed: bool) -> str:
    meal = slot.removesuffix("_shake")
    return f"{'Logged' if confirmed else 'Skipped'} your {meal} shake."
AMBIGUOUS_ACTIVE_MEAL_PLAN_REPLY_TEXT = (
    "I have more than one active meal plan, so I can't tell which meal this reports. "
    "Please say which meal you are reporting."
)

_LOGGER = logging.getLogger(__name__)


class MealReportContextError(RuntimeError):
    """Base error for a transport event without a safe meal-plan context."""


class NoActiveMealPlanContextError(MealReportContextError):
    """Raised when the current service date has no active meal plan."""


class AmbiguousMealPlanContextError(MealReportContextError):
    """Raised when more than one current active context could answer a report."""

    def __init__(self, message: str, meal_slots: tuple[MealSlot, ...] = ()) -> None:
        self.meal_slots = meal_slots
        super().__init__(message)


class NoReportableMealSlotContextError(MealReportContextError):
    """Raised when explicit report scope has no reportable plan for today."""

    def __init__(self, meal_slot: MealSlot) -> None:
        self.meal_slot = require_meal_slot(meal_slot)
        super().__init__(f"no reportable {self.meal_slot} meal plan exists for the current date")


class MealWorkflowMessageError(RuntimeError):
    """Raised when a valid transport event must remain retryable."""


class MealPlanDeliveryError(RuntimeError):
    """Raised after a persisted recommendation could not be sent."""

    def __init__(self, plan_id: str) -> None:
        self.plan_id = plan_id
        super().__init__(
            "outbound recommendation delivery failed after meal plan "
            f"{plan_id} was persisted; use the explicit resend-plan command"
        )


class MealPlanResendError(RuntimeError):
    """Raised when an explicit resend cannot safely use a persisted plan."""


def _ambiguous_meal_plan_reply_text(meal_slots: tuple[MealSlot, ...]) -> str:
    if not meal_slots:
        return AMBIGUOUS_ACTIVE_MEAL_PLAN_REPLY_TEXT
    if len(meal_slots) == 1:
        choices = meal_slots[0]
    elif len(meal_slots) == 2:
        choices = f"{meal_slots[0]} or {meal_slots[1]}"
    else:
        choices = f"{', '.join(meal_slots[:-1])}, or {meal_slots[-1]}"
    return f"Which meal are you reporting: {choices}?"


def _unavailable_meal_slot_reply_text(meal_slot: MealSlot) -> str:
    return (
        f"I don't have a reportable {meal_slot} recommendation for today. "
        "I haven't logged anything."
    )


class MealReportProcessor(Protocol):
    """The existing transport-agnostic inbound application boundary."""

    def process(
        self,
        service_date: date,
        meal: str | int,
        raw_user_text: str,
        source_event_id: str,
    ) -> MealReportProcessingResult:
        """Apply or replay exactly one meal-report event."""


class MealRecommendationPreparer(Protocol):
    """The existing transport-agnostic outbound application boundary."""

    def prepare(
        self,
        service_date: date,
        meal: str | int,
        targets: DailyTargets,
    ) -> PreparedMealRecommendation:
        """Prepare and durably persist one recommendation."""


class ActiveMealPlanContextResolver:
    """Resolve a report from replay or one current reportable active plan.

    BlueBubbles' GUID is checked against durable report applications first.  A
    confirmation-send retry therefore recovers the original applied plan even
    though it is no longer active. New events consider only reportable active
    plans for the supplied current service date, never a globally newest
    historical plan. When no explicit date provider is supplied, that date
    comes from the Nutrition application clock in America/Detroit rather than
    the host's local calendar.
    """

    def __init__(
        self,
        state: DurableMealState,
        *,
        service_date_provider: Callable[[], date] | None = None,
        clock: NutritionApplicationClock | None = None,
        service_calendar: PhelpsServiceCalendar | None = None,
    ) -> None:
        if not isinstance(state, DurableMealState):
            raise TypeError("state must be a DurableMealState")
        if service_date_provider is not None and not callable(service_date_provider):
            raise TypeError("service_date_provider must be callable")
        if clock is not None and not isinstance(clock, NutritionApplicationClock):
            raise TypeError("clock must be a NutritionApplicationClock")
        if service_calendar is not None and not isinstance(
            service_calendar, PhelpsServiceCalendar
        ):
            raise TypeError("service_calendar must be a PhelpsServiceCalendar")
        if service_date_provider is not None and clock is not None:
            raise ValueError("provide service_date_provider or clock, not both")
        self._state = state
        self._service_date_provider = service_date_provider
        self._clock = clock or DEFAULT_NUTRITION_APPLICATION_CLOCK
        self._service_calendar = service_calendar

    def current_service_date(self) -> date:
        """Return the same Detroit date used for new meal context routing."""

        return (
            self._service_date_provider()
            if self._service_date_provider is not None
            else self._clock.service_date()
        )

    def resolve(
        self,
        source_event_id: str,
        *,
        meal_slot: MealSlot | str | None = None,
    ) -> PersistedMealPlan:
        """Return the one plan that can safely receive this transport event."""

        if not isinstance(source_event_id, str) or not source_event_id.strip():
            raise ValueError("source_event_id must be non-empty text")

        requested_slot = None if meal_slot is None else require_meal_slot(meal_slot)
        replayed_plan = self._state.load_meal_plan_for_source_event(source_event_id)
        if replayed_plan is not None:
            if replayed_plan.status != "applied":
                raise MealReportContextError(
                    "source event references a meal plan with invalid lifecycle state"
                )
            return replayed_plan

        local_now = self._clock.now()
        service_date = (
            self._service_date_provider()
            if self._service_date_provider is not None
            else local_now.date()
        )
        if not isinstance(service_date, date) or isinstance(service_date, datetime):
            raise TypeError("service_date_provider must return a date")
        if self._service_calendar is not None:
            self._state.retire_stale_active_meal_plans(
                self._service_calendar,
                evaluated_at=local_now,
            )
        if self._service_calendar is not None:
            active_plans = self._state.list_reportable_active_meal_plans(
                service_date,
                self._service_calendar,
                evaluated_at=local_now,
            )
        else:
            active_plans = self._state.list_active_meal_plans(service_date)
        if not active_plans:
            if requested_slot is not None:
                raise NoReportableMealSlotContextError(requested_slot)
            raise NoActiveMealPlanContextError("no active meal plan exists for the current date")
        if requested_slot is not None:
            active_plans = tuple(
                plan for plan in active_plans if plan.meal_slot == requested_slot
            )
            if not active_plans:
                raise NoReportableMealSlotContextError(requested_slot)
        if len(active_plans) != 1:
            slots = tuple(
                slot
                for slot in MEAL_SLOTS
                if any(plan.meal_slot == slot for plan in active_plans)
            )
            raise AmbiguousMealPlanContextError(
                "multiple active meal plans exist for the current date",
                slots,
            )
        active_plan = active_plans[0]
        if canonical_meal_id(active_plan.plan.meal) is None:
            raise MealReportContextError("active meal plan has an unknown meal identity")
        return active_plan


class MealReportMessageHandler:
    """Filter BlueBubbles events and delegate accepted reports unchanged.

    The handler intentionally has no multi-turn clarification state.  Each
    later incoming message is a fresh full report against the still-active
    plan; a short answer to an earlier clarification must therefore be restated
    with what was eaten and skipped rather than inferred from conversation
    history.
    """

    def __init__(
        self,
        outbound_sender: OutboundTextSender,
        config: ApplicationMessagingConfig,
        meal_report_orchestrator: MealReportProcessor,
        context_resolver: ActiveMealPlanContextResolver,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        if not callable(getattr(outbound_sender, "send_text", None)):
            raise TypeError("outbound_sender must provide send_text")
        if not isinstance(config, ApplicationMessagingConfig):
            raise TypeError("config must be ApplicationMessagingConfig")
        if not callable(getattr(meal_report_orchestrator, "process", None)):
            raise TypeError("meal_report_orchestrator must provide process")
        if not isinstance(context_resolver, ActiveMealPlanContextResolver):
            raise TypeError("context_resolver must be an ActiveMealPlanContextResolver")
        self.outbound_sender = outbound_sender
        self.config = config
        self.meal_report_orchestrator = meal_report_orchestrator
        self.context_resolver = context_resolver
        self._logger = logger or _LOGGER

    def __call__(self, message: IncomingMessage) -> None:
        """Handle one normalized incoming message."""

        self.handle(message)

    def handle(self, message: IncomingMessage) -> bool:
        """Process one eligible report and send its application-produced reply.

        ``False`` means filtering safely ignored the message.  An exception
        means no successful transport completion occurred, allowing the shared
        GUID cache and durable ROWID cursor to retry it.
        """

        if not self.config.accepts(message):
            return False
        chat_guid = message.chat_guid
        if not isinstance(chat_guid, str) or not chat_guid:
            self._logger.warning("Meal report message ignored (missing chat GUID)")
            return False
        source_event_id = message.message_guid
        if not isinstance(source_event_id, str) or not source_event_id.strip():
            self._logger.warning("Meal report message ignored (missing BlueBubbles GUID)")
            return False
        assert isinstance(message.text, str)

        try:
            persisted_plan = self.context_resolver.resolve(source_event_id)
        except NoReportableMealSlotContextError as exc:
            self._send_reply(chat_guid, _unavailable_meal_slot_reply_text(exc.meal_slot))
            return True
        except NoActiveMealPlanContextError:
            self._send_reply(chat_guid, NO_ACTIVE_MEAL_PLAN_REPLY_TEXT)
            return True
        except AmbiguousMealPlanContextError as exc:
            self._send_reply(chat_guid, _ambiguous_meal_plan_reply_text(exc.meal_slots))
            return True
        except Exception as exc:
            self._logger.warning(
                "Meal report context resolution failed (%s)", type(exc).__name__
            )
            raise MealWorkflowMessageError("meal report context resolution failed") from None

        try:
            result = self.meal_report_orchestrator.process(
                persisted_plan.plan.service_date,
                persisted_plan.plan.meal,
                message.text,
                source_event_id,
            )
            if not isinstance(result, MealReportProcessingResult):
                raise TypeError("meal report orchestrator returned invalid output")
        except MealReportNoActivePlanError:
            # A concurrent replacement/application can make a just-resolved
            # plan ineligible.  Do not select another plan heuristically.
            self._send_reply(chat_guid, NO_ACTIVE_MEAL_PLAN_REPLY_TEXT)
            return True
        except Exception as exc:
            self._logger.warning(
                "Meal report application failed (%s)", type(exc).__name__
            )
            raise MealWorkflowMessageError("meal report application failed") from None

        self._send_reply(chat_guid, result.message)
        return True

    def _send_reply(self, chat_guid: str, reply_text: str) -> None:
        try:
            self.outbound_sender.send_text(chat_guid, reply_text)
        except Exception as exc:
            # Do not include user content, GUIDs, BlueBubbles request details,
            # or credentials in diagnostics.  Raising keeps the event retryable.
            self._logger.warning("Meal report outbound reply failed (%s)", type(exc).__name__)
            raise MealWorkflowMessageError("outbound reply failed") from None


class ConversationalMealReportMessageHandler:
    """Transport adapter for durable multi-turn meal-report conversations.

    The legacy :class:`MealReportMessageHandler` remains a compact direct
    adapter for transport-independent callers.  Production composition uses
    this handler so every valid GUID is first checked against durable draft or
    interaction state before it can be treated as a new report.
    """

    def __init__(
        self,
        outbound_sender: OutboundTextSender,
        config: ApplicationMessagingConfig,
        conversation_orchestrator: MealReportConversationOrchestrator,
        context_resolver: ActiveMealPlanContextResolver,
        *,
        replacement_delivery: MealRecommendationReplacementDelivery | None = None,
        no_active_meal_request_processor: NoActiveMealRequestProcessor | None = None,
        immediate_meal_request_delivery: ImmediateMealRequestDelivery | None = None,
        shake_state: DurableMealState | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not callable(getattr(outbound_sender, "send_text", None)):
            raise TypeError("outbound_sender must provide send_text")
        if not isinstance(config, ApplicationMessagingConfig):
            raise TypeError("config must be an ApplicationMessagingConfig")
        if not isinstance(conversation_orchestrator, MealReportConversationOrchestrator):
            raise TypeError("conversation_orchestrator must be a MealReportConversationOrchestrator")
        if not isinstance(context_resolver, ActiveMealPlanContextResolver):
            raise TypeError("context_resolver must be an ActiveMealPlanContextResolver")
        if replacement_delivery is not None and not isinstance(
            replacement_delivery, MealRecommendationReplacementDelivery
        ):
            raise TypeError("replacement_delivery must be a MealRecommendationReplacementDelivery")
        if no_active_meal_request_processor is not None and not isinstance(
            no_active_meal_request_processor, NoActiveMealRequestProcessor
        ):
            raise TypeError("no_active_meal_request_processor must be a NoActiveMealRequestProcessor")
        if immediate_meal_request_delivery is not None and not isinstance(
            immediate_meal_request_delivery, ImmediateMealRequestDelivery
        ):
            raise TypeError("immediate_meal_request_delivery must be an ImmediateMealRequestDelivery")
        if (no_active_meal_request_processor is None) != (
            immediate_meal_request_delivery is None
        ):
            raise ValueError(
                "no_active_meal_request_processor and immediate_meal_request_delivery must be configured together"
            )
        self.outbound_sender = outbound_sender
        self.config = config
        self.conversation_orchestrator = conversation_orchestrator
        self.context_resolver = context_resolver
        self._replacement_delivery = replacement_delivery
        self._no_active_meal_request_processor = no_active_meal_request_processor
        self._immediate_meal_request_delivery = immediate_meal_request_delivery
        self._shake_state = shake_state
        self._logger = logger or _LOGGER

    def __call__(self, message: IncomingMessage) -> None:
        self.handle(message)

    def handle(self, message: IncomingMessage) -> bool:
        """Process one inbound turn and leave failed outbound replies retryable."""

        if not self.config.accepts(message):
            return False
        chat_guid = message.chat_guid
        source_event_id = message.message_guid
        if not isinstance(chat_guid, str) or not chat_guid.strip():
            self._logger.warning("Meal report message ignored (missing chat GUID)")
            return False
        if not isinstance(source_event_id, str) or not source_event_id.strip():
            self._logger.warning("Meal report message ignored (missing BlueBubbles GUID)")
            return False
        assert isinstance(message.text, str)
        if self._handle_shake_reply(chat_guid, source_event_id, message.text):
            return True
        try:
            result = self.conversation_orchestrator.process(
                chat_guid=chat_guid,
                user_text=message.text,
                source_event_id=source_event_id,
                context_resolver=self.context_resolver,
            )
        except NoReportableMealSlotContextError as exc:
            replay = self.conversation_orchestrator.record_pre_route_outcome(
                chat_guid=chat_guid,
                source_event_id=source_event_id,
                outcome_type="unavailable_meal_slot",
                reply_text=_unavailable_meal_slot_reply_text(exc.meal_slot),
            )
            self._send_reply(chat_guid, replay.message)
            return True
        except NoActiveMealPlanContextError:
            return self._process_no_active_meal_request(
                chat_guid=chat_guid,
                user_text=message.text,
                source_event_id=source_event_id,
            )
        except AmbiguousMealPlanContextError as exc:
            replay = self.conversation_orchestrator.record_pre_route_outcome(
                chat_guid=chat_guid,
                source_event_id=source_event_id,
                outcome_type="ambiguous_meal_context",
                reply_text=_ambiguous_meal_plan_reply_text(exc.meal_slots),
            )
            self._send_reply(chat_guid, replay.message)
            return True
        except MealReportNoActivePlanError:
            replay = self.conversation_orchestrator.record_pre_route_outcome(
                chat_guid=chat_guid,
                source_event_id=source_event_id,
                outcome_type="no_reportable_context",
                reply_text=NO_ACTIVE_MEAL_PLAN_REPLY_TEXT,
            )
            self._send_reply(chat_guid, replay.message)
            return True
        except Exception as exc:
            self._logger.warning(
                "Meal report conversation failed (%s)", type(exc).__name__
            )
            raise MealWorkflowMessageError("meal report conversation failed") from None
        if result.outcome == "replayed":
            replayed_request = self._replay_no_active_meal_request(
                chat_guid=chat_guid,
                source_event_id=source_event_id,
            )
            if replayed_request is not None:
                return self._complete_no_active_meal_request(
                    chat_guid,
                    replayed_request,
                )
        if result.replacement_dispatch is not None:
            if self._replacement_delivery is None:
                raise MealWorkflowMessageError("replacement delivery is not configured")
            try:
                self._replacement_delivery.deliver(result.replacement_dispatch)
            except MealRecommendationReplacementDeliveryError as exc:
                self._logger.warning(
                    "Replacement recommendation outbound delivery failed (%s)",
                    exc.status,
                )
                raise MealWorkflowMessageError(
                    "replacement recommendation delivery failed"
                ) from None
            return True
        self._send_reply(chat_guid, result.message)
        if (
            self._shake_state is not None
            and result.persisted_plan is not None
            and (
                result.application is not None
                or (
                    result.persisted_plan.plan.service_date == self.context_resolver.current_service_date()
                    and self._shake_state.load_applied_meal_report(source_event_id) is not None
                )
            )
            and self._shake_state.claim_shake_reminder(
                result.persisted_plan.plan_id, chat_guid
            )
        ):
            self._send_reply(chat_guid, "Drink a shake")
            self._shake_state.mark_shake_reminder_delivered(
                result.persisted_plan.plan_id, chat_guid
            )
        return True

    def _handle_shake_reply(
        self, chat_guid: str, source_event_id: str, user_text: str
    ) -> bool:
        state = self._shake_state
        if state is None:
            return False
        if (
            state.load_meal_report_message_event(source_event_id) is not None
            or state.load_meal_plan_for_source_event(source_event_id) is not None
        ):
            return False
        replay = state.load_shake_for_source_event(source_event_id)
        if replay is not None:
            if replay["chat_guid"] != chat_guid:
                raise MealWorkflowMessageError("shake event belongs to another chat")
            self._send_reply(
                chat_guid,
                _shake_acknowledgment(str(replay["shake_slot"]), replay["status"] == "confirmed"),
            )
            return True
        intent = _shake_reply_intent(user_text)
        if intent is None:
            return False
        explicit, confirmed = intent
        drafts = state.list_active_meal_report_drafts(chat_guid)
        if drafts:
            if len(drafts) == 1 and re.search(r"\bshake\b", user_text, re.IGNORECASE):
                self._send_reply(chat_guid, drafts[0].last_prompt)
                return True
            return False
        local_date = self.context_resolver.current_service_date()
        pending = state.pending_shakes(local_date, chat_guid)
        if explicit is None and len(pending) > 1:
            self._send_reply(chat_guid, "Which shake: breakfast or dinner?")
            return True
        slot = explicit if explicit in pending else pending[0] if explicit is None and len(pending) == 1 else None
        if slot is None:
            return False
        result = state.resolve_shake(
            local_date, slot, chat_guid, source_event_id, confirmed=confirmed
        )
        self._send_reply(
            chat_guid,
            _shake_acknowledgment(slot, result["status"] == "confirmed"),
        )
        return True

    def _replay_no_active_meal_request(
        self,
        *,
        chat_guid: str,
        source_event_id: str,
    ) -> NoActiveMealRequestResult | None:
        if self._no_active_meal_request_processor is None:
            return None
        try:
            return self._no_active_meal_request_processor.replay(
                chat_guid=chat_guid,
                source_event_id=source_event_id,
            )
        except Exception as exc:
            self._logger.warning(
                "Meal request replay lookup failed (%s)", type(exc).__name__
            )
            raise MealWorkflowMessageError("meal request replay lookup failed") from None

    def _process_no_active_meal_request(
        self,
        *,
        chat_guid: str,
        user_text: str,
        source_event_id: str,
    ) -> bool:
        if self._no_active_meal_request_processor is None:
            replay = self.conversation_orchestrator.record_pre_route_outcome(
                chat_guid=chat_guid,
                source_event_id=source_event_id,
                outcome_type="no_reportable_context",
                reply_text=NO_ACTIVE_MEAL_PLAN_REPLY_TEXT,
            )
            self._send_reply(chat_guid, replay.message)
            return True
        try:
            result = self._no_active_meal_request_processor.process(
                chat_guid=chat_guid,
                user_text=user_text,
                source_event_id=source_event_id,
            )
        except Exception as exc:
            self._logger.warning(
                "No-active meal request processing failed (%s)", type(exc).__name__
            )
            raise MealWorkflowMessageError("no-active meal request processing failed") from None
        if result is None:
            replay = self.conversation_orchestrator.record_pre_route_outcome(
                chat_guid=chat_guid,
                source_event_id=source_event_id,
                outcome_type="no_reportable_context",
                reply_text=NO_ACTIVE_MEAL_PLAN_REPLY_TEXT,
            )
            self._send_reply(chat_guid, replay.message)
            return True
        if result.outcome == "reply":
            replay = self.conversation_orchestrator.record_pre_route_outcome(
                chat_guid=chat_guid,
                source_event_id=source_event_id,
                outcome_type="routing_failure",
                reply_text=result.reply_text,
            )
            self._send_reply(chat_guid, replay.message)
            return True
        return self._complete_no_active_meal_request(chat_guid, result)

    def _complete_no_active_meal_request(
        self,
        chat_guid: str,
        result: NoActiveMealRequestResult,
    ) -> bool:
        if not isinstance(result, NoActiveMealRequestResult):
            raise MealWorkflowMessageError("no-active meal request returned invalid output")
        if result.immediate_dispatch is None:
            self._send_reply(chat_guid, result.reply_text)
            return True
        if self._immediate_meal_request_delivery is None:
            raise MealWorkflowMessageError("immediate meal request delivery is not configured")
        try:
            self._immediate_meal_request_delivery.deliver(result.immediate_dispatch)
        except ImmediateMealRequestDeliveryError as exc:
            self._logger.warning(
                "Immediate meal request outbound delivery failed (%s)", exc.status
            )
            raise MealWorkflowMessageError(
                "immediate meal request delivery failed"
            ) from None
        return True

    def _send_reply(self, chat_guid: str, reply_text: str) -> None:
        try:
            self.outbound_sender.send_text(chat_guid, reply_text)
        except Exception as exc:
            self._logger.warning("Meal report outbound reply failed (%s)", type(exc).__name__)
            raise MealWorkflowMessageError("outbound reply failed") from None


class MessageScopedMealReportMessageHandler:
    """Build SQLite-backed report resources inside each handler invocation.

    The webhook receiver and polling worker both retain this callback for the
    service lifetime, but they can invoke it from arbitrary threads.  A
    catalog connection therefore cannot be retained here: every eligible
    message creates its catalog, durable state, resolver, reconciler, and
    orchestrator on the executing thread and closes the catalog before that
    invocation returns or raises.

    The semantic adapters are deliberately shared.  Their production OpenClaw
    implementations retain immutable invocation configuration and launch a
    subprocess for each interpretation, without retaining SQLite state.

    An optional explicit service-date provider remains available to deterministic
    callers.  Otherwise each scoped resolver obtains its date from the supplied
    Nutrition clock, or the Detroit application-clock default.
    """

    def __init__(
        self,
        outbound_sender: OutboundTextSender,
        config: ApplicationMessagingConfig,
        semantic_interpreter: MealReportSemanticInterpreter,
        *,
        food_semantic_matcher: FoodSemanticMatcher | None = None,
        catalog_path: str | Path = DEFAULT_CATALOG_PATH,
        service_date_provider: Callable[[], date] | None = None,
        clock: NutritionApplicationClock | None = None,
        service_calendar: PhelpsServiceCalendar | None = None,
        replacement_target_loader: Callable[[], DailyTargets] | None = None,
        replacement_renderer: RecommendedPortionRenderer | None = None,
        is_known_replacement_delivery_failure: Callable[[BaseException], bool] | None = None,
        catalog_factory: Callable[[str | Path], OfficialNutritionCatalog] = OfficialNutritionCatalog,
        logger: logging.Logger | None = None,
    ) -> None:
        if not callable(getattr(outbound_sender, "send_text", None)):
            raise TypeError("outbound_sender must provide send_text")
        if not isinstance(config, ApplicationMessagingConfig):
            raise TypeError("config must be ApplicationMessagingConfig")
        if not callable(getattr(semantic_interpreter, "interpret", None)):
            raise TypeError("semantic_interpreter must provide interpret")
        if food_semantic_matcher is not None and not callable(
            getattr(food_semantic_matcher, "decide", None)
        ):
            raise TypeError("food_semantic_matcher must provide decide")
        if not isinstance(catalog_path, (str, Path)):
            raise TypeError("catalog_path must be a string or Path")
        if service_date_provider is not None and not callable(service_date_provider):
            raise TypeError("service_date_provider must be callable")
        if clock is not None and not isinstance(clock, NutritionApplicationClock):
            raise TypeError("clock must be a NutritionApplicationClock")
        if service_calendar is not None and not isinstance(
            service_calendar, PhelpsServiceCalendar
        ):
            raise TypeError("service_calendar must be a PhelpsServiceCalendar")
        if replacement_target_loader is not None and not callable(replacement_target_loader):
            raise TypeError("replacement_target_loader must be callable")
        if replacement_renderer is not None and not isinstance(
            replacement_renderer, RecommendedPortionRenderer
        ):
            raise TypeError("replacement_renderer must be a RecommendedPortionRenderer")
        if is_known_replacement_delivery_failure is not None and not callable(
            is_known_replacement_delivery_failure
        ):
            raise TypeError("is_known_replacement_delivery_failure must be callable")
        if service_date_provider is not None and clock is not None:
            raise ValueError("provide service_date_provider or clock, not both")
        if not callable(catalog_factory):
            raise TypeError("catalog_factory must be callable")

        self.outbound_sender = outbound_sender
        self.config = config
        self.semantic_interpreter = semantic_interpreter
        self.food_semantic_matcher = food_semantic_matcher
        self.catalog_path = Path(catalog_path).expanduser()
        self._service_date_provider = service_date_provider
        self._clock = clock
        self._service_calendar = service_calendar
        self._replacement_target_loader = replacement_target_loader
        self._replacement_renderer = replacement_renderer
        self._is_known_replacement_delivery_failure = (
            is_known_replacement_delivery_failure
        )
        self._catalog_factory = catalog_factory
        self._logger = logger or _LOGGER

    def __call__(self, message: IncomingMessage) -> None:
        """Handle one normalized message with execution-scoped SQLite state."""

        self.handle(message)

    def handle(self, message: IncomingMessage) -> bool:
        """Process one eligible event without crossing SQLite thread boundaries."""

        # Avoid opening SQLite at all for traffic that the application would
        # safely ignore.  The composed handler repeats this check so its
        # public behavior stays identical to direct use elsewhere.
        if not self.config.accepts(message):
            return False

        with self._catalog_factory(self.catalog_path) as catalog:
            food_resolver = LocalFDFoodResolver(
                catalog,
                semantic_matcher=self.food_semantic_matcher,
            )
            reconciler = MealReportReconciler(
                self.semantic_interpreter,
                food_resolver,
                NaturalPortionInterpreter(),
            )
            state = DurableMealState(catalog)
            context_resolver = ActiveMealPlanContextResolver(
                state,
                service_date_provider=self._service_date_provider,
                clock=self._clock,
                service_calendar=self._service_calendar,
            )
            recommendation_orchestrator = (
                MealRecommendationOrchestrator(
                    catalog,
                    self._replacement_renderer,
                    service_calendar=self._service_calendar,
                    clock=self._clock,
                )
                if self._replacement_target_loader is not None
                else None
            )
            replacement_preparer = (
                MealRecommendationReplacementPreparer(
                    state,
                    recommendation_orchestrator,
                    self._replacement_target_loader,
                )
                if recommendation_orchestrator is not None
                else None
            )
            replacement_delivery = (
                MealRecommendationReplacementDelivery(
                    state,
                    self.outbound_sender,
                    is_known_delivery_failure=self._is_known_replacement_delivery_failure,
                    logger=self._logger,
                )
                if replacement_preparer is not None
                else None
            )
            no_active_meal_request_processor = (
                NoActiveMealRequestProcessor(
                    state,
                    self.semantic_interpreter,
                    food_resolver,
                    ImmediateMealRequestPreparer(
                        state,
                        recommendation_orchestrator,
                        self._replacement_target_loader,
                    ),
                    clock=self._clock or DEFAULT_NUTRITION_APPLICATION_CLOCK,
                    service_calendar=self._service_calendar,
                )
                if (
                    recommendation_orchestrator is not None
                    and self._service_calendar is not None
                    and callable(
                        getattr(self.semantic_interpreter, "interpret_meal_request", None)
                    )
                )
                else None
            )
            immediate_meal_request_delivery = (
                ImmediateMealRequestDelivery(
                    state,
                    self.outbound_sender,
                    is_known_delivery_failure=self._is_known_replacement_delivery_failure,
                    logger=self._logger,
                )
                if no_active_meal_request_processor is not None
                else None
            )
            handler = ConversationalMealReportMessageHandler(
                self.outbound_sender,
                self.config,
                MealReportConversationOrchestrator(
                    state,
                    reconciler,
                    service_calendar=self._service_calendar,
                    clock=self._clock,
                    replacement_preparer=replacement_preparer,
                ),
                context_resolver,
                replacement_delivery=replacement_delivery,
                no_active_meal_request_processor=no_active_meal_request_processor,
                immediate_meal_request_delivery=immediate_meal_request_delivery,
                shake_state=state,
                logger=self._logger,
            )
            return handler.handle(message)


class ManualMealRecommendationDelivery:
    """Prepare once, then send or explicitly resend an active persisted plan.

    ``prepare_and_send`` never retries preparation after a send failure because
    preparation itself creates a new active plan.  ``resend_active_plan``
    instead reconstructs the exact existing deterministic message from the
    persisted ``MealPlan`` fields used by ``format_meal_message``.
    """

    def __init__(
        self,
        outbound_sender: OutboundTextSender,
        chat_guid: str,
        recommendation_orchestrator: MealRecommendationPreparer,
        state: DurableMealState,
        *,
        service_calendar: PhelpsServiceCalendar | None = None,
        clock: NutritionApplicationClock | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not callable(getattr(outbound_sender, "send_text", None)):
            raise TypeError("outbound_sender must provide send_text")
        if not isinstance(chat_guid, str) or not chat_guid.strip():
            raise ValueError("chat_guid must be non-empty text")
        if not callable(getattr(recommendation_orchestrator, "prepare", None)):
            raise TypeError("recommendation_orchestrator must provide prepare")
        if not isinstance(state, DurableMealState):
            raise TypeError("state must be a DurableMealState")
        if service_calendar is not None and not isinstance(
            service_calendar, PhelpsServiceCalendar
        ):
            raise TypeError("service_calendar must be a PhelpsServiceCalendar")
        if clock is not None and not isinstance(clock, NutritionApplicationClock):
            raise TypeError("clock must be a NutritionApplicationClock")
        self.outbound_sender = outbound_sender
        self.chat_guid = chat_guid
        self.recommendation_orchestrator = recommendation_orchestrator
        self._state = state
        self._service_calendar = service_calendar
        self._clock = clock or DEFAULT_NUTRITION_APPLICATION_CLOCK
        self._logger = logger or _LOGGER

    def prepare_and_send(
        self,
        service_date: date,
        meal: str | int,
        targets: DailyTargets,
    ) -> PreparedMealRecommendation:
        """Run the existing preparation boundary once and attempt one delivery."""

        prepared = self.recommendation_orchestrator.prepare(service_date, meal, targets)
        if not isinstance(prepared, PreparedMealRecommendation):
            raise TypeError("recommendation_orchestrator returned invalid output")
        self._send_plan_message(prepared.plan_id, prepared.message)
        return prepared

    def resend_active_plan(self, plan_id: str) -> PersistedMealPlan:
        """Explicitly re-send one still-answerable persisted recommendation."""

        if not isinstance(plan_id, str) or not plan_id.strip():
            raise ValueError("plan_id must be non-empty text")
        if self._service_calendar is not None:
            local_now = self._clock.now()
            self._state.retire_stale_active_meal_plans(
                self._service_calendar,
                evaluated_at=local_now,
            )
        persisted_plan = self._state.load_meal_plan(plan_id)
        if persisted_plan is None:
            raise MealPlanResendError("meal plan does not exist")
        if persisted_plan.status != "active":
            raise MealPlanResendError("only an active meal plan can be resent")
        if self._service_calendar is not None and not self._service_calendar.is_meal_context_eligible_at(
            persisted_plan.plan.service_date,
            persisted_plan.plan.meal,
            self._clock.now(),
        ):
            raise MealPlanResendError("meal plan is not eligible to send at the current Phelps service time")
        self._send_plan_message(
            persisted_plan.plan_id,
            format_meal_message(persisted_plan.plan),
        )
        return persisted_plan

    def _send_plan_message(self, plan_id: str, message: str) -> None:
        try:
            self.outbound_sender.send_text(self.chat_guid, message)
        except Exception as exc:
            self._logger.warning(
                "Meal recommendation outbound delivery failed for persisted plan (%s)",
                type(exc).__name__,
            )
            raise MealPlanDeliveryError(plan_id) from None


def send_manual_recommendation(
    delivery: ManualMealRecommendationDelivery,
    service_date: date,
    meal: str | int,
    *,
    target_loader: Callable[[], DailyTargets] = load_production_daily_targets,
) -> PreparedMealRecommendation:
    """Load strict production targets, then prepare and send one recommendation."""

    if not isinstance(delivery, ManualMealRecommendationDelivery):
        raise TypeError("delivery must be a ManualMealRecommendationDelivery")
    if not callable(target_loader):
        raise TypeError("target_loader must be callable")
    return delivery.prepare_and_send(service_date, meal, target_loader())


@dataclass(slots=True)
class ProductionMealReportRuntime:
    """Own service-lifetime non-SQLite inbound report composition.

    ``handler`` owns no catalog connection between invocations.  It opens and
    closes the SQLite-backed message resources in whichever polling or HTTP
    request thread delivers each event.
    """

    handler: MessageScopedMealReportMessageHandler

    def close(self) -> None:
        """Finish shutdown; each completed handler invocation already closed SQLite."""


def open_production_meal_report_runtime(
    outbound_sender: OutboundTextSender,
    *,
    environ: Mapping[str, str] | None = None,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    service_date_provider: Callable[[], date] | None = None,
    clock: NutritionApplicationClock | None = None,
    service_calendar: PhelpsServiceCalendar | None = DEFAULT_PHELPS_SERVICE_CALENDAR,
    catalog_factory: Callable[[str | Path], OfficialNutritionCatalog] = OfficialNutritionCatalog,
) -> ProductionMealReportRuntime:
    """Construct the live inbound adapter without weakening runtime configuration.

    The long-lived returned runtime retains only thread-safe, non-SQLite
    configuration and the OpenClaw interpreter.  SQLite resources are instead
    created by the handler for each eligible inbound message.  Each scoped
    handler composes the durable conversation boundary, so a short
    clarification answer can recover its plan-scoped draft after a webhook,
    poller, or process restart. Without an explicit date provider, inbound
    active-plan lookup uses the Detroit application clock. The default Phelps
    calendar also retires objectively stale active plans only after durable
    replay lookup and filters new reports to an eligible current service
    opportunity.
    """

    config = ApplicationMessagingConfig.from_env(environ)
    semantic_interpreter = OpenClawMealReportInterpreter.from_env(environ)
    semantic_interpreter.ensure_executable_available()
    food_semantic_matcher = OpenClawFoodSemanticMatcher.from_env(environ)
    food_semantic_matcher.ensure_executable_available()
    replacement_renderer = RecommendedPortionRenderer(
        OpenClawRecommendedPortionRenderer.from_env(environ)
    )
    return ProductionMealReportRuntime(
        MessageScopedMealReportMessageHandler(
            outbound_sender,
            config,
            semantic_interpreter,
            food_semantic_matcher=food_semantic_matcher,
            catalog_path=catalog_path,
            service_date_provider=service_date_provider,
            clock=clock,
            service_calendar=service_calendar,
            replacement_target_loader=lambda: load_production_daily_targets(environ),
            replacement_renderer=replacement_renderer,
            is_known_replacement_delivery_failure=lambda exc: isinstance(
                exc, BlueBubblesAPIError
            ),
            catalog_factory=catalog_factory,
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Run explicit operator commands for recommendation delivery and lifecycle."""

    parser = argparse.ArgumentParser(
        description="Manually send a persisted Nutrition Optimizer meal recommendation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    send_parser = subparsers.add_parser(
        "send-recommendation",
        help="prepare one new recommendation and attempt one BlueBubbles send",
    )
    send_parser.add_argument("--service-date", required=True, type=_parse_service_date)
    send_parser.add_argument("--meal", required=True)
    send_parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)

    resend_parser = subparsers.add_parser(
        "resend-plan",
        help="re-send one already prepared active plan without preparing again",
    )
    resend_parser.add_argument("--plan-id", required=True)
    resend_parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)

    status_parser = subparsers.add_parser(
        "service-status",
        help="inspect Detroit Phelps availability and active-plan staleness without retiring",
    )
    status_parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)

    retire_parser = subparsers.add_parser(
        "retire-stale-plans",
        help="explicitly supersede objectively stale active plans without sending a message",
    )
    retire_parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)

    args = parser.parse_args(argv)
    try:
        if args.command in {"service-status", "retire-stale-plans"}:
            local_now = DEFAULT_NUTRITION_APPLICATION_CLOCK.now()
            with OfficialNutritionCatalog(args.catalog_path) as catalog:
                state = DurableMealState(catalog)
                if args.command == "service-status":
                    availability = DEFAULT_PHELPS_SERVICE_CALENDAR.availability_at(local_now)
                    active = state.list_all_active_meal_plans()
                    stale = state.list_stale_active_meal_plans(
                        DEFAULT_PHELPS_SERVICE_CALENDAR,
                        evaluated_at=local_now,
                    )
                    print(f"Detroit now: {availability.local_now.isoformat()}", flush=True)
                    print(f"Service date: {availability.service_date.isoformat()}", flush=True)
                    print(f"Phelps open: {'yes' if availability.is_open else 'no'}", flush=True)
                    window_text = "; ".join(
                        f"{window.label} {window.starts_at.strftime('%H:%M')}-{window.ends_at.strftime('%H:%M')}"
                        + (f" (fd:{window.meal_id})" if window.meal_id is not None else "")
                        for window in availability.windows
                    ) or "none"
                    print(f"Applicable windows: {window_text}", flush=True)
                    active_text = ", ".join(
                        f"{plan.plan_id}:{plan.plan.service_date.isoformat()}:{plan.plan.meal}"
                        for plan in active
                    ) or "none"
                    stale_text = ", ".join(plan.plan_id for plan in stale) or "none"
                    print(f"Active plans: {active_text}", flush=True)
                    print(f"Stale active plans: {stale_text}", flush=True)
                    return 0

                retirement = state.retire_stale_active_meal_plans(
                    DEFAULT_PHELPS_SERVICE_CALENDAR,
                    evaluated_at=local_now,
                )
            print(
                "Stale active meal plans retired "
                f"(count={len(retirement.retired_plan_ids)})",
                flush=True,
            )
            return 0

        # Target loading is deliberately the first runtime action for a new
        # recommendation.  Bad target configuration cannot prepare a plan,
        # send a message, or open the catalog for migration.
        targets = (
            load_production_daily_targets()
            if args.command == "send-recommendation"
            else None
        )
        config = ApplicationMessagingConfig.from_env()
        chat_guid = config.expected_chat_guid
        if chat_guid is None:
            raise ValueError(f"{CHAT_GUID_ENV_VAR} must be set for outbound delivery")
        client = BlueBubblesClient.from_env()

        if args.command == "send-recommendation":
            assert targets is not None
            with OfficialNutritionCatalog(args.catalog_path) as catalog:
                # The optional Luna renderer sees only the official food name
                # and already-computed canonical physical quantity. Its output
                # can improve the serving-line wording, but the outer renderer
                # preserves deterministic text whenever a descriptor is absent,
                # malformed, or unavailable.
                renderer = RecommendedPortionRenderer(
                    OpenClawRecommendedPortionRenderer.from_env()
                )
                delivery = ManualMealRecommendationDelivery(
                    client,
                    chat_guid,
                    MealRecommendationOrchestrator(
                        catalog,
                        renderer,
                        service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
                    ),
                    DurableMealState(catalog),
                    service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
                )
                prepared = delivery.prepare_and_send(args.service_date, args.meal, targets)
            print(f"Meal recommendation sent (plan_id={prepared.plan_id})", flush=True)
            return 0

        with OfficialNutritionCatalog(args.catalog_path) as catalog:
            delivery = ManualMealRecommendationDelivery(
                client,
                chat_guid,
                MealRecommendationOrchestrator(
                    catalog,
                    service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
                ),
                DurableMealState(catalog),
                service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            )
            resent = delivery.resend_active_plan(args.plan_id)
        print(f"Meal recommendation resent (plan_id={resent.plan_id})", flush=True)
        return 0
    except MealPlanDeliveryError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    except (RuntimeError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 2  # pragma: no cover - argparse.error always raises SystemExit.


def _parse_service_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError("service date must use YYYY-MM-DD") from None


if __name__ == "__main__":
    raise SystemExit(main())
