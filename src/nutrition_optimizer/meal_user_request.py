"""Durable preparation and delivery for a no-prior-plan meal request.

This is deliberately a small transport-adjacent coordinator around the
existing semantic interpreter, local menu resolver, and deterministic
recommendation orchestrator.  The semantic result supplies only user wording;
authoritative identity, service context, quantities, nutrition, and durable
state remain in Python.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
import logging
from typing import Literal, Protocol

from .application import (
    MealRecommendationOrchestrator,
    MealRecommendationRequestedFoodError,
    MealRecommendationServiceUnavailableError,
    MealRecommendationUnavailableError,
    PreparedMealRecommendationRequest,
)
from .durable_state import (
    DurableMealState,
    ImmediateMealRequestDispatch,
    ImmediateMealRequestDispatchStatus,
    MealReportApplicationError,
    PendingMealRequest,
)
from .food_resolution import (
    AmbiguousFood,
    FoodResolutionRequest,
    LocalFDFoodResolver,
    ResolvedFood,
)
from .meal_identity import MealSlot, canonical_meal_id, meal_name_for_display
from .meal_request import requested_meal_food_from_resolved
from .meal_report import MealPlan
from .meal_request_structured import MealRequestSemanticResult
from .nutrition import DailyTargets
from .application_clock import NUTRITION_APPLICATION_TIMEZONE, NutritionApplicationClock
from .phelps_service_calendar import PhelpsServiceCalendar
from .recommendation_scheduler import RecommendationDispatchSchedule, ScheduledMealOpportunity
from .recommendation_rendering import format_meal_message


__all__ = [
    "ImmediateMealRequestDelivery",
    "ImmediateMealRequestDeliveryError",
    "ImmediateMealRequestPreparer",
    "NoActiveMealRequestProcessor",
    "NoActiveMealRequestResult",
    "format_immediate_meal_request_message",
]


_LOGGER = logging.getLogger(__name__)


class ImmediateRequestOutboundSender(Protocol):
    def send_text(
        self,
        chat_guid: str,
        message: str,
        *,
        temp_guid: str | None = None,
    ) -> object:
        """Send one stable-token text message."""


class ContextFreeMealRequestInterpreter(Protocol):
    def interpret_meal_request(self, user_text: str) -> MealRequestSemanticResult:
        """Classify a no-active-plan message without menu authority."""


class ImmediateMealRequestDeliveryError(RuntimeError):
    def __init__(self, status: ImmediateMealRequestDispatchStatus) -> None:
        self.status = status
        super().__init__("immediate meal request delivery is not complete")


class ImmediateMealRequestPreparer:
    """Calculate and atomically persist an immediate user-requested plan."""

    def __init__(
        self,
        state: DurableMealState,
        recommendation_orchestrator: MealRecommendationOrchestrator,
        target_loader: Callable[[], DailyTargets],
    ) -> None:
        if not isinstance(state, DurableMealState):
            raise TypeError("state must be a DurableMealState")
        if not isinstance(recommendation_orchestrator, MealRecommendationOrchestrator):
            raise TypeError("recommendation_orchestrator must be a MealRecommendationOrchestrator")
        if not callable(target_loader):
            raise TypeError("target_loader must be callable")
        self._state = state
        self._recommendation_orchestrator = recommendation_orchestrator
        self._target_loader = target_loader

    def prepare(
        self,
        *,
        service_date: date,
        meal: str | int,
        meal_slot: MealSlot,
        requested_foods: tuple[ResolvedFood, ...],
        whole_meal: bool,
        source_event_id: str,
        chat_guid: str,
    ) -> ImmediateMealRequestDispatch:
        """Return one immutable plan dispatch, reusing a GUID after restart."""

        if not isinstance(source_event_id, str) or not source_event_id.strip():
            raise ValueError("source_event_id must be non-empty text")
        if not isinstance(chat_guid, str) or not chat_guid.strip():
            raise ValueError("chat_guid must be non-empty text")
        if not isinstance(requested_foods, tuple) or not all(
            isinstance(food, ResolvedFood) for food in requested_foods
        ):
            raise TypeError("requested_foods must contain ResolvedFood values")
        if not isinstance(whole_meal, bool):
            raise TypeError("whole_meal must be a bool")
        if not whole_meal and not requested_foods:
            raise ValueError("targeted request needs requested_foods")

        existing = self._state.load_immediate_meal_request_dispatch(source_event_id)
        if existing is not None:
            if existing.chat_guid != chat_guid:
                raise MealReportApplicationError("request source event belongs to another chat")
            return existing
        supplied_targets = self._target_loader()
        if not isinstance(supplied_targets, DailyTargets):
            raise TypeError("target_loader returned invalid targets")
        prepared = self._recommendation_orchestrator.prepare_requested_meal(
            service_date,
            meal,
            requested_foods,
            supplied_targets,
            whole_meal=whole_meal,
        )
        if not isinstance(prepared, PreparedMealRecommendationRequest):
            raise TypeError("meal request orchestrator returned invalid output")
        return self._state.save_immediate_meal_request_plan(
            prepared.rendering.meal_plan,
            meal_slot=meal_slot,
            source_event_id=source_event_id,
            chat_guid=chat_guid,
            requested_foods=prepared.requested_foods,
            whole_meal=prepared.whole_meal,
            reply_text=format_immediate_meal_request_message(prepared.rendering.meal_plan),
        )


class ImmediateMealRequestDelivery:
    """Send an immediate request exactly once with replacement-like safety."""

    def __init__(
        self,
        state: DurableMealState,
        outbound_sender: ImmediateRequestOutboundSender,
        *,
        is_known_delivery_failure: Callable[[BaseException], bool] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(state, DurableMealState):
            raise TypeError("state must be a DurableMealState")
        if not callable(getattr(outbound_sender, "send_text", None)):
            raise TypeError("outbound_sender must provide send_text")
        if is_known_delivery_failure is not None and not callable(is_known_delivery_failure):
            raise TypeError("is_known_delivery_failure must be callable")
        self._state = state
        self._outbound_sender = outbound_sender
        self._is_known_delivery_failure = is_known_delivery_failure or (lambda exc: True)
        self._logger = logger or _LOGGER

    def deliver(self, dispatch: ImmediateMealRequestDispatch) -> ImmediateMealRequestDispatch:
        if not isinstance(dispatch, ImmediateMealRequestDispatch):
            raise TypeError("dispatch must be an ImmediateMealRequestDispatch")
        current = self._state.load_immediate_meal_request_dispatch(dispatch.source_event_id)
        if current is None or current.plan_id != dispatch.plan_id:
            raise ImmediateMealRequestDeliveryError(dispatch.status)
        if current.status == "delivered":
            return current
        if current.persisted_plan.status != "active":
            return current
        if current.status == "sending":
            raise ImmediateMealRequestDeliveryError("sending")
        if current.status != "pending_delivery":
            raise ImmediateMealRequestDeliveryError(current.status)
        if not self._state.claim_immediate_meal_request_delivery(current):
            refreshed = self._state.load_immediate_meal_request_dispatch(
                current.source_event_id
            )
            if refreshed is not None and refreshed.status == "delivered":
                return refreshed
            raise ImmediateMealRequestDeliveryError(
                current.status if refreshed is None else refreshed.status
            )
        try:
            self._outbound_sender.send_text(
                current.chat_guid,
                format_immediate_meal_request_message(current.persisted_plan.plan),
                temp_guid=current.delivery_token,
            )
        except Exception as exc:
            self._logger.warning(
                "Immediate meal request outbound delivery failed (%s)", type(exc).__name__
            )
            if not self._is_known_delivery_failure(exc):
                raise ImmediateMealRequestDeliveryError("sending") from None
            try:
                self._state.release_immediate_meal_request_delivery(current)
            except Exception:
                raise ImmediateMealRequestDeliveryError("sending") from None
            raise ImmediateMealRequestDeliveryError("pending_delivery") from None
        try:
            return self._state.mark_immediate_meal_request_delivered(current)
        except Exception:
            raise ImmediateMealRequestDeliveryError("sending") from None


def format_immediate_meal_request_message(plan: MealPlan) -> str:
    """Mark an otherwise standard plan as the requested meal recommendation."""

    if not isinstance(plan, MealPlan):
        raise TypeError("plan must be a MealPlan")
    return f"Here's your recommendation:\n{format_meal_message(plan)}"


@dataclass(frozen=True, slots=True)
class _MealRequestTarget:
    service_date: date
    meal_id: int
    meal_slot: MealSlot
    mode: Literal["immediate", "pending"]


@dataclass(frozen=True, slots=True)
class NoActiveMealRequestResult:
    """One no-active-plan request outcome for the message transport adapter."""

    outcome: Literal["pending", "immediate", "reply"]
    reply_text: str
    pending_request: PendingMealRequest | None = None
    immediate_dispatch: ImmediateMealRequestDispatch | None = None

    def __post_init__(self) -> None:
        if self.outcome not in {"pending", "immediate", "reply"}:
            raise ValueError("no-active request outcome is invalid")
        if not isinstance(self.reply_text, str) or not self.reply_text.strip():
            raise ValueError("reply_text must be non-empty text")
        if self.pending_request is not None and not isinstance(
            self.pending_request, PendingMealRequest
        ):
            raise TypeError("pending_request must be a PendingMealRequest or None")
        if self.immediate_dispatch is not None and not isinstance(
            self.immediate_dispatch, ImmediateMealRequestDispatch
        ):
            raise TypeError("immediate_dispatch must be an ImmediateMealRequestDispatch or None")
        if self.outcome == "pending" and self.pending_request is None:
            raise ValueError("pending request outcome needs pending_request")
        if self.outcome == "immediate" and self.immediate_dispatch is None:
            raise ValueError("immediate request outcome needs immediate_dispatch")
        if self.outcome == "reply" and (
            self.pending_request is not None or self.immediate_dispatch is not None
        ):
            raise ValueError("reply request outcome cannot retain a dispatch")


class NoActiveMealRequestProcessor:
    """Resolve/store/prepare one no-active-plan request without transport I/O."""

    def __init__(
        self,
        state: DurableMealState,
        semantic_interpreter: ContextFreeMealRequestInterpreter,
        food_resolver: LocalFDFoodResolver,
        immediate_preparer: ImmediateMealRequestPreparer,
        *,
        clock: NutritionApplicationClock,
        service_calendar: PhelpsServiceCalendar,
        schedule: RecommendationDispatchSchedule | None = None,
    ) -> None:
        if not isinstance(state, DurableMealState):
            raise TypeError("state must be a DurableMealState")
        if not callable(getattr(semantic_interpreter, "interpret_meal_request", None)):
            raise TypeError("semantic_interpreter must provide interpret_meal_request")
        if not isinstance(food_resolver, LocalFDFoodResolver):
            raise TypeError("food_resolver must be a LocalFDFoodResolver")
        if not isinstance(immediate_preparer, ImmediateMealRequestPreparer):
            raise TypeError("immediate_preparer must be an ImmediateMealRequestPreparer")
        if not isinstance(clock, NutritionApplicationClock):
            raise TypeError("clock must be a NutritionApplicationClock")
        if not isinstance(service_calendar, PhelpsServiceCalendar):
            raise TypeError("service_calendar must be a PhelpsServiceCalendar")
        if schedule is not None and not isinstance(schedule, RecommendationDispatchSchedule):
            raise TypeError("schedule must be a RecommendationDispatchSchedule or None")
        self._state = state
        self._semantic_interpreter = semantic_interpreter
        self._food_resolver = food_resolver
        self._immediate_preparer = immediate_preparer
        self._clock = clock
        self._service_calendar = service_calendar
        self._schedule = schedule or RecommendationDispatchSchedule()

    def process(
        self,
        *,
        chat_guid: str,
        user_text: str,
        source_event_id: str,
    ) -> NoActiveMealRequestResult | None:
        """Return ``None`` when this is not a meal request for this route."""

        if not isinstance(chat_guid, str) or not chat_guid.strip():
            raise ValueError("chat_guid must be non-empty text")
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must be non-empty text")
        if not isinstance(source_event_id, str) or not source_event_id.strip():
            raise ValueError("source_event_id must be non-empty text")

        replayed = self.replay(chat_guid=chat_guid, source_event_id=source_event_id)
        if replayed is not None:
            return replayed

        try:
            semantic = self._semantic_interpreter.interpret_meal_request(user_text)
        except Exception:
            return None
        if not isinstance(semantic, MealRequestSemanticResult) or semantic.intent != "meal_request":
            return None
        if semantic.request_mode == "ambiguous":
            return NoActiveMealRequestResult(
                "reply", "Which meal and current menu food would you like me to plan?"
            )
        target = self._target_for(semantic.requested_meal)
        if target is None:
            return NoActiveMealRequestResult(
                "reply",
                "Please say which upcoming meal you mean, such as lunch or dinner.",
            )
        if semantic.request_mode == "targeted" or semantic.requested_food_texts:
            resolved, reply = self._resolve_requested_foods(
                semantic.requested_food_texts,
                target,
            )
            if reply is not None:
                return NoActiveMealRequestResult("reply", reply)
        elif semantic.request_mode == "whole_meal":
            resolved = ()
        else:
            return NoActiveMealRequestResult(
                "reply", "Which meal and current menu food would you like me to plan?"
            )

        if target.mode == "pending":
            meal_name = meal_name_for_display(target.meal_id)
            if resolved:
                names = " and ".join(food.nutrition_record.name for food in resolved)
                reply_text = f"Got it — I'll include {names} in your {meal_name} recommendation."
            else:
                reply_text = f"Got it — I'll make you a different {meal_name} recommendation."
            saved = self._state.save_pending_meal_request(
                source_event_id=source_event_id,
                chat_guid=chat_guid,
                service_date=target.service_date,
                meal=target.meal_id,
                meal_slot=target.meal_slot,
                requested_foods=tuple(
                    requested_meal_food_from_resolved(food) for food in resolved
                ),
                whole_meal=semantic.request_mode == "whole_meal",
                reply_text=reply_text,
            )
            return NoActiveMealRequestResult("pending", saved.reply_text, pending_request=saved)

        try:
            dispatch = self._immediate_preparer.prepare(
                service_date=target.service_date,
                meal=target.meal_id,
                meal_slot=target.meal_slot,
                requested_foods=resolved,
                whole_meal=semantic.request_mode == "whole_meal",
                source_event_id=source_event_id,
                chat_guid=chat_guid,
            )
        except MealRecommendationRequestedFoodError:
            return NoActiveMealRequestResult(
                "reply", "I couldn't include that requested food from the current menu."
            )
        except MealRecommendationServiceUnavailableError:
            return NoActiveMealRequestResult(
                "reply", "That Phelps meal opportunity is not available right now."
            )
        except MealRecommendationUnavailableError:
            return NoActiveMealRequestResult(
                "reply", "I couldn't make an actionable recommendation from the current local menu."
            )
        return NoActiveMealRequestResult(
            "immediate",
            format_immediate_meal_request_message(dispatch.persisted_plan.plan),
            immediate_dispatch=dispatch,
        )

    def replay(
        self,
        *,
        chat_guid: str,
        source_event_id: str,
    ) -> NoActiveMealRequestResult | None:
        """Recover an existing request before any active-plan interpretation.

        A pending request may already have been consumed by the scheduler when
        BlueBubbles retries its original inbound GUID.  It remains a request
        replay, rather than becoming a positive replacement against that new
        active plan.
        """

        if not isinstance(chat_guid, str) or not chat_guid.strip():
            raise ValueError("chat_guid must be non-empty text")
        if not isinstance(source_event_id, str) or not source_event_id.strip():
            raise ValueError("source_event_id must be non-empty text")
        immediate = self._state.load_immediate_meal_request_dispatch(source_event_id)
        if immediate is not None:
            if immediate.chat_guid != chat_guid:
                raise MealReportApplicationError("request source event belongs to another chat")
            return NoActiveMealRequestResult(
                "immediate",
                format_immediate_meal_request_message(immediate.persisted_plan.plan),
                immediate_dispatch=immediate,
            )
        pending = self._state.load_pending_meal_request_for_source_event(source_event_id)
        if pending is not None:
            if pending.chat_guid != chat_guid:
                raise MealReportApplicationError("request source event belongs to another chat")
            return NoActiveMealRequestResult("pending", pending.reply_text, pending_request=pending)
        return None

    def _target_for(self, requested_meal: str | None) -> _MealRequestTarget | None:
        now = self._clock.now()
        service_date = now.date()
        opportunities = self._schedule.opportunities_for(service_date)
        if requested_meal is not None:
            meal_id = canonical_meal_id(requested_meal)
            opportunity = next(
                (item for item in opportunities if item.meal_id == meal_id),
                None,
            )
            if opportunity is None or meal_id is None:
                return None
            activation = _activation_at(service_date, opportunity)
            if now < activation and self._service_calendar.is_meal_context_eligible_at(
                service_date, meal_id, activation
            ):
                return _MealRequestTarget(service_date, meal_id, opportunity.meal_slot, "pending")
            if now >= activation and self._service_calendar.is_meal_context_eligible_at(
                service_date, meal_id, now
            ):
                return _MealRequestTarget(service_date, meal_id, opportunity.meal_slot, "immediate")
            return None

        # Without an explicit meal name, choose only a currently open specific
        # weekend context or a sole future opportunity. Weekday continuous
        # facility hours intentionally do not invent a meal transition.
        availability = self._service_calendar.availability_at(now)
        specific_open = tuple(
            window.meal_id for window in availability.open_windows if window.meal_id is not None
        )
        if len(specific_open) == 1:
            opportunity = next(
                item for item in opportunities if item.meal_id == specific_open[0]
            )
            return _MealRequestTarget(
                service_date, specific_open[0], opportunity.meal_slot, "immediate"
            )
        due = tuple(
            opportunity
            for opportunity in opportunities
            if self._schedule.timing_at(service_date, opportunity, now) == "due"
            and self._service_calendar.is_meal_context_eligible_at(
                service_date,
                opportunity.meal_id,
                now,
            )
        )
        if len(due) == 1:
            return _MealRequestTarget(
                service_date, due[0].meal_id, due[0].meal_slot, "immediate"
            )
        future = tuple(
            opportunity
            for opportunity in opportunities
            if now < _activation_at(service_date, opportunity)
            and self._service_calendar.is_meal_context_eligible_at(
                service_date,
                opportunity.meal_id,
                _activation_at(service_date, opportunity),
            )
        )
        if len(future) == 1:
            return _MealRequestTarget(
                service_date, future[0].meal_id, future[0].meal_slot, "pending"
            )
        return None

    def _resolve_requested_foods(
        self,
        food_texts: tuple[str, ...],
        target: _MealRequestTarget,
    ) -> tuple[tuple[ResolvedFood, ...], str | None]:
        resolved_foods: list[ResolvedFood] = []
        seen: set[tuple[str, str, str]] = set()
        for food_text in food_texts:
            resolved = self._food_resolver.resolve(
                FoodResolutionRequest(food_text, target.service_date, meal=target.meal_id)
            )
            if isinstance(resolved, AmbiguousFood):
                names = ", ".join(
                    candidate.official_display_name
                    + (f" at {candidate.occurrence.station_name}" if candidate.occurrence.station_name else "")
                    for candidate in resolved.candidates[:3]
                )
                return (), (
                    f"I found more than one current menu item for {food_text}: {names}. "
                    "Which one did you mean? Please request its menu name and station."
                )
            if not isinstance(resolved, ResolvedFood):
                return (), (
                    f"{food_text.strip().capitalize()} isn't available for "
                    f"{meal_name_for_display(target.meal_id)}. Want me to make you something else?"
                )
            key = (
                resolved.source_identifier.kind,
                resolved.source_identifier.value,
                resolved.content_signature,
            )
            if key not in seen:
                seen.add(key)
                resolved_foods.append(resolved)
        return tuple(resolved_foods), None


def _activation_at(service_date: date, opportunity: ScheduledMealOpportunity) -> datetime:
    return datetime.combine(
        service_date,
        opportunity.activation_time,
        tzinfo=NUTRITION_APPLICATION_TIMEZONE,
    )
