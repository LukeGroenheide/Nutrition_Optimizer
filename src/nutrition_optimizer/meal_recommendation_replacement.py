"""Durable pre-meal recommendation replacement delivery.

This module is intentionally small glue between the existing deterministic
recommendation orchestrator, durable lifecycle state, and one outbound send.
It never interprets user references or chooses nutrition independently.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Literal, Protocol

from .application import (
    MealRecommendationOrchestrator,
    MealRecommendationReplacementUnavailableError,
    PreparedMealRecommendationReplacement,
    PreparedMealRecommendationRequest,
)
from .durable_state import (
    DurableMealState,
    MealRecommendationReplacementDispatch,
    MealRecommendationReplacementDispatchStatus,
)
from .durable_state import MealRecommendationReplacementConflictError
from .meal_report import MealPlan
from .food_resolution import ResolvedFood
from .nutrition import DailyTargets
from .recommendation_rendering import format_meal_message


__all__ = [
    "MealRecommendationReplacementDelivery",
    "MealRecommendationReplacementDeliveryError",
    "MealRecommendationReplacementPreparer",
    "ReplacementOutboundSender",
    "format_replacement_recommendation_message",
]


_LOGGER = logging.getLogger(__name__)


class ReplacementOutboundSender(Protocol):
    """The stable-id outbound capability used by a replacement delivery."""

    def send_text(
        self,
        chat_guid: str,
        message: str,
        *,
        temp_guid: str | None = None,
    ) -> object:
        """Send one message using the durable BlueBubbles temporary GUID."""


class MealRecommendationReplacementDeliveryError(RuntimeError):
    """Raised when a persisted replacement cannot be safely completed."""

    def __init__(self, status: MealRecommendationReplacementDispatchStatus) -> None:
        self.status = status
        super().__init__("replacement recommendation delivery is not complete")


class MealRecommendationReplacementPreparer:
    """Calculate and atomically persist one user-requested replacement.

    The durable inbound GUID is checked before target loading or optimization,
    so a webhook/poller retry cannot create a second plan even after process
    restart.  The state transaction owns plan supersession and message-event
    persistence; this class is otherwise side-effect free.
    """

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
        prior_plan_id: str,
        rejected_plan_item_ids: tuple[str, ...],
        whole_meal: bool,
        source_event_id: str,
        chat_guid: str,
        request_kind: Literal["rejection", "meal_request"] = "rejection",
    ) -> MealRecommendationReplacementDispatch:
        """Return one persisted dispatch, reusing it exactly on GUID replay."""

        if not isinstance(prior_plan_id, str) or not prior_plan_id.strip():
            raise ValueError("prior_plan_id must be non-empty text")
        if not isinstance(rejected_plan_item_ids, tuple) or not rejected_plan_item_ids:
            raise ValueError("rejected_plan_item_ids must be a non-empty tuple")
        if not isinstance(whole_meal, bool):
            raise TypeError("whole_meal must be a bool")
        if not isinstance(source_event_id, str) or not source_event_id.strip():
            raise ValueError("source_event_id must be non-empty text")
        if not isinstance(chat_guid, str) or not chat_guid.strip():
            raise ValueError("chat_guid must be non-empty text")
        if request_kind not in {"rejection", "meal_request"}:
            raise ValueError("request_kind is invalid")
        if request_kind == "meal_request" and not whole_meal:
            raise ValueError("positive targeted requests must use prepare_requested")

        existing = self._state.load_meal_recommendation_replacement_dispatch(
            source_event_id
        )
        if existing is not None:
            if existing.chat_guid != chat_guid:
                raise MealRecommendationReplacementConflictError(
                    "replacement source event belongs to another chat"
                )
            return existing

        prior_plan = self._state.load_meal_plan(prior_plan_id)
        if prior_plan is None:
            raise MealRecommendationReplacementUnavailableError("prior plan is missing")
        supplied_targets = self._target_loader()
        if not isinstance(supplied_targets, DailyTargets):
            raise TypeError("target_loader returned invalid targets")
        try:
            prepared = self._recommendation_orchestrator.prepare_replacement(
                prior_plan,
                rejected_plan_item_ids,
                whole_meal=whole_meal,
                targets=supplied_targets,
            )
        except MealRecommendationReplacementUnavailableError:
            # A concurrent webhook/poller delivery of this exact GUID may
            # have committed the replacement while this invocation was still
            # calculating. Reuse that durable work rather than recording a
            # contradictory "unavailable" interaction.
            concurrent = self._state.load_meal_recommendation_replacement_dispatch(
                source_event_id
            )
            if concurrent is not None and concurrent.chat_guid == chat_guid:
                return concurrent
            raise
        if not isinstance(prepared, PreparedMealRecommendationReplacement):
            raise TypeError("replacement orchestrator returned invalid output")
        return self._state.save_replacement_meal_plan(
            prior_plan,
            prepared.rendering.meal_plan,
            source_event_id=source_event_id,
            chat_guid=chat_guid,
            rejected_plan_item_ids=(
                prepared.rejected_plan_item_ids if request_kind == "rejection" else ()
            ),
            whole_meal=prepared.whole_meal,
            request_kind=request_kind,
            requested_foods=(),
            reply_text=format_replacement_recommendation_message(
                prepared.rendering.meal_plan
            ),
        )

    def prepare_requested(
        self,
        *,
        prior_plan_id: str,
        requested_foods: tuple[ResolvedFood, ...],
        whole_meal: bool,
        source_event_id: str,
        chat_guid: str,
    ) -> MealRecommendationReplacementDispatch:
        """Persist one positive active-plan request with existing delivery safety."""

        if not isinstance(prior_plan_id, str) or not prior_plan_id.strip():
            raise ValueError("prior_plan_id must be non-empty text")
        if not isinstance(requested_foods, tuple) or not requested_foods:
            raise ValueError("requested_foods must be a non-empty tuple")
        if not all(isinstance(food, ResolvedFood) for food in requested_foods):
            raise TypeError("requested_foods must contain ResolvedFood values")
        if not isinstance(whole_meal, bool):
            raise TypeError("whole_meal must be a bool")
        if not isinstance(source_event_id, str) or not source_event_id.strip():
            raise ValueError("source_event_id must be non-empty text")
        if not isinstance(chat_guid, str) or not chat_guid.strip():
            raise ValueError("chat_guid must be non-empty text")

        existing = self._state.load_meal_recommendation_replacement_dispatch(source_event_id)
        if existing is not None:
            if existing.chat_guid != chat_guid:
                raise MealRecommendationReplacementConflictError(
                    "replacement source event belongs to another chat"
                )
            return existing
        prior_plan = self._state.load_meal_plan(prior_plan_id)
        if prior_plan is None:
            raise MealRecommendationReplacementUnavailableError("prior plan is missing")
        supplied_targets = self._target_loader()
        if not isinstance(supplied_targets, DailyTargets):
            raise TypeError("target_loader returned invalid targets")
        try:
            prepared = self._recommendation_orchestrator.prepare_requested_replacement(
                prior_plan,
                requested_foods,
                whole_meal=whole_meal,
                targets=supplied_targets,
            )
        except MealRecommendationReplacementUnavailableError:
            concurrent = self._state.load_meal_recommendation_replacement_dispatch(
                source_event_id
            )
            if concurrent is not None and concurrent.chat_guid == chat_guid:
                return concurrent
            raise
        if not isinstance(prepared, PreparedMealRecommendationRequest):
            raise TypeError("requested replacement orchestrator returned invalid output")
        return self._state.save_replacement_meal_plan(
            prior_plan,
            prepared.rendering.meal_plan,
            source_event_id=source_event_id,
            chat_guid=chat_guid,
            rejected_plan_item_ids=(),
            whole_meal=prepared.whole_meal,
            request_kind="meal_request",
            requested_foods=prepared.requested_foods,
            reply_text=format_replacement_recommendation_message(
                prepared.rendering.meal_plan
            ),
        )


class MealRecommendationReplacementDelivery:
    """Send a persisted replacement once, with scheduler-like retry safety."""

    def __init__(
        self,
        state: DurableMealState,
        outbound_sender: ReplacementOutboundSender,
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

    def deliver(
        self,
        dispatch: MealRecommendationReplacementDispatch,
    ) -> MealRecommendationReplacementDispatch:
        """Deliver a one-time replacement or preserve conservative uncertainty."""

        if not isinstance(dispatch, MealRecommendationReplacementDispatch):
            raise TypeError("dispatch must be a MealRecommendationReplacementDispatch")
        current = self._state.load_meal_recommendation_replacement_dispatch(
            dispatch.source_event_id
        )
        if current is None or current.plan_id != dispatch.plan_id:
            raise MealRecommendationReplacementDeliveryError(dispatch.status)
        if current.status == "delivered":
            return current
        if current.persisted_plan.status != "active":
            # A later successfully delivered scheduler recommendation (or an
            # explicit valid replacement) retired this unsent plan. Never
            # revive it merely because an old inbound GUID retries. Returning
            # the durable row lets the inbound cursor complete without a
            # retry loop or a stale outbound message.
            return current
        if current.status == "sending":
            # The external hand-off may already have happened.  Do not turn an
            # unknown outcome into a second recommendation message.
            raise MealRecommendationReplacementDeliveryError("sending")
        if current.status != "pending_delivery":
            raise MealRecommendationReplacementDeliveryError(current.status)
        if not self._state.claim_replacement_recommendation_delivery(current):
            refreshed = self._state.load_meal_recommendation_replacement_dispatch(
                current.source_event_id
            )
            if refreshed is not None and refreshed.status == "delivered":
                return refreshed
            raise MealRecommendationReplacementDeliveryError(
                current.status if refreshed is None else refreshed.status
            )
        try:
            self._outbound_sender.send_text(
                current.chat_guid,
                format_replacement_recommendation_message(current.persisted_plan.plan),
                temp_guid=current.delivery_token,
            )
        except Exception as exc:
            self._logger.warning(
                "Replacement recommendation outbound delivery failed (%s)",
                type(exc).__name__,
            )
            if not self._is_known_delivery_failure(exc):
                raise MealRecommendationReplacementDeliveryError("sending") from None
            try:
                self._state.release_replacement_recommendation_delivery(current)
            except Exception:
                # The state is still at least conservative (sending); do not
                # assert a retryable pending state we failed to persist.
                raise MealRecommendationReplacementDeliveryError("sending") from None
            raise MealRecommendationReplacementDeliveryError("pending_delivery") from None
        try:
            return self._state.mark_replacement_recommendation_delivered(current)
        except Exception:
            # A successful transport call without durable acknowledgement is
            # necessarily uncertain.  The row stays sending and never resends
            # automatically.
            raise MealRecommendationReplacementDeliveryError("sending") from None


def format_replacement_recommendation_message(plan: MealPlan) -> str:
    """Mark an otherwise standard persisted plan as an explicit update."""

    if not isinstance(plan, MealPlan):
        raise TypeError("plan must be a MealPlan")
    return f"Updated recommendation:\n{format_meal_message(plan)}"
