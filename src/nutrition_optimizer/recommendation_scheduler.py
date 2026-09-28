"""Durable, Detroit-local dispatch for automatic meal recommendations.

The scheduler is deliberately an application boundary rather than a timer
implementation.  A lightweight external runner may call :meth:`run_once`
regularly, while this module owns the service-date, availability, durable
idempotency, and retry decisions.  It never reimplements optimization: new
plans still flow through :class:`MealRecommendationOrchestrator`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import logging
from types import MappingProxyType
from typing import Literal, Protocol

from .application import (
    MealRecommendationPreparationError,
    MealRecommendationRequestedFoodError,
    MealRecommendationUnavailableError,
    PreparedMealRecommendation,
)
from .application_clock import (
    DEFAULT_NUTRITION_APPLICATION_CLOCK,
    NUTRITION_APPLICATION_TIMEZONE,
    NutritionApplicationClock,
)
from .durable_state import (
    DurableMealState,
    PendingMealRequest,
    ScheduledRecommendationAlreadyExistsError,
    ScheduledRecommendationDispatch,
    ScheduledRecommendationPlanConflictError,
    StaleMealPlanRetirement,
)
from .meal_identity import MealSlot, canonical_meal_id, require_meal_slot
from .nutrition import DailyTargets
from .phelps_service_calendar import (
    DEFAULT_PHELPS_SERVICE_CALENDAR,
    PhelpsServiceCalendar,
)
from .recommendation_rendering import format_meal_message


__all__ = [
    "DEFAULT_RECOMMENDATION_DISPATCH_SCHEDULE",
    "RECOMMENDATION_DISPATCH_WEEKLY_OPPORTUNITIES",
    "RecommendationDispatchOutcome",
    "RecommendationDispatchResult",
    "RecommendationDispatchSchedule",
    "RecommendationScheduler",
    "RecommendationSchedulerRunResult",
    "RecommendationSchedulerStatus",
    "RecommendationSchedulerStatusItem",
    "ScheduledMealOpportunity",
]


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScheduledMealOpportunity:
    """One product-level recommendation activation, not a facility schedule."""

    meal_id: int
    activation_time: time
    meal_slot: MealSlot

    def __post_init__(self) -> None:
        if canonical_meal_id(self.meal_id) != self.meal_id:
            raise ValueError("meal_id must be a known canonical FD meal ID")
        if not isinstance(self.activation_time, time):
            raise TypeError("activation_time must be a time")
        if self.activation_time.tzinfo is not None:
            raise ValueError("activation_time must be a naive Detroit-local time")
        require_meal_slot(self.meal_slot)

    @property
    def label(self) -> str:
        """Retain the existing diagnostic label as the canonical product slot."""

        return self.meal_slot


# These are deliberately product delivery times, separate from the published
# Phelps facility windows.  Weekday dining is continuous, so the weekday
# lunch/dinner values below are product activation boundaries rather than
# claims about facility meal transitions. Sunday has no fd:1 menu opportunity;
# current FD cache semantics map Hope's Sunday brunch to fd:2.
_WEEKDAY_RECOMMENDATION_OPPORTUNITIES = (
    ScheduledMealOpportunity(1, time(7, 30), "breakfast"),
    ScheduledMealOpportunity(2, time(11, 30), "lunch"),
    ScheduledMealOpportunity(3, time(17), "dinner"),
)
_SATURDAY_RECOMMENDATION_OPPORTUNITIES = (
    ScheduledMealOpportunity(1, time(7, 30), "breakfast"),
    ScheduledMealOpportunity(2, time(10, 30), "lunch"),
    ScheduledMealOpportunity(3, time(17), "dinner"),
)
_SUNDAY_RECOMMENDATION_OPPORTUNITIES = (
    ScheduledMealOpportunity(2, time(10, 30), "brunch"),
    ScheduledMealOpportunity(3, time(17), "dinner"),
)

RECOMMENDATION_DISPATCH_WEEKLY_OPPORTUNITIES: Mapping[
    int, tuple[ScheduledMealOpportunity, ...]
] = MappingProxyType(
    {
        0: _WEEKDAY_RECOMMENDATION_OPPORTUNITIES,
        1: _WEEKDAY_RECOMMENDATION_OPPORTUNITIES,
        2: _WEEKDAY_RECOMMENDATION_OPPORTUNITIES,
        3: _WEEKDAY_RECOMMENDATION_OPPORTUNITIES,
        4: _WEEKDAY_RECOMMENDATION_OPPORTUNITIES,
        5: _SATURDAY_RECOMMENDATION_OPPORTUNITIES,
        6: _SUNDAY_RECOMMENDATION_OPPORTUNITIES,
    }
)


class RecommendationDispatchSchedule:
    """Replaceable Detroit-local product schedule with a bounded catch-up window."""

    def __init__(
        self,
        weekly_opportunities: Mapping[int, Sequence[ScheduledMealOpportunity]] = RECOMMENDATION_DISPATCH_WEEKLY_OPPORTUNITIES,
        *,
        date_overrides: Mapping[date, Sequence[ScheduledMealOpportunity]] | None = None,
        catch_up_window: timedelta = timedelta(minutes=60),
    ) -> None:
        self._weekly_opportunities = _normalize_weekly_opportunities(weekly_opportunities)
        self._date_overrides = _normalize_date_overrides(date_overrides)
        if not isinstance(catch_up_window, timedelta) or catch_up_window <= timedelta(0):
            raise ValueError("catch_up_window must be a positive timedelta")
        self.catch_up_window = catch_up_window

    def opportunities_for(self, service_date: date) -> tuple[ScheduledMealOpportunity, ...]:
        """Return the configured product opportunities for an explicit date."""

        _require_service_date(service_date)
        return self._date_overrides.get(
            service_date,
            self._weekly_opportunities[service_date.weekday()],
        )

    def timing_at(
        self,
        service_date: date,
        opportunity: ScheduledMealOpportunity,
        instant: datetime,
    ) -> Literal["not_due", "due", "too_late"]:
        """Classify an absolute instant against one product activation window."""

        _require_service_date(service_date)
        if not isinstance(opportunity, ScheduledMealOpportunity):
            raise TypeError("opportunity must be a ScheduledMealOpportunity")
        local_now = _local_datetime(instant)
        if local_now.date() < service_date:
            return "not_due"
        if local_now.date() > service_date:
            return "too_late"
        starts_at = datetime.combine(
            service_date,
            opportunity.activation_time,
            tzinfo=NUTRITION_APPLICATION_TIMEZONE,
        )
        if local_now < starts_at:
            return "not_due"
        if local_now <= starts_at + self.catch_up_window:
            return "due"
        return "too_late"


RecommendationDispatchOutcome = Literal[
    "not_due",
    "too_late",
    "unavailable",
    "manual_plan_conflict",
    "already_delivered",
    "delivery_in_progress",
    "pending_delivery",
    "expired",
    "dispatched",
    "delivery_failed",
    "delivery_state_unknown",
    "target_loading_failed",
    "preparation_failed",
]


@dataclass(frozen=True, slots=True)
class RecommendationDispatchResult:
    """One scheduler decision with no user text or transport credentials."""

    opportunity: ScheduledMealOpportunity
    outcome: RecommendationDispatchOutcome
    plan_id: str | None = None
    dispatch_status: str | None = None
    diagnostic_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.opportunity, ScheduledMealOpportunity):
            raise TypeError("opportunity must be a ScheduledMealOpportunity")
        if self.outcome not in {
            "not_due",
            "too_late",
            "unavailable",
            "manual_plan_conflict",
            "already_delivered",
            "delivery_in_progress",
            "pending_delivery",
            "expired",
            "dispatched",
            "delivery_failed",
            "delivery_state_unknown",
            "target_loading_failed",
            "preparation_failed",
        }:
            raise ValueError("outcome is invalid")
        if self.plan_id is not None and (not isinstance(self.plan_id, str) or not self.plan_id.strip()):
            raise ValueError("plan_id must be non-empty text when supplied")
        if self.dispatch_status is not None and (
            not isinstance(self.dispatch_status, str) or not self.dispatch_status.strip()
        ):
            raise ValueError("dispatch_status must be non-empty text when supplied")
        if self.diagnostic_reason is not None and (
            not isinstance(self.diagnostic_reason, str) or not self.diagnostic_reason.strip()
        ):
            raise ValueError("diagnostic_reason must be non-empty text when supplied")


@dataclass(frozen=True, slots=True)
class RecommendationSchedulerRunResult:
    """The complete result of one mutable scheduler evaluation."""

    local_now: datetime
    stale_retirement: StaleMealPlanRetirement
    decisions: tuple[RecommendationDispatchResult, ...]

    def __post_init__(self) -> None:
        _local_datetime(self.local_now)
        if not isinstance(self.stale_retirement, StaleMealPlanRetirement):
            raise TypeError("stale_retirement must be a StaleMealPlanRetirement")
        if not isinstance(self.decisions, tuple) or not all(
            isinstance(decision, RecommendationDispatchResult) for decision in self.decisions
        ):
            raise TypeError("decisions must be RecommendationDispatchResult values")

    @property
    def has_retryable_failure(self) -> bool:
        """Return whether the next periodic invocation should try again."""

        return any(
            decision.outcome
            in {"delivery_failed", "target_loading_failed", "preparation_failed"}
            for decision in self.decisions
        )

    @property
    def has_operational_failure(self) -> bool:
        """Return whether a one-shot runner should surface an operator failure."""

        return self.has_retryable_failure or any(
            decision.outcome == "delivery_state_unknown" for decision in self.decisions
        )


@dataclass(frozen=True, slots=True)
class RecommendationSchedulerStatusItem:
    """Read-only status for one configured product opportunity."""

    opportunity: ScheduledMealOpportunity
    timing: Literal["not_due", "due", "too_late"]
    phelps_eligible: bool
    reportable: bool
    dispatch_status: str | None
    plan_id: str | None
    active_plan_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.opportunity, ScheduledMealOpportunity):
            raise TypeError("opportunity must be a ScheduledMealOpportunity")
        if self.timing not in {"not_due", "due", "too_late"}:
            raise ValueError("timing is invalid")
        if not isinstance(self.phelps_eligible, bool):
            raise TypeError("phelps_eligible must be a bool")
        if not isinstance(self.reportable, bool):
            raise TypeError("reportable must be a bool")
        for value, name in (
            (self.dispatch_status, "dispatch_status"),
            (self.plan_id, "plan_id"),
            (self.active_plan_id, "active_plan_id"),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be non-empty text when supplied")


@dataclass(frozen=True, slots=True)
class RecommendationSchedulerStatus:
    """Read-only scheduler diagnostic suitable for an operator command."""

    local_now: datetime
    items: tuple[RecommendationSchedulerStatusItem, ...]

    def __post_init__(self) -> None:
        _local_datetime(self.local_now)
        if not isinstance(self.items, tuple) or not all(
            isinstance(item, RecommendationSchedulerStatusItem) for item in self.items
        ):
            raise TypeError("items must be RecommendationSchedulerStatusItem values")


class ScheduledRecommendationPreparer(Protocol):
    """The existing optimizer/rendering boundary with scheduler persistence."""

    def prepare_scheduled(
        self,
        service_date: date,
        meal: str | int,
        targets: DailyTargets,
        *,
        meal_slot: MealSlot,
    ) -> PreparedMealRecommendation:
        """Create one pending scheduler-owned plan without replacement semantics."""

    def prepare_scheduled_pending_request(
        self,
        pending_request: PendingMealRequest,
        targets: DailyTargets,
    ) -> PreparedMealRecommendation:
        """Create and atomically consume one stored positive meal request."""


class ScheduledRecommendationSender(Protocol):
    """Narrow outbound capability, including BlueBubbles' stable temp GUID."""

    def send_text(
        self,
        chat_guid: str,
        message: str,
        *,
        temp_guid: str | None = None,
    ) -> object:
        """Send one deterministic rendered message."""


class RecommendationScheduler:
    """Evaluate due Detroit-local opportunities and deliver each plan once.

    A missing durable dispatch means preparation has not happened.  The first
    successful preparation atomically creates both the plan and a
    ``pending_delivery`` dispatch.  A later send failure returns that exact
    row to pending, so the next invocation reuses its immutable plan rather
    than recalculating or superseding it.
    """

    def __init__(
        self,
        state: DurableMealState,
        *,
        schedule: RecommendationDispatchSchedule | None = None,
        service_calendar: PhelpsServiceCalendar = DEFAULT_PHELPS_SERVICE_CALENDAR,
        clock: NutritionApplicationClock | None = None,
        recommendation_preparer: ScheduledRecommendationPreparer | None = None,
        target_loader: Callable[[], DailyTargets] | None = None,
        outbound_sender: ScheduledRecommendationSender | None = None,
        chat_guid: str | None = None,
        is_known_delivery_failure: Callable[[BaseException], bool] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(state, DurableMealState):
            raise TypeError("state must be a DurableMealState")
        if schedule is not None and not isinstance(schedule, RecommendationDispatchSchedule):
            raise TypeError("schedule must be a RecommendationDispatchSchedule")
        if not isinstance(service_calendar, PhelpsServiceCalendar):
            raise TypeError("service_calendar must be a PhelpsServiceCalendar")
        if clock is not None and not isinstance(clock, NutritionApplicationClock):
            raise TypeError("clock must be a NutritionApplicationClock")
        if recommendation_preparer is not None and not callable(
            getattr(recommendation_preparer, "prepare_scheduled", None)
        ):
            raise TypeError("recommendation_preparer must provide prepare_scheduled")
        if target_loader is not None and not callable(target_loader):
            raise TypeError("target_loader must be callable")
        if outbound_sender is not None and not callable(getattr(outbound_sender, "send_text", None)):
            raise TypeError("outbound_sender must provide send_text")
        if chat_guid is not None and (not isinstance(chat_guid, str) or not chat_guid.strip()):
            raise ValueError("chat_guid must be non-empty text when supplied")
        if is_known_delivery_failure is not None and not callable(is_known_delivery_failure):
            raise TypeError("is_known_delivery_failure must be callable")
        self._state = state
        self._schedule = schedule or DEFAULT_RECOMMENDATION_DISPATCH_SCHEDULE
        self._service_calendar = service_calendar
        self._clock = clock or DEFAULT_NUTRITION_APPLICATION_CLOCK
        self._recommendation_preparer = recommendation_preparer
        self._target_loader = target_loader
        self._outbound_sender = outbound_sender
        self._chat_guid = chat_guid
        self._is_known_delivery_failure = (
            is_known_delivery_failure or _default_known_delivery_failure
        )
        self._logger = logger or _LOGGER

    def status(self) -> RecommendationSchedulerStatus:
        """Inspect today's schedule/dispatch state without any mutation or send."""

        local_now = self._clock.now()
        service_date = local_now.date()
        items: list[RecommendationSchedulerStatusItem] = []
        for opportunity in self._schedule.opportunities_for(service_date):
            dispatch = self._state.load_scheduled_recommendation_dispatch(
                service_date, opportunity.meal_id, meal_slot=opportunity.meal_slot
            )
            active_plan = self._state.load_active_meal_plan(
                service_date,
                opportunity.meal_id,
                meal_slot=opportunity.meal_slot,
            )
            reportable = active_plan is not None and self._state.is_active_meal_plan_reportable(
                active_plan,
                self._service_calendar,
                evaluated_at=local_now,
            )
            items.append(
                RecommendationSchedulerStatusItem(
                    opportunity=opportunity,
                    timing=self._schedule.timing_at(service_date, opportunity, local_now),
                    phelps_eligible=self._service_calendar.is_meal_context_eligible_at(
                        service_date, opportunity.meal_id, local_now
                    ),
                    reportable=reportable,
                    dispatch_status=None if dispatch is None else dispatch.status,
                    plan_id=None if dispatch is None else dispatch.plan_id,
                    active_plan_id=None if active_plan is None else active_plan.plan_id,
                )
            )
        return RecommendationSchedulerStatus(local_now, tuple(items))

    def run_once(self) -> RecommendationSchedulerRunResult:
        """Perform one bounded scheduler evaluation and at most one send per meal."""

        self._require_dispatch_dependencies()
        local_now = self._clock.now()
        service_date = local_now.date()
        stale_retirement = self._state.retire_stale_active_meal_plans(
            self._service_calendar,
            evaluated_at=local_now,
        )
        self._state.retire_stale_pending_meal_requests(service_date)
        decisions = tuple(
            self._dispatch_opportunity(service_date, opportunity, local_now)
            for opportunity in self._schedule.opportunities_for(service_date)
        )
        return RecommendationSchedulerRunResult(local_now, stale_retirement, decisions)

    def _require_dispatch_dependencies(self) -> None:
        if self._recommendation_preparer is None:
            raise RuntimeError("recommendation_preparer is required to run the scheduler")
        if self._target_loader is None:
            raise RuntimeError("target_loader is required to run the scheduler")
        if self._outbound_sender is None:
            raise RuntimeError("outbound_sender is required to run the scheduler")
        if self._chat_guid is None:
            raise RuntimeError("chat_guid is required to run the scheduler")

    def _dispatch_opportunity(
        self,
        service_date: date,
        opportunity: ScheduledMealOpportunity,
        local_now: datetime,
    ) -> RecommendationDispatchResult:
        dispatch = self._state.load_scheduled_recommendation_dispatch(
            service_date, opportunity.meal_id, meal_slot=opportunity.meal_slot
        )
        if dispatch is not None:
            terminal = self._terminal_dispatch_result(opportunity, dispatch)
            if terminal is not None:
                return terminal

        timing = self._schedule.timing_at(service_date, opportunity, local_now)
        if timing == "not_due":
            return self._result(opportunity, "not_due", dispatch)
        if timing == "too_late":
            if dispatch is not None and dispatch.status == "pending_delivery":
                dispatch = self._state.expire_scheduled_recommendation_dispatch(
                    dispatch,
                    expired_at=local_now,
                )
                return self._result(opportunity, "expired", dispatch)
            return self._result(opportunity, "too_late", dispatch)

        if not self._service_calendar.is_meal_context_eligible_at(
            service_date,
            opportunity.meal_id,
            local_now,
        ):
            return self._result(
                opportunity,
                "unavailable",
                dispatch,
                diagnostic_reason="phelps_service_unavailable",
            )

        if dispatch is None:
            active_plan = self._state.load_active_meal_plan(
                service_date,
                opportunity.meal_id,
                meal_slot=opportunity.meal_slot,
            )
            if active_plan is not None:
                return RecommendationDispatchResult(
                    opportunity,
                    "manual_plan_conflict",
                    active_plan.plan_id,
                    None,
                )
            assert self._chat_guid is not None
            pending_request = self._state.load_pending_meal_request(
                service_date,
                opportunity.meal_id,
                self._chat_guid,
                meal_slot=opportunity.meal_slot,
            )
            dispatch, preparation_result = self._prepare_dispatch(
                service_date,
                opportunity,
                pending_request=pending_request,
            )
            if preparation_result is not None:
                return preparation_result
            assert dispatch is not None

        if dispatch.persisted_plan.status != "active":
            if dispatch.status == "pending_delivery":
                dispatch = self._state.expire_scheduled_recommendation_dispatch(
                    dispatch,
                    expired_at=local_now,
                )
            return self._result(opportunity, "expired", dispatch)
        return self._send_pending_dispatch(opportunity, dispatch)

    def _prepare_dispatch(
        self,
        service_date: date,
        opportunity: ScheduledMealOpportunity,
        *,
        pending_request: PendingMealRequest | None = None,
    ) -> tuple[ScheduledRecommendationDispatch | None, RecommendationDispatchResult | None]:
        """Prepare exactly once, leaving errors unpersisted and classified."""

        assert self._target_loader is not None
        assert self._recommendation_preparer is not None
        try:
            supplied_targets = self._target_loader()
        except Exception as exc:
            self._logger.warning("Scheduled target loading failed (%s)", type(exc).__name__)
            return None, RecommendationDispatchResult(opportunity, "target_loading_failed")
        if not isinstance(supplied_targets, DailyTargets):
            return None, RecommendationDispatchResult(opportunity, "target_loading_failed")
        try:
            if pending_request is None:
                prepared = self._recommendation_preparer.prepare_scheduled(
                    service_date,
                    opportunity.meal_id,
                    supplied_targets,
                    meal_slot=opportunity.meal_slot,
                )
            else:
                prepare_pending = getattr(
                    self._recommendation_preparer,
                    "prepare_scheduled_pending_request",
                    None,
                )
                if not callable(prepare_pending):
                    raise TypeError(
                        "recommendation_preparer must provide prepare_scheduled_pending_request"
                    )
                prepared = prepare_pending(pending_request, supplied_targets)
            if not isinstance(prepared, PreparedMealRecommendation):
                raise TypeError("recommendation_preparer returned invalid output")
        except ScheduledRecommendationAlreadyExistsError:
            dispatch = self._state.load_scheduled_recommendation_dispatch(
                service_date, opportunity.meal_id, meal_slot=opportunity.meal_slot
            )
            if dispatch is not None:
                return dispatch, None
            return None, RecommendationDispatchResult(opportunity, "preparation_failed")
        except ScheduledRecommendationPlanConflictError:
            active_plan = self._state.load_active_meal_plan(
                service_date,
                opportunity.meal_id,
                meal_slot=opportunity.meal_slot,
            )
            return None, RecommendationDispatchResult(
                opportunity,
                "manual_plan_conflict",
                None if active_plan is None else active_plan.plan_id,
            )
        except MealRecommendationRequestedFoodError as exc:
            return None, RecommendationDispatchResult(
                opportunity,
                "unavailable",
                diagnostic_reason=(
                    "requested_food_infeasible"
                    if exc.reason == "required_food_infeasible"
                    else "requested_food_unavailable"
                ),
            )
        except MealRecommendationUnavailableError as exc:
            # ``empty_menu`` is the deterministic local optimizer result when
            # the current FD occurrence cache has no usable rows.  Preserve
            # the existing unavailable outcome/idempotency behavior while
            # making the operational cause explicit for the five-minute log.
            reason = (
                "menu_coverage_missing"
                if exc.outcome == "empty_menu"
                else "recommendation_unavailable"
            )
            return None, RecommendationDispatchResult(
                opportunity,
                "unavailable",
                diagnostic_reason=reason,
            )
        except MealRecommendationPreparationError as exc:
            self._logger.warning("Scheduled recommendation preparation failed (%s)", type(exc).__name__)
            return None, RecommendationDispatchResult(opportunity, "preparation_failed")
        except Exception as exc:
            self._logger.warning("Scheduled recommendation preparation failed (%s)", type(exc).__name__)
            return None, RecommendationDispatchResult(opportunity, "preparation_failed")
        dispatch = self._state.load_scheduled_recommendation_dispatch(
            service_date,
            opportunity.meal_id,
            meal_slot=opportunity.meal_slot,
        )
        if dispatch is None or dispatch.plan_id != prepared.plan_id:
            return None, RecommendationDispatchResult(opportunity, "preparation_failed")
        return dispatch, None

    def _send_pending_dispatch(
        self,
        opportunity: ScheduledMealOpportunity,
        dispatch: ScheduledRecommendationDispatch,
    ) -> RecommendationDispatchResult:
        terminal = self._terminal_dispatch_result(opportunity, dispatch)
        if terminal is not None:
            return terminal
        if dispatch.status != "pending_delivery":
            return self._result(opportunity, "pending_delivery", dispatch)
        if not self._state.claim_scheduled_recommendation_delivery(dispatch):
            refreshed = self._state.load_scheduled_recommendation_dispatch(
                dispatch.service_date,
                opportunity.meal_id,
                meal_slot=opportunity.meal_slot,
            )
            if refreshed is None:
                return RecommendationDispatchResult(opportunity, "delivery_state_unknown")
            terminal = self._terminal_dispatch_result(opportunity, refreshed)
            if terminal is not None:
                return terminal
            return self._result(opportunity, "pending_delivery", refreshed)

        assert self._outbound_sender is not None
        assert self._chat_guid is not None
        message = format_meal_message(dispatch.persisted_plan.plan)
        try:
            self._outbound_sender.send_text(
                self._chat_guid,
                message,
                temp_guid=dispatch.delivery_token,
            )
        except Exception as exc:
            self._logger.warning(
                "Scheduled recommendation outbound delivery failed (%s)", type(exc).__name__
            )
            if not self._is_known_delivery_failure(exc):
                # A transport interruption can occur after the remote service
                # accepted the request.  Preserve ``sending`` and require an
                # explicit operator decision rather than turning uncertainty
                # into an automatic duplicate message.
                return self._result(opportunity, "delivery_state_unknown", dispatch)
            try:
                pending = self._state.release_scheduled_recommendation_delivery(dispatch)
            except Exception as state_exc:
                self._logger.error(
                    "Scheduled recommendation delivery state could not be released (%s)",
                    type(state_exc).__name__,
                )
                return self._result(opportunity, "delivery_state_unknown", dispatch)
            return self._result(opportunity, "delivery_failed", pending)
        try:
            delivered = self._state.mark_scheduled_recommendation_delivered(dispatch)
        except Exception as exc:
            # The outbound call returned, but the durable acknowledgement did
            # not.  Keep the row in ``sending`` and fail closed rather than
            # risking a second iMessage on a later timer invocation.
            self._logger.error(
                "Scheduled recommendation delivery state could not be completed (%s)",
                type(exc).__name__,
            )
            return self._result(opportunity, "delivery_state_unknown", dispatch)
        return self._result(opportunity, "dispatched", delivered)

    @staticmethod
    def _terminal_dispatch_result(
        opportunity: ScheduledMealOpportunity,
        dispatch: ScheduledRecommendationDispatch,
    ) -> RecommendationDispatchResult | None:
        if dispatch.status == "delivered":
            return RecommendationDispatchResult(
                opportunity, "already_delivered", dispatch.plan_id, dispatch.status
            )
        if dispatch.status == "sending":
            return RecommendationDispatchResult(
                opportunity, "delivery_in_progress", dispatch.plan_id, dispatch.status
            )
        if dispatch.status == "expired":
            return RecommendationDispatchResult(
                opportunity, "expired", dispatch.plan_id, dispatch.status
            )
        return None

    @staticmethod
    def _result(
        opportunity: ScheduledMealOpportunity,
        outcome: RecommendationDispatchOutcome,
        dispatch: ScheduledRecommendationDispatch | None = None,
        *,
        diagnostic_reason: str | None = None,
    ) -> RecommendationDispatchResult:
        return RecommendationDispatchResult(
            opportunity,
            outcome,
            None if dispatch is None else dispatch.plan_id,
            None if dispatch is None else dispatch.status,
            diagnostic_reason,
        )


def _normalize_weekly_opportunities(
    weekly_opportunities: Mapping[int, Sequence[ScheduledMealOpportunity]],
) -> Mapping[int, tuple[ScheduledMealOpportunity, ...]]:
    if not isinstance(weekly_opportunities, Mapping):
        raise TypeError("weekly_opportunities must be a mapping")
    if set(weekly_opportunities) != set(range(7)):
        raise ValueError("weekly_opportunities keys must be weekdays 0 through 6")
    normalized: dict[int, tuple[ScheduledMealOpportunity, ...]] = {}
    for weekday in range(7):
        values = weekly_opportunities[weekday]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError("weekly opportunities must be sequences")
        opportunities = tuple(values)
        if not all(isinstance(value, ScheduledMealOpportunity) for value in opportunities):
            raise TypeError("weekly opportunities must use ScheduledMealOpportunity")
        meal_slots = tuple(value.meal_slot for value in opportunities)
        if len(set(meal_slots)) != len(meal_slots):
            raise ValueError("a weekday cannot schedule the same product meal slot twice")
        normalized[weekday] = opportunities
    return MappingProxyType(normalized)


def _normalize_date_overrides(
    date_overrides: Mapping[date, Sequence[ScheduledMealOpportunity]] | None,
) -> Mapping[date, tuple[ScheduledMealOpportunity, ...]]:
    if date_overrides is None:
        return MappingProxyType({})
    if not isinstance(date_overrides, Mapping):
        raise TypeError("date_overrides must be a mapping")
    normalized: dict[date, tuple[ScheduledMealOpportunity, ...]] = {}
    for service_date, values in date_overrides.items():
        _require_service_date(service_date)
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError("date override opportunities must be sequences")
        opportunities = tuple(values)
        if not all(isinstance(value, ScheduledMealOpportunity) for value in opportunities):
            raise TypeError("date override opportunities must use ScheduledMealOpportunity")
        meal_slots = tuple(value.meal_slot for value in opportunities)
        if len(set(meal_slots)) != len(meal_slots):
            raise ValueError("a date override cannot schedule the same product meal slot twice")
        normalized[service_date] = opportunities
    return MappingProxyType(normalized)


def _require_service_date(value: object) -> None:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise TypeError("service_date must be a date")


def _local_datetime(instant: object) -> datetime:
    if not isinstance(instant, datetime) or instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("instant must be timezone-aware")
    return instant.astimezone(NUTRITION_APPLICATION_TIMEZONE)


def _default_known_delivery_failure(exc: BaseException) -> bool:
    """Treat injected/test sender errors as known failures by default.

    The production composition supplies a narrower predicate for its concrete
    HTTP transport, where a lost connection has an objectively uncertain remote
    outcome.
    """

    return True


DEFAULT_RECOMMENDATION_DISPATCH_SCHEDULE = RecommendationDispatchSchedule()
