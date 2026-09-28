"""Deterministic SQLite persistence for meal plans, report drafts, and intake.

This is intentionally downstream of reconciliation: it does not resolve food,
interpret portions, invoke AI, refresh FD data, calculate meal choices, or
send messages.  It persists immutable catalog references and reconstructs the
existing nutrition ledger from those exact snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import sqlite3
from typing import Literal
from uuid import uuid4

from .presentation_binding import PresentationBinding
from .application_clock import NutritionApplicationClock
from .fdmealplanner.catalog import (
    FDMenuOccurrence,
    NutritionCatalogError,
    OfficialNutritionCatalog,
)
from .food_resolution import ResolvedFood
from .meal_identity import (
    MealSlot,
    legacy_meal_slot,
    meal_context_key,
    meal_slot_for_provider_meal,
    meal_values_equal,
    require_meal_slot,
)
from .meal_request import RequestedMealFood
from .meal_report import (
    ClarificationItem,
    MealPlan,
    PlannedMealItem,
    ProposedEatenMealItem,
    ProposedUnplannedMealItem,
    ReconciledMealReport,
    SkippedMealItem,
)
from .nutrition import DailyBalance, DailyLedger, DailyTargets, IntakeEntry, NutritionRecord, NutrientProfile, calculate_daily_balance
from .nutrition.arithmetic import add_nutrients
from .nutrition.models import SourceIdentifier
from .phelps_service_calendar import PhelpsServiceCalendar


__all__ = [
    "AcceptedIntakeEntry",
    "ActiveMealPlanInvariantError",
    "AppliedMealReport",
    "DraftClarification",
    "DraftPlannedItem",
    "DraftUnplannedItem",
    "DurableMealState",
    "MealReportDraft",
    "MealReportDraftConflictError",
    "MealReportCommitTimeRejection",
    "MealReportMessageEvent",
    "MealReportApplicationError",
    "MealRecommendationReplacementConflictError",
    "MealRecommendationReplacementDispatch",
    "MealRecommendationReplacementDispatchStatus",
    "MealRecommendationReplacementDraftConflictError",
    "MealPlanStatus",
    "ImmediateMealRequestDispatch",
    "ImmediateMealRequestDispatchStatus",
    "PendingMealRequest",
    "PendingMealRequestStatus",
    "PersistedMealPlan",
    "ScheduledRecommendationAlreadyExistsError",
    "ScheduledRecommendationDispatch",
    "ScheduledRecommendationDispatchStatus",
    "ScheduledRecommendationPlanConflictError",
    "StaleMealPlanRetirement",
]


IntakeOrigin = Literal[
    "planned_quantity",
    "explicit_deterministic",
    "explicit_semantic_estimate",
    "manual_other",
]

MealPlanStatus = Literal["active", "applied", "superseded"]
ScheduledRecommendationDispatchStatus = Literal[
    "pending_delivery",
    "sending",
    "delivered",
    "expired",
]
MealRecommendationReplacementDispatchStatus = Literal[
    "pending_delivery",
    "sending",
    "delivered",
]
ImmediateMealRequestDispatchStatus = Literal[
    "pending_delivery",
    "sending",
    "delivered",
]
PendingMealRequestStatus = Literal["pending", "consumed", "superseded"]
MealReportDraftStatus = Literal[
    "draft",
    "awaiting_clarification",
    "completed",
    "cancelled",
]
MealReportDraftItemAction = Literal["eaten", "skipped"]
DraftIdentityStatus = Literal["resolved", "unresolved"]
DraftQuantityStatus = Literal["resolved", "unresolved", "not_applicable"]
MealReportIntent = Literal[
    "meal_report",
    "clarification_answer",
    "location_question",
    "replacement_request",
    "meal_request",
    "unsupported_or_ambiguous",
]
MealReportOutcomeType = Literal[
    "routed_interaction",
    "draft_update",
    "draft_cancelled",
    "final_application",
    "ambiguous_meal_context",
    "unavailable_meal_slot",
    "no_reportable_context",
    "routing_failure",
    "pending_meal_request",
]


class MealReportApplicationError(RuntimeError):
    """Raised when a reconciliation cannot be safely or atomically persisted."""


class ActiveMealPlanInvariantError(MealReportApplicationError):
    """Raised when storage contains more than one answerable plan for a context."""


class ScheduledRecommendationAlreadyExistsError(MealReportApplicationError):
    """Raised when a scheduler race finds a durable dispatch already created."""


class ScheduledRecommendationPlanConflictError(MealReportApplicationError):
    """Raised when a non-scheduled active plan owns a scheduled context."""


class MealReportDraftConflictError(MealReportApplicationError):
    """Raised when another message changed a durable draft before this update."""


class MealReportCommitTimeRejection(MealReportApplicationError):
    """Signals a durably recorded final-commit reportability rejection."""

    def __init__(self, event: "MealReportMessageEvent") -> None:
        if not isinstance(event, MealReportMessageEvent):
            raise TypeError("event must be a MealReportMessageEvent")
        self.event = event
        super().__init__(event.reply_text)


class MealRecommendationReplacementConflictError(MealReportApplicationError):
    """Raised when a replacement no longer owns its exact active plan context."""


class MealRecommendationReplacementDraftConflictError(
    MealRecommendationReplacementConflictError
):
    """Raised when an unresolved intake draft blocks a pre-meal replacement."""


@dataclass(frozen=True, slots=True)
class PersistedMealPlan:
    """A durable plan identifier paired with its reconstructed immutable plan."""

    plan_id: str
    plan: MealPlan
    meal_slot: MealSlot
    created_at: datetime
    status: MealPlanStatus

    def __post_init__(self) -> None:
        _text(self.plan_id, "plan_id")
        if not isinstance(self.plan, MealPlan):
            raise TypeError("plan must be a MealPlan")
        require_meal_slot(self.meal_slot)
        _timestamp(self.created_at, "created_at")
        if self.status not in {"active", "applied", "superseded"}:
            raise ValueError("status is invalid")


@dataclass(frozen=True, slots=True)
class ScheduledRecommendationDispatch:
    """One scheduler-owned persisted plan and its outbound delivery state.

    No row for a service-date/context means the scheduler has not prepared an
    opportunity.  The immutable ``delivery_token`` is retained across a known
    delivery retry and is supplied to the outbound client as its message token.
    ``sending`` deliberately represents an uncertain external hand-off; it is
    never automatically resent after a process interruption.
    """

    service_date: date
    meal_context: str
    meal_slot: MealSlot
    persisted_plan: PersistedMealPlan
    delivery_token: str
    status: ScheduledRecommendationDispatchStatus
    created_at: datetime
    delivery_started_at: datetime | None = None
    delivered_at: datetime | None = None
    expired_at: datetime | None = None

    def __post_init__(self) -> None:
        _date(self.service_date, "service_date")
        _text(self.meal_context, "meal_context")
        require_meal_slot(self.meal_slot)
        if not isinstance(self.persisted_plan, PersistedMealPlan):
            raise TypeError("persisted_plan must be a PersistedMealPlan")
        if self.persisted_plan.plan.service_date != self.service_date:
            raise ValueError("scheduled dispatch service date must match its plan")
        if meal_context_key(self.persisted_plan.plan.meal) != self.meal_context:
            raise ValueError("scheduled dispatch meal context must match its plan")
        if self.persisted_plan.meal_slot != self.meal_slot:
            raise ValueError("scheduled dispatch meal slot must match its plan")
        _text(self.delivery_token, "delivery_token")
        if self.status not in {"pending_delivery", "sending", "delivered", "expired"}:
            raise ValueError("scheduled dispatch status is invalid")
        _timestamp(self.created_at, "created_at")
        for value, name in (
            (self.delivery_started_at, "delivery_started_at"),
            (self.delivered_at, "delivered_at"),
            (self.expired_at, "expired_at"),
        ):
            if value is not None:
                _timestamp(value, name)

    @property
    def plan_id(self) -> str:
        """Return the exact scheduler-owned plan identifier."""

        return self.persisted_plan.plan_id


@dataclass(frozen=True, slots=True)
class MealRecommendationReplacementDispatch:
    """One user-requested replacement plan and durable outbound hand-off state.

    The row preserves a precise plan lineage.  It never overwrites the
    scheduler's original dispatch row, so periodic scheduling remains terminal
    for that service opportunity while the replacement is the active plan that
    the user can later report.
    """

    source_event_id: str
    chat_guid: str
    prior_plan: PersistedMealPlan
    persisted_plan: PersistedMealPlan
    rejected_plan_item_ids: tuple[str, ...]
    requested_foods: tuple[RequestedMealFood, ...]
    request_kind: Literal["rejection", "meal_request"]
    whole_meal: bool
    delivery_token: str
    status: MealRecommendationReplacementDispatchStatus
    created_at: datetime
    scheduled_origin_plan_id: str | None = None
    delivery_started_at: datetime | None = None
    delivered_at: datetime | None = None

    def __post_init__(self) -> None:
        _text(self.source_event_id, "source_event_id")
        _text(self.chat_guid, "chat_guid")
        if not isinstance(self.prior_plan, PersistedMealPlan):
            raise TypeError("prior_plan must be a PersistedMealPlan")
        if not isinstance(self.persisted_plan, PersistedMealPlan):
            raise TypeError("persisted_plan must be a PersistedMealPlan")
        if self.prior_plan.plan_id == self.persisted_plan.plan_id:
            raise ValueError("replacement plan must differ from prior plan")
        if (
            self.prior_plan.plan.service_date != self.persisted_plan.plan.service_date
            or not meal_values_equal(self.prior_plan.plan.meal, self.persisted_plan.plan.meal)
        ):
            raise ValueError("replacement plan must retain the prior service context")
        if self.request_kind not in {"rejection", "meal_request"}:
            raise ValueError("replacement request kind is invalid")
        if not isinstance(self.rejected_plan_item_ids, tuple):
            raise TypeError("replacement rejected plan item IDs must be a tuple")
        if len(set(self.rejected_plan_item_ids)) != len(self.rejected_plan_item_ids):
            raise ValueError("replacement rejected plan item IDs must not repeat")
        for item_id in self.rejected_plan_item_ids:
            _text(item_id, "rejected plan item ID")
            if self.prior_plan.plan.item_for_id(item_id) is None:
                raise ValueError("replacement rejected plan item is not in prior plan")
        if not isinstance(self.requested_foods, tuple) or not all(
            isinstance(food, RequestedMealFood) for food in self.requested_foods
        ):
            raise TypeError("requested_foods must contain RequestedMealFood values")
        if len({food.identity_with_signature for food in self.requested_foods}) != len(
            self.requested_foods
        ):
            raise ValueError("replacement requested foods must not repeat")
        if self.request_kind == "rejection":
            if not self.rejected_plan_item_ids:
                raise ValueError("rejection replacement must retain rejected plan item IDs")
            if self.requested_foods:
                raise ValueError("rejection replacement cannot retain requested foods")
        elif self.rejected_plan_item_ids:
            raise ValueError("positive meal request cannot retain rejected plan item IDs")
        if not isinstance(self.whole_meal, bool):
            raise TypeError("whole_meal must be a bool")
        if self.request_kind == "meal_request" and not self.whole_meal and not self.requested_foods:
            raise ValueError("targeted positive meal request needs requested foods")
        _text(self.delivery_token, "delivery_token")
        if self.status not in {"pending_delivery", "sending", "delivered"}:
            raise ValueError("replacement dispatch status is invalid")
        _timestamp(self.created_at, "created_at")
        if self.scheduled_origin_plan_id is not None:
            _text(self.scheduled_origin_plan_id, "scheduled_origin_plan_id")
        for value, name in (
            (self.delivery_started_at, "delivery_started_at"),
            (self.delivered_at, "delivered_at"),
        ):
            if value is not None:
                _timestamp(value, name)

    @property
    def plan_id(self) -> str:
        """Return the exact replacement plan ID."""

        return self.persisted_plan.plan_id


@dataclass(frozen=True, slots=True)
class ImmediateMealRequestDispatch:
    """One no-prior-plan meal request and its durable outbound hand-off."""

    source_event_id: str
    chat_guid: str
    persisted_plan: PersistedMealPlan
    requested_foods: tuple[RequestedMealFood, ...]
    whole_meal: bool
    delivery_token: str
    status: ImmediateMealRequestDispatchStatus
    created_at: datetime
    delivery_started_at: datetime | None = None
    delivered_at: datetime | None = None

    def __post_init__(self) -> None:
        _text(self.source_event_id, "source_event_id")
        _text(self.chat_guid, "chat_guid")
        if not isinstance(self.persisted_plan, PersistedMealPlan):
            raise TypeError("persisted_plan must be a PersistedMealPlan")
        if not isinstance(self.requested_foods, tuple) or not all(
            isinstance(food, RequestedMealFood) for food in self.requested_foods
        ):
            raise TypeError("requested_foods must contain RequestedMealFood values")
        if len({food.identity_with_signature for food in self.requested_foods}) != len(
            self.requested_foods
        ):
            raise ValueError("requested_foods must not repeat")
        if not isinstance(self.whole_meal, bool):
            raise TypeError("whole_meal must be a bool")
        if not self.whole_meal and not self.requested_foods:
            raise ValueError("targeted immediate request needs requested foods")
        _text(self.delivery_token, "delivery_token")
        if self.status not in {"pending_delivery", "sending", "delivered"}:
            raise ValueError("immediate meal request dispatch status is invalid")
        _timestamp(self.created_at, "created_at")
        for value, name in (
            (self.delivery_started_at, "delivery_started_at"),
            (self.delivered_at, "delivered_at"),
        ):
            if value is not None:
                _timestamp(value, name)

    @property
    def plan_id(self) -> str:
        return self.persisted_plan.plan_id


@dataclass(frozen=True, slots=True)
class PendingMealRequest:
    """One scoped request retained only until its scheduled plan is created."""

    source_event_id: str
    chat_guid: str
    service_date: date
    meal: str | int
    meal_context: str
    meal_slot: MealSlot
    requested_foods: tuple[RequestedMealFood, ...]
    whole_meal: bool
    status: PendingMealRequestStatus
    reply_text: str
    created_at: datetime
    consumed_at: datetime | None = None
    consumed_plan_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.source_event_id, "source_event_id")
        _text(self.chat_guid, "chat_guid")
        _date(self.service_date, "service_date")
        if isinstance(self.meal, bool) or not isinstance(self.meal, (str, int)):
            raise TypeError("meal must be text or an integer")
        if isinstance(self.meal, str):
            _text(self.meal, "meal")
        _text(self.meal_context, "meal_context")
        require_meal_slot(self.meal_slot)
        if meal_context_key(self.meal) != self.meal_context:
            raise ValueError("pending request meal context must match its meal")
        if not isinstance(self.requested_foods, tuple) or not all(
            isinstance(food, RequestedMealFood) for food in self.requested_foods
        ):
            raise TypeError("requested_foods must contain RequestedMealFood values")
        if len({food.identity_with_signature for food in self.requested_foods}) != len(
            self.requested_foods
        ):
            raise ValueError("pending requested foods must not repeat")
        if not isinstance(self.whole_meal, bool):
            raise TypeError("whole_meal must be a bool")
        if not self.whole_meal and not self.requested_foods:
            raise ValueError("targeted pending request needs requested foods")
        if self.status not in {"pending", "consumed", "superseded"}:
            raise ValueError("pending meal request status is invalid")
        _text(self.reply_text, "reply_text")
        _timestamp(self.created_at, "created_at")
        if self.consumed_at is not None:
            _timestamp(self.consumed_at, "consumed_at")
        if self.consumed_plan_id is not None:
            _text(self.consumed_plan_id, "consumed_plan_id")
        if self.status == "consumed":
            if self.consumed_at is None or self.consumed_plan_id is None:
                raise ValueError("consumed pending request needs plan and timestamp")
        elif self.consumed_at is not None or self.consumed_plan_id is not None:
            raise ValueError("only consumed pending request may retain a plan")


@dataclass(frozen=True, slots=True)
class DraftClarification:
    """One minimal, structured unresolved report fact.

    This intentionally records only a deterministic reason and an optional
    authoritative plan-item or menu phrase reference.  It is not a transcript
    or model chain-of-thought.
    """

    reason: str
    plan_item_id: str | None = None
    food_text: str | None = None
    quantity_text: str | None = None

    def __post_init__(self) -> None:
        _text(self.reason, "reason")
        if self.plan_item_id is not None:
            _text(self.plan_item_id, "plan_item_id")
        if self.food_text is not None:
            _text(self.food_text, "food_text")
        if self.quantity_text is not None:
            _text(self.quantity_text, "quantity_text")


@dataclass(frozen=True, slots=True)
class DraftPlannedItem:
    """One effective planned-food fact with an independent quantity axis."""

    plan_item_id: str
    action: MealReportDraftItemAction
    official_servings: Decimal | None
    quantity_source: IntakeOrigin | None
    original_reference_text: str
    original_quantity_text: str | None = None
    quantity_status: DraftQuantityStatus | None = None
    unresolved_reason: str | None = None
    last_source_event_id: str | None = None
    fact_revision: int = 0

    def __post_init__(self) -> None:
        _text(self.plan_item_id, "plan_item_id")
        if self.action not in {"eaten", "skipped"}:
            raise ValueError("draft planned item action is invalid")
        _text(self.original_reference_text, "original_reference_text")
        if self.original_quantity_text is not None:
            _text(self.original_quantity_text, "original_quantity_text")
        status: DraftQuantityStatus = self.quantity_status or (
            "not_applicable" if self.action == "skipped" else "resolved"
        )
        object.__setattr__(self, "quantity_status", status)
        if self.unresolved_reason is not None:
            _text(self.unresolved_reason, "unresolved_reason")
        if self.last_source_event_id is not None:
            _text(self.last_source_event_id, "last_source_event_id")
        if (
            isinstance(self.fact_revision, bool)
            or not isinstance(self.fact_revision, int)
            or self.fact_revision < 0
        ):
            raise ValueError("draft fact revision is invalid")
        if self.action == "skipped":
            if (
                status != "not_applicable"
                or self.official_servings is not None
                or self.quantity_source is not None
                or self.unresolved_reason is not None
            ):
                raise ValueError("skipped draft items cannot have a quantity")
            return
        if status == "unresolved":
            if (
                self.official_servings is not None
                or self.quantity_source is not None
                or self.unresolved_reason is None
            ):
                raise ValueError("unresolved draft quantity has invalid authoritative state")
            return
        if status != "resolved" or self.unresolved_reason is not None:
            raise ValueError("eaten draft item has an invalid quantity status")
        _positive_decimal(self.official_servings, "official_servings")
        if self.quantity_source not in {
            "planned_quantity",
            "explicit_deterministic",
            "explicit_semantic_estimate",
        }:
            raise ValueError("draft eaten item has an invalid quantity source")


@dataclass(frozen=True, slots=True)
class DraftUnplannedItem:
    """One effective additional-food fact with independent identity and quantity."""

    food_text: str
    quantity_text: str | None
    food: ResolvedFood | None
    official_servings: Decimal | None
    quantity_source: Literal["explicit_deterministic", "explicit_semantic_estimate"] | None
    confidence: Literal["high", "medium", "low"] | None = None
    fact_id: str = field(default_factory=lambda: str(uuid4()))
    identity_status: DraftIdentityStatus | None = None
    identity_unresolved_reason: str | None = None
    quantity_status: Literal["resolved", "unresolved"] | None = None
    quantity_unresolved_reason: str | None = None
    last_source_event_id: str | None = None
    fact_revision: int = 0

    def __post_init__(self) -> None:
        _text(self.food_text, "food_text")
        if self.quantity_text is not None:
            _text(self.quantity_text, "quantity_text")
        _text(self.fact_id, "fact_id")
        identity_status: DraftIdentityStatus = self.identity_status or (
            "resolved" if self.food is not None else "unresolved"
        )
        quantity_status: Literal["resolved", "unresolved"] = self.quantity_status or (
            "resolved" if self.official_servings is not None else "unresolved"
        )
        object.__setattr__(self, "identity_status", identity_status)
        object.__setattr__(self, "quantity_status", quantity_status)
        if self.identity_unresolved_reason is not None:
            _text(self.identity_unresolved_reason, "identity_unresolved_reason")
        if self.quantity_unresolved_reason is not None:
            _text(self.quantity_unresolved_reason, "quantity_unresolved_reason")
        if self.last_source_event_id is not None:
            _text(self.last_source_event_id, "last_source_event_id")
        if (
            isinstance(self.fact_revision, bool)
            or not isinstance(self.fact_revision, int)
            or self.fact_revision < 0
        ):
            raise ValueError("draft fact revision is invalid")
        if identity_status == "resolved":
            if not isinstance(self.food, ResolvedFood) or self.identity_unresolved_reason is not None:
                raise ValueError("resolved unplanned identity is invalid")
        elif self.food is not None or self.identity_unresolved_reason is None:
            raise ValueError("unresolved unplanned identity is invalid")
        if quantity_status == "resolved":
            _positive_decimal(self.official_servings, "official_servings")
            if (
                self.quantity_source not in {
                    "explicit_deterministic",
                    "explicit_semantic_estimate",
                }
                or self.quantity_unresolved_reason is not None
            ):
                raise ValueError("resolved unplanned quantity is invalid")
        elif (
            self.official_servings is not None
            or self.quantity_source is not None
            or self.quantity_unresolved_reason is None
        ):
            raise ValueError("unresolved unplanned quantity is invalid")
        if self.confidence is not None and self.confidence not in {"high", "medium", "low"}:
            raise ValueError("draft unplanned item confidence is invalid")
        if quantity_status == "unresolved" and self.confidence is not None:
            raise ValueError("unresolved unplanned quantity cannot have confidence")


@dataclass(frozen=True, slots=True)
class MealReportDraft:
    """Durable, plan-scoped facts accumulated across report messages."""

    draft_id: str
    chat_guid: str
    persisted_plan: PersistedMealPlan
    status: MealReportDraftStatus
    revision: int
    planned_items: tuple[DraftPlannedItem, ...]
    unplanned_items: tuple[DraftUnplannedItem, ...]
    clarifications: tuple[DraftClarification, ...]
    last_prompt: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None

    def __post_init__(self) -> None:
        _text(self.draft_id, "draft_id")
        _text(self.chat_guid, "chat_guid")
        if not isinstance(self.persisted_plan, PersistedMealPlan):
            raise TypeError("persisted_plan must be a PersistedMealPlan")
        if self.status not in {"draft", "awaiting_clarification", "completed", "cancelled"}:
            raise ValueError("draft status is invalid")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise ValueError("draft revision is invalid")
        if not isinstance(self.planned_items, tuple) or not all(
            isinstance(item, DraftPlannedItem) for item in self.planned_items
        ):
            raise TypeError("planned_items must be DraftPlannedItem values")
        if not isinstance(self.unplanned_items, tuple) or not all(
            isinstance(item, DraftUnplannedItem) for item in self.unplanned_items
        ):
            raise TypeError("unplanned_items must be DraftUnplannedItem values")
        if not isinstance(self.clarifications, tuple) or not all(
            isinstance(item, DraftClarification) for item in self.clarifications
        ):
            raise TypeError("clarifications must be DraftClarification values")
        if len({item.plan_item_id for item in self.planned_items}) != len(self.planned_items):
            raise ValueError("draft planned items must not repeat a plan item")
        if self.last_prompt is not None:
            _text(self.last_prompt, "last_prompt")
        _timestamp(self.created_at, "created_at")
        _timestamp(self.updated_at, "updated_at")
        if self.completed_at is not None:
            _timestamp(self.completed_at, "completed_at")
        if self.cancelled_at is not None:
            _timestamp(self.cancelled_at, "cancelled_at")

    @property
    def plan_id(self) -> str:
        return self.persisted_plan.plan_id


@dataclass(frozen=True, slots=True)
class MealReportMessageEvent:
    """One idempotently incorporated inbound GUID and its deterministic reply."""

    source_event_id: str
    chat_guid: str
    outcome_type: MealReportOutcomeType
    plan_id: str | None
    draft_id: str | None
    application_source_event_id: str | None
    intent: MealReportIntent
    reply_text: str
    processed_at: datetime

    def __post_init__(self) -> None:
        _text(self.source_event_id, "source_event_id")
        _text(self.chat_guid, "chat_guid")
        if self.outcome_type not in {
            "routed_interaction",
            "draft_update",
            "draft_cancelled",
            "final_application",
            "ambiguous_meal_context",
            "unavailable_meal_slot",
            "no_reportable_context",
            "routing_failure",
            "pending_meal_request",
        }:
            raise ValueError("message event outcome type is invalid")
        if self.plan_id is not None:
            _text(self.plan_id, "plan_id")
        if self.draft_id is not None:
            _text(self.draft_id, "draft_id")
        if self.application_source_event_id is not None:
            _text(self.application_source_event_id, "application_source_event_id")
        if self.intent not in {
            "meal_report",
            "clarification_answer",
            "location_question",
            "replacement_request",
            "meal_request",
            "unsupported_or_ambiguous",
        }:
            raise ValueError("message event intent is invalid")
        _text(self.reply_text, "reply_text")
        _timestamp(self.processed_at, "processed_at")
        pre_route = self.outcome_type in {
            "ambiguous_meal_context",
            "unavailable_meal_slot",
            "no_reportable_context",
            "routing_failure",
            "pending_meal_request",
        }
        if pre_route:
            if (
                self.plan_id is not None
                or self.draft_id is not None
                or self.application_source_event_id is not None
            ):
                raise ValueError("pre-route message event cannot retain routed identity")
        elif self.plan_id is None:
            raise ValueError("routed message event needs a plan")
        if self.outcome_type == "final_application":
            if self.application_source_event_id != self.source_event_id:
                raise ValueError("final application event needs its application identity")
        elif self.application_source_event_id is not None:
            raise ValueError("only final application events retain an application identity")


@dataclass(frozen=True, slots=True)
class StaleMealPlanRetirement:
    """The deterministic outcome of one atomic stale-plan maintenance pass."""

    evaluated_at: datetime
    retired_plan_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _timestamp(self.evaluated_at, "evaluated_at")
        if not isinstance(self.retired_plan_ids, tuple) or not all(
            isinstance(plan_id, str) and plan_id.strip() for plan_id in self.retired_plan_ids
        ):
            raise TypeError("retired_plan_ids must be non-empty text values")


@dataclass(frozen=True, slots=True)
class AcceptedIntakeEntry:
    """One persisted exact snapshot reference and positive consumed serving count."""

    intake_id: str
    source_event_id: str
    service_date: date
    recorded_at: datetime
    meal: str | None
    plan_id: str | None
    plan_item_id: str | None
    occurrence: FDMenuOccurrence
    record: NutritionRecord
    official_servings: Decimal
    quantity_source: IntakeOrigin
    original_reference_text: str | None
    original_quantity_text: str | None

    def __post_init__(self) -> None:
        _text(self.intake_id, "intake_id")
        _text(self.source_event_id, "source_event_id")
        if not isinstance(self.service_date, date) or isinstance(self.service_date, datetime):
            raise TypeError("service_date must be a date")
        _timestamp(self.recorded_at, "recorded_at")
        if self.meal is not None:
            _text(self.meal, "meal")
        if self.plan_id is not None:
            _text(self.plan_id, "plan_id")
        if self.plan_item_id is not None:
            _text(self.plan_item_id, "plan_item_id")
        if not isinstance(self.occurrence, FDMenuOccurrence):
            raise TypeError("occurrence must be an FDMenuOccurrence")
        if not isinstance(self.record, NutritionRecord):
            raise TypeError("record must be a NutritionRecord")
        if self.occurrence.nutrition_record != self.record:
            raise ValueError("intake record must match its exact occurrence snapshot")
        _positive_decimal(self.official_servings, "official_servings")
        if self.quantity_source not in {
            "planned_quantity",
            "explicit_deterministic",
            "explicit_semantic_estimate",
            "manual_other",
        }:
            raise ValueError("quantity_source is invalid")
        if self.original_reference_text is not None:
            _text(self.original_reference_text, "original_reference_text")
        if self.original_quantity_text is not None:
            _text(self.original_quantity_text, "original_quantity_text")

    def as_ledger_entry(self) -> IntakeEntry:
        """Adapt this exact stored snapshot to existing deterministic arithmetic."""

        return IntakeEntry(self.record, self.official_servings)


@dataclass(frozen=True, slots=True)
class AppliedMealReport:
    """Deterministic result for future confirmation rendering, never prose."""

    plan_id: str
    source_event_id: str
    already_applied: bool
    accepted_intake_entries: tuple[AcceptedIntakeEntry, ...]
    skipped_items: tuple[SkippedMealItem, ...]

    def __post_init__(self) -> None:
        _text(self.plan_id, "plan_id")
        _text(self.source_event_id, "source_event_id")
        if not isinstance(self.already_applied, bool):
            raise TypeError("already_applied must be a bool")
        if not isinstance(self.accepted_intake_entries, tuple) or not all(
            isinstance(entry, AcceptedIntakeEntry) for entry in self.accepted_intake_entries
        ):
            raise TypeError("accepted_intake_entries must be AcceptedIntakeEntry values")
        if not isinstance(self.skipped_items, tuple) or not all(
            isinstance(item, SkippedMealItem) for item in self.skipped_items
        ):
            raise TypeError("skipped_items must be SkippedMealItem values")


class DurableMealState:
    """Own deterministic application state in the catalog's migrated SQLite file."""

    def __init__(self, catalog: OfficialNutritionCatalog) -> None:
        if not isinstance(catalog, OfficialNutritionCatalog):
            raise TypeError("catalog must be an OfficialNutritionCatalog")
        self._catalog = catalog

    @staticmethod
    def _save_presentation_binding(
        connection: sqlite3.Connection,
        plan_id: str,
        item_id: str,
        item: PlannedMealItem,
        timestamp: datetime,
    ) -> None:
        binding = item.presentation_binding
        if binding is None:
            return  # Legacy/manual plans have no inferred presentation authority.
        binding.validate(item.food, item.recommended_official_servings, item.natural_quantity_text)
        connection.execute(
            "INSERT INTO meal_plan_item_presentation_bindings "
            "(plan_id, plan_item_id, binding_json, created_at) VALUES (?, ?, ?, ?)",
            (plan_id, item_id, binding.to_json(), _timestamp_text(timestamp)),
        )

    def _load_presentation_binding(self, plan_id: str, item_id: str) -> PresentationBinding | None:
        row = self._connection().execute(
            "SELECT binding_json FROM meal_plan_item_presentation_bindings WHERE plan_id = ? AND plan_item_id = ?",
            (plan_id, item_id),
        ).fetchone()
        return None if row is None else PresentationBinding.from_json(row["binding_json"])

    def save_meal_plan(
        self,
        plan: MealPlan,
        *,
        meal_slot: MealSlot | str | None = None,
        plan_id: str | None = None,
        created_at: datetime | None = None,
    ) -> PersistedMealPlan:
        """Persist one plan and atomically supersede its prior active context.

        Rendering and immutable food-link validation happen before the write.
        The old active plan is superseded inside the same SQLite transaction as
        the new plan and its items, so an insertion failure rolls that state
        change back rather than leaving a service context without an answerable
        recommendation.
        """

        if not isinstance(plan, MealPlan):
            raise TypeError("plan must be a MealPlan")
        identifier = plan_id or str(uuid4())
        _text(identifier, "plan_id")
        timestamp = _utc_now() if created_at is None else _timestamp(created_at, "created_at")
        try:
            context = meal_context_key(plan.meal)
            slot = _resolved_meal_slot(plan.meal, meal_slot)
            for item in plan.items:
                self._validate_plan_item_link(item)
        except (TypeError, ValueError, NutritionCatalogError, MealReportApplicationError) as exc:
            raise MealReportApplicationError("unable to persist meal plan") from exc
        connection = self._connection()
        try:
            with connection:
                active_rows = connection.execute(
                    """
                    SELECT plan_id FROM meal_plans
                    WHERE service_date = ? AND meal_slot = ? AND status = 'active'
                    """,
                    (plan.service_date.isoformat(), slot),
                ).fetchall()
                if len(active_rows) > 1:
                    raise ActiveMealPlanInvariantError(
                        "multiple active meal plans exist for one product meal slot"
                    )
                connection.execute(
                    """
                    UPDATE meal_plans
                    SET status = 'superseded'
                    WHERE service_date = ? AND meal_slot = ? AND status = 'active'
                    """,
                    (plan.service_date.isoformat(), slot),
                )
                connection.execute(
                    """
                    INSERT INTO meal_plans
                    (plan_id, service_date, meal, meal_kind, meal_context, meal_slot, created_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
                    """,
                    (
                        identifier,
                        plan.service_date.isoformat(),
                        str(plan.meal),
                        "integer" if isinstance(plan.meal, int) else "text",
                        context,
                        slot,
                        _timestamp_text(timestamp),
                    ),
                )
                for position, item in enumerate(plan.items):
                    food = item.food
                    connection.execute(
                        """
                        INSERT INTO meal_plan_items
                        (plan_id, plan_item_id, item_position, occurrence_id,
                         nutrition_snapshot_id, source_kind, source_value,
                         content_signature, recommended_official_servings,
                         natural_quantity_text, display_food_name)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            identifier,
                            plan.item_id(item),
                            position,
                            food.occurrence.occurrence_id,
                            food.nutrition_snapshot_id,
                            food.source_identifier.kind,
                            food.source_identifier.value,
                            food.content_signature,
                            str(item.recommended_official_servings),
                            item.natural_quantity_text,
                            item.display_food_name,
                        ),
                    )
                    self._save_presentation_binding(connection, identifier, plan.item_id(item), item, timestamp)
        except ActiveMealPlanInvariantError:
            raise
        except (sqlite3.Error, NutritionCatalogError) as exc:
            raise MealReportApplicationError("unable to persist meal plan") from exc
        return PersistedMealPlan(identifier, plan, slot, timestamp, "active")

    def save_scheduled_meal_plan(
        self,
        plan: MealPlan,
        *,
        meal_slot: MealSlot | str | None = None,
        plan_id: str | None = None,
        delivery_token: str | None = None,
        created_at: datetime | None = None,
        pending_request: PendingMealRequest | None = None,
    ) -> PersistedMealPlan:
        """Persist one scheduler-owned plan and pending delivery atomically.

        Unlike :meth:`save_meal_plan`, this method never supersedes an active
        plan.  An existing dispatch wins a scheduler race, and an active plan
        without a dispatch is treated as an explicit/manual-plan conflict.
        The new plan, its immutable items, and its pending dispatch row commit
        together, so a failed optimizer/rendering/persistence path cannot
        leave a timer retry with a partially owned recommendation.
        """

        if not isinstance(plan, MealPlan):
            raise TypeError("plan must be a MealPlan")
        if pending_request is not None and not isinstance(pending_request, PendingMealRequest):
            raise TypeError("pending_request must be a PendingMealRequest or None")
        identifier = plan_id or str(uuid4())
        token = delivery_token or str(uuid4())
        _text(identifier, "plan_id")
        _text(token, "delivery_token")
        timestamp = _utc_now() if created_at is None else _timestamp(created_at, "created_at")
        try:
            context = meal_context_key(plan.meal)
            slot = (
                legacy_meal_slot(plan.service_date, plan.meal)
                if meal_slot is None
                else require_meal_slot(meal_slot)
            )
            for item in plan.items:
                self._validate_plan_item_link(item)
            if pending_request is not None and (
                pending_request.status != "pending"
                or pending_request.service_date != plan.service_date
                or pending_request.meal_context != context
                or pending_request.meal_slot != slot
            ):
                raise ValueError("pending request does not match scheduled plan context")
            if pending_request is not None and not {
                food.identity_with_signature for food in pending_request.requested_foods
            } <= {
                (
                    item.food.source_identifier.kind,
                    item.food.source_identifier.value,
                    item.food.content_signature,
                )
                for item in plan.items
            }:
                raise ValueError("scheduled plan does not retain every requested food")
        except (TypeError, ValueError, NutritionCatalogError, MealReportApplicationError) as exc:
            raise MealReportApplicationError("unable to persist scheduled meal plan") from exc

        connection = self._connection()
        try:
            # Reserve the database write lock before examining either an
            # existing dispatch or active plan.  Two timer invocations can
            # both optimize, but only one can durably create this context;
            # the losing transaction writes neither a replacement plan nor a
            # second dispatch.
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT plan_id FROM scheduled_recommendation_dispatches
                WHERE service_date = ? AND meal_slot = ?
                """,
                (plan.service_date.isoformat(), slot),
            ).fetchone()
            if existing is not None:
                raise ScheduledRecommendationAlreadyExistsError(
                    "scheduled recommendation dispatch already exists"
                )
            active_rows = connection.execute(
                """
                SELECT plan_id FROM meal_plans
                WHERE service_date = ? AND meal_slot = ? AND status = 'active'
                """,
                (plan.service_date.isoformat(), slot),
            ).fetchall()
            if len(active_rows) > 1:
                raise ActiveMealPlanInvariantError(
                    "multiple active meal plans exist for one product meal slot"
                )
            if active_rows:
                raise ScheduledRecommendationPlanConflictError(
                    "an active non-scheduled meal plan already owns this service context"
                )
            connection.execute(
                """
                INSERT INTO meal_plans
                (plan_id, service_date, meal, meal_kind, meal_context, meal_slot, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                (
                    identifier,
                    plan.service_date.isoformat(),
                    str(plan.meal),
                    "integer" if isinstance(plan.meal, int) else "text",
                    context,
                    slot,
                    _timestamp_text(timestamp),
                ),
            )
            for position, item in enumerate(plan.items):
                food = item.food
                connection.execute(
                    """
                    INSERT INTO meal_plan_items
                    (plan_id, plan_item_id, item_position, occurrence_id,
                     nutrition_snapshot_id, source_kind, source_value,
                     content_signature, recommended_official_servings,
                     natural_quantity_text, display_food_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        plan.item_id(item),
                        position,
                        food.occurrence.occurrence_id,
                        food.nutrition_snapshot_id,
                        food.source_identifier.kind,
                        food.source_identifier.value,
                        food.content_signature,
                        str(item.recommended_official_servings),
                        item.natural_quantity_text,
                        item.display_food_name,
                    ),
                )
                self._save_presentation_binding(connection, identifier, plan.item_id(item), item, timestamp)
            connection.execute(
                """
                INSERT INTO scheduled_recommendation_dispatches
                (service_date, meal_context, meal_slot, plan_id, delivery_token, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'pending_delivery', ?)
                """,
                (
                    plan.service_date.isoformat(),
                    context,
                    slot,
                    identifier,
                    token,
                    _timestamp_text(timestamp),
                ),
            )
            if pending_request is not None:
                consumed = connection.execute(
                    """
                    UPDATE pending_meal_requests
                    SET status = 'consumed', consumed_at = ?, consumed_plan_id = ?
                    WHERE source_event_id = ? AND chat_guid = ?
                      AND service_date = ? AND meal_slot = ? AND status = 'pending'
                    """,
                    (
                        _timestamp_text(timestamp),
                        identifier,
                        pending_request.source_event_id,
                        pending_request.chat_guid,
                        pending_request.service_date.isoformat(),
                        pending_request.meal_slot,
                    ),
                )
                if consumed.rowcount != 1:
                    raise MealReportApplicationError(
                        "pending meal request changed before scheduled plan persistence"
                    )
            connection.commit()
        except (
            ActiveMealPlanInvariantError,
            ScheduledRecommendationAlreadyExistsError,
            ScheduledRecommendationPlanConflictError,
            MealReportApplicationError,
        ):
            connection.rollback()
            raise
        except (sqlite3.Error, NutritionCatalogError) as exc:
            connection.rollback()
            raise MealReportApplicationError("unable to persist scheduled meal plan") from exc
        return PersistedMealPlan(identifier, plan, slot, timestamp, "active")

    def save_immediate_meal_request_plan(
        self,
        plan: MealPlan,
        *,
        meal_slot: MealSlot | str | None = None,
        source_event_id: str,
        chat_guid: str,
        requested_foods: tuple[RequestedMealFood, ...],
        whole_meal: bool,
        reply_text: str,
        plan_id: str | None = None,
        delivery_token: str | None = None,
        created_at: datetime | None = None,
    ) -> ImmediateMealRequestDispatch:
        """Persist a no-prior-plan request and delivery state atomically."""

        if not isinstance(plan, MealPlan):
            raise TypeError("plan must be a MealPlan")
        _text(source_event_id, "source_event_id")
        _text(chat_guid, "chat_guid")
        if not isinstance(requested_foods, tuple) or not all(
            isinstance(food, RequestedMealFood) for food in requested_foods
        ):
            raise TypeError("requested_foods must contain RequestedMealFood values")
        if len({food.identity_with_signature for food in requested_foods}) != len(
            requested_foods
        ):
            raise ValueError("requested_foods must not repeat")
        if not isinstance(whole_meal, bool):
            raise TypeError("whole_meal must be a bool")
        if not whole_meal and not requested_foods:
            raise ValueError("targeted immediate request needs requested foods")
        _text(reply_text, "reply_text")
        identifier = plan_id or str(uuid4())
        token = delivery_token or str(uuid4())
        _text(identifier, "plan_id")
        _text(token, "delivery_token")
        timestamp = _utc_now() if created_at is None else _timestamp(created_at, "created_at")
        try:
            context = meal_context_key(plan.meal)
            slot = _resolved_meal_slot(plan.meal, meal_slot)
            for item in plan.items:
                self._validate_plan_item_link(item)
            required_identities = {
                food.identity_with_signature for food in requested_foods
            }
            actual_identities = {
                (
                    item.food.source_identifier.kind,
                    item.food.source_identifier.value,
                    item.food.content_signature,
                )
                for item in plan.items
            }
            if not required_identities <= actual_identities:
                raise ValueError("immediate plan does not retain every requested food")
        except (TypeError, ValueError, NutritionCatalogError, MealReportApplicationError) as exc:
            raise MealReportApplicationError("unable to persist immediate meal request") from exc

        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT source_event_id, chat_guid, plan_id, whole_meal,
                       delivery_token, status, created_at, delivery_started_at, delivered_at
                FROM immediate_meal_request_dispatches WHERE source_event_id = ?
                """,
                (source_event_id,),
            ).fetchone()
            if existing is not None:
                dispatch = self._immediate_meal_request_dispatch_from_row(existing)
                if dispatch.chat_guid != chat_guid:
                    raise MealReportApplicationError(
                        "immediate request source event belongs to another chat"
                    )
                connection.rollback()
                return dispatch
            if self._message_event_row(connection, source_event_id) is not None:
                raise MealReportApplicationError(
                    "source event was already used by another conversation interaction"
                )
            if connection.execute(
                "SELECT 1 FROM pending_meal_requests WHERE source_event_id = ?",
                (source_event_id,),
            ).fetchone() is not None or connection.execute(
                "SELECT 1 FROM meal_recommendation_replacements WHERE source_event_id = ?",
                (source_event_id,),
            ).fetchone() is not None:
                raise MealReportApplicationError(
                    "source event was already used by another meal request"
                )
            active_rows = connection.execute(
                """
                SELECT plan_id FROM meal_plans
                WHERE service_date = ? AND meal_slot = ? AND status = 'active'
                """,
                (plan.service_date.isoformat(), slot),
            ).fetchall()
            if active_rows:
                raise MealReportApplicationError(
                    "an active meal plan already owns this immediate request context"
                )
            connection.execute(
                """
                INSERT INTO meal_plans
                (plan_id, service_date, meal, meal_kind, meal_context, meal_slot, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                (
                    identifier,
                    plan.service_date.isoformat(),
                    str(plan.meal),
                    "integer" if isinstance(plan.meal, int) else "text",
                    context,
                    slot,
                    _timestamp_text(timestamp),
                ),
            )
            for position, item in enumerate(plan.items):
                food = item.food
                connection.execute(
                    """
                    INSERT INTO meal_plan_items
                    (plan_id, plan_item_id, item_position, occurrence_id,
                     nutrition_snapshot_id, source_kind, source_value,
                     content_signature, recommended_official_servings,
                     natural_quantity_text, display_food_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        plan.item_id(item),
                        position,
                        food.occurrence.occurrence_id,
                        food.nutrition_snapshot_id,
                        food.source_identifier.kind,
                        food.source_identifier.value,
                        food.content_signature,
                        str(item.recommended_official_servings),
                        item.natural_quantity_text,
                        item.display_food_name,
                    ),
                )
                self._save_presentation_binding(connection, identifier, plan.item_id(item), item, timestamp)
            connection.execute(
                """
                INSERT INTO immediate_meal_request_dispatches
                (source_event_id, chat_guid, plan_id, whole_meal, delivery_token, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'pending_delivery', ?)
                """,
                (
                    source_event_id,
                    chat_guid,
                    identifier,
                    1 if whole_meal else 0,
                    token,
                    _timestamp_text(timestamp),
                ),
            )
            for position, food in enumerate(requested_foods):
                connection.execute(
                    """
                    INSERT INTO immediate_meal_request_requested_foods
                    (source_event_id, food_position, food_text, source_kind,
                     source_value, content_signature)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_event_id,
                        position,
                        food.food_text,
                        food.source_kind,
                        food.source_value,
                        food.content_signature,
                    ),
                )
            connection.execute(
                """
                INSERT INTO meal_report_message_events
                (source_event_id, chat_guid, outcome_type, plan_id, draft_id,
                 application_source_event_id, intent, reply_text, processed_at)
                VALUES (?, ?, 'routed_interaction', ?, NULL, NULL,
                        'meal_request', ?, ?)
                """,
                (source_event_id, chat_guid, identifier, reply_text, _timestamp_text(timestamp)),
            )
            connection.execute(
                """
                UPDATE pending_meal_requests SET status = 'superseded'
                WHERE chat_guid = ? AND service_date = ? AND meal_slot = ?
                  AND status = 'pending'
                """,
                (chat_guid, plan.service_date.isoformat(), slot),
            )
            connection.commit()
        except MealReportApplicationError:
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise MealReportApplicationError("unable to persist immediate meal request") from exc
        saved = self.load_immediate_meal_request_dispatch(source_event_id)
        if saved is None or saved.plan_id != identifier:
            raise MealReportApplicationError("immediate meal request disappeared after persistence")
        return saved

    def load_immediate_meal_request_dispatch(
        self,
        source_event_id: str,
    ) -> ImmediateMealRequestDispatch | None:
        """Load an immediate request by its idempotent inbound GUID."""

        _text(source_event_id, "source_event_id")
        row = self._connection().execute(
            """
            SELECT source_event_id, chat_guid, plan_id, whole_meal,
                   delivery_token, status, created_at, delivery_started_at, delivered_at
            FROM immediate_meal_request_dispatches WHERE source_event_id = ?
            """,
            (source_event_id,),
        ).fetchone()
        return None if row is None else self._immediate_meal_request_dispatch_from_row(row)

    def load_immediate_meal_request_dispatch_for_plan(
        self,
        plan_id: str,
    ) -> ImmediateMealRequestDispatch | None:
        """Load delivery state before allowing reports against a new request."""

        _text(plan_id, "plan_id")
        row = self._connection().execute(
            """
            SELECT source_event_id, chat_guid, plan_id, whole_meal,
                   delivery_token, status, created_at, delivery_started_at, delivered_at
            FROM immediate_meal_request_dispatches WHERE plan_id = ?
            """,
            (plan_id,),
        ).fetchone()
        return None if row is None else self._immediate_meal_request_dispatch_from_row(row)

    def claim_immediate_meal_request_delivery(
        self,
        dispatch: ImmediateMealRequestDispatch,
        *,
        started_at: datetime | None = None,
    ) -> bool:
        """Atomically claim one immediate request delivery."""

        if not isinstance(dispatch, ImmediateMealRequestDispatch):
            raise TypeError("dispatch must be an ImmediateMealRequestDispatch")
        if dispatch.status != "pending_delivery":
            return False
        timestamp = _utc_now() if started_at is None else _timestamp(started_at, "started_at")
        connection = self._connection()
        try:
            with connection:
                claimed = connection.execute(
                    """
                    UPDATE immediate_meal_request_dispatches
                    SET status = 'sending', delivery_started_at = ?
                    WHERE source_event_id = ? AND plan_id = ? AND status = 'pending_delivery'
                    """,
                    (_timestamp_text(timestamp), dispatch.source_event_id, dispatch.plan_id),
                )
        except sqlite3.Error as exc:
            raise MealReportApplicationError("unable to claim immediate request delivery") from exc
        return claimed.rowcount == 1

    def release_immediate_meal_request_delivery(
        self,
        dispatch: ImmediateMealRequestDispatch,
    ) -> ImmediateMealRequestDispatch:
        return self._transition_immediate_meal_request_dispatch(
            dispatch,
            expected_status="sending",
            next_status="pending_delivery",
            delivered_at=None,
        )

    def mark_immediate_meal_request_delivered(
        self,
        dispatch: ImmediateMealRequestDispatch,
        *,
        delivered_at: datetime | None = None,
    ) -> ImmediateMealRequestDispatch:
        timestamp = _utc_now() if delivered_at is None else _timestamp(delivered_at, "delivered_at")
        return self._transition_immediate_meal_request_dispatch(
            dispatch,
            expected_status="sending",
            next_status="delivered",
            delivered_at=timestamp,
        )

    def save_pending_meal_request(
        self,
        *,
        source_event_id: str,
        chat_guid: str,
        service_date: date,
        meal: str | int,
        meal_slot: MealSlot | str | None = None,
        requested_foods: tuple[RequestedMealFood, ...],
        whole_meal: bool,
        reply_text: str,
        created_at: datetime | None = None,
    ) -> PendingMealRequest:
        """Persist one future single-meal request with GUID idempotency.

        A later distinct request for the same chat/date/meal scope supersedes
        the earlier pending constraint.  It does not affect any other meal or
        future date, and it never creates a plan or intake entry by itself.
        """

        _text(source_event_id, "source_event_id")
        _text(chat_guid, "chat_guid")
        _date(service_date, "service_date")
        context = meal_context_key(meal)
        slot = (
            legacy_meal_slot(service_date, meal)
            if meal_slot is None
            else require_meal_slot(meal_slot)
        )
        if not isinstance(requested_foods, tuple) or not all(
            isinstance(food, RequestedMealFood) for food in requested_foods
        ):
            raise TypeError("requested_foods must contain RequestedMealFood values")
        if len({food.identity_with_signature for food in requested_foods}) != len(
            requested_foods
        ):
            raise ValueError("requested_foods must not repeat")
        if not isinstance(whole_meal, bool):
            raise TypeError("whole_meal must be a bool")
        if not requested_foods and not whole_meal:
            raise ValueError("a targeted pending request needs requested foods")
        _text(reply_text, "reply_text")
        timestamp = _utc_now() if created_at is None else _timestamp(created_at, "created_at")
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT source_event_id, chat_guid, service_date, meal, meal_kind,
                       meal_context, meal_slot, whole_meal, status, reply_text, created_at,
                       consumed_at, consumed_plan_id
                FROM pending_meal_requests WHERE source_event_id = ?
                """,
                (source_event_id,),
            ).fetchone()
            if existing is not None:
                saved = self._pending_meal_request_from_row(existing)
                if saved.chat_guid != chat_guid:
                    raise MealReportApplicationError(
                        "pending request source event belongs to another chat"
                    )
                connection.rollback()
                return saved
            if self._message_event_row(connection, source_event_id) is not None:
                raise MealReportApplicationError(
                    "source event was already used by another conversation interaction"
                )
            if connection.execute(
                """
                SELECT 1 FROM meal_recommendation_replacements
                WHERE source_event_id = ?
                """,
                (source_event_id,),
            ).fetchone() is not None:
                raise MealReportApplicationError(
                    "source event was already used by a replacement"
                )
            if connection.execute(
                """
                SELECT 1 FROM meal_plans
                WHERE service_date = ? AND meal_slot = ? AND status = 'active'
                """,
                (service_date.isoformat(), slot),
            ).fetchone() is not None or connection.execute(
                """
                SELECT 1 FROM scheduled_recommendation_dispatches
                WHERE service_date = ? AND meal_slot = ?
                """,
                (service_date.isoformat(), slot),
            ).fetchone() is not None:
                raise MealReportApplicationError(
                    "a persisted plan already owns this pending request context"
                )
            connection.execute(
                """
                UPDATE pending_meal_requests SET status = 'superseded'
                WHERE chat_guid = ? AND service_date = ? AND meal_slot = ?
                  AND status = 'pending'
                """,
                (chat_guid, service_date.isoformat(), slot),
            )
            connection.execute(
                """
                INSERT INTO pending_meal_requests
                (source_event_id, chat_guid, service_date, meal, meal_kind,
                 meal_context, meal_slot, whole_meal, status, reply_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    source_event_id,
                    chat_guid,
                    service_date.isoformat(),
                    str(meal),
                    "integer" if isinstance(meal, int) else "text",
                    context,
                    slot,
                    1 if whole_meal else 0,
                    reply_text,
                    _timestamp_text(timestamp),
                ),
            )
            for position, food in enumerate(requested_foods):
                connection.execute(
                    """
                    INSERT INTO pending_meal_request_foods
                    (source_event_id, food_position, food_text, source_kind,
                     source_value, content_signature)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_event_id,
                        position,
                        food.food_text,
                        food.source_kind,
                        food.source_value,
                        food.content_signature,
                    ),
                )
            connection.execute(
                """
                INSERT INTO meal_report_message_events
                (source_event_id, chat_guid, outcome_type, plan_id, draft_id,
                 application_source_event_id, intent, reply_text, processed_at)
                VALUES (?, ?, 'pending_meal_request', NULL, NULL, NULL,
                        'meal_request', ?, ?)
                """,
                (source_event_id, chat_guid, reply_text, _timestamp_text(timestamp)),
            )
            connection.commit()
        except MealReportApplicationError:
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise MealReportApplicationError("unable to persist pending meal request") from exc
        saved = self.load_pending_meal_request_for_source_event(source_event_id)
        if saved is None:
            raise MealReportApplicationError("pending meal request disappeared after persistence")
        return saved

    def load_pending_meal_request(
        self,
        service_date: date,
        meal: str | int,
        chat_guid: str,
        *,
        meal_slot: MealSlot | str | None = None,
    ) -> PendingMealRequest | None:
        """Load only the still-open request for one exact scheduler scope."""

        _date(service_date, "service_date")
        _text(chat_guid, "chat_guid")
        slot = (
            legacy_meal_slot(service_date, meal)
            if meal_slot is None
            else require_meal_slot(meal_slot)
        )
        row = self._connection().execute(
            """
            SELECT source_event_id, chat_guid, service_date, meal, meal_kind,
                   meal_context, meal_slot, whole_meal, status, reply_text, created_at,
                   consumed_at, consumed_plan_id
            FROM pending_meal_requests
            WHERE chat_guid = ? AND service_date = ? AND meal_slot = ?
              AND status = 'pending'
            """,
            (chat_guid, service_date.isoformat(), slot),
        ).fetchone()
        return None if row is None else self._pending_meal_request_from_row(row)

    def load_pending_meal_request_for_source_event(
        self,
        source_event_id: str,
    ) -> PendingMealRequest | None:
        """Load any lifecycle state for an inbound request GUID."""

        _text(source_event_id, "source_event_id")
        row = self._connection().execute(
            """
            SELECT source_event_id, chat_guid, service_date, meal, meal_kind,
                   meal_context, meal_slot, whole_meal, status, reply_text, created_at,
                   consumed_at, consumed_plan_id
            FROM pending_meal_requests WHERE source_event_id = ?
            """,
            (source_event_id,),
        ).fetchone()
        return None if row is None else self._pending_meal_request_from_row(row)

    def retire_stale_pending_meal_requests(self, current_service_date: date) -> int:
        """Close requests from earlier Detroit dates without touching plans/intake."""

        _date(current_service_date, "current_service_date")
        connection = self._connection()
        try:
            with connection:
                updated = connection.execute(
                    """
                    UPDATE pending_meal_requests SET status = 'superseded'
                    WHERE status = 'pending' AND service_date < ?
                    """,
                    (current_service_date.isoformat(),),
                )
        except sqlite3.Error as exc:
            raise MealReportApplicationError("unable to retire stale pending meal requests") from exc
        return int(updated.rowcount)

    def load_scheduled_recommendation_dispatch(
        self,
        service_date: date,
        meal: str | int,
        *,
        meal_slot: MealSlot | str | None = None,
    ) -> ScheduledRecommendationDispatch | None:
        """Load one scheduler dispatch for an exact canonical meal context."""

        _date(service_date, "service_date")
        slot = (
            legacy_meal_slot(service_date, meal)
            if meal_slot is None
            else require_meal_slot(meal_slot)
        )
        row = self._connection().execute(
            """
            SELECT service_date, meal_context, meal_slot, plan_id, delivery_token, status,
                   created_at, delivery_started_at, delivered_at, expired_at
            FROM scheduled_recommendation_dispatches
            WHERE service_date = ? AND meal_slot = ?
            """,
            (service_date.isoformat(), slot),
        ).fetchone()
        return None if row is None else self._scheduled_dispatch_from_row(row)

    def load_scheduled_recommendation_dispatch_for_plan(
        self,
        plan_id: str,
    ) -> ScheduledRecommendationDispatch | None:
        """Load scheduler delivery state owned by one exact persisted plan.

        Looking up by immutable plan ID distinguishes a currently active manual
        replacement from an older scheduler dispatch in the same meal context.
        """

        _text(plan_id, "plan_id")
        row = self._connection().execute(
            """
            SELECT service_date, meal_context, meal_slot, plan_id, delivery_token, status,
                   created_at, delivery_started_at, delivered_at, expired_at
            FROM scheduled_recommendation_dispatches
            WHERE plan_id = ?
            """,
            (plan_id,),
        ).fetchone()
        return None if row is None else self._scheduled_dispatch_from_row(row)

    def list_scheduled_recommendation_dispatches(
        self,
        service_date: date,
    ) -> tuple[ScheduledRecommendationDispatch, ...]:
        """Return scheduler dispatch state for one explicit service date."""

        _date(service_date, "service_date")
        rows = self._connection().execute(
            """
            SELECT service_date, meal_context, meal_slot, plan_id, delivery_token, status,
                   created_at, delivery_started_at, delivered_at, expired_at
            FROM scheduled_recommendation_dispatches
            WHERE service_date = ?
            ORDER BY meal_slot, plan_id
            """,
            (service_date.isoformat(),),
        ).fetchall()
        return tuple(self._scheduled_dispatch_from_row(row) for row in rows)

    def claim_scheduled_recommendation_delivery(
        self,
        dispatch: ScheduledRecommendationDispatch,
        *,
        started_at: datetime | None = None,
    ) -> bool:
        """Atomically change one pending delivery to ``sending``.

        ``False`` means another invocation already owns or completed the
        delivery; callers must reload the durable row instead of sending.
        """

        if not isinstance(dispatch, ScheduledRecommendationDispatch):
            raise TypeError("dispatch must be a ScheduledRecommendationDispatch")
        if dispatch.status != "pending_delivery":
            return False
        timestamp = _utc_now() if started_at is None else _timestamp(started_at, "started_at")
        connection = self._connection()
        try:
            with connection:
                claimed = connection.execute(
                    """
                    UPDATE scheduled_recommendation_dispatches
                    SET status = 'sending', delivery_started_at = ?
                    WHERE service_date = ? AND meal_slot = ? AND plan_id = ?
                      AND status = 'pending_delivery'
                    """,
                    (
                        _timestamp_text(timestamp),
                        dispatch.service_date.isoformat(),
                        dispatch.meal_slot,
                        dispatch.plan_id,
                    ),
                )
        except sqlite3.Error as exc:
            raise MealReportApplicationError("unable to claim scheduled delivery") from exc
        return claimed.rowcount == 1

    def mark_scheduled_recommendation_delivered(
        self,
        dispatch: ScheduledRecommendationDispatch,
        *,
        delivered_at: datetime | None = None,
    ) -> ScheduledRecommendationDispatch:
        """Record delivery and retire only older same-slot recommendation lineage.

        A later slot never mutates an earlier same-day slot. Replacements and
        immediate-request lineage are retired only when they belong to this
        exact product slot; the provider meal period is retained separately.
        """

        if not isinstance(dispatch, ScheduledRecommendationDispatch):
            raise TypeError("dispatch must be a ScheduledRecommendationDispatch")
        timestamp = (
            _utc_now()
            if delivered_at is None
            else _timestamp(delivered_at, "delivered_at")
        )
        connection = self._connection()
        try:
            with connection:
                transitioned = connection.execute(
                    """
                    UPDATE scheduled_recommendation_dispatches
                    SET status = 'delivered', delivered_at = ?
                    WHERE service_date = ? AND meal_slot = ? AND plan_id = ?
                      AND status = 'sending'
                    """,
                    (
                        _timestamp_text(timestamp),
                        dispatch.service_date.isoformat(),
                        dispatch.meal_slot,
                        dispatch.plan_id,
                    ),
                )
                if transitioned.rowcount != 1:
                    raise MealReportApplicationError(
                        "scheduled delivery state changed concurrently"
                    )
                connection.execute(
                    """
                    WITH RECURSIVE requested_lineage(plan_id) AS (
                        SELECT plan_id FROM immediate_meal_request_dispatches
                        WHERE status = 'delivered'
                        UNION
                        SELECT replacement.replacement_plan_id
                        FROM meal_recommendation_replacements AS replacement
                        JOIN requested_lineage
                          ON replacement.prior_plan_id = requested_lineage.plan_id
                    )
                    UPDATE meal_plans
                    SET status = 'superseded'
                    WHERE service_date = ? AND meal_slot = ?
                      AND plan_id != ? AND status = 'active'
                      AND plan_id IN (
                          SELECT plan_id
                          FROM scheduled_recommendation_dispatches
                          WHERE service_date = ?
                          UNION
                          SELECT replacement_plan_id
                          FROM meal_recommendation_replacements
                          WHERE scheduled_origin_plan_id IS NOT NULL
                          UNION
                          SELECT plan_id FROM requested_lineage
                      )
                    """,
                    (
                        dispatch.service_date.isoformat(),
                        dispatch.meal_slot,
                        dispatch.plan_id,
                        dispatch.service_date.isoformat(),
                    ),
                )
        except MealReportApplicationError:
            raise
        except sqlite3.Error as exc:
            raise MealReportApplicationError("unable to update scheduled delivery") from exc
        updated = self.load_scheduled_recommendation_dispatch(
            dispatch.service_date,
            dispatch.persisted_plan.plan.meal,
            meal_slot=dispatch.meal_slot,
        )
        if updated is None:
            raise MealReportApplicationError("scheduled delivery disappeared after update")
        return updated

    def release_scheduled_recommendation_delivery(
        self,
        dispatch: ScheduledRecommendationDispatch,
    ) -> ScheduledRecommendationDispatch:
        """Return a known failed ``sending`` attempt to pending delivery."""

        return self._transition_scheduled_dispatch(
            dispatch,
            expected_status="sending",
            next_status="pending_delivery",
            timestamp_column=None,
            timestamp=None,
        )

    def expire_scheduled_recommendation_dispatch(
        self,
        dispatch: ScheduledRecommendationDispatch,
        *,
        expired_at: datetime | None = None,
    ) -> ScheduledRecommendationDispatch:
        """Close an unsent plan after its configured delivery window ends.

        The scheduler-owned plan was never delivered, so it must not remain an
        answerable active plan after the product delivery window.  The dispatch
        transition and active-to-superseded lifecycle update commit together;
        this is distinct from facility-hour staleness and does not invent a
        weekday meal hand-off for a plan the user never received.
        """

        if not isinstance(dispatch, ScheduledRecommendationDispatch):
            raise TypeError("dispatch must be a ScheduledRecommendationDispatch")
        if dispatch.status != "pending_delivery":
            return dispatch
        timestamp = _utc_now() if expired_at is None else _timestamp(expired_at, "expired_at")
        connection = self._connection()
        try:
            with connection:
                transitioned = connection.execute(
                    """
                    UPDATE scheduled_recommendation_dispatches
                    SET status = 'expired', expired_at = ?
                    WHERE service_date = ? AND meal_slot = ? AND plan_id = ?
                      AND status = 'pending_delivery'
                    """,
                    (
                        _timestamp_text(timestamp),
                        dispatch.service_date.isoformat(),
                        dispatch.meal_slot,
                        dispatch.plan_id,
                    ),
                )
                if transitioned.rowcount != 1:
                    raise MealReportApplicationError(
                        "scheduled delivery state changed concurrently"
                    )
                # A concurrent explicit replacement may already have
                # superseded this plan.  In that case zero rows is valid; an
                # active row, however, must be retired with the expiry.
                connection.execute(
                    """
                    UPDATE meal_plans
                    SET status = 'superseded'
                    WHERE plan_id = ? AND status = 'active'
                    """,
                    (dispatch.plan_id,),
                )
        except MealReportApplicationError:
            raise
        except sqlite3.Error as exc:
            raise MealReportApplicationError("unable to expire scheduled delivery") from exc
        updated = self.load_scheduled_recommendation_dispatch(
            dispatch.service_date,
            dispatch.persisted_plan.plan.meal,
        )
        if updated is None:
            raise MealReportApplicationError("scheduled delivery disappeared after expiry")
        return updated

    def save_replacement_meal_plan(
        self,
        prior_plan: PersistedMealPlan,
        replacement_plan: MealPlan,
        *,
        source_event_id: str,
        chat_guid: str,
        rejected_plan_item_ids: tuple[str, ...],
        whole_meal: bool,
        reply_text: str,
        request_kind: Literal["rejection", "meal_request"] = "rejection",
        requested_foods: tuple[RequestedMealFood, ...] = (),
        replacement_plan_id: str | None = None,
        delivery_token: str | None = None,
        created_at: datetime | None = None,
    ) -> MealRecommendationReplacementDispatch:
        """Atomically replace one exact active plan and reserve its delivery.

        The replacement plan, prior-plan supersession, lineage row, rejected
        authoritative item references, and inbound GUID/reply event commit as
        one transaction.  Therefore a retry after a known send failure can
        only ever resend this immutable replacement; it cannot re-optimize or
        create another plan.
        """

        if not isinstance(prior_plan, PersistedMealPlan):
            raise TypeError("prior_plan must be a PersistedMealPlan")
        if not isinstance(replacement_plan, MealPlan):
            raise TypeError("replacement_plan must be a MealPlan")
        _text(source_event_id, "source_event_id")
        _text(chat_guid, "chat_guid")
        if request_kind not in {"rejection", "meal_request"}:
            raise ValueError("request_kind is invalid")
        if not isinstance(rejected_plan_item_ids, tuple):
            raise TypeError("rejected_plan_item_ids must be a tuple")
        if len(set(rejected_plan_item_ids)) != len(rejected_plan_item_ids):
            raise ValueError("rejected_plan_item_ids must not repeat")
        for item_id in rejected_plan_item_ids:
            _text(item_id, "rejected plan item ID")
            if prior_plan.plan.item_for_id(item_id) is None:
                raise ValueError("rejected plan item does not belong to prior plan")
        if not isinstance(requested_foods, tuple) or not all(
            isinstance(food, RequestedMealFood) for food in requested_foods
        ):
            raise TypeError("requested_foods must contain RequestedMealFood values")
        if len({food.identity_with_signature for food in requested_foods}) != len(
            requested_foods
        ):
            raise ValueError("requested_foods must not repeat")
        if request_kind == "rejection":
            if not rejected_plan_item_ids:
                raise ValueError("rejected_plan_item_ids must be a non-empty tuple")
            if requested_foods:
                raise ValueError("rejection replacement cannot retain requested foods")
        elif rejected_plan_item_ids:
            raise ValueError("positive meal request cannot retain rejected plan item IDs")
        if not isinstance(whole_meal, bool):
            raise TypeError("whole_meal must be a bool")
        if request_kind == "meal_request" and not whole_meal and not requested_foods:
            raise ValueError("targeted positive meal request needs requested foods")
        if request_kind == "rejection" and whole_meal and set(rejected_plan_item_ids) != {
            prior_plan.plan.item_id(item) for item in prior_plan.plan.items
        }:
            raise ValueError("whole-meal replacement must reject every prior plan item")
        _text(reply_text, "reply_text")
        if (
            replacement_plan.service_date != prior_plan.plan.service_date
            or not meal_values_equal(replacement_plan.meal, prior_plan.plan.meal)
        ):
            raise ValueError("replacement plan must keep the prior service context")
        identifier = replacement_plan_id or str(uuid4())
        token = delivery_token or str(uuid4())
        _text(identifier, "replacement_plan_id")
        _text(token, "delivery_token")
        timestamp = _utc_now() if created_at is None else _timestamp(created_at, "created_at")
        rejected_identities = {
            _plan_item_food_identity(prior_plan.plan.item_for_id(item_id))
            for item_id in rejected_plan_item_ids
        }
        prior_identities = {_plan_item_food_identity(item) for item in prior_plan.plan.items}
        replacement_identities = {
            _plan_item_food_identity(item) for item in replacement_plan.items
        }
        if request_kind == "rejection" and rejected_identities & replacement_identities:
            raise ValueError("replacement plan contains a rejected authoritative food")
        if request_kind == "rejection" and replacement_identities <= prior_identities:
            raise ValueError("replacement plan must contain an authoritative new food")
        if request_kind == "meal_request" and not {
            food.identity_with_signature for food in requested_foods
        } <= {
            (
                item.food.source_identifier.kind,
                item.food.source_identifier.value,
                item.food.content_signature,
            )
            for item in replacement_plan.items
        }:
            raise ValueError("replacement plan does not retain every requested food")
        try:
            context = meal_context_key(replacement_plan.meal)
            slot = prior_plan.meal_slot
            for item in replacement_plan.items:
                self._validate_plan_item_link(item)
        except (TypeError, ValueError, NutritionCatalogError, MealReportApplicationError) as exc:
            raise MealReportApplicationError("unable to persist replacement meal plan") from exc

        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT source_event_id, chat_guid, prior_plan_id, replacement_plan_id,
                       scheduled_origin_plan_id, request_kind, whole_meal, delivery_token, status,
                       created_at, delivery_started_at, delivered_at
                FROM meal_recommendation_replacements
                WHERE source_event_id = ?
                """,
                (source_event_id,),
            ).fetchone()
            if existing is not None:
                existing_dispatch = self._replacement_dispatch_from_row(existing)
                if existing_dispatch.chat_guid != chat_guid:
                    raise MealRecommendationReplacementConflictError(
                        "replacement source event belongs to another chat"
                    )
                connection.rollback()
                return existing_dispatch
            existing_event = self._message_event_row(connection, source_event_id)
            if existing_event is not None:
                raise MealRecommendationReplacementConflictError(
                    "source event was already used by another conversation interaction"
                )

            prior_row = connection.execute(
                """
                SELECT service_date, meal_context, meal_slot, status FROM meal_plans
                WHERE plan_id = ?
                """,
                (prior_plan.plan_id,),
            ).fetchone()
            if (
                prior_row is None
                or str(prior_row["status"]) != "active"
                or str(prior_row["service_date"]) != replacement_plan.service_date.isoformat()
                or str(prior_row["meal_context"]) != context
                or str(prior_row["meal_slot"]) != slot
            ):
                raise MealRecommendationReplacementConflictError(
                    "prior meal plan is no longer the active replacement context"
                )
            active_rows = connection.execute(
                """
                SELECT plan_id FROM meal_plans
                WHERE service_date = ? AND meal_slot = ? AND status = 'active'
                """,
                (replacement_plan.service_date.isoformat(), slot),
            ).fetchall()
            if len(active_rows) != 1 or str(active_rows[0]["plan_id"]) != prior_plan.plan_id:
                raise MealRecommendationReplacementConflictError(
                    "replacement context changed concurrently"
                )
            open_draft = connection.execute(
                """
                SELECT draft_id FROM meal_report_drafts
                WHERE plan_id = ? AND status IN ('draft', 'awaiting_clarification')
                """,
                (prior_plan.plan_id,),
            ).fetchone()
            if open_draft is not None:
                raise MealRecommendationReplacementDraftConflictError(
                    "an unresolved meal report draft blocks replacement"
                )
            scheduled_origin = connection.execute(
                """
                SELECT plan_id FROM scheduled_recommendation_dispatches
                WHERE plan_id = ? AND status = 'delivered'
                """,
                (prior_plan.plan_id,),
            ).fetchone()
            if scheduled_origin is not None:
                scheduled_origin_plan_id: str | None = str(scheduled_origin["plan_id"])
            else:
                inherited_origin = connection.execute(
                    """
                    SELECT scheduled_origin_plan_id
                    FROM meal_recommendation_replacements
                    WHERE replacement_plan_id = ?
                    """,
                    (prior_plan.plan_id,),
                ).fetchone()
                scheduled_origin_plan_id = (
                    None
                    if inherited_origin is None
                    or inherited_origin["scheduled_origin_plan_id"] is None
                    else str(inherited_origin["scheduled_origin_plan_id"])
                )

            retired = connection.execute(
                """
                UPDATE meal_plans SET status = 'superseded'
                WHERE plan_id = ? AND status = 'active'
                """,
                (prior_plan.plan_id,),
            )
            if retired.rowcount != 1:
                raise MealRecommendationReplacementConflictError(
                    "prior meal plan could not be superseded"
                )
            connection.execute(
                """
                INSERT INTO meal_plans
                (plan_id, service_date, meal, meal_kind, meal_context, meal_slot, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                (
                    identifier,
                    replacement_plan.service_date.isoformat(),
                    str(replacement_plan.meal),
                    "integer" if isinstance(replacement_plan.meal, int) else "text",
                    context,
                    slot,
                    _timestamp_text(timestamp),
                ),
            )
            for position, item in enumerate(replacement_plan.items):
                food = item.food
                connection.execute(
                    """
                    INSERT INTO meal_plan_items
                    (plan_id, plan_item_id, item_position, occurrence_id,
                     nutrition_snapshot_id, source_kind, source_value,
                     content_signature, recommended_official_servings,
                     natural_quantity_text, display_food_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        replacement_plan.item_id(item),
                        position,
                        food.occurrence.occurrence_id,
                        food.nutrition_snapshot_id,
                        food.source_identifier.kind,
                        food.source_identifier.value,
                        food.content_signature,
                        str(item.recommended_official_servings),
                        item.natural_quantity_text,
                        item.display_food_name,
                    ),
                )
                self._save_presentation_binding(connection, identifier, replacement_plan.item_id(item), item, timestamp)
            connection.execute(
                """
                INSERT INTO meal_recommendation_replacements
                (source_event_id, chat_guid, prior_plan_id, replacement_plan_id,
                 scheduled_origin_plan_id, request_kind, whole_meal, delivery_token, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending_delivery', ?)
                """,
                (
                    source_event_id,
                    chat_guid,
                    prior_plan.plan_id,
                    identifier,
                    scheduled_origin_plan_id,
                    request_kind,
                    1 if whole_meal else 0,
                    token,
                    _timestamp_text(timestamp),
                ),
            )
            for item_id in rejected_plan_item_ids:
                connection.execute(
                    """
                    INSERT INTO meal_recommendation_replacement_rejections
                    (source_event_id, plan_item_id) VALUES (?, ?)
                    """,
                    (source_event_id, item_id),
                )
            for position, food in enumerate(requested_foods):
                connection.execute(
                    """
                    INSERT INTO meal_recommendation_replacement_requested_foods
                    (source_event_id, food_position, food_text, source_kind, source_value,
                     content_signature)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_event_id,
                        position,
                        food.food_text,
                        food.source_kind,
                        food.source_value,
                        food.content_signature,
                    ),
                )
            connection.execute(
                """
                INSERT INTO meal_report_message_events
                (source_event_id, chat_guid, outcome_type, plan_id, draft_id,
                 application_source_event_id, intent, reply_text, processed_at)
                VALUES (?, ?, 'routed_interaction', ?, NULL, NULL, ?, ?, ?)
                """,
                (
                    source_event_id,
                    chat_guid,
                    identifier,
                    "meal_request" if request_kind == "meal_request" else "replacement_request",
                    reply_text,
                    _timestamp_text(timestamp),
                ),
            )
            connection.commit()
        except (
            MealRecommendationReplacementConflictError,
            MealRecommendationReplacementDraftConflictError,
        ):
            connection.rollback()
            raise
        except (sqlite3.Error, NutritionCatalogError) as exc:
            connection.rollback()
            raise MealReportApplicationError("unable to persist replacement meal plan") from exc
        saved = self.load_meal_recommendation_replacement_dispatch(source_event_id)
        if saved is None or saved.plan_id != identifier:
            raise MealReportApplicationError("replacement dispatch disappeared after persistence")
        return saved

    def load_meal_recommendation_replacement_dispatch(
        self,
        source_event_id: str,
    ) -> MealRecommendationReplacementDispatch | None:
        """Load one replacement by its idempotent inbound source GUID."""

        _text(source_event_id, "source_event_id")
        row = self._connection().execute(
            """
            SELECT source_event_id, chat_guid, prior_plan_id, replacement_plan_id,
                   scheduled_origin_plan_id, request_kind, whole_meal, delivery_token, status,
                   created_at, delivery_started_at, delivered_at
            FROM meal_recommendation_replacements
            WHERE source_event_id = ?
            """,
            (source_event_id,),
        ).fetchone()
        return None if row is None else self._replacement_dispatch_from_row(row)

    def load_meal_recommendation_replacement_dispatch_for_plan(
        self,
        plan_id: str,
    ) -> MealRecommendationReplacementDispatch | None:
        """Load replacement delivery state for one exact replacement plan."""

        _text(plan_id, "plan_id")
        row = self._connection().execute(
            """
            SELECT source_event_id, chat_guid, prior_plan_id, replacement_plan_id,
                   scheduled_origin_plan_id, request_kind, whole_meal, delivery_token, status,
                   created_at, delivery_started_at, delivered_at
            FROM meal_recommendation_replacements
            WHERE replacement_plan_id = ?
            """,
            (plan_id,),
        ).fetchone()
        return None if row is None else self._replacement_dispatch_from_row(row)

    def claim_replacement_recommendation_delivery(
        self,
        dispatch: MealRecommendationReplacementDispatch,
        *,
        started_at: datetime | None = None,
    ) -> bool:
        """Atomically claim one known-pending replacement delivery."""

        if not isinstance(dispatch, MealRecommendationReplacementDispatch):
            raise TypeError("dispatch must be a MealRecommendationReplacementDispatch")
        if dispatch.status != "pending_delivery":
            return False
        timestamp = _utc_now() if started_at is None else _timestamp(started_at, "started_at")
        connection = self._connection()
        try:
            with connection:
                claimed = connection.execute(
                    """
                    UPDATE meal_recommendation_replacements
                    SET status = 'sending', delivery_started_at = ?
                    WHERE source_event_id = ? AND replacement_plan_id = ?
                      AND status = 'pending_delivery'
                    """,
                    (
                        _timestamp_text(timestamp),
                        dispatch.source_event_id,
                        dispatch.plan_id,
                    ),
                )
        except sqlite3.Error as exc:
            raise MealReportApplicationError("unable to claim replacement delivery") from exc
        return claimed.rowcount == 1

    def release_replacement_recommendation_delivery(
        self,
        dispatch: MealRecommendationReplacementDispatch,
    ) -> MealRecommendationReplacementDispatch:
        """Return one known-failed replacement send to pending delivery."""

        return self._transition_replacement_dispatch(
            dispatch,
            expected_status="sending",
            next_status="pending_delivery",
            timestamp_column=None,
            timestamp=None,
        )

    def mark_replacement_recommendation_delivered(
        self,
        dispatch: MealRecommendationReplacementDispatch,
        *,
        delivered_at: datetime | None = None,
    ) -> MealRecommendationReplacementDispatch:
        """Mark the immutable replacement recommendation as delivered once."""

        timestamp = _utc_now() if delivered_at is None else _timestamp(delivered_at, "delivered_at")
        return self._transition_replacement_dispatch(
            dispatch,
            expected_status="sending",
            next_status="delivered",
            timestamp_column="delivered_at",
            timestamp=timestamp,
        )

    def load_meal_plan(self, plan_id: str) -> PersistedMealPlan | None:
        """Reconstruct a plan from historical links, without current-menu lookup."""

        _text(plan_id, "plan_id")
        row = self._connection().execute(
            """
            SELECT plan_id, service_date, meal, meal_kind, meal_context, meal_slot,
                   created_at, status
            FROM meal_plans WHERE plan_id = ?
            """,
            (plan_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            service_date = date.fromisoformat(str(row["service_date"]))
            meal = int(row["meal"]) if row["meal_kind"] == "integer" else str(row["meal"])
            if str(row["meal_context"]) != meal_context_key(meal):
                raise MealReportApplicationError("persisted meal plan context is corrupt")
            created_at = _parse_timestamp(row["created_at"], "created_at")
            status = str(row["status"])
            item_rows = self._connection().execute(
                """
                SELECT plan_item_id, item_position, occurrence_id, nutrition_snapshot_id,
                       source_kind, source_value, content_signature,
                       recommended_official_servings, natural_quantity_text, display_food_name
                FROM meal_plan_items WHERE plan_id = ? ORDER BY item_position
                """,
                (plan_id,),
            ).fetchall()
            items: list[PlannedMealItem] = []
            for expected_position, item_row in enumerate(item_rows):
                if int(item_row["item_position"]) != expected_position or item_row["plan_item_id"] != f"item_{expected_position + 1}":
                    raise MealReportApplicationError("persisted plan item order is corrupt")
                food = self._historical_food(
                    occurrence_id=int(item_row["occurrence_id"]),
                    snapshot_id=int(item_row["nutrition_snapshot_id"]),
                    source_identifier=SourceIdentifier(item_row["source_kind"], item_row["source_value"]),
                    content_signature=str(item_row["content_signature"]),
                )
                items.append(
                    PlannedMealItem(
                        food,
                        _decimal(item_row["recommended_official_servings"], "recommended_official_servings"),
                        str(item_row["natural_quantity_text"]),
                        item_row["display_food_name"],
                        self._load_presentation_binding(plan_id, item_row["plan_item_id"]),
                    )
                )
            return PersistedMealPlan(
                str(row["plan_id"]),
                MealPlan(service_date, meal, tuple(items)),
                require_meal_slot(row["meal_slot"]),
                created_at,
                status,  # type: ignore[arg-type]
            )
        except (TypeError, ValueError, NutritionCatalogError, MealReportApplicationError):
            raise MealReportApplicationError("persisted meal plan is corrupt") from None

    def load_active_meal_plan(
        self,
        service_date: date,
        meal: str | int,
        *,
        meal_slot: MealSlot | str | None = None,
    ) -> PersistedMealPlan | None:
        """Load one active lifecycle plan for a provider context and optional slot.

        ``None`` is the explicit no-active-plan result. Reportability is a
        stricter later decision because a scheduler plan can be pending or
        sending before the user has actually received it. A corrupt database
        or a provider period shared by multiple product slots fails closed
        instead of selecting a newest plan heuristically when no slot is given.
        """

        _date(service_date, "service_date")
        context = meal_context_key(meal)
        slot = None if meal_slot is None else require_meal_slot(meal_slot)
        rows = self._connection().execute(
            """
            SELECT plan_id, meal, meal_kind, meal_context, meal_slot FROM meal_plans
            WHERE service_date = ? AND status = 'active'
            ORDER BY meal_slot, plan_id
            """,
            (service_date.isoformat(),),
        ).fetchall()
        matching_rows: list[sqlite3.Row] = []
        try:
            for row in rows:
                stored_meal = (
                    int(str(row["meal"]))
                    if row["meal_kind"] == "integer"
                    else str(row["meal"])
                )
                if str(row["meal_context"]) != meal_context_key(stored_meal):
                    raise MealReportApplicationError("active meal plan context is corrupt")
                stored_slot = require_meal_slot(row["meal_slot"])
                if str(row["meal_context"]) == context and (
                    slot is None or stored_slot == slot
                ):
                    matching_rows.append(row)
        except (TypeError, ValueError):
            raise MealReportApplicationError("active meal plan context is corrupt") from None
        if len(matching_rows) > 1:
            raise ActiveMealPlanInvariantError(
                "multiple active meal plans match the requested product context"
            )
        if not matching_rows:
            return None
        persisted = self.load_meal_plan(str(matching_rows[0]["plan_id"]))
        if persisted is None or persisted.status != "active":
            raise MealReportApplicationError("active meal plan row is corrupt")
        return persisted

    def list_active_meal_plans(self, service_date: date) -> tuple[PersistedMealPlan, ...]:
        """Return every active lifecycle plan for one exact service date.

        This deliberately does not select a newest plan.  Callers that need to
        correlate an inbound transport event can decide only when the returned
        set is unambiguous. Rows are ordered by durable product slot and plan
        ID, and a corrupt duplicate slot fails closed even if a damaged
        database no longer enforces the partial unique index.
        """

        _date(service_date, "service_date")
        rows = self._connection().execute(
            """
            SELECT plan_id, meal_context, meal_slot FROM meal_plans
            WHERE service_date = ? AND status = 'active'
            ORDER BY meal_slot, plan_id
            """,
            (service_date.isoformat(),),
        ).fetchall()

        persisted_plans: list[PersistedMealPlan] = []
        seen_slots: set[MealSlot] = set()
        for row in rows:
            context = str(row["meal_context"])
            try:
                slot = require_meal_slot(row["meal_slot"])
            except ValueError:
                raise MealReportApplicationError("active meal plan slot is corrupt") from None
            if slot in seen_slots:
                raise ActiveMealPlanInvariantError(
                    "multiple active meal plans exist for one product meal slot"
                )
            seen_slots.add(slot)
            persisted = self.load_meal_plan(str(row["plan_id"]))
            if (
                persisted is None
                or persisted.status != "active"
                or persisted.plan.service_date != service_date
                or meal_context_key(persisted.plan.meal) != context
                or persisted.meal_slot != slot
            ):
                raise MealReportApplicationError("active meal plan row is corrupt")
            persisted_plans.append(persisted)
        return tuple(persisted_plans)

    def is_active_meal_plan_reportable(
        self,
        plan: PersistedMealPlan,
        service_calendar: PhelpsServiceCalendar,
        *,
        evaluated_at: datetime,
    ) -> bool:
        """Return whether one current-version plan can receive a new report.

        Reportability is independent of generation/service eligibility. A plan
        must be active for the Detroit calendar date and have a successfully
        delivered dispatch when its delivery path records one. The legacy
        manual-send path has no separate v10 delivery marker and therefore
        retains its existing conservative service-calendar fallback.
        """

        if not isinstance(plan, PersistedMealPlan):
            raise TypeError("plan must be a PersistedMealPlan")
        if not isinstance(service_calendar, PhelpsServiceCalendar):
            raise TypeError("service_calendar must be a PhelpsServiceCalendar")
        local_now = service_calendar.local_datetime_at(evaluated_at)
        if plan.status != "active" or plan.plan.service_date != local_now.date():
            return False
        delivery_state = self._reportable_delivery_state(plan.plan_id)
        if delivery_state is not None:
            return delivery_state
        # Schema v10 has no durable outbound marker for the legacy manual-send
        # path. Preserve its prior fail-closed service eligibility behavior;
        # every delivery-tracked product path above is window-independent.
        return service_calendar.is_meal_context_eligible_at(
            plan.plan.service_date,
            plan.plan.meal,
            local_now,
        )

    def list_reportable_active_meal_plans(
        self,
        service_date: date,
        service_calendar: PhelpsServiceCalendar,
        *,
        evaluated_at: datetime,
    ) -> tuple[PersistedMealPlan, ...]:
        """Return current-date active plans that can receive a new report.

        The result deliberately does not choose a recent historical plan.  If
        multiple reportable contexts survive, the inbound resolver still fails
        closed instead of guessing which meal the user means.
        """

        _date(service_date, "service_date")
        if not isinstance(service_calendar, PhelpsServiceCalendar):
            raise TypeError("service_calendar must be a PhelpsServiceCalendar")
        local_now = service_calendar.local_datetime_at(evaluated_at)
        return tuple(
            plan
            for plan in self.list_active_meal_plans(service_date)
            if self.is_active_meal_plan_reportable(
                plan,
                service_calendar,
                evaluated_at=local_now,
            )
        )

    def list_stale_active_meal_plans(
        self,
        service_calendar: PhelpsServiceCalendar,
        *,
        evaluated_at: datetime,
    ) -> tuple[PersistedMealPlan, ...]:
        """Inspect active plans that are definitely stale without mutating them.

        The calendar converts the supplied absolute instant into Detroit local
        time.  This read-only surface is intentionally useful for an operator
        diagnostic before requesting the explicit retirement action.
        """

        if not isinstance(service_calendar, PhelpsServiceCalendar):
            raise TypeError("service_calendar must be a PhelpsServiceCalendar")
        local_now = service_calendar.local_datetime_at(evaluated_at)
        return self._stale_active_meal_plans(service_calendar, local_now)

    def list_all_active_meal_plans(self) -> tuple[PersistedMealPlan, ...]:
        """Return every active plan across service dates without selecting one.

        This is a read-only lifecycle diagnostic.  It preserves the same
        duplicate-slot corruption checks as context-specific lookup rather
        than returning a potentially arbitrary row.
        """

        return self._list_all_active_meal_plans()

    def _stale_active_meal_plans(
        self,
        service_calendar: PhelpsServiceCalendar,
        local_now: datetime,
    ) -> tuple[PersistedMealPlan, ...]:
        """Select stale plans while preserving same-day reportability.

        This private helper is shared by the read-only diagnostic and the
        locked retirement transaction. Service-window closure cannot retire a
        current-date reportable plan; prior-date plans remain historical and
        are eligible for the existing stale-plan retirement policy.
        """

        return tuple(
            plan
            for plan in self._list_all_active_meal_plans()
            if not self._is_current_reportable_meal_plan(plan, local_now)
            and service_calendar.has_plan_become_stale(
                plan.plan.service_date,
                plan.plan.meal,
                local_now,
            )
        )

    def _is_current_reportable_meal_plan(
        self,
        plan: PersistedMealPlan,
        local_now: datetime,
    ) -> bool:
        """Return whether a plan remains normally reportable before midnight."""

        if plan.status != "active" or plan.plan.service_date != local_now.date():
            return False
        return self._reportable_delivery_state(plan.plan_id) is True

    def _reportable_delivery_state(self, plan_id: str) -> bool | None:
        """Evaluate durable delivery state without consulting service windows.

        Scheduler, immediate-request, and replacement paths have explicit
        delivery state. A plan with one of those rows is reportable only after
        successful delivery. ``None`` identifies the legacy manual path, which
        has no separate v10 delivery table and is handled conservatively by
        its existing service-calendar rule.
        """

        immediate = self.load_immediate_meal_request_dispatch_for_plan(plan_id)
        if immediate is not None:
            return immediate.status == "delivered"
        dispatch = self.load_scheduled_recommendation_dispatch_for_plan(plan_id)
        if dispatch is not None:
            return dispatch.status == "delivered"
        replacement = self.load_meal_recommendation_replacement_dispatch_for_plan(
            plan_id
        )
        if replacement is not None:
            return replacement.status == "delivered"
        return None

    def _is_plan_reportable_for_final_application(
        self,
        plan: PersistedMealPlan,
        service_calendar: PhelpsServiceCalendar | None,
        *,
        evaluated_at: datetime,
    ) -> bool:
        """Recheck current-date reportability while the final write lock is held."""

        if not isinstance(plan, PersistedMealPlan):
            raise TypeError("plan must be a PersistedMealPlan")
        if not isinstance(evaluated_at, datetime) or evaluated_at.tzinfo is None:
            raise TypeError("evaluated_at must be a timezone-aware datetime")
        if plan.status != "active" or plan.plan.service_date != evaluated_at.date():
            return False
        delivery_state = self._reportable_delivery_state(plan.plan_id)
        if delivery_state is not None:
            return delivery_state
        if service_calendar is None:
            return True
        return service_calendar.is_meal_context_eligible_at(
            plan.plan.service_date,
            plan.plan.meal,
            evaluated_at,
        )

    def retire_stale_active_meal_plans(
        self,
        service_calendar: PhelpsServiceCalendar,
        *,
        evaluated_at: datetime,
    ) -> StaleMealPlanRetirement:
        """Atomically transition every objectively stale active plan to superseded.

        Historical plans and plan items are not deleted or rewritten.  The
        candidate inventory completes before any update, and every update is
        performed in one SQLite transaction.  A damaged lifecycle row or any
        write failure therefore rolls back the whole pass instead of leaving a
        partially retired set of active plans.
        """

        if not isinstance(service_calendar, PhelpsServiceCalendar):
            raise TypeError("service_calendar must be a PhelpsServiceCalendar")
        local_now = service_calendar.local_datetime_at(evaluated_at)
        connection = self._connection()
        try:
            # Take the write lock before selecting candidates.  A concurrent
            # sender that records a successful delivery at the closing instant
            # therefore cannot be selected as stale and then retired by an
            # outdated maintenance snapshot.
            connection.execute("BEGIN IMMEDIATE")
            stale = self._stale_active_meal_plans(service_calendar, local_now)
            for persisted in stale:
                retired = connection.execute(
                    """
                    UPDATE meal_plans
                    SET status = 'superseded'
                    WHERE plan_id = ? AND status = 'active'
                    """,
                    (persisted.plan_id,),
                )
                if retired.rowcount != 1:
                    raise MealReportApplicationError(
                        "stale meal plan could not be retired"
                    )
            connection.commit()
        except MealReportApplicationError:
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise MealReportApplicationError(
                "stale meal plan retirement failed atomically"
            ) from exc
        return StaleMealPlanRetirement(
            local_now,
            tuple(persisted.plan_id for persisted in stale),
        )

    def load_meal_plan_for_source_event(self, source_event_id: str) -> PersistedMealPlan | None:
        """Return the historical plan already applied for one transport event."""

        _text(source_event_id, "source_event_id")
        row = self._connection().execute(
            "SELECT plan_id FROM meal_report_applications WHERE source_event_id = ?",
            (source_event_id,),
        ).fetchone()
        if row is None:
            return None
        persisted = self.load_meal_plan(str(row["plan_id"]))
        if persisted is None:
            raise MealReportApplicationError("meal report application references a missing plan")
        return persisted

    def load_applied_meal_report(self, source_event_id: str) -> AppliedMealReport | None:
        """Load an already-applied durable event for transport replay handling.

        The legacy application table intentionally stores accepted intake but
        not a second conversational report transcript.  Therefore this result
        exposes exact accepted entries and the plan identity, while a replay
        renderer uses a generic acknowledgment instead of reconstructing
        skipped or unspecified assertions from incomplete history.
        """

        _text(source_event_id, "source_event_id")
        row = self._connection().execute(
            "SELECT plan_id FROM meal_report_applications WHERE source_event_id = ?",
            (source_event_id,),
        ).fetchone()
        if row is None:
            return None
        return AppliedMealReport(
            str(row["plan_id"]),
            source_event_id,
            True,
            self._load_event_intake(source_event_id),
            (),
        )

    def load_meal_report_message_event(
        self,
        source_event_id: str,
    ) -> MealReportMessageEvent | None:
        """Load an already incorporated inbound message GUID, if any."""

        _text(source_event_id, "source_event_id")
        row = self._connection().execute(
            """
            SELECT source_event_id, chat_guid, outcome_type, plan_id, draft_id,
                   application_source_event_id, intent, reply_text, processed_at
            FROM meal_report_message_events
            WHERE source_event_id = ?
            """,
            (source_event_id,),
        ).fetchone()
        return None if row is None else self._meal_report_message_event_from_row(row)

    def load_meal_report_draft(self, draft_id: str) -> MealReportDraft | None:
        """Reconstruct one historical or open draft from exact persisted facts."""

        _text(draft_id, "draft_id")
        row = self._connection().execute(
            """
            SELECT draft_id, chat_guid, plan_id, status, revision, last_prompt,
                   created_at, updated_at, completed_at, cancelled_at
            FROM meal_report_drafts
            WHERE draft_id = ?
            """,
            (draft_id,),
        ).fetchone()
        return None if row is None else self._meal_report_draft_from_row(row)

    def list_active_meal_report_drafts(self, chat_guid: str) -> tuple[MealReportDraft, ...]:
        """Return every open plan-pinned draft for one chat in stable order."""

        _text(chat_guid, "chat_guid")
        rows = self._connection().execute(
            """
            SELECT draft_id FROM meal_report_drafts
            WHERE chat_guid = ? AND status IN ('draft', 'awaiting_clarification')
            ORDER BY updated_at, draft_id
            """,
            (chat_guid,),
        ).fetchall()
        drafts = tuple(
            self.load_meal_report_draft(str(row["draft_id"])) for row in rows
        )
        if any(
            draft is None or draft.status not in {"draft", "awaiting_clarification"}
            for draft in drafts
        ):
            raise MealReportApplicationError("open meal report draft is corrupt")
        return tuple(draft for draft in drafts if draft is not None)

    def load_active_meal_report_draft(
        self,
        chat_guid: str,
        *,
        plan_id: str | None = None,
    ) -> MealReportDraft | None:
        """Return one open draft, optionally pinned to an exact persisted plan."""

        drafts = self.list_active_meal_report_drafts(chat_guid)
        if plan_id is not None:
            _text(plan_id, "plan_id")
            drafts = tuple(draft for draft in drafts if draft.plan_id == plan_id)
        if len(drafts) > 1:
            raise MealReportApplicationError("multiple open meal report drafts require meal scope")
        return None if not drafts else drafts[0]

    def record_meal_report_message_event(
        self,
        *,
        source_event_id: str,
        chat_guid: str,
        persisted_plan: PersistedMealPlan,
        intent: MealReportIntent,
        reply_text: str,
        draft_id: str | None = None,
        outcome_type: Literal[
            "routed_interaction",
            "draft_update",
            "draft_cancelled",
        ] = "routed_interaction",
        processed_at: datetime | None = None,
    ) -> MealReportMessageEvent:
        """Durably record a non-intake inbound interaction and its reply.

        Location, replacement, and unsupported messages must be retry-safe too,
        but they intentionally create neither intake nor a report draft.
        """

        _text(source_event_id, "source_event_id")
        _text(chat_guid, "chat_guid")
        if not isinstance(persisted_plan, PersistedMealPlan):
            raise TypeError("persisted_plan must be a PersistedMealPlan")
        _intent(intent)
        if outcome_type not in {"routed_interaction", "draft_update", "draft_cancelled"}:
            raise ValueError("routed message outcome type is invalid")
        _text(reply_text, "reply_text")
        if draft_id is not None:
            _text(draft_id, "draft_id")
        timestamp = _utc_now() if processed_at is None else _timestamp(processed_at, "processed_at")
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._message_event_row(connection, source_event_id)
            if existing is not None:
                event = self._meal_report_message_event_from_row(existing)
                self._validate_existing_message_event(event, chat_guid, persisted_plan.plan_id)
                connection.rollback()
                return event
            if draft_id is not None:
                draft_row = connection.execute(
                    "SELECT plan_id FROM meal_report_drafts WHERE draft_id = ?",
                    (draft_id,),
                ).fetchone()
                if draft_row is None or str(draft_row["plan_id"]) != persisted_plan.plan_id:
                    raise MealReportApplicationError("message event draft does not belong to its plan")
            connection.execute(
                """
                INSERT INTO meal_report_message_events
                (source_event_id, chat_guid, outcome_type, plan_id, draft_id,
                 application_source_event_id, intent, reply_text, processed_at)
                VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?)
                """,
                (
                    source_event_id,
                    chat_guid,
                    outcome_type,
                    persisted_plan.plan_id,
                    draft_id,
                    intent,
                    reply_text,
                    _timestamp_text(timestamp),
                ),
            )
            connection.commit()
        except MealReportApplicationError:
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise MealReportApplicationError("unable to persist meal report message event") from exc
        return MealReportMessageEvent(
            source_event_id,
            chat_guid,
            outcome_type,
            persisted_plan.plan_id,
            draft_id,
            None,
            intent,
            reply_text,
            timestamp,
        )

    def record_pre_route_meal_report_outcome(
        self,
        *,
        source_event_id: str,
        chat_guid: str,
        outcome_type: Literal[
            "ambiguous_meal_context",
            "unavailable_meal_slot",
            "no_reportable_context",
            "routing_failure",
        ],
        reply_text: str,
        processed_at: datetime | None = None,
    ) -> MealReportMessageEvent:
        """Persist a deterministic plan-free reply before transport delivery."""

        _text(source_event_id, "source_event_id")
        _text(chat_guid, "chat_guid")
        if outcome_type not in {
            "ambiguous_meal_context",
            "unavailable_meal_slot",
            "no_reportable_context",
            "routing_failure",
        }:
            raise ValueError("pre-route outcome type is invalid")
        _text(reply_text, "reply_text")
        timestamp = _utc_now() if processed_at is None else _timestamp(processed_at, "processed_at")
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._message_event_row(connection, source_event_id)
            if existing is not None:
                event = self._meal_report_message_event_from_row(existing)
                if event.chat_guid != chat_guid:
                    raise MealReportApplicationError(
                        "source_event_id belongs to another conversation"
                    )
                connection.rollback()
                return event
            connection.execute(
                """
                INSERT INTO meal_report_message_events
                (source_event_id, chat_guid, outcome_type, plan_id, draft_id,
                 application_source_event_id, intent, reply_text, processed_at)
                VALUES (?, ?, ?, NULL, NULL, NULL, 'unsupported_or_ambiguous', ?, ?)
                """,
                (
                    source_event_id,
                    chat_guid,
                    outcome_type,
                    reply_text,
                    _timestamp_text(timestamp),
                ),
            )
            connection.commit()
        except MealReportApplicationError:
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise MealReportApplicationError(
                "unable to persist pre-route meal report outcome"
            ) from exc
        return MealReportMessageEvent(
            source_event_id,
            chat_guid,
            outcome_type,
            None,
            None,
            None,
            "unsupported_or_ambiguous",
            reply_text,
            timestamp,
        )

    def save_meal_report_draft(
        self,
        *,
        chat_guid: str,
        persisted_plan: PersistedMealPlan,
        current_draft: MealReportDraft | None,
        planned_items: tuple[DraftPlannedItem, ...],
        unplanned_items: tuple[DraftUnplannedItem, ...],
        clarifications: tuple[DraftClarification, ...],
        last_prompt: str,
        source_event_id: str,
        intent: MealReportIntent,
        processed_at: datetime | None = None,
    ) -> tuple[MealReportDraft, MealReportMessageEvent]:
        """Create or revision-guardedly update one draft and record its GUID.

        The draft facts, clarification state, and source GUID commit together.
        A stale concurrent update raises instead of silently overwriting another
        message's accumulated report facts.
        """

        _text(chat_guid, "chat_guid")
        if not isinstance(persisted_plan, PersistedMealPlan):
            raise TypeError("persisted_plan must be a PersistedMealPlan")
        if current_draft is not None:
            if not isinstance(current_draft, MealReportDraft):
                raise TypeError("current_draft must be a MealReportDraft or None")
            if current_draft.chat_guid != chat_guid or current_draft.plan_id != persisted_plan.plan_id:
                raise MealReportApplicationError("draft does not belong to this chat and plan")
            if current_draft.status not in {"draft", "awaiting_clarification"}:
                raise MealReportApplicationError("draft is no longer open")
        if not isinstance(planned_items, tuple) or not all(
            isinstance(item, DraftPlannedItem) for item in planned_items
        ):
            raise TypeError("planned_items must be DraftPlannedItem values")
        if not isinstance(unplanned_items, tuple) or not all(
            isinstance(item, DraftUnplannedItem) for item in unplanned_items
        ):
            raise TypeError("unplanned_items must be DraftUnplannedItem values")
        if not isinstance(clarifications, tuple) or not all(
            isinstance(item, DraftClarification) for item in clarifications
        ):
            raise TypeError("clarifications must be DraftClarification values")
        if not clarifications:
            raise ValueError("an open draft must retain at least one clarification or completion prompt")
        _text(last_prompt, "last_prompt")
        _text(source_event_id, "source_event_id")
        _intent(intent)
        self._validate_draft_contents(persisted_plan, planned_items, unplanned_items, clarifications)
        timestamp = _utc_now() if processed_at is None else _timestamp(processed_at, "processed_at")
        identifier = current_draft.draft_id if current_draft is not None else str(uuid4())
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing_event = self._message_event_row(connection, source_event_id)
            if existing_event is not None:
                event = self._meal_report_message_event_from_row(existing_event)
                self._validate_existing_message_event(event, chat_guid, persisted_plan.plan_id)
                if event.draft_id is None:
                    raise MealReportDraftConflictError("message event has no report draft")
                existing_draft = self.load_meal_report_draft(event.draft_id)
                if existing_draft is None:
                    raise MealReportApplicationError("message event references a missing report draft")
                connection.rollback()
                return existing_draft, event
            lifecycle = connection.execute(
                "SELECT status FROM meal_plans WHERE plan_id = ?",
                (persisted_plan.plan_id,),
            ).fetchone()
            if lifecycle is None:
                raise MealReportApplicationError("draft plan is missing")
            if lifecycle["status"] != "active":
                raise MealReportApplicationError("draft plan is no longer active")
            if current_draft is None:
                competing = connection.execute(
                    """
                    SELECT draft_id FROM meal_report_drafts
                    WHERE chat_guid = ? AND plan_id = ?
                      AND status IN ('draft', 'awaiting_clarification')
                    """,
                    (chat_guid, persisted_plan.plan_id),
                ).fetchall()
                if competing:
                    raise MealReportDraftConflictError(
                        "another open report draft exists for this chat and plan"
                    )
                connection.execute(
                    """
                    INSERT INTO meal_report_drafts
                    (draft_id, chat_guid, plan_id, status, revision, last_prompt, created_at, updated_at)
                    VALUES (?, ?, ?, 'awaiting_clarification', 0, ?, ?, ?)
                    """,
                    (
                        identifier,
                        chat_guid,
                        persisted_plan.plan_id,
                        last_prompt,
                        _timestamp_text(timestamp),
                        _timestamp_text(timestamp),
                    ),
                )
                next_revision = 0
            else:
                updated = connection.execute(
                    """
                    UPDATE meal_report_drafts
                    SET status = 'awaiting_clarification', revision = revision + 1,
                        last_prompt = ?, updated_at = ?
                    WHERE draft_id = ? AND plan_id = ?
                      AND revision = ? AND status IN ('draft', 'awaiting_clarification')
                    """,
                    (
                        last_prompt,
                        _timestamp_text(timestamp),
                        identifier,
                        persisted_plan.plan_id,
                        current_draft.revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise MealReportDraftConflictError("meal report draft changed concurrently")
                next_revision = current_draft.revision + 1
                self._archive_corrected_draft_facts(
                    connection,
                    current_draft,
                    planned_items,
                    unplanned_items,
                    superseded_by_source_event_id=source_event_id,
                    superseded_at=timestamp,
                )
            for table_name in (
                "meal_report_draft_planned_items",
                "meal_report_draft_unplanned_items",
                "meal_report_draft_clarifications",
            ):
                connection.execute(f"DELETE FROM {table_name} WHERE draft_id = ?", (identifier,))
            self._insert_draft_contents(
                connection,
                identifier,
                planned_items,
                unplanned_items,
                clarifications,
            )
            connection.execute(
                """
                INSERT INTO meal_report_message_events
                (source_event_id, chat_guid, outcome_type, plan_id, draft_id,
                 application_source_event_id, intent, reply_text, processed_at)
                VALUES (?, ?, 'draft_update', ?, ?, NULL, ?, ?, ?)
                """,
                (
                    source_event_id,
                    chat_guid,
                    persisted_plan.plan_id,
                    identifier,
                    intent,
                    last_prompt,
                    _timestamp_text(timestamp),
                ),
            )
            connection.commit()
        except (MealReportApplicationError, MealReportDraftConflictError):
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise MealReportApplicationError("unable to persist meal report draft") from exc
        saved = self.load_meal_report_draft(identifier)
        if saved is None or saved.revision != next_revision:
            raise MealReportApplicationError("saved meal report draft disappeared")
        return saved, MealReportMessageEvent(
            source_event_id,
            chat_guid,
            "draft_update",
            persisted_plan.plan_id,
            identifier,
            None,
            intent,
            last_prompt,
            timestamp,
        )

    def cancel_meal_report_draft(
        self,
        draft: MealReportDraft,
        *,
        source_event_id: str,
        reply_text: str,
        processed_at: datetime | None = None,
    ) -> MealReportMessageEvent:
        """Cancel an obsolete open draft without attaching it to another plan."""

        if not isinstance(draft, MealReportDraft):
            raise TypeError("draft must be a MealReportDraft")
        if draft.status not in {"draft", "awaiting_clarification"}:
            raise MealReportApplicationError("only an open report draft can be cancelled")
        _text(source_event_id, "source_event_id")
        _text(reply_text, "reply_text")
        timestamp = _utc_now() if processed_at is None else _timestamp(processed_at, "processed_at")
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._message_event_row(connection, source_event_id)
            if existing is not None:
                event = self._meal_report_message_event_from_row(existing)
                self._validate_existing_message_event(event, draft.chat_guid, draft.plan_id)
                connection.rollback()
                return event
            cancelled = connection.execute(
                """
                UPDATE meal_report_drafts
                SET status = 'cancelled', revision = revision + 1, updated_at = ?, cancelled_at = ?
                WHERE draft_id = ? AND revision = ?
                  AND status IN ('draft', 'awaiting_clarification')
                """,
                (
                    _timestamp_text(timestamp),
                    _timestamp_text(timestamp),
                    draft.draft_id,
                    draft.revision,
                ),
            )
            if cancelled.rowcount != 1:
                raise MealReportDraftConflictError("meal report draft changed concurrently")
            connection.execute(
                """
                INSERT INTO meal_report_message_events
                (source_event_id, chat_guid, outcome_type, plan_id, draft_id,
                 application_source_event_id, intent, reply_text, processed_at)
                VALUES (?, ?, 'draft_cancelled', ?, ?, NULL,
                        'unsupported_or_ambiguous', ?, ?)
                """,
                (
                    source_event_id,
                    draft.chat_guid,
                    draft.plan_id,
                    draft.draft_id,
                    reply_text,
                    _timestamp_text(timestamp),
                ),
            )
            connection.commit()
        except (MealReportApplicationError, MealReportDraftConflictError):
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise MealReportApplicationError("unable to cancel meal report draft") from exc
        return MealReportMessageEvent(
            source_event_id,
            draft.chat_guid,
            "draft_cancelled",
            draft.plan_id,
            draft.draft_id,
            None,
            "unsupported_or_ambiguous",
            reply_text,
            timestamp,
        )

    def _archive_corrected_draft_facts(
        self,
        connection: sqlite3.Connection,
        current_draft: MealReportDraft,
        planned_items: tuple[DraftPlannedItem, ...],
        unplanned_items: tuple[DraftUnplannedItem, ...],
        *,
        superseded_by_source_event_id: str,
        superseded_at: datetime,
    ) -> None:
        """Archive only facts explicitly replaced by a higher fact revision."""

        planned_by_id = {item.plan_item_id: item for item in planned_items}
        for prior in current_draft.planned_items:
            replacement = planned_by_id.get(prior.plan_item_id)
            if replacement is None or replacement.fact_revision <= prior.fact_revision:
                continue
            connection.execute(
                """
                INSERT INTO meal_report_draft_planned_item_history
                (draft_id, plan_item_id, fact_revision, action, official_servings,
                 quantity_source, original_reference_text, original_quantity_text,
                 quantity_status, unresolved_reason, last_source_event_id,
                 superseded_by_source_event_id, superseded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    current_draft.draft_id,
                    prior.plan_item_id,
                    prior.fact_revision,
                    prior.action,
                    None if prior.official_servings is None else str(prior.official_servings),
                    prior.quantity_source,
                    prior.original_reference_text,
                    prior.original_quantity_text,
                    prior.quantity_status,
                    prior.unresolved_reason,
                    prior.last_source_event_id,
                    superseded_by_source_event_id,
                    _timestamp_text(superseded_at),
                ),
            )
        unplanned_by_id = {item.fact_id: item for item in unplanned_items}
        for position, prior in enumerate(current_draft.unplanned_items):
            replacement = unplanned_by_id.get(prior.fact_id)
            if replacement is None or replacement.fact_revision <= prior.fact_revision:
                continue
            connection.execute(
                """
                INSERT INTO meal_report_draft_unplanned_item_history
                (draft_id, fact_id, fact_revision, item_position, action, food_text,
                 quantity_text, identity_status, identity_unresolved_reason,
                 occurrence_id, nutrition_snapshot_id, source_kind, source_value,
                 content_signature, quantity_status, official_servings,
                 quantity_source, quantity_unresolved_reason, confidence,
                 last_source_event_id, superseded_by_source_event_id, superseded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    current_draft.draft_id,
                    prior.fact_id,
                    prior.fact_revision,
                    position,
                    "eaten",
                    prior.food_text,
                    prior.quantity_text,
                    prior.identity_status,
                    prior.identity_unresolved_reason,
                    None if prior.food is None else prior.food.occurrence.occurrence_id,
                    None if prior.food is None else prior.food.nutrition_snapshot_id,
                    None if prior.food is None else prior.food.source_identifier.kind,
                    None if prior.food is None else prior.food.source_identifier.value,
                    None if prior.food is None else prior.food.content_signature,
                    prior.quantity_status,
                    None if prior.official_servings is None else str(prior.official_servings),
                    prior.quantity_source,
                    prior.quantity_unresolved_reason,
                    prior.confidence,
                    prior.last_source_event_id,
                    superseded_by_source_event_id,
                    _timestamp_text(superseded_at),
                ),
            )

    def apply_draft_reconciled_meal_report(
        self,
        draft: MealReportDraft,
        report: ReconciledMealReport,
        *,
        source_event_id: str,
        reply_text: str,
        intent: Literal["meal_report", "clarification_answer"],
        effective_planned_items: tuple[DraftPlannedItem, ...] | None = None,
        effective_unplanned_items: tuple[DraftUnplannedItem, ...] | None = None,
        application_clock: NutritionApplicationClock | None = None,
        service_calendar: PhelpsServiceCalendar | None = None,
        unavailable_reply_text: str = (
            "That earlier recommendation is no longer reportable. I haven't logged anything."
        ),
        recorded_at: datetime | None = None,
    ) -> AppliedMealReport:
        """Atomically apply one complete draft, close it, and record its GUID.

        Accepted intake rows, the legacy successful-application idempotency
        record, plan lifecycle, final draft state, and user-facing event reply
        are one SQLite transaction.  A confirmation transport failure happens
        after this method, so replay can resend the persisted reply without
        duplicating intake.
        """

        if not isinstance(draft, MealReportDraft):
            raise TypeError("draft must be a MealReportDraft")
        if draft.status not in {"draft", "awaiting_clarification"}:
            raise MealReportApplicationError("meal report draft is no longer open")
        if not isinstance(report, ReconciledMealReport):
            raise TypeError("report must be a ReconciledMealReport")
        _text(source_event_id, "source_event_id")
        _text(reply_text, "reply_text")
        if intent not in {"meal_report", "clarification_answer"}:
            raise ValueError("completed draft intent is invalid")
        if application_clock is not None and not isinstance(
            application_clock, NutritionApplicationClock
        ):
            raise TypeError("application_clock must be a NutritionApplicationClock or None")
        if service_calendar is not None and not isinstance(
            service_calendar, PhelpsServiceCalendar
        ):
            raise TypeError("service_calendar must be a PhelpsServiceCalendar or None")
        if application_clock is None and service_calendar is not None:
            raise ValueError("service_calendar requires application_clock")
        _text(unavailable_reply_text, "unavailable_reply_text")
        if report.clarification_items:
            raise MealReportApplicationError("completed draft still requires clarification")
        planned_items = (
            draft.planned_items
            if effective_planned_items is None
            else effective_planned_items
        )
        unplanned_items = (
            draft.unplanned_items if effective_unplanned_items is None else effective_unplanned_items
        )
        self._validate_draft_contents(
            draft.persisted_plan,
            planned_items,
            unplanned_items,
            (),
        )
        if any(item.quantity_status == "unresolved" for item in planned_items) or any(
            item.identity_status == "unresolved" or item.quantity_status == "unresolved"
            for item in unplanned_items
        ):
            raise MealReportApplicationError("completed draft contains unresolved effective facts")
        if any(
            item.quantity_source == "explicit_semantic_estimate"
            for item in (*planned_items, *unplanned_items)
        ):
            raise MealReportApplicationError(
                "semantic quantity estimates are not authoritative for intake"
            )
        if not _same_plan(draft.persisted_plan.plan, report.plan):
            raise MealReportApplicationError("completed draft report belongs to another plan")
        self._validate_draft_matches_report(
            draft,
            report,
            planned_items=planned_items,
            unplanned_items=unplanned_items,
        )
        timestamp = _utc_now() if recorded_at is None else _timestamp(recorded_at, "recorded_at")
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing_event = self._message_event_row(connection, source_event_id)
            if existing_event is not None:
                event = self._meal_report_message_event_from_row(existing_event)
                self._validate_existing_message_event(event, draft.chat_guid, draft.plan_id)
                applied = self.load_applied_meal_report(source_event_id)
                if applied is None:
                    raise MealReportApplicationError("message GUID belongs to a non-applied interaction")
                connection.rollback()
                return applied
            existing_application = connection.execute(
                "SELECT plan_id FROM meal_report_applications WHERE source_event_id = ?",
                (source_event_id,),
            ).fetchone()
            if existing_application is not None:
                if str(existing_application["plan_id"]) != draft.plan_id:
                    raise MealReportApplicationError("source_event_id belongs to another meal plan")
                applied = self.load_applied_meal_report(source_event_id)
                if applied is None:
                    raise MealReportApplicationError("meal report application is missing")
                connection.rollback()
                return applied
            draft_row = connection.execute(
                """
                SELECT plan_id, status, revision FROM meal_report_drafts
                WHERE draft_id = ?
                """,
                (draft.draft_id,),
            ).fetchone()
            if (
                draft_row is None
                or str(draft_row["plan_id"]) != draft.plan_id
                or str(draft_row["status"]) not in {"draft", "awaiting_clarification"}
                or int(draft_row["revision"]) != draft.revision
            ):
                raise MealReportDraftConflictError("meal report draft changed concurrently")
            current_draft = self.load_meal_report_draft(draft.draft_id)
            if current_draft != draft:
                raise MealReportDraftConflictError("meal report draft changed concurrently")
            current_plan = self.load_meal_plan(draft.plan_id)
            if current_plan is None:
                raise MealReportApplicationError("persisted meal plan is missing")
            local_now = None if application_clock is None else application_clock.now()
            if local_now is not None and not self._is_plan_reportable_for_final_application(
                current_plan,
                service_calendar,
                evaluated_at=local_now,
            ):
                if current_plan.plan.service_date != local_now.date():
                    connection.execute(
                        """
                        UPDATE meal_plans SET status = 'superseded'
                        WHERE plan_id = ? AND status = 'active'
                        """,
                        (draft.plan_id,),
                    )
                cancelled = connection.execute(
                    """
                    UPDATE meal_report_drafts
                    SET status = 'cancelled', revision = revision + 1,
                        updated_at = ?, cancelled_at = ?
                    WHERE draft_id = ? AND revision = ?
                      AND status IN ('draft', 'awaiting_clarification')
                    """,
                    (
                        _timestamp_text(timestamp),
                        _timestamp_text(timestamp),
                        draft.draft_id,
                        draft.revision,
                    ),
                )
                if cancelled.rowcount != 1:
                    raise MealReportDraftConflictError(
                        "meal report draft changed during commit-time rejection"
                    )
                connection.execute(
                    """
                    INSERT INTO meal_report_message_events
                    (source_event_id, chat_guid, outcome_type, plan_id, draft_id,
                     application_source_event_id, intent, reply_text, processed_at)
                    VALUES (?, ?, 'draft_cancelled', ?, ?, NULL,
                            'unsupported_or_ambiguous', ?, ?)
                    """,
                    (
                        source_event_id,
                        draft.chat_guid,
                        draft.plan_id,
                        draft.draft_id,
                        unavailable_reply_text,
                        _timestamp_text(timestamp),
                    ),
                )
                connection.commit()
                raise MealReportCommitTimeRejection(
                    MealReportMessageEvent(
                        source_event_id,
                        draft.chat_guid,
                        "draft_cancelled",
                        draft.plan_id,
                        draft.draft_id,
                        None,
                        "unsupported_or_ambiguous",
                        unavailable_reply_text,
                        timestamp,
                    )
                )
            self._validate_draft_contents(
                current_plan,
                planned_items,
                unplanned_items,
                (),
            )
            if any(item.quantity_status == "unresolved" for item in planned_items) or any(
                item.identity_status == "unresolved" or item.quantity_status == "unresolved"
                for item in unplanned_items
            ):
                raise MealReportApplicationError(
                    "completed draft contains unresolved effective facts"
                )
            if any(
                item.quantity_source == "explicit_semantic_estimate"
                for item in (*planned_items, *unplanned_items)
            ):
                raise MealReportApplicationError(
                    "semantic quantity estimates are not authoritative for intake"
                )
            for item in current_plan.plan.items:
                self._validate_plan_item_link(item)
            self._archive_corrected_draft_facts(
                connection,
                draft,
                planned_items,
                unplanned_items,
                superseded_by_source_event_id=source_event_id,
                superseded_at=timestamp,
            )
            for table_name in (
                "meal_report_draft_planned_items",
                "meal_report_draft_unplanned_items",
                "meal_report_draft_clarifications",
            ):
                connection.execute(
                    f"DELETE FROM {table_name} WHERE draft_id = ?",
                    (draft.draft_id,),
                )
            self._insert_draft_contents(
                connection,
                draft.draft_id,
                planned_items,
                unplanned_items,
                (),
            )
            proposed = self._accepted_items(draft.persisted_plan, report)
            connection.execute(
                """
                INSERT INTO meal_report_applications (source_event_id, plan_id, applied_at)
                VALUES (?, ?, ?)
                """,
                (source_event_id, draft.plan_id, _timestamp_text(timestamp)),
            )
            for position, item in enumerate(proposed):
                self._insert_intake(item, source_event_id, timestamp, position)
            completed_plan = connection.execute(
                """
                UPDATE meal_plans
                SET status = 'applied'
                WHERE plan_id = ? AND status = 'active'
                """,
                (draft.plan_id,),
            )
            if completed_plan.rowcount != 1:
                raise MealReportApplicationError("meal plan could not be completed")
            completed_draft = connection.execute(
                """
                UPDATE meal_report_drafts
                SET status = 'completed', revision = revision + 1, updated_at = ?, completed_at = ?
                WHERE draft_id = ? AND revision = ?
                  AND status IN ('draft', 'awaiting_clarification')
                """,
                (
                    _timestamp_text(timestamp),
                    _timestamp_text(timestamp),
                    draft.draft_id,
                    draft.revision,
                ),
            )
            if completed_draft.rowcount != 1:
                raise MealReportDraftConflictError("meal report draft could not be completed")
            connection.execute(
                """
                INSERT INTO meal_report_message_events
                (source_event_id, chat_guid, outcome_type, plan_id, draft_id,
                 application_source_event_id, intent, reply_text, processed_at)
                VALUES (?, ?, 'final_application', ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_event_id,
                    draft.chat_guid,
                    draft.plan_id,
                    draft.draft_id,
                    source_event_id,
                    intent,
                    reply_text,
                    _timestamp_text(timestamp),
                ),
            )
            connection.commit()
        except MealReportCommitTimeRejection:
            raise
        except (MealReportApplicationError, MealReportDraftConflictError):
            connection.rollback()
            raise
        except (sqlite3.Error, NutritionCatalogError) as exc:
            connection.rollback()
            raise MealReportApplicationError("meal report draft application failed atomically") from exc
        return AppliedMealReport(
            draft.plan_id,
            source_event_id,
            False,
            self._load_event_intake(source_event_id),
            report.skipped_items,
        )

    def apply_reconciled_meal_report(
        self,
        persisted_plan: PersistedMealPlan,
        report: ReconciledMealReport,
        *,
        source_event_id: str,
        application_clock: NutritionApplicationClock | None = None,
        service_calendar: PhelpsServiceCalendar | None = None,
        chat_guid: str | None = None,
        reply_text: str | None = None,
        intent: Literal["meal_report", "clarification_answer"] = "meal_report",
        unavailable_reply_text: str | None = None,
        recorded_at: datetime | None = None,
    ) -> AppliedMealReport:
        """Atomically persist accepted intake and complete one active plan.

        Existing source-event applications are intentionally checked before
        lifecycle eligibility.  That ordering makes an exact transport replay
        idempotent even after the original application has completed the plan.
        """

        if not isinstance(persisted_plan, PersistedMealPlan):
            raise TypeError("persisted_plan must be a PersistedMealPlan")
        if not isinstance(report, ReconciledMealReport):
            raise TypeError("report must be a ReconciledMealReport")
        _text(source_event_id, "source_event_id")
        if application_clock is not None and not isinstance(
            application_clock, NutritionApplicationClock
        ):
            raise TypeError("application_clock must be a NutritionApplicationClock or None")
        if service_calendar is not None and not isinstance(
            service_calendar, PhelpsServiceCalendar
        ):
            raise TypeError("service_calendar must be a PhelpsServiceCalendar or None")
        if application_clock is None and service_calendar is not None:
            raise ValueError("service_calendar requires application_clock")
        message_event_requested = chat_guid is not None or reply_text is not None
        if message_event_requested:
            _text(chat_guid, "chat_guid")
            _text(reply_text, "reply_text")
            if unavailable_reply_text is None:
                raise ValueError("conversation application requires unavailable_reply_text")
            _text(unavailable_reply_text, "unavailable_reply_text")
        elif unavailable_reply_text is not None:
            raise ValueError("unavailable_reply_text requires conversation message context")
        if intent not in {"meal_report", "clarification_answer"}:
            raise ValueError("application message intent is invalid")
        if not _same_plan(persisted_plan.plan, report.plan):
            raise MealReportApplicationError("reconciled report does not belong to persisted plan")
        timestamp = _utc_now() if recorded_at is None else _timestamp(recorded_at, "recorded_at")
        connection = self._connection()
        existing = connection.execute(
            "SELECT plan_id FROM meal_report_applications WHERE source_event_id = ?",
            (source_event_id,),
        ).fetchone()
        if existing is not None:
            if existing["plan_id"] != persisted_plan.plan_id:
                raise MealReportApplicationError("source_event_id belongs to another meal plan")
            return AppliedMealReport(
                persisted_plan.plan_id,
                source_event_id,
                True,
                self._load_event_intake(source_event_id),
                report.skipped_items,
            )

        if report.clarification_items:
            raise MealReportApplicationError("meal report still requires clarification")
        if persisted_plan.status != "active":
            raise MealReportApplicationError("meal plan is no longer active")
        for item in persisted_plan.plan.items:
            self._validate_plan_item_link(item)
        proposed = self._accepted_items(persisted_plan, report)
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT plan_id FROM meal_report_applications WHERE source_event_id = ?",
                    (source_event_id,),
                ).fetchone()
                if existing is not None:
                    if existing["plan_id"] != persisted_plan.plan_id:
                        raise MealReportApplicationError(
                            "source_event_id belongs to another meal plan"
                        )
                    return AppliedMealReport(
                        persisted_plan.plan_id,
                        source_event_id,
                        True,
                        self._load_event_intake(source_event_id),
                        report.skipped_items,
                    )
                current_plan = self.load_meal_plan(persisted_plan.plan_id)
                if current_plan is None:
                    raise MealReportApplicationError("persisted meal plan is missing")
                if current_plan.status != "active":
                    raise MealReportApplicationError("meal plan is no longer active")
                if application_clock is not None:
                    local_now = application_clock.now()
                    if not self._is_plan_reportable_for_final_application(
                        current_plan,
                        service_calendar,
                        evaluated_at=local_now,
                    ):
                        if current_plan.plan.service_date != local_now.date():
                            connection.execute(
                                """
                                UPDATE meal_plans SET status = 'superseded'
                                WHERE plan_id = ? AND status = 'active'
                                """,
                                (persisted_plan.plan_id,),
                            )
                        if message_event_requested:
                            assert chat_guid is not None
                            assert unavailable_reply_text is not None
                            connection.execute(
                                """
                                INSERT INTO meal_report_message_events
                                (source_event_id, chat_guid, outcome_type, plan_id, draft_id,
                                 application_source_event_id, intent, reply_text, processed_at)
                                VALUES (?, ?, 'routed_interaction', ?, NULL, NULL,
                                        'unsupported_or_ambiguous', ?, ?)
                                """,
                                (
                                    source_event_id,
                                    chat_guid,
                                    persisted_plan.plan_id,
                                    unavailable_reply_text,
                                    _timestamp_text(timestamp),
                                ),
                            )
                            connection.commit()
                            raise MealReportCommitTimeRejection(
                                MealReportMessageEvent(
                                    source_event_id,
                                    chat_guid,
                                    "routed_interaction",
                                    persisted_plan.plan_id,
                                    None,
                                    None,
                                    "unsupported_or_ambiguous",
                                    unavailable_reply_text,
                                    timestamp,
                                )
                            )
                        raise MealReportApplicationError(
                            "meal plan is no longer reportable at final commit"
                        )
                if not _same_plan(current_plan.plan, report.plan):
                    raise MealReportApplicationError(
                        "reconciled report no longer matches persisted plan"
                    )
                if report.clarification_items:
                    raise MealReportApplicationError(
                        "meal report still requires clarification"
                    )
                proposed = self._accepted_items(current_plan, report)
                try:
                    connection.execute(
                        "INSERT INTO meal_report_applications (source_event_id, plan_id, applied_at) VALUES (?, ?, ?)",
                        (source_event_id, persisted_plan.plan_id, _timestamp_text(timestamp)),
                    )
                except sqlite3.IntegrityError:
                    row = connection.execute(
                        "SELECT plan_id FROM meal_report_applications WHERE source_event_id = ?",
                        (source_event_id,),
                    ).fetchone()
                    if row is None or row["plan_id"] != persisted_plan.plan_id:
                        raise MealReportApplicationError("source_event_id belongs to another meal plan")
                    return AppliedMealReport(
                        persisted_plan.plan_id,
                        source_event_id,
                        True,
                        self._load_event_intake(source_event_id),
                        report.skipped_items,
                    )
                for position, item in enumerate(proposed):
                    self._insert_intake(item, source_event_id, timestamp, position)
                completed = connection.execute(
                    """
                    UPDATE meal_plans
                    SET status = 'applied'
                    WHERE plan_id = ? AND status = 'active'
                    """,
                    (persisted_plan.plan_id,),
                )
                if completed.rowcount != 1:
                    raise MealReportApplicationError("meal plan could not be completed")
                if message_event_requested:
                    assert chat_guid is not None and reply_text is not None
                    connection.execute(
                        """
                        INSERT INTO meal_report_message_events
                        (source_event_id, chat_guid, outcome_type, plan_id, draft_id,
                         application_source_event_id, intent, reply_text, processed_at)
                        VALUES (?, ?, 'final_application', ?, NULL, ?, ?, ?, ?)
                        """,
                        (
                            source_event_id,
                            chat_guid,
                            persisted_plan.plan_id,
                            source_event_id,
                            intent,
                            reply_text,
                            _timestamp_text(timestamp),
                        ),
                    )
        except MealReportCommitTimeRejection:
            raise
        except MealReportApplicationError:
            raise
        except (sqlite3.Error, NutritionCatalogError) as exc:
            raise MealReportApplicationError("meal report application failed atomically") from exc
        return AppliedMealReport(
            persisted_plan.plan_id,
            source_event_id,
            False,
            self._load_event_intake(source_event_id),
            report.skipped_items,
        )

    def load_daily_ledger(self, service_date: date) -> DailyLedger:
        """Reconstruct meal intake only, used for recommendation accounting."""

        _date(service_date, "service_date")
        entries = tuple(entry.as_ledger_entry() for entry in self.get_daily_intake(service_date))
        return DailyLedger(entries)

    def load_recommendation_ledger(self, service_date: date) -> DailyLedger:
        """Return only meal intake for the optimizer's remaining-target calculation."""

        return self.load_daily_ledger(service_date)

    def claim_shake_reminder(
        self,
        plan_id: str,
        chat_guid: str,
        *,
        claimed_at: datetime | None = None,
    ) -> bool:
        """Claim the one send attempt for an applied breakfast/dinner plan.

        The claim commits before transport delivery. The outbound transport
        has no idempotency key, so an uncertain send cannot safely be retried.
        """

        _text(plan_id, "plan_id")
        _text(chat_guid, "chat_guid")
        timestamp = claimed_at or datetime.now(timezone.utc)
        _timestamp(timestamp, "claimed_at")
        connection = self._connection()
        with connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO shake_intake
                    (local_date, shake_slot, chat_guid, reminder_claimed_at, status)
                SELECT service_date, meal_slot || '_shake', ?, ?, 'pending'
                FROM meal_plans
                WHERE plan_id = ? AND status = 'applied'
                  AND meal_slot IN ('breakfast', 'dinner')
                """,
                (chat_guid, _timestamp_text(timestamp), plan_id),
            )
        return cursor.rowcount == 1

    def mark_shake_reminder_delivered(
        self, plan_id: str, chat_guid: str, *, delivered_at: datetime | None = None
    ) -> None:
        """Record a successful transport return for a previously claimed send."""

        _text(plan_id, "plan_id")
        _text(chat_guid, "chat_guid")
        timestamp = delivered_at or datetime.now(timezone.utc)
        _timestamp(timestamp, "delivered_at")
        connection = self._connection()
        with connection:
            connection.execute(
                """
                UPDATE shake_intake SET reminder_delivered_at = ?
                WHERE chat_guid = ? AND reminder_delivered_at IS NULL
                  AND (local_date, shake_slot) IN (
                    SELECT service_date, meal_slot || '_shake' FROM meal_plans
                    WHERE plan_id = ? AND status = 'applied'
                  )
                """,
                (_timestamp_text(timestamp), chat_guid, plan_id),
            )

    def pending_shakes(self, local_date: date, chat_guid: str) -> tuple[str, ...]:
        """Return only today's unresolved shake contexts for this chat."""

        _date(local_date, "local_date")
        _text(chat_guid, "chat_guid")
        rows = self._connection().execute(
            """
            SELECT shake_slot FROM shake_intake
            WHERE local_date = ? AND chat_guid = ? AND status = 'pending'
            ORDER BY shake_slot
            """,
            (local_date.isoformat(), chat_guid),
        ).fetchall()
        return tuple(str(row["shake_slot"]) for row in rows)

    def load_shake(self, local_date: date, shake_slot: str) -> sqlite3.Row | None:
        """Read one independently recorded shake context and its frozen estimate."""

        _date(local_date, "local_date")
        if shake_slot not in {"breakfast_shake", "dinner_shake"}:
            raise ValueError("invalid shake slot")
        return self._connection().execute(
            "SELECT * FROM shake_intake WHERE local_date = ? AND shake_slot = ?",
            (local_date.isoformat(), shake_slot),
        ).fetchone()

    def load_shake_for_source_event(self, source_event_id: str) -> sqlite3.Row | None:
        _text(source_event_id, "source_event_id")
        return self._connection().execute(
            "SELECT * FROM shake_intake WHERE resolution_source_event_id = ?",
            (source_event_id,),
        ).fetchone()

    def resolve_shake(
        self,
        local_date: date,
        shake_slot: str,
        chat_guid: str,
        source_event_id: str,
        *,
        confirmed: bool,
        resolved_at: datetime | None = None,
    ) -> sqlite3.Row:
        """Atomically resolve a pending shake once, freezing nutrition if drunk."""

        _date(local_date, "local_date")
        _text(chat_guid, "chat_guid")
        _text(source_event_id, "source_event_id")
        if shake_slot not in {"breakfast_shake", "dinner_shake"}:
            raise ValueError("invalid shake slot")
        timestamp = resolved_at or datetime.now(timezone.utc)
        _timestamp(timestamp, "resolved_at")
        connection = self._connection()
        with connection:
            replay = self.load_shake_for_source_event(source_event_id)
            if replay is not None:
                if replay["chat_guid"] != chat_guid:
                    raise MealReportApplicationError("shake source event belongs to another chat")
                return replay
            values = (
                "confirmed" if confirmed else "skipped",
                _timestamp_text(timestamp),
                source_event_id,
                "Serious Mass Chocolate shake" if confirmed else None,
                "580" if confirmed else None,
                "23" if confirmed else None,
                "116" if confirmed else None,
                "3" if confirmed else None,
                local_date.isoformat(),
                shake_slot,
                chat_guid,
            )
            cursor = connection.execute(
                """
                UPDATE shake_intake SET status = ?, resolved_at = ?,
                    resolution_source_event_id = ?, product_name = ?,
                    calories_kcal = ?, protein_g = ?, carbohydrates_g = ?, fat_g = ?
                WHERE local_date = ? AND shake_slot = ? AND chat_guid = ?
                  AND status = 'pending'
                """,
                values,
            )
            if cursor.rowcount != 1:
                raise MealReportApplicationError("shake is no longer pending")
        row = self.load_shake(local_date, shake_slot)
        assert row is not None
        return row

    def load_actual_daily_nutrients(self, service_date: date) -> NutrientProfile:
        """Return meal intake plus confirmed shakes; unestimated nutrients stay unknown."""

        meal_total = self.load_recommendation_ledger(service_date).total_consumed_nutrients
        rows = self._connection().execute(
            """
            SELECT calories_kcal, protein_g, carbohydrates_g, fat_g
            FROM shake_intake WHERE local_date = ? AND status = 'confirmed'
            ORDER BY shake_slot
            """,
            (service_date.isoformat(),),
        ).fetchall()
        profiles = tuple(
            NutrientProfile(
                calories_kcal=Decimal(row["calories_kcal"]),
                protein_g=Decimal(row["protein_g"]),
                carbohydrates_g=Decimal(row["carbohydrates_g"]),
                fat_g=Decimal(row["fat_g"]),
            )
            for row in rows
        )
        return add_nutrients(meal_total, *profiles)

    def get_daily_intake(self, service_date: date) -> tuple[AcceptedIntakeEntry, ...]:
        """Load accepted intake in recorded order without recalculating nutrition."""

        _date(service_date, "service_date")
        rows = self._connection().execute(
            """
            SELECT intake_id, source_event_id, service_date, recorded_at, meal,
                   plan_id, plan_item_id, occurrence_id, nutrition_snapshot_id,
                   source_kind, source_value, content_signature, official_servings,
                   quantity_source, original_reference_text, original_quantity_text
            FROM accepted_intake_entries
            WHERE service_date = ?
            ORDER BY recorded_at, item_position, intake_id
            """,
            (service_date.isoformat(),),
        ).fetchall()
        try:
            return tuple(self._intake_from_row(row) for row in rows)
        except (TypeError, ValueError, NutritionCatalogError):
            raise MealReportApplicationError("persisted intake is corrupt") from None

    def calculate_daily_balance(self, service_date: date, targets: DailyTargets) -> DailyBalance:
        """Apply caller-provided targets to actual meal and confirmed shake intake."""

        if not isinstance(targets, DailyTargets):
            raise TypeError("targets must be a DailyTargets")
        return calculate_daily_balance(targets, self.load_actual_daily_nutrients(service_date))

    def _accepted_items(
        self,
        persisted_plan: PersistedMealPlan,
        report: ReconciledMealReport,
    ) -> tuple[tuple[ResolvedFood, Decimal, IntakeOrigin, str, str | None, str | None], ...]:
        accepted: list[tuple[ResolvedFood, Decimal, IntakeOrigin, str, str | None, str | None]] = []
        for eaten in report.eaten_items:
            if eaten.quantity_source == "explicit_semantic_estimate":
                raise MealReportApplicationError(
                    "semantic quantity estimates are not authoritative for intake"
                )
            self._validate_plan_item_link(eaten.plan_item)
            accepted.append(
                (
                    eaten.food,
                    eaten.official_servings,
                    eaten.quantity_source,
                    persisted_plan.plan.item_id(eaten.plan_item),
                    eaten.original_user_phrase,
                    None,
                )
            )
        for unplanned in report.unplanned_items:
            if unplanned.resolved_food is None or unplanned.official_servings is None:
                raise MealReportApplicationError("unplanned intake is not fully resolved")
            assert unplanned.quantity_source is not None
            if unplanned.quantity_source == "explicit_semantic_estimate":
                raise MealReportApplicationError(
                    "semantic quantity estimates are not authoritative for intake"
                )
            self._validate_food_link(unplanned.resolved_food)
            accepted.append(
                (
                    unplanned.resolved_food,
                    unplanned.official_servings,
                    unplanned.quantity_source,
                    None,
                    unplanned.food_text,
                    unplanned.quantity_text,
                )
            )
        return tuple(accepted)

    def _insert_intake(
        self,
        proposed: tuple[ResolvedFood, Decimal, IntakeOrigin, str | None, str | None, str | None],
        source_event_id: str,
        recorded_at: datetime,
        position: int,
    ) -> None:
        food, servings, source, plan_item_id, reference_text, quantity_text = proposed
        _positive_decimal(servings, "official_servings")
        self._validate_food_link(food)
        self._connection().execute(
            """
            INSERT INTO accepted_intake_entries
            (intake_id, source_event_id, service_date, recorded_at, meal, plan_id,
             plan_item_id, occurrence_id, nutrition_snapshot_id, source_kind,
             source_value, content_signature, official_servings, quantity_source,
             original_reference_text, original_quantity_text, item_position)
            SELECT ?, ?, ?, ?, ?, application.plan_id, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            FROM meal_report_applications AS application
            WHERE application.source_event_id = ?
            """,
            (
                str(uuid4()),
                source_event_id,
                food.occurrence.service_date.isoformat(),
                _timestamp_text(recorded_at),
                food.occurrence.meal_period_name,
                plan_item_id,
                food.occurrence.occurrence_id,
                food.nutrition_snapshot_id,
                food.source_identifier.kind,
                food.source_identifier.value,
                food.content_signature,
                str(servings),
                source,
                reference_text,
                quantity_text,
                position,
                source_event_id,
            ),
        )

    def _message_event_row(
        self,
        connection: sqlite3.Connection,
        source_event_id: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT source_event_id, chat_guid, outcome_type, plan_id, draft_id,
                   application_source_event_id, intent, reply_text, processed_at
            FROM meal_report_message_events
            WHERE source_event_id = ?
            """,
            (source_event_id,),
        ).fetchone()

    def _meal_report_message_event_from_row(
        self,
        row: sqlite3.Row,
    ) -> MealReportMessageEvent:
        try:
            return MealReportMessageEvent(
                str(row["source_event_id"]),
                str(row["chat_guid"]),
                str(row["outcome_type"]),  # type: ignore[arg-type]
                None if row["plan_id"] is None else str(row["plan_id"]),
                None if row["draft_id"] is None else str(row["draft_id"]),
                None
                if row["application_source_event_id"] is None
                else str(row["application_source_event_id"]),
                str(row["intent"]),  # type: ignore[arg-type]
                str(row["reply_text"]),
                _parse_timestamp(row["processed_at"], "processed_at"),
            )
        except (TypeError, ValueError):
            raise MealReportApplicationError("meal report message event is corrupt") from None

    def _validate_existing_message_event(
        self,
        event: MealReportMessageEvent,
        chat_guid: str,
        plan_id: str | None,
    ) -> None:
        if event.chat_guid != chat_guid or event.plan_id != plan_id:
            raise MealReportApplicationError("source_event_id belongs to another conversation")

    def _meal_report_draft_from_row(self, row: sqlite3.Row) -> MealReportDraft:
        try:
            draft_id = str(row["draft_id"])
            persisted_plan = self.load_meal_plan(str(row["plan_id"]))
            if persisted_plan is None:
                raise MealReportApplicationError("meal report draft references a missing plan")
            planned_rows = self._connection().execute(
                """
                SELECT plan_item_id, action, official_servings, quantity_source,
                       original_reference_text, original_quantity_text,
                       quantity_status, unresolved_reason, last_source_event_id,
                       fact_revision
                FROM meal_report_draft_planned_items
                WHERE draft_id = ?
                ORDER BY plan_item_id
                """,
                (draft_id,),
            ).fetchall()
            planned_items = tuple(
                DraftPlannedItem(
                    str(item_row["plan_item_id"]),
                    str(item_row["action"]),  # type: ignore[arg-type]
                    None
                    if item_row["official_servings"] is None
                    else _decimal(item_row["official_servings"], "draft official_servings"),
                    None
                    if item_row["quantity_source"] is None
                    else str(item_row["quantity_source"]),  # type: ignore[arg-type]
                    str(item_row["original_reference_text"]),
                    None
                    if item_row["original_quantity_text"] is None
                    else str(item_row["original_quantity_text"]),
                    str(item_row["quantity_status"]),  # type: ignore[arg-type]
                    None
                    if item_row["unresolved_reason"] is None
                    else str(item_row["unresolved_reason"]),
                    None
                    if item_row["last_source_event_id"] is None
                    else str(item_row["last_source_event_id"]),
                    int(item_row["fact_revision"]),
                )
                for item_row in planned_rows
            )
            unplanned_rows = self._connection().execute(
                """
                SELECT item_position, fact_id, action, food_text, quantity_text,
                       identity_status, identity_unresolved_reason, occurrence_id,
                       nutrition_snapshot_id, source_kind, source_value,
                       content_signature, quantity_status, official_servings,
                       quantity_source, quantity_unresolved_reason, confidence,
                       last_source_event_id, fact_revision
                FROM meal_report_draft_unplanned_items
                WHERE draft_id = ?
                ORDER BY item_position
                """,
                (draft_id,),
            ).fetchall()
            unplanned_items: list[DraftUnplannedItem] = []
            for position, item_row in enumerate(unplanned_rows):
                if int(item_row["item_position"]) != position:
                    raise MealReportApplicationError("draft unplanned item order is corrupt")
                food = (
                    self._historical_food(
                        occurrence_id=int(item_row["occurrence_id"]),
                        snapshot_id=int(item_row["nutrition_snapshot_id"]),
                        source_identifier=SourceIdentifier(
                            str(item_row["source_kind"]), str(item_row["source_value"])
                        ),
                        content_signature=str(item_row["content_signature"]),
                    )
                    if str(item_row["identity_status"]) == "resolved"
                    else None
                )
                unplanned_items.append(
                    DraftUnplannedItem(
                        str(item_row["food_text"]),
                        None if item_row["quantity_text"] is None else str(item_row["quantity_text"]),
                        food,
                        None
                        if item_row["official_servings"] is None
                        else _decimal(item_row["official_servings"], "draft official_servings"),
                        None
                        if item_row["quantity_source"] is None
                        else str(item_row["quantity_source"]),  # type: ignore[arg-type]
                        None if item_row["confidence"] is None else str(item_row["confidence"]),  # type: ignore[arg-type]
                        str(item_row["fact_id"]),
                        str(item_row["identity_status"]),  # type: ignore[arg-type]
                        None
                        if item_row["identity_unresolved_reason"] is None
                        else str(item_row["identity_unresolved_reason"]),
                        str(item_row["quantity_status"]),  # type: ignore[arg-type]
                        None
                        if item_row["quantity_unresolved_reason"] is None
                        else str(item_row["quantity_unresolved_reason"]),
                        None
                        if item_row["last_source_event_id"] is None
                        else str(item_row["last_source_event_id"]),
                        int(item_row["fact_revision"]),
                    )
                )
            clarification_rows = self._connection().execute(
                """
                SELECT clarification_position, reason, plan_item_id, food_text,
                       quantity_text
                FROM meal_report_draft_clarifications
                WHERE draft_id = ?
                ORDER BY clarification_position
                """,
                (draft_id,),
            ).fetchall()
            clarifications: list[DraftClarification] = []
            for position, clarification_row in enumerate(clarification_rows):
                if int(clarification_row["clarification_position"]) != position:
                    raise MealReportApplicationError("draft clarification order is corrupt")
                clarifications.append(
                    DraftClarification(
                        str(clarification_row["reason"]),
                        None
                        if clarification_row["plan_item_id"] is None
                        else str(clarification_row["plan_item_id"]),
                        None
                        if clarification_row["food_text"] is None
                        else str(clarification_row["food_text"]),
                        None
                        if clarification_row["quantity_text"] is None
                        else str(clarification_row["quantity_text"]),
                    )
                )
            draft = MealReportDraft(
                draft_id,
                str(row["chat_guid"]),
                persisted_plan,
                str(row["status"]),  # type: ignore[arg-type]
                int(row["revision"]),
                planned_items,
                tuple(unplanned_items),
                tuple(clarifications),
                None if row["last_prompt"] is None else str(row["last_prompt"]),
                _parse_timestamp(row["created_at"], "draft created_at"),
                _parse_timestamp(row["updated_at"], "draft updated_at"),
                _optional_timestamp(row["completed_at"], "draft completed_at"),
                _optional_timestamp(row["cancelled_at"], "draft cancelled_at"),
            )
            self._validate_draft_contents(
                persisted_plan,
                draft.planned_items,
                draft.unplanned_items,
                draft.clarifications,
            )
            return draft
        except (TypeError, ValueError, NutritionCatalogError, MealReportApplicationError):
            raise MealReportApplicationError("meal report draft is corrupt") from None

    def _validate_draft_contents(
        self,
        persisted_plan: PersistedMealPlan,
        planned_items: tuple[DraftPlannedItem, ...],
        unplanned_items: tuple[DraftUnplannedItem, ...],
        clarifications: tuple[DraftClarification, ...],
    ) -> None:
        seen: set[str] = set()
        for item in planned_items:
            if item.plan_item_id in seen or persisted_plan.plan.item_for_id(item.plan_item_id) is None:
                raise MealReportApplicationError("draft planned item does not belong to its plan")
            seen.add(item.plan_item_id)
        for item in unplanned_items:
            if item.food is not None:
                self._validate_food_link(item.food)
        for clarification in clarifications:
            if (
                clarification.plan_item_id is not None
                and persisted_plan.plan.item_for_id(clarification.plan_item_id) is None
            ):
                raise MealReportApplicationError("draft clarification does not belong to its plan")

    def _insert_draft_contents(
        self,
        connection: sqlite3.Connection,
        draft_id: str,
        planned_items: tuple[DraftPlannedItem, ...],
        unplanned_items: tuple[DraftUnplannedItem, ...],
        clarifications: tuple[DraftClarification, ...],
    ) -> None:
        for item in planned_items:
            connection.execute(
                """
                INSERT INTO meal_report_draft_planned_items
                (draft_id, plan_item_id, action, official_servings, quantity_source,
                 original_reference_text, original_quantity_text, quantity_status,
                 unresolved_reason, last_source_event_id, fact_revision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    draft_id,
                    item.plan_item_id,
                    item.action,
                    None if item.official_servings is None else str(item.official_servings),
                    item.quantity_source,
                    item.original_reference_text,
                    item.original_quantity_text,
                    item.quantity_status,
                    item.unresolved_reason,
                    item.last_source_event_id,
                    item.fact_revision,
                ),
            )
        for position, item in enumerate(unplanned_items):
            connection.execute(
                """
                INSERT INTO meal_report_draft_unplanned_items
                (draft_id, item_position, fact_id, action, food_text, quantity_text,
                 identity_status, identity_unresolved_reason, occurrence_id,
                 nutrition_snapshot_id, source_kind, source_value, content_signature,
                 quantity_status, official_servings, quantity_source,
                 quantity_unresolved_reason, confidence, last_source_event_id,
                 fact_revision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    draft_id,
                    position,
                    item.fact_id,
                    "eaten",
                    item.food_text,
                    item.quantity_text,
                    item.identity_status,
                    item.identity_unresolved_reason,
                    None if item.food is None else item.food.occurrence.occurrence_id,
                    None if item.food is None else item.food.nutrition_snapshot_id,
                    None if item.food is None else item.food.source_identifier.kind,
                    None if item.food is None else item.food.source_identifier.value,
                    None if item.food is None else item.food.content_signature,
                    item.quantity_status,
                    None if item.official_servings is None else str(item.official_servings),
                    item.quantity_source,
                    item.quantity_unresolved_reason,
                    item.confidence,
                    item.last_source_event_id,
                    item.fact_revision,
                ),
            )
        for position, clarification in enumerate(clarifications):
            connection.execute(
                """
                INSERT INTO meal_report_draft_clarifications
                (draft_id, clarification_position, reason, plan_item_id, food_text,
                 quantity_text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    draft_id,
                    position,
                    clarification.reason,
                    clarification.plan_item_id,
                    clarification.food_text,
                    clarification.quantity_text,
                ),
            )

    def _validate_draft_matches_report(
        self,
        draft: MealReportDraft,
        report: ReconciledMealReport,
        *,
        planned_items: tuple[DraftPlannedItem, ...] | None = None,
        unplanned_items: tuple[DraftUnplannedItem, ...] | None = None,
    ) -> None:
        actual: dict[str, tuple[str, Decimal | None, str | None]] = {}
        for eaten in report.eaten_items:
            item_id = report.plan.item_id(eaten.plan_item)
            actual[item_id] = ("eaten", eaten.official_servings, eaten.quantity_source)
        for skipped in report.skipped_items:
            item_id = report.plan.item_id(skipped.plan_item)
            actual[item_id] = ("skipped", None, None)
        effective_planned = draft.planned_items if planned_items is None else planned_items
        effective_unplanned = draft.unplanned_items if unplanned_items is None else unplanned_items
        expected = {
            item.plan_item_id: (item.action, item.official_servings, item.quantity_source)
            for item in effective_planned
        }
        if actual != expected:
            raise MealReportApplicationError("completed report does not match its draft")
        if len(report.unplanned_items) != len(effective_unplanned):
            raise MealReportApplicationError("completed report unplanned items do not match its draft")
        for actual_item, expected_item in zip(
            report.unplanned_items,
            effective_unplanned,
            strict=True,
        ):
            if (
                actual_item.resolved_food is None
                or actual_item.official_servings is None
                or actual_item.quantity_source is None
                or actual_item.food_text != expected_item.food_text
                or actual_item.quantity_text != expected_item.quantity_text
                or actual_item.official_servings != expected_item.official_servings
                or actual_item.quantity_source != expected_item.quantity_source
                or actual_item.confidence != expected_item.confidence
                or not _same_food(actual_item.resolved_food, expected_item.food)
            ):
                raise MealReportApplicationError("completed report unplanned items do not match its draft")

    def _load_event_intake(self, source_event_id: str) -> tuple[AcceptedIntakeEntry, ...]:
        rows = self._connection().execute(
            """
            SELECT intake_id, source_event_id, service_date, recorded_at, meal,
                   plan_id, plan_item_id, occurrence_id, nutrition_snapshot_id,
                   source_kind, source_value, content_signature, official_servings,
                   quantity_source, original_reference_text, original_quantity_text
            FROM accepted_intake_entries WHERE source_event_id = ?
            ORDER BY item_position, intake_id
            """,
            (source_event_id,),
        ).fetchall()
        return tuple(self._intake_from_row(row) for row in rows)

    def _intake_from_row(self, row: sqlite3.Row) -> AcceptedIntakeEntry:
        source = SourceIdentifier(str(row["source_kind"]), str(row["source_value"]))
        food = self._historical_food(
            occurrence_id=int(row["occurrence_id"]),
            snapshot_id=int(row["nutrition_snapshot_id"]),
            source_identifier=source,
            content_signature=str(row["content_signature"]),
        )
        return AcceptedIntakeEntry(
            str(row["intake_id"]),
            str(row["source_event_id"]),
            date.fromisoformat(str(row["service_date"])),
            _parse_timestamp(row["recorded_at"], "recorded_at"),
            row["meal"],
            row["plan_id"],
            row["plan_item_id"],
            food.occurrence,
            food.nutrition_record,
            _decimal(row["official_servings"], "official_servings"),
            row["quantity_source"],  # type: ignore[arg-type]
            row["original_reference_text"],
            row["original_quantity_text"],
        )

    def _historical_food(
        self,
        *,
        occurrence_id: int,
        snapshot_id: int,
        source_identifier: SourceIdentifier,
        content_signature: str,
    ) -> ResolvedFood:
        occurrence = self._catalog.get_occurrence_by_id(occurrence_id)
        snapshot = self._catalog.get_snapshot_by_id(snapshot_id)
        if occurrence is None or snapshot is None:
            raise MealReportApplicationError("persisted food link is missing")
        if (
            occurrence.nutrition_snapshot_id != snapshot_id
            or occurrence.source_identifier != source_identifier
            or occurrence.content_signature != content_signature
            or snapshot.source_identifier != source_identifier
            or snapshot.content_signature != content_signature
            or occurrence.nutrition_record != snapshot.record
        ):
            raise MealReportApplicationError("persisted food link is inconsistent")
        return ResolvedFood(
            snapshot.record.name,
            occurrence,
            (occurrence,),
            source_identifier,
            content_signature,
            snapshot_id,
            snapshot.record,
            "exact_name",
        )

    def _validate_plan_item_link(self, item: PlannedMealItem) -> None:
        self._validate_food_link(item.food)

    def _scheduled_dispatch_from_row(
        self,
        row: sqlite3.Row,
    ) -> ScheduledRecommendationDispatch:
        """Reconstruct and validate one scheduler row and its immutable plan."""

        try:
            service_date = date.fromisoformat(str(row["service_date"]))
            meal_context = str(row["meal_context"])
            plan_id = str(row["plan_id"])
            persisted_plan = self.load_meal_plan(plan_id)
            if persisted_plan is None:
                raise MealReportApplicationError(
                    "scheduled dispatch references a missing meal plan"
                )
            return ScheduledRecommendationDispatch(
                service_date=service_date,
                meal_context=meal_context,
                meal_slot=require_meal_slot(row["meal_slot"]),
                persisted_plan=persisted_plan,
                delivery_token=str(row["delivery_token"]),
                status=str(row["status"]),  # type: ignore[arg-type]
                created_at=_parse_timestamp(row["created_at"], "created_at"),
                delivery_started_at=_optional_timestamp(
                    row["delivery_started_at"], "delivery_started_at"
                ),
                delivered_at=_optional_timestamp(row["delivered_at"], "delivered_at"),
                expired_at=_optional_timestamp(row["expired_at"], "expired_at"),
            )
        except (TypeError, ValueError, MealReportApplicationError):
            raise MealReportApplicationError("scheduled recommendation dispatch is corrupt") from None

    def _pending_meal_request_from_row(self, row: sqlite3.Row) -> PendingMealRequest:
        """Reconstruct one scoped future request and its stable food identities."""

        try:
            source_event_id = str(row["source_event_id"])
            meal = int(row["meal"]) if row["meal_kind"] == "integer" else str(row["meal"])
            food_rows = self._connection().execute(
                """
                SELECT food_text, source_kind, source_value, content_signature
                FROM pending_meal_request_foods
                WHERE source_event_id = ? ORDER BY food_position
                """,
                (source_event_id,),
            ).fetchall()
            return PendingMealRequest(
                source_event_id=source_event_id,
                chat_guid=str(row["chat_guid"]),
                service_date=date.fromisoformat(str(row["service_date"])),
                meal=meal,
                meal_context=str(row["meal_context"]),
                meal_slot=require_meal_slot(row["meal_slot"]),
                requested_foods=tuple(
                    RequestedMealFood(
                        str(food_row["food_text"]),
                        str(food_row["source_kind"]),
                        str(food_row["source_value"]),
                        str(food_row["content_signature"]),
                    )
                    for food_row in food_rows
                ),
                whole_meal=bool(int(row["whole_meal"])),
                status=str(row["status"]),  # type: ignore[arg-type]
                reply_text=str(row["reply_text"]),
                created_at=_parse_timestamp(row["created_at"], "pending request created_at"),
                consumed_at=_optional_timestamp(
                    row["consumed_at"], "pending request consumed_at"
                ),
                consumed_plan_id=(
                    None if row["consumed_plan_id"] is None else str(row["consumed_plan_id"])
                ),
            )
        except (TypeError, ValueError, MealReportApplicationError):
            raise MealReportApplicationError("pending meal request is corrupt") from None

    def _transition_scheduled_dispatch(
        self,
        dispatch: ScheduledRecommendationDispatch,
        *,
        expected_status: ScheduledRecommendationDispatchStatus,
        next_status: ScheduledRecommendationDispatchStatus,
        timestamp_column: Literal["delivered_at", "expired_at"] | None,
        timestamp: datetime | None,
    ) -> ScheduledRecommendationDispatch:
        """Apply one guarded scheduler state transition and reload its row."""

        if not isinstance(dispatch, ScheduledRecommendationDispatch):
            raise TypeError("dispatch must be a ScheduledRecommendationDispatch")
        if timestamp_column is None:
            if timestamp is not None:
                raise ValueError("timestamp must be omitted without a timestamp column")
            statement = """
                UPDATE scheduled_recommendation_dispatches
                SET status = ?
                WHERE service_date = ? AND meal_slot = ? AND plan_id = ? AND status = ?
            """
            parameters: tuple[object, ...] = (
                next_status,
                dispatch.service_date.isoformat(),
                dispatch.meal_slot,
                dispatch.plan_id,
                expected_status,
            )
        else:
            if timestamp is None:
                raise ValueError("timestamp is required for a timestamped transition")
            statement = f"""
                UPDATE scheduled_recommendation_dispatches
                SET status = ?, {timestamp_column} = ?
                WHERE service_date = ? AND meal_slot = ? AND plan_id = ? AND status = ?
            """
            parameters = (
                next_status,
                _timestamp_text(timestamp),
                dispatch.service_date.isoformat(),
                dispatch.meal_slot,
                dispatch.plan_id,
                expected_status,
            )
        connection = self._connection()
        try:
            with connection:
                transitioned = connection.execute(statement, parameters)
        except sqlite3.Error as exc:
            raise MealReportApplicationError("unable to update scheduled delivery") from exc
        if transitioned.rowcount != 1:
            raise MealReportApplicationError("scheduled delivery state changed concurrently")
        updated = self.load_scheduled_recommendation_dispatch(
            dispatch.service_date,
            dispatch.persisted_plan.plan.meal,
            meal_slot=dispatch.meal_slot,
        )
        if updated is None:
            raise MealReportApplicationError("scheduled delivery disappeared after update")
        return updated

    def _replacement_dispatch_from_row(
        self,
        row: sqlite3.Row,
    ) -> MealRecommendationReplacementDispatch:
        """Reconstruct one replacement row and its immutable plan lineage."""

        try:
            source_event_id = str(row["source_event_id"])
            prior_plan = self.load_meal_plan(str(row["prior_plan_id"]))
            replacement_plan = self.load_meal_plan(str(row["replacement_plan_id"]))
            if prior_plan is None or replacement_plan is None:
                raise MealReportApplicationError("replacement dispatch references a missing plan")
            rejected_rows = self._connection().execute(
                """
                SELECT plan_item_id FROM meal_recommendation_replacement_rejections
                WHERE source_event_id = ? ORDER BY plan_item_id
                """,
                (source_event_id,),
            ).fetchall()
            requested_rows = self._connection().execute(
                """
                SELECT food_text, source_kind, source_value, content_signature
                FROM meal_recommendation_replacement_requested_foods
                WHERE source_event_id = ? ORDER BY food_position
                """,
                (source_event_id,),
            ).fetchall()
            return MealRecommendationReplacementDispatch(
                source_event_id=source_event_id,
                chat_guid=str(row["chat_guid"]),
                prior_plan=prior_plan,
                persisted_plan=replacement_plan,
                rejected_plan_item_ids=tuple(
                    str(rejected_row["plan_item_id"]) for rejected_row in rejected_rows
                ),
                requested_foods=tuple(
                    RequestedMealFood(
                        str(requested_row["food_text"]),
                        str(requested_row["source_kind"]),
                        str(requested_row["source_value"]),
                        str(requested_row["content_signature"]),
                    )
                    for requested_row in requested_rows
                ),
                request_kind=str(row["request_kind"]),  # type: ignore[arg-type]
                whole_meal=bool(int(row["whole_meal"])),
                delivery_token=str(row["delivery_token"]),
                status=str(row["status"]),  # type: ignore[arg-type]
                created_at=_parse_timestamp(row["created_at"], "replacement created_at"),
                scheduled_origin_plan_id=(
                    None
                    if row["scheduled_origin_plan_id"] is None
                    else str(row["scheduled_origin_plan_id"])
                ),
                delivery_started_at=_optional_timestamp(
                    row["delivery_started_at"], "replacement delivery_started_at"
                ),
                delivered_at=_optional_timestamp(
                    row["delivered_at"], "replacement delivered_at"
                ),
            )
        except (TypeError, ValueError, MealReportApplicationError):
            raise MealReportApplicationError("meal recommendation replacement is corrupt") from None

    def _immediate_meal_request_dispatch_from_row(
        self,
        row: sqlite3.Row,
    ) -> ImmediateMealRequestDispatch:
        """Reconstruct one immediate-request dispatch and its immutable plan."""

        try:
            source_event_id = str(row["source_event_id"])
            plan = self.load_meal_plan(str(row["plan_id"]))
            if plan is None:
                raise MealReportApplicationError("immediate request references a missing plan")
            food_rows = self._connection().execute(
                """
                SELECT food_text, source_kind, source_value, content_signature
                FROM immediate_meal_request_requested_foods
                WHERE source_event_id = ? ORDER BY food_position
                """,
                (source_event_id,),
            ).fetchall()
            return ImmediateMealRequestDispatch(
                source_event_id=source_event_id,
                chat_guid=str(row["chat_guid"]),
                persisted_plan=plan,
                requested_foods=tuple(
                    RequestedMealFood(
                        str(food_row["food_text"]),
                        str(food_row["source_kind"]),
                        str(food_row["source_value"]),
                        str(food_row["content_signature"]),
                    )
                    for food_row in food_rows
                ),
                whole_meal=bool(int(row["whole_meal"])),
                delivery_token=str(row["delivery_token"]),
                status=str(row["status"]),  # type: ignore[arg-type]
                created_at=_parse_timestamp(row["created_at"], "immediate request created_at"),
                delivery_started_at=_optional_timestamp(
                    row["delivery_started_at"], "immediate request delivery_started_at"
                ),
                delivered_at=_optional_timestamp(
                    row["delivered_at"], "immediate request delivered_at"
                ),
            )
        except (TypeError, ValueError, MealReportApplicationError):
            raise MealReportApplicationError("immediate meal request dispatch is corrupt") from None

    def _transition_immediate_meal_request_dispatch(
        self,
        dispatch: ImmediateMealRequestDispatch,
        *,
        expected_status: ImmediateMealRequestDispatchStatus,
        next_status: ImmediateMealRequestDispatchStatus,
        delivered_at: datetime | None,
    ) -> ImmediateMealRequestDispatch:
        """Apply one guarded immediate-request delivery transition."""

        if not isinstance(dispatch, ImmediateMealRequestDispatch):
            raise TypeError("dispatch must be an ImmediateMealRequestDispatch")
        if delivered_at is None:
            statement = """
                UPDATE immediate_meal_request_dispatches
                SET status = ?
                WHERE source_event_id = ? AND plan_id = ? AND status = ?
            """
            parameters: tuple[object, ...] = (
                next_status,
                dispatch.source_event_id,
                dispatch.plan_id,
                expected_status,
            )
        else:
            statement = """
                UPDATE immediate_meal_request_dispatches
                SET status = ?, delivered_at = ?
                WHERE source_event_id = ? AND plan_id = ? AND status = ?
            """
            parameters = (
                next_status,
                _timestamp_text(delivered_at),
                dispatch.source_event_id,
                dispatch.plan_id,
                expected_status,
            )
        connection = self._connection()
        try:
            with connection:
                transitioned = connection.execute(statement, parameters)
        except sqlite3.Error as exc:
            raise MealReportApplicationError("unable to update immediate request delivery") from exc
        if transitioned.rowcount != 1:
            raise MealReportApplicationError("immediate request delivery state changed concurrently")
        updated = self.load_immediate_meal_request_dispatch(dispatch.source_event_id)
        if updated is None:
            raise MealReportApplicationError("immediate request delivery disappeared after update")
        return updated

    def _transition_replacement_dispatch(
        self,
        dispatch: MealRecommendationReplacementDispatch,
        *,
        expected_status: MealRecommendationReplacementDispatchStatus,
        next_status: MealRecommendationReplacementDispatchStatus,
        timestamp_column: Literal["delivered_at"] | None,
        timestamp: datetime | None,
    ) -> MealRecommendationReplacementDispatch:
        """Apply one guarded replacement delivery transition and reload it."""

        if not isinstance(dispatch, MealRecommendationReplacementDispatch):
            raise TypeError("dispatch must be a MealRecommendationReplacementDispatch")
        if timestamp_column is None:
            if timestamp is not None:
                raise ValueError("timestamp must be omitted without a timestamp column")
            statement = """
                UPDATE meal_recommendation_replacements
                SET status = ?
                WHERE source_event_id = ? AND replacement_plan_id = ? AND status = ?
            """
            parameters: tuple[object, ...] = (
                next_status,
                dispatch.source_event_id,
                dispatch.plan_id,
                expected_status,
            )
        else:
            if timestamp is None:
                raise ValueError("timestamp is required for a timestamped transition")
            statement = """
                UPDATE meal_recommendation_replacements
                SET status = ?, delivered_at = ?
                WHERE source_event_id = ? AND replacement_plan_id = ? AND status = ?
            """
            parameters = (
                next_status,
                _timestamp_text(timestamp),
                dispatch.source_event_id,
                dispatch.plan_id,
                expected_status,
            )
        connection = self._connection()
        try:
            with connection:
                transitioned = connection.execute(statement, parameters)
        except sqlite3.Error as exc:
            raise MealReportApplicationError("unable to update replacement delivery") from exc
        if transitioned.rowcount != 1:
            raise MealReportApplicationError("replacement delivery state changed concurrently")
        updated = self.load_meal_recommendation_replacement_dispatch(
            dispatch.source_event_id
        )
        if updated is None:
            raise MealReportApplicationError("replacement delivery disappeared after update")
        return updated

    def _validate_food_link(self, food: ResolvedFood) -> None:
        historical = self._historical_food(
            occurrence_id=food.occurrence.occurrence_id,
            snapshot_id=food.nutrition_snapshot_id,
            source_identifier=food.source_identifier,
            content_signature=food.content_signature,
        )
        if historical.nutrition_record != food.nutrition_record:
            raise MealReportApplicationError("food snapshot does not match persisted catalog")

    def _list_all_active_meal_plans(self) -> tuple[PersistedMealPlan, ...]:
        """Load all active rows while preserving the one-slot invariant."""

        rows = self._connection().execute(
            """
            SELECT plan_id, service_date, meal_context, meal_slot FROM meal_plans
            WHERE status = 'active'
            ORDER BY service_date, meal_slot, plan_id
            """
        ).fetchall()
        persisted_plans: list[PersistedMealPlan] = []
        seen_slots: set[tuple[str, MealSlot]] = set()
        for row in rows:
            service_date_text = str(row["service_date"])
            context = str(row["meal_context"])
            try:
                slot = require_meal_slot(row["meal_slot"])
            except ValueError:
                raise MealReportApplicationError("active meal plan slot is corrupt") from None
            key = (service_date_text, slot)
            if key in seen_slots:
                raise ActiveMealPlanInvariantError(
                    "multiple active meal plans exist for one product meal slot"
                )
            seen_slots.add(key)
            persisted = self.load_meal_plan(str(row["plan_id"]))
            if (
                persisted is None
                or persisted.status != "active"
                or persisted.plan.service_date.isoformat() != service_date_text
                or meal_context_key(persisted.plan.meal) != context
                or persisted.meal_slot != slot
            ):
                raise MealReportApplicationError("active meal plan row is corrupt")
            persisted_plans.append(persisted)
        return tuple(persisted_plans)

    def _connection(self) -> sqlite3.Connection:
        # The catalog owns this single SQLite connection and its migration.
        return self._catalog._require_connection()


def _resolved_meal_slot(
    provider_meal: str | int,
    supplied: MealSlot | str | None,
) -> MealSlot:
    """Resolve a caller-supplied product slot independently of provider identity."""

    return (
        meal_slot_for_provider_meal(provider_meal)
        if supplied is None
        else require_meal_slot(supplied)
    )


def _same_plan(left: MealPlan, right: MealPlan) -> bool:
    if (
        left.service_date != right.service_date
        or not meal_values_equal(left.meal, right.meal)
        or len(left.items) != len(right.items)
    ):
        return False
    return all(
        left.item_id(left_item) == right.item_id(right_item)
        and left_item.recommended_official_servings == right_item.recommended_official_servings
        and left_item.natural_quantity_text == right_item.natural_quantity_text
        and left_item.display_food_name == right_item.display_food_name
        and left_item.presentation_binding == right_item.presentation_binding
        and _same_food(left_item.food, right_item.food)
        for left_item, right_item in zip(left.items, right.items, strict=True)
    )


def _same_food(left: ResolvedFood, right: ResolvedFood) -> bool:
    return (
        left.occurrence.occurrence_id == right.occurrence.occurrence_id
        and left.nutrition_snapshot_id == right.nutrition_snapshot_id
        and left.source_identifier == right.source_identifier
        and left.content_signature == right.content_signature
        and left.nutrition_record == right.nutrition_record
    )


def _plan_item_food_identity(item: PlannedMealItem | None) -> tuple[str, str]:
    """Return the stable authoritative food identity used for exclusion.

    Source identifiers are provider-backed immutable identifiers, unlike a
    display name or a menu station.  The replacement boundary deliberately
    uses this pair so a rejected food cannot be picked again through a
    duplicate occurrence or a spelling variant.
    """

    if not isinstance(item, PlannedMealItem):
        raise ValueError("plan item is missing")
    return (item.food.source_identifier.kind, item.food.source_identifier.value)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return _timestamp(value, "timestamp").isoformat(timespec="microseconds")


def _parse_timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} is invalid")
    try:
        return _timestamp(datetime.fromisoformat(value), name)
    except ValueError:
        raise ValueError(f"{name} is invalid") from None


def _optional_timestamp(value: object, name: str) -> datetime | None:
    if value is None:
        return None
    return _parse_timestamp(value, name)


def _decimal(value: object, name: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f"{name} is invalid")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError):
        raise ValueError(f"{name} is invalid") from None
    _positive_decimal(parsed, name)
    return parsed


def _positive_decimal(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
        raise ValueError(f"{name} must be a positive finite Decimal")


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _intent(value: MealReportIntent) -> None:
    if value not in {
        "meal_report",
        "clarification_answer",
        "location_question",
        "replacement_request",
        "meal_request",
        "unsupported_or_ambiguous",
    }:
        raise ValueError("meal report intent is invalid")


def _date(value: date, name: str) -> None:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise TypeError(f"{name} must be a date")
