"""Durable, plan-scoped conversation orchestration for inbound meal reports.

This boundary deliberately keeps AI on the semantic side of the line.  The
model classifies intent and references only supplied items; deterministic
Python reconstructs exact food links, quantities, draft state, lifecycle, and
the final transactional application.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Literal, Protocol

from .application_clock import (
    DEFAULT_NUTRITION_APPLICATION_CLOCK,
    NutritionApplicationClock,
)
from .durable_state import (
    AppliedMealReport,
    DraftClarification,
    DraftPlannedItem,
    DraftUnplannedItem,
    DurableMealState,
    MealReportDraft,
    MealReportCommitTimeRejection,
    MealReportDraftConflictError,
    MealReportIntent,
    MealRecommendationReplacementConflictError,
    MealRecommendationReplacementDispatch,
    MealRecommendationReplacementDraftConflictError,
    PersistedMealPlan,
)
from .food_resolution import AmbiguousFood, ResolvedFood
from .meal_identity import MealSlot, meal_name_for_display, meal_values_equal
from .meal_report import (
    ClarificationItem,
    MealPlan,
    MealReportReconciler,
    ProposedEatenMealItem,
    ProposedUnplannedMealItem,
    ReconciledMealReport,
    SkippedMealItem,
)
from .meal_report_rendering import (
    format_meal_report_confirmation,
)
from .meal_report_structured import MealReportSemanticResult
from .meal_recommendation_replacement import (
    MealRecommendationReplacementPreparer,
)
from .application import (
    MealRecommendationReplacementUnavailableError,
    MealRecommendationRequestedFoodError,
    MealRecommendationServiceUnavailableError,
    MealRecommendationUnavailableError,
)
from .phelps_service_calendar import PhelpsServiceCalendar


__all__ = [
    "ConversationPlanResolver",
    "MealReportConversationError",
    "MealReportConversationOrchestrator",
    "MealReportConversationResult",
    "format_meal_report_draft_prompt",
]


REPLACEMENT_NOT_YET_SUPPORTED_REPLY_TEXT = (
    "I understand you want to replace part of that recommendation, but replacements "
    "aren't available yet. I haven't logged anything."
)
REPLACEMENT_AMBIGUITY_REPLY_TEXT = (
    "Which recommended item would you like to replace?"
)
REPLACEMENT_DRAFT_CONFLICT_REPLY_TEXT = (
    "Please finish or cancel the meal report you started before changing this recommendation."
)
REPLACEMENT_UNAVAILABLE_REPLY_TEXT = (
    "I couldn't find a different current-menu replacement, so your recommendation is unchanged."
)
REPLACEMENT_CONTEXT_CHANGED_REPLY_TEXT = (
    "That recommendation changed before I could replace it. I haven't changed anything else."
)
REPLACEMENT_SERVICE_UNAVAILABLE_REPLY_TEXT = (
    "I can't change that recommendation while this Phelps meal opportunity is unavailable. "
    "Your current recommendation is unchanged."
)
MEAL_REQUEST_AMBIGUITY_REPLY_TEXT = (
    "Which current menu item would you like me to include?"
)
MEAL_REQUEST_DRAFT_CONFLICT_REPLY_TEXT = (
    "Please finish or cancel the meal report you started before changing this recommendation."
)
MEAL_REQUEST_UNAVAILABLE_REPLY_TEXT = (
    "I couldn't include that requested food from the current menu, so your recommendation is unchanged."
)
UNSUPPORTED_INTERACTION_REPLY_TEXT = (
    "I couldn't tell whether that was a meal update. Please say what you ate, skipped, "
    "or ask where a recommended item is."
)
STALE_DRAFT_REPLY_TEXT = (
    "That earlier recommendation was replaced or is no longer reportable before the "
    "meal report was complete. I haven't logged anything from the draft."
)
REPORT_CONTEXT_UNAVAILABLE_REPLY_TEXT = (
    "I don't have an active meal plan to log right now. "
    "Please ask for a new meal recommendation first."
)
class MealReportConversationError(RuntimeError):
    """Raised when a conversation turn cannot preserve durable invariants."""


class ConversationPlanResolver(Protocol):
    """Minimal context resolver needed only for a new draft-free report."""

    def resolve(
        self,
        source_event_id: str,
        *,
        meal_slot: MealSlot | None = None,
    ) -> PersistedMealPlan:
        """Return the one reportable active plan for an inbound GUID."""


@dataclass(frozen=True, slots=True)
class MealReportConversationResult:
    """Transport-neutral result for one conversation turn."""

    outcome: Literal[
        "applied",
        "clarification_required",
        "replayed",
        "location_question",
        "replacement_request",
        "meal_request",
        "unsupported_or_ambiguous",
        "draft_cancelled",
    ]
    persisted_plan: PersistedMealPlan | None
    message: str
    application: AppliedMealReport | None = None
    draft: MealReportDraft | None = None
    replacement_dispatch: MealRecommendationReplacementDispatch | None = None

    def __post_init__(self) -> None:
        if self.outcome not in {
            "applied",
            "clarification_required",
            "replayed",
            "location_question",
            "replacement_request",
            "meal_request",
            "unsupported_or_ambiguous",
            "draft_cancelled",
        }:
            raise ValueError("conversation outcome is invalid")
        if self.persisted_plan is not None and not isinstance(
            self.persisted_plan, PersistedMealPlan
        ):
            raise TypeError("persisted_plan must be a PersistedMealPlan or None")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("conversation message must be non-empty text")
        if self.application is not None and not isinstance(self.application, AppliedMealReport):
            raise TypeError("application must be an AppliedMealReport or None")
        if self.draft is not None and not isinstance(self.draft, MealReportDraft):
            raise TypeError("draft must be a MealReportDraft or None")
        if self.replacement_dispatch is not None and not isinstance(
            self.replacement_dispatch, MealRecommendationReplacementDispatch
        ):
            raise TypeError(
                "replacement_dispatch must be a MealRecommendationReplacementDispatch or None"
            )
        if self.outcome == "applied" and self.application is None:
            raise ValueError("applied conversation result needs an application")
        if self.outcome != "replayed" and self.persisted_plan is None:
            raise ValueError("only a replayed pre-route result may omit its plan")
        if self.outcome == "clarification_required" and self.draft is None:
            raise ValueError("clarification conversation result needs a draft")
        if self.replacement_dispatch is not None and self.outcome not in {
            "replacement_request",
            "meal_request",
        }:
            raise ValueError("replacement dispatch requires a replacement-style outcome")


class MealReportConversationOrchestrator:
    """Continue one plan-scoped report draft until a complete report is safe.

    A draft belongs to exactly one chat and exact persisted plan. A pre-meal
    replacement is blocked while such a draft is unresolved, rather than
    moving or cancelling facts the user may still intend to log.
    """

    def __init__(
        self,
        state: DurableMealState,
        reconciler: MealReportReconciler,
        *,
        service_calendar: PhelpsServiceCalendar | None = None,
        clock: NutritionApplicationClock | None = None,
        replacement_preparer: MealRecommendationReplacementPreparer | None = None,
    ) -> None:
        if not isinstance(state, DurableMealState):
            raise TypeError("state must be a DurableMealState")
        if not isinstance(reconciler, MealReportReconciler):
            raise TypeError("reconciler must be a MealReportReconciler")
        if service_calendar is not None and not isinstance(
            service_calendar, PhelpsServiceCalendar
        ):
            raise TypeError("service_calendar must be a PhelpsServiceCalendar")
        if clock is not None and not isinstance(clock, NutritionApplicationClock):
            raise TypeError("clock must be a NutritionApplicationClock")
        if replacement_preparer is not None and not isinstance(
            replacement_preparer, MealRecommendationReplacementPreparer
        ):
            raise TypeError("replacement_preparer must be a MealRecommendationReplacementPreparer")
        self._state = state
        self._reconciler = reconciler
        self._service_calendar = service_calendar
        self._clock = clock or DEFAULT_NUTRITION_APPLICATION_CLOCK
        self._enforce_commit_reportability = service_calendar is not None or clock is not None
        self._replacement_preparer = replacement_preparer

    def process(
        self,
        *,
        chat_guid: str,
        user_text: str,
        source_event_id: str,
        context_resolver: ConversationPlanResolver,
    ) -> MealReportConversationResult:
        """Interpret, accumulate, or atomically apply one inbound message."""

        _text(chat_guid, "chat_guid")
        _text(user_text, "user_text")
        _text(source_event_id, "source_event_id")
        if not callable(getattr(context_resolver, "resolve", None)):
            raise TypeError("context_resolver must provide resolve")

        existing_event = self._state.load_meal_report_message_event(source_event_id)
        if existing_event is not None:
            if existing_event.chat_guid != chat_guid:
                raise MealReportConversationError("source_event_id belongs to another chat")
            replacement_dispatch = self._state.load_meal_recommendation_replacement_dispatch(
                source_event_id
            )
            if replacement_dispatch is not None:
                return MealReportConversationResult(
                    "meal_request"
                    if replacement_dispatch.request_kind == "meal_request"
                    else "replacement_request",
                    replacement_dispatch.persisted_plan,
                    existing_event.reply_text,
                    replacement_dispatch=replacement_dispatch,
                )
            replayed = (
                None
                if existing_event.plan_id is None
                else self._state.load_meal_plan(existing_event.plan_id)
            )
            if existing_event.plan_id is None:
                return MealReportConversationResult(
                    "replayed",
                    None,
                    existing_event.reply_text,
                )
            if replayed is None:
                raise MealReportConversationError("message event references a missing plan")
            return MealReportConversationResult(
                "replayed",
                replayed,
                existing_event.reply_text,
            )

        # Retain the legacy successful-application replay invariant for rows
        # created before conversation-event persistence or by direct callers.
        historical = self._state.load_meal_plan_for_source_event(source_event_id)
        if historical is not None:
            return MealReportConversationResult(
                "replayed",
                historical,
                "Got it — that meal report was already logged.",
                self._state.load_applied_meal_report(source_event_id),
            )

        drafts = self._state.list_active_meal_report_drafts(chat_guid)
        routing = self._reconciler.interpret_report_routing(user_text)
        if routing is None:
            raise MealReportConversationError("meal report routing interpretation failed")

        draft: MealReportDraft | None
        persisted_plan: PersistedMealPlan
        if routing.explicit_meal_slot is not None:
            persisted_plan = context_resolver.resolve(
                source_event_id,
                meal_slot=routing.explicit_meal_slot,
            )
            draft = next(
                (candidate for candidate in drafts if candidate.plan_id == persisted_plan.plan_id),
                None,
            )
        elif len(drafts) == 1:
            draft = drafts[0]
            persisted_plan = draft.persisted_plan
        else:
            # With no draft, this retains the convenient single-plan route. With
            # multiple drafts it deliberately asks for meal scope rather than
            # guessing newest/focused state. Stale drafts never become candidates
            # because the resolver evaluates today's reportable plans.
            persisted_plan = context_resolver.resolve(source_event_id)
            draft = next(
                (candidate for candidate in drafts if candidate.plan_id == persisted_plan.plan_id),
                None,
            )

        if draft is not None:
            persisted_plan = self._reportable_draft_plan_or_cancel(
                draft,
                source_event_id=source_event_id,
            )
            if persisted_plan is None:
                return MealReportConversationResult(
                    "draft_cancelled",
                    draft.persisted_plan,
                    STALE_DRAFT_REPLY_TEXT,
                )
            # Reload after deterministic reportability maintenance so the
            # draft and its plan share one current durable lifecycle view.
            draft = self._state.load_meal_report_draft(draft.draft_id)
            if draft is None:
                raise MealReportConversationError("meal report draft disappeared")
        else:
            if not self._is_persisted_plan_reportable(persisted_plan):
                return self._record_nonreport_interaction(
                    persisted_plan,
                    None,
                    source_event_id,
                    chat_guid,
                    "unsupported_or_ambiguous",
                    REPORT_CONTEXT_UNAVAILABLE_REPLY_TEXT,
                )

        context = _model_context(persisted_plan.plan, draft)
        semantic = self._reconciler.interpret_semantic(
            persisted_plan.plan,
            user_text,
            conversation_context=context,
        )
        if semantic is None:
            return self._record_nonreport_interaction(
                persisted_plan,
                draft,
                source_event_id,
                chat_guid,
                "unsupported_or_ambiguous",
                UNSUPPORTED_INTERACTION_REPLY_TEXT,
            )

        if semantic.intent == "location_question":
            return self._handle_location_question(
                persisted_plan,
                draft,
                semantic,
                source_event_id,
                chat_guid,
            )
        if semantic.intent == "replacement_request":
            return self._handle_replacement_request(
                persisted_plan,
                draft,
                semantic,
                source_event_id,
                chat_guid,
            )
        if semantic.intent == "meal_request":
            return self._handle_meal_request(
                persisted_plan,
                draft,
                semantic,
                source_event_id,
                chat_guid,
            )
        if semantic.intent == "unsupported_or_ambiguous":
            return self._record_nonreport_interaction(
                persisted_plan,
                draft,
                source_event_id,
                chat_guid,
                "unsupported_or_ambiguous",
                UNSUPPORTED_INTERACTION_REPLY_TEXT,
            )

        inherited_quantities = (
            {
                item.plan_item_id: item.official_servings
                for item in draft.planned_items
                if item.action == "eaten" and item.official_servings is not None
            }
            if draft is not None
            else {}
        )
        report = self._reconciler.reconcile_semantic(
            persisted_plan.plan,
            semantic,
            user_text,
            inherited_plan_item_quantities=inherited_quantities,
            quantity_reference_item_id=_quantity_reference_item_id(draft),
        )
        return self._handle_report_semantics(
            persisted_plan,
            draft,
            report,
            semantic,
            source_event_id,
            chat_guid,
            user_text,
        )

    def _handle_meal_request(
        self,
        persisted_plan: PersistedMealPlan,
        draft: MealReportDraft | None,
        semantic: MealReportSemanticResult,
        source_event_id: str,
        chat_guid: str,
    ) -> MealReportConversationResult:
        """Resolve positive food wording, then reuse replacement optimization."""

        if semantic.requested_meal is not None and not meal_values_equal(
            semantic.requested_meal, persisted_plan.plan.meal,
        ):
            return self._record_nonreport_interaction(
                persisted_plan, draft, source_event_id, chat_guid, "meal_request",
                "Your active recommendation is for "
                f"{meal_name_for_display(persisted_plan.plan.meal)}. "
                "Please report that meal first, then ask for "
                f"{semantic.requested_meal} again. Your recommendation is unchanged.",
            )
        if draft is not None:
            return self._record_nonreport_interaction(
                persisted_plan,
                draft,
                source_event_id,
                chat_guid,
                "meal_request",
                MEAL_REQUEST_DRAFT_CONFLICT_REPLY_TEXT,
            )
        if self._replacement_preparer is None:
            return self._record_nonreport_interaction(
                persisted_plan,
                None,
                source_event_id,
                chat_guid,
                "meal_request",
                REPLACEMENT_NOT_YET_SUPPORTED_REPLY_TEXT,
            )
        if semantic.meal_request_mode == "whole_meal" and not semantic.requested_food_texts:
            # A request for a wholly different meal without named foods uses
            # the established whole-meal replacement lifecycle unchanged.
            rejected_item_ids = tuple(
                persisted_plan.plan.item_id(item) for item in persisted_plan.plan.items
            )
            try:
                dispatch = self._replacement_preparer.prepare(
                    prior_plan_id=persisted_plan.plan_id,
                    rejected_plan_item_ids=rejected_item_ids,
                    whole_meal=True,
                    source_event_id=source_event_id,
                    chat_guid=chat_guid,
                    request_kind="meal_request",
                )
            except MealRecommendationReplacementUnavailableError:
                return self._record_nonreport_interaction(
                    persisted_plan,
                    None,
                    source_event_id,
                    chat_guid,
                    "meal_request",
                    REPLACEMENT_UNAVAILABLE_REPLY_TEXT,
                )
            except (MealRecommendationUnavailableError, MealRecommendationServiceUnavailableError):
                return self._record_nonreport_interaction(
                    persisted_plan,
                    None,
                    source_event_id,
                    chat_guid,
                    "meal_request",
                    REPLACEMENT_SERVICE_UNAVAILABLE_REPLY_TEXT,
                )
            except MealRecommendationReplacementDraftConflictError:
                return self._record_nonreport_interaction(
                    persisted_plan,
                    None,
                    source_event_id,
                    chat_guid,
                    "meal_request",
                    MEAL_REQUEST_DRAFT_CONFLICT_REPLY_TEXT,
                )
            except MealRecommendationReplacementConflictError:
                return self._record_nonreport_interaction(
                    persisted_plan,
                    None,
                    source_event_id,
                    chat_guid,
                    "meal_request",
                    REPLACEMENT_CONTEXT_CHANGED_REPLY_TEXT,
                )
            event = self._state.load_meal_report_message_event(source_event_id)
            if event is None or event.plan_id != dispatch.plan_id:
                raise MealReportConversationError("whole-meal request was not recorded safely")
            return MealReportConversationResult(
                "meal_request",
                dispatch.persisted_plan,
                event.reply_text,
                replacement_dispatch=dispatch,
            )

        if semantic.meal_request_mode not in {"targeted", "whole_meal"} or not semantic.requested_food_texts:
            return self._record_nonreport_interaction(
                persisted_plan,
                None,
                source_event_id,
                chat_guid,
                "meal_request",
                MEAL_REQUEST_AMBIGUITY_REPLY_TEXT,
            )

        resolved_foods: list[ResolvedFood] = []
        seen_identities: set[tuple[str, str, str]] = set()
        for food_text in semantic.requested_food_texts:
            resolved = self._reconciler.resolve_current_menu_food(
                persisted_plan.plan,
                food_text,
            )
            if isinstance(resolved, AmbiguousFood):
                names = tuple(
                    candidate.official_display_name
                    + (f" at {candidate.occurrence.station_name}" if candidate.occurrence.station_name else "")
                    for candidate in resolved.candidates
                )
                reply = (
                    f"I found more than one current menu item for {food_text}: "
                    + ", ".join(names[:3])
                    + ". Which one did you mean? Please request its menu name and station."
                )
                return self._record_nonreport_interaction(
                    persisted_plan,
                    None,
                    source_event_id,
                    chat_guid,
                    "meal_request",
                    reply,
                )
            if not isinstance(resolved, ResolvedFood):
                reply = (
                    f"{food_text.strip().capitalize()} isn't available for "
                    f"{meal_name_for_display(persisted_plan.plan.meal)} right now. "
                    "Want me to make you something else?"
                )
                return self._record_nonreport_interaction(
                    persisted_plan,
                    None,
                    source_event_id,
                    chat_guid,
                    "meal_request",
                    reply,
                )
            key = (
                resolved.source_identifier.kind,
                resolved.source_identifier.value,
                resolved.content_signature,
            )
            if key not in seen_identities:
                seen_identities.add(key)
                resolved_foods.append(resolved)
        try:
            dispatch = self._replacement_preparer.prepare_requested(
                prior_plan_id=persisted_plan.plan_id,
                requested_foods=tuple(resolved_foods),
                whole_meal=semantic.meal_request_mode == "whole_meal",
                source_event_id=source_event_id,
                chat_guid=chat_guid,
            )
        except MealRecommendationRequestedFoodError:
            return self._record_nonreport_interaction(
                persisted_plan,
                None,
                source_event_id,
                chat_guid,
                "meal_request",
                MEAL_REQUEST_UNAVAILABLE_REPLY_TEXT,
            )
        except MealRecommendationReplacementUnavailableError:
            return self._record_nonreport_interaction(
                persisted_plan,
                None,
                source_event_id,
                chat_guid,
                "meal_request",
                REPLACEMENT_UNAVAILABLE_REPLY_TEXT,
            )
        except (MealRecommendationUnavailableError, MealRecommendationServiceUnavailableError):
            return self._record_nonreport_interaction(
                persisted_plan,
                None,
                source_event_id,
                chat_guid,
                "meal_request",
                REPLACEMENT_SERVICE_UNAVAILABLE_REPLY_TEXT,
            )
        except MealRecommendationReplacementDraftConflictError:
            return self._record_nonreport_interaction(
                persisted_plan,
                None,
                source_event_id,
                chat_guid,
                "meal_request",
                MEAL_REQUEST_DRAFT_CONFLICT_REPLY_TEXT,
            )
        except MealRecommendationReplacementConflictError:
            return self._record_nonreport_interaction(
                persisted_plan,
                None,
                source_event_id,
                chat_guid,
                "meal_request",
                REPLACEMENT_CONTEXT_CHANGED_REPLY_TEXT,
            )
        event = self._state.load_meal_report_message_event(source_event_id)
        if event is None or event.plan_id != dispatch.plan_id:
            raise MealReportConversationError("meal request dispatch was not recorded safely")
        return MealReportConversationResult(
            "meal_request",
            dispatch.persisted_plan,
            event.reply_text,
            replacement_dispatch=dispatch,
        )

    def _handle_replacement_request(
        self,
        persisted_plan: PersistedMealPlan,
        draft: MealReportDraft | None,
        semantic: MealReportSemanticResult,
        source_event_id: str,
        chat_guid: str,
    ) -> MealReportConversationResult:
        """Turn a safe semantic rejection into one deterministic new plan."""

        if draft is not None:
            return self._record_nonreport_interaction(
                persisted_plan,
                draft,
                source_event_id,
                chat_guid,
                "replacement_request",
                REPLACEMENT_DRAFT_CONFLICT_REPLY_TEXT,
            )
        if self._replacement_preparer is None:
            # Retain a safe response for direct/test composition sites that
            # have not opted into replacement delivery yet.
            return self._record_nonreport_interaction(
                persisted_plan,
                None,
                source_event_id,
                chat_guid,
                "replacement_request",
                REPLACEMENT_NOT_YET_SUPPORTED_REPLY_TEXT,
            )
        if semantic.replacement_mode == "whole_meal":
            rejected_item_ids = tuple(
                persisted_plan.plan.item_id(item) for item in persisted_plan.plan.items
            )
            whole_meal = True
        elif semantic.replacement_mode == "targeted":
            rejected_item_ids = tuple(semantic.replacement_plan_item_ids)
            whole_meal = False
        else:
            rejected_item_ids = ()
            whole_meal = False
        if (
            not rejected_item_ids
            or len(set(rejected_item_ids)) != len(rejected_item_ids)
            or any(
                persisted_plan.plan.item_for_id(item_id) is None
                for item_id in rejected_item_ids
            )
        ):
            return self._record_nonreport_interaction(
                persisted_plan,
                None,
                source_event_id,
                chat_guid,
                "replacement_request",
                REPLACEMENT_AMBIGUITY_REPLY_TEXT,
            )
        try:
            dispatch = self._replacement_preparer.prepare(
                prior_plan_id=persisted_plan.plan_id,
                rejected_plan_item_ids=rejected_item_ids,
                whole_meal=whole_meal,
                source_event_id=source_event_id,
                chat_guid=chat_guid,
            )
        except MealRecommendationReplacementUnavailableError:
            return self._record_nonreport_interaction(
                persisted_plan,
                None,
                source_event_id,
                chat_guid,
                "replacement_request",
                REPLACEMENT_UNAVAILABLE_REPLY_TEXT,
            )
        except (MealRecommendationUnavailableError, MealRecommendationServiceUnavailableError):
            return self._record_nonreport_interaction(
                persisted_plan,
                None,
                source_event_id,
                chat_guid,
                "replacement_request",
                REPLACEMENT_SERVICE_UNAVAILABLE_REPLY_TEXT,
            )
        except MealRecommendationReplacementDraftConflictError:
            return self._record_nonreport_interaction(
                persisted_plan,
                None,
                source_event_id,
                chat_guid,
                "replacement_request",
                REPLACEMENT_DRAFT_CONFLICT_REPLY_TEXT,
            )
        except MealRecommendationReplacementConflictError:
            return self._record_nonreport_interaction(
                persisted_plan,
                None,
                source_event_id,
                chat_guid,
                "replacement_request",
                REPLACEMENT_CONTEXT_CHANGED_REPLY_TEXT,
            )
        event = self._state.load_meal_report_message_event(source_event_id)
        if event is None or event.plan_id != dispatch.plan_id:
            raise MealReportConversationError("replacement dispatch was not recorded safely")
        return MealReportConversationResult(
            "replacement_request",
            dispatch.persisted_plan,
            event.reply_text,
            replacement_dispatch=dispatch,
        )

    def _reportable_draft_plan_or_cancel(
        self,
        draft: MealReportDraft,
        *,
        source_event_id: str,
    ) -> PersistedMealPlan | None:
        """Return an open draft's plan only while it remains reportable."""

        persisted = self._state.load_meal_plan(draft.plan_id)
        if persisted is None:
            raise MealReportConversationError("meal report draft references a missing plan")
        if self._is_persisted_plan_reportable(persisted):
            refreshed = self._state.load_meal_plan(draft.plan_id)
            if refreshed is None:
                raise MealReportConversationError("meal report draft references a missing plan")
            return refreshed
        self._state.cancel_meal_report_draft(
            draft,
            source_event_id=source_event_id,
            reply_text=STALE_DRAFT_REPLY_TEXT,
        )
        return None

    def _is_persisted_plan_reportable(self, persisted_plan: PersistedMealPlan) -> bool:
        """Recheck lifecycle/reportability at a durable write boundary."""

        current = self._state.load_meal_plan(persisted_plan.plan_id)
        if current is None:
            raise MealReportConversationError("persisted plan disappeared")
        if self._service_calendar is None:
            return current.status == "active"
        local_now = self._clock.now()
        self._state.retire_stale_active_meal_plans(
            self._service_calendar,
            evaluated_at=local_now,
        )
        current = self._state.load_meal_plan(persisted_plan.plan_id)
        if current is None:
            raise MealReportConversationError("persisted plan disappeared")
        return self._state.is_active_meal_plan_reportable(
            current,
            self._service_calendar,
            evaluated_at=local_now,
        )

    def _handle_location_question(
        self,
        persisted_plan: PersistedMealPlan,
        draft: MealReportDraft | None,
        semantic: MealReportSemanticResult,
        source_event_id: str,
        chat_guid: str,
    ) -> MealReportConversationResult:
        item = (
            persisted_plan.plan.item_for_id(semantic.location_plan_item_id)
            if semantic.location_plan_item_id is not None
            else None
        )
        resolved_menu_food = (
            self._reconciler.resolve_current_menu_food(
                persisted_plan.plan,
                semantic.location_food_text,
            )
            if item is None and semantic.location_food_text is not None
            else None
        )
        if item is not None:
            food_name = item.display_name
            station_name = item.food.occurrence.station_name
            ambiguous = False
        elif isinstance(resolved_menu_food, ResolvedFood):
            food_name = resolved_menu_food.nutrition_record.name
            station_name = resolved_menu_food.occurrence.station_name
            ambiguous = False
        elif isinstance(resolved_menu_food, AmbiguousFood):
            food_name = None
            station_name = None
            ambiguous = True
        else:
            food_name = None
            station_name = None
            ambiguous = False
        if food_name is None:
            reply = "Which recommended item would you like to locate?"
            outcome: Literal["location_question", "unsupported_or_ambiguous"] = (
                "unsupported_or_ambiguous"
            )
            intent: MealReportIntent = "unsupported_or_ambiguous"
            if ambiguous:
                reply = "Which current menu item would you like to locate?"
        elif station_name:
            reply = f"{food_name} is at {station_name}."
            outcome = "location_question"
            intent = "location_question"
        else:
            reply = f"I couldn't find a current station listed for {food_name}."
            outcome = "location_question"
            intent = "location_question"
        result = self._record_nonreport_interaction(
            persisted_plan,
            draft,
            source_event_id,
            chat_guid,
            intent,
            reply,
        )
        return MealReportConversationResult(
            outcome,
            result.persisted_plan,
            result.message,
            draft=result.draft,
        )

    def _record_nonreport_interaction(
        self,
        persisted_plan: PersistedMealPlan,
        draft: MealReportDraft | None,
        source_event_id: str,
        chat_guid: str,
        intent: MealReportIntent,
        reply: str,
    ) -> MealReportConversationResult:
        event = self._state.record_meal_report_message_event(
            source_event_id=source_event_id,
            chat_guid=chat_guid,
            persisted_plan=persisted_plan,
            intent=intent,
            reply_text=reply,
            draft_id=None if draft is None else draft.draft_id,
        )
        return MealReportConversationResult(
            _outcome_for_nonreport_intent(event.intent),
            persisted_plan,
            event.reply_text,
            draft=draft,
        )

    def _handle_report_semantics(
        self,
        persisted_plan: PersistedMealPlan,
        draft: MealReportDraft | None,
        report: ReconciledMealReport,
        semantic: MealReportSemanticResult,
        source_event_id: str,
        chat_guid: str,
        user_text: str,
    ) -> MealReportConversationResult:
        if not self._is_persisted_plan_reportable(persisted_plan):
            if draft is not None:
                self._state.cancel_meal_report_draft(
                    draft,
                    source_event_id=source_event_id,
                    reply_text=STALE_DRAFT_REPLY_TEXT,
                )
                return MealReportConversationResult(
                    "draft_cancelled",
                    draft.persisted_plan,
                    STALE_DRAFT_REPLY_TEXT,
                )
            return self._record_nonreport_interaction(
                persisted_plan,
                None,
                source_event_id,
                chat_guid,
                "unsupported_or_ambiguous",
                REPORT_CONTEXT_UNAVAILABLE_REPLY_TEXT,
            )

        planned_items, unplanned_items, clarifications = _merge_draft_facts(
            persisted_plan.plan,
            draft,
            report,
            semantic,
            source_event_id,
            user_text,
        )

        if not clarifications and semantic.report_scope == "complete":
            complete_report = _report_from_draft_facts(
                persisted_plan.plan,
                planned_items,
                unplanned_items,
            )
            if not _has_report_content(complete_report):
                clarifications = (DraftClarification("report_content_required"),)
            elif draft is None:
                # Preserve the established direct self-contained happy path:
                # there is no need to create a draft solely to apply a complete
                # report.  The legacy application transaction remains the
                # source-event idempotency record for this path.
                reply = format_meal_report_confirmation(complete_report)
                try:
                    application = self._state.apply_reconciled_meal_report(
                        persisted_plan,
                        complete_report,
                        source_event_id=source_event_id,
                        application_clock=(
                            self._clock if self._enforce_commit_reportability else None
                        ),
                        service_calendar=self._service_calendar,
                        chat_guid=chat_guid,
                        reply_text=reply,
                        intent=semantic.intent,
                        unavailable_reply_text=STALE_DRAFT_REPLY_TEXT,
                    )
                except MealReportCommitTimeRejection as exc:
                    return MealReportConversationResult(
                        "unsupported_or_ambiguous",
                        persisted_plan,
                        exc.event.reply_text,
                    )
                completed_plan = self._state.load_meal_plan(persisted_plan.plan_id)
                if completed_plan is None or completed_plan.status != "applied":
                    raise MealReportConversationError("meal report application did not complete")
                return MealReportConversationResult(
                    "applied",
                    completed_plan,
                    reply,
                    application,
                )
            else:
                reply = format_meal_report_confirmation(complete_report)
                try:
                    application = self._state.apply_draft_reconciled_meal_report(
                        draft,
                        complete_report,
                        source_event_id=source_event_id,
                        reply_text=reply,
                        intent=semantic.intent,
                        effective_planned_items=planned_items,
                        effective_unplanned_items=unplanned_items,
                        application_clock=(
                            self._clock if self._enforce_commit_reportability else None
                        ),
                        service_calendar=self._service_calendar,
                        unavailable_reply_text=STALE_DRAFT_REPLY_TEXT,
                    )
                except MealReportCommitTimeRejection as exc:
                    cancelled = self._state.load_meal_report_draft(draft.draft_id)
                    return MealReportConversationResult(
                        "draft_cancelled",
                        draft.persisted_plan,
                        exc.event.reply_text,
                        draft=cancelled,
                    )
                completed_plan = self._state.load_meal_plan(persisted_plan.plan_id)
                completed_draft = self._state.load_meal_report_draft(draft.draft_id)
                if completed_plan is None or completed_plan.status != "applied":
                    raise MealReportConversationError("meal report application did not complete")
                return MealReportConversationResult(
                    "applied",
                    completed_plan,
                    reply,
                    application,
                    completed_draft,
                )

        # A report that is not explicitly complete always retains a focused
        # completion question even when every amount mentioned so far resolved.
        if not clarifications:
            clarifications = (DraftClarification("completion_required"),)
        prompt = format_meal_report_draft_prompt(persisted_plan.plan, clarifications)
        try:
            saved_draft, event = self._state.save_meal_report_draft(
                chat_guid=chat_guid,
                persisted_plan=persisted_plan,
                current_draft=draft,
                planned_items=planned_items,
                unplanned_items=unplanned_items,
                clarifications=clarifications,
                last_prompt=prompt,
                source_event_id=source_event_id,
                intent=semantic.intent,
            )
        except MealReportDraftConflictError:
            # Two independent webhook/poll threads can both finish semantic
            # interpretation before either obtains SQLite's write lock.  Their
            # structured output remains valid for this immutable plan, so
            # reload once and merge it against the winner's draft instead of
            # overwriting facts or turning an otherwise safe clarification
            # into a transport 500.  Draft-relative references that were not
            # available to the first interpretation remain fail-closed in the
            # reconciled report; this path never invents a quantity.
            latest_draft = self._state.load_active_meal_report_draft(
                chat_guid,
                plan_id=persisted_plan.plan_id,
            )
            if latest_draft is None or latest_draft.plan_id != persisted_plan.plan_id:
                raise
            planned_items, unplanned_items, clarifications = _merge_draft_facts(
                persisted_plan.plan,
                latest_draft,
                report,
                semantic,
                source_event_id,
                user_text,
            )
            if not clarifications:
                clarifications = (DraftClarification("completion_required"),)
            prompt = format_meal_report_draft_prompt(persisted_plan.plan, clarifications)
            saved_draft, event = self._state.save_meal_report_draft(
                chat_guid=chat_guid,
                persisted_plan=persisted_plan,
                current_draft=latest_draft,
                planned_items=planned_items,
                unplanned_items=unplanned_items,
                clarifications=clarifications,
                last_prompt=prompt,
                source_event_id=source_event_id,
                intent=semantic.intent,
            )
        return MealReportConversationResult(
            "clarification_required",
            persisted_plan,
            event.reply_text,
            draft=saved_draft,
        )

    def record_pre_route_outcome(
        self,
        *,
        chat_guid: str,
        source_event_id: str,
        outcome_type: Literal[
            "ambiguous_meal_context",
            "unavailable_meal_slot",
            "no_reportable_context",
            "routing_failure",
        ],
        reply_text: str,
    ) -> MealReportConversationResult:
        """Store one plan-free routing result before its reply is delivered."""

        event = self._state.record_pre_route_meal_report_outcome(
            source_event_id=source_event_id,
            chat_guid=chat_guid,
            outcome_type=outcome_type,
            reply_text=reply_text,
        )
        plan = None if event.plan_id is None else self._state.load_meal_plan(event.plan_id)
        if event.plan_id is not None and plan is None:
            raise MealReportConversationError("message event references a missing plan")
        return MealReportConversationResult("replayed", plan, event.reply_text)


def format_meal_report_draft_prompt(
    plan: MealPlan,
    clarifications: tuple[DraftClarification, ...],
) -> str:
    """Render only the still-unresolved deterministic draft facts."""

    if not isinstance(plan, MealPlan):
        raise TypeError("plan must be a MealPlan")
    if not isinstance(clarifications, tuple) or not clarifications:
        raise ValueError("clarifications must be a non-empty tuple")
    if not all(isinstance(item, DraftClarification) for item in clarifications):
        raise TypeError("clarifications must contain DraftClarification values")

    completion = [item for item in clarifications if item.reason == "completion_required"]
    substantive = [item for item in clarifications if item.reason != "completion_required"]
    lines: list[str] = []
    if substantive:
        lines.append("I need a little clarification before I log that:")
        for item in substantive:
            planned = plan.item_for_id(item.plan_item_id) if item.plan_item_id else None
            if planned is not None:
                if item.quantity_text is not None:
                    lines.append(
                        f"- I have “{item.quantity_text}” for {planned.display_name}, "
                        "but I still need an authoritative amount in its menu serving unit."
                    )
                else:
                    lines.append(
                        f"- How much {planned.display_name} did you eat? "
                        f"I had recommended {planned.natural_quantity_text}."
                    )
            elif item.food_text is not None:
                if item.reason == "unplanned_food_ambiguous":
                    lines.append(f"- Which current menu item did you mean by {item.food_text}?")
                elif item.reason == "unplanned_food_unresolved":
                    lines.append(f"- What current menu item did you mean by {item.food_text}?")
                else:
                    lines.append(f"- How much {item.food_text} did you eat?")
            elif item.reason == "report_content_required":
                lines.append("- What did you have from the recommendation?")
            elif item.reason in {"ambiguous_reference", "unrecognized_statement"}:
                lines.append("- Which recommended item did you mean?")
            else:
                lines.append("- Please clarify the remaining meal item.")
    if completion:
        if lines:
            lines.append("")
        lines.append(
            "Anything else from the recommendation? You can say “that's all” when you're done."
        )
    return "\n".join(lines)


def _quantity_reference_item_id(draft: MealReportDraft | None) -> str | None:
    """A pronoun can answer one outstanding planned-item quantity question."""
    if draft is None:
        return None
    questions = [q for q in draft.clarifications if q.reason != "completion_required"]
    ids = {q.plan_item_id for q in questions}
    if len(ids) == 1 and None not in ids:
        return next(iter(ids))
    return None


def _model_context(plan: MealPlan, draft: MealReportDraft | None) -> Mapping[str, object]:
    """Build nutrition-free state needed for pronouns and short replies."""

    if draft is None:
        return {
            "draft": {
                "state": "none",
                "resolved_items": [],
                "outstanding_questions": [],
            }
        }
    resolved: list[dict[str, str]] = []
    for item in draft.planned_items:
        planned = plan.item_for_id(item.plan_item_id)
        if planned is None:
            continue
        resolved.append(
            {
                "plan_item_id": item.plan_item_id,
                "name": planned.display_name,
                "action": item.action,
                "quantity_status": str(item.quantity_status),
                **(
                    {"quantity_text": item.original_quantity_text}
                    if item.original_quantity_text is not None
                    else {}
                ),
            }
        )
    questions: list[dict[str, str]] = []
    for clarification in draft.clarifications:
        question: dict[str, str] = {"reason": clarification.reason}
        if clarification.plan_item_id is not None:
            planned = plan.item_for_id(clarification.plan_item_id)
            if planned is not None:
                question["plan_item_id"] = clarification.plan_item_id
                question["name"] = planned.display_name
        if clarification.food_text is not None:
            question["food_text"] = clarification.food_text
        if clarification.quantity_text is not None:
            question["quantity_text"] = clarification.quantity_text
        questions.append(question)
    return {
        "draft": {
            "state": draft.status,
            "resolved_items": resolved,
            "outstanding_questions": questions,
            "last_clarification": draft.last_prompt,
        }
    }


def _merge_draft_facts(
    plan: MealPlan,
    draft: MealReportDraft | None,
    report: ReconciledMealReport,
    semantic: MealReportSemanticResult,
    source_event_id: str,
    user_text: str,
) -> tuple[
    tuple[DraftPlannedItem, ...],
    tuple[DraftUnplannedItem, ...],
    tuple[DraftClarification, ...],
]:
    """Merge effective fact axes without treating omission as deletion."""

    if report.plan != plan:
        raise MealReportConversationError("report does not belong to its plan")
    _text(source_event_id, "source_event_id")
    existing = {} if draft is None else {item.plan_item_id: item for item in draft.planned_items}
    conflicts: list[DraftClarification] = []
    eaten_by_id = {plan.item_id(item.plan_item): item for item in report.eaten_items}
    skipped_by_id = {plan.item_id(item.plan_item): item for item in report.skipped_items}
    clarification_by_id = {
        plan.item_id(item.plan_item): item
        for item in report.clarification_items
        if item.plan_item is not None
    }
    settled_plan_ids: set[str] = set()
    for stated in semantic.planned_items:
        item_id = stated.plan_item_id
        if item_id in eaten_by_id:
            eaten = eaten_by_id[item_id]
            candidate = DraftPlannedItem(
                item_id,
                "eaten",
                eaten.official_servings,
                eaten.quantity_source,
                eaten.original_user_phrase,
                (
                    eaten.original_user_phrase
                    if stated.quantity_relation == "modified"
                    and eaten.original_user_phrase != stated.reference_text
                    else stated.quantity_text
                ),
                "resolved",
                None,
                source_event_id,
            )
        elif item_id in skipped_by_id:
            skipped = skipped_by_id[item_id]
            candidate = DraftPlannedItem(
                item_id,
                "skipped",
                None,
                None,
                skipped.original_user_phrase,
                None,
                "not_applicable",
                None,
                source_event_id,
            )
        else:
            unresolved = clarification_by_id.get(item_id)
            reason = "planned_quantity_unresolved" if unresolved is None else unresolved.reason
            original_quantity_text = (
                unresolved.original_user_phrase
                if unresolved is not None and reason == "reported_quantity_not_grounded"
                else stated.quantity_text or (
                    unresolved.original_user_phrase
                    if unresolved is not None and reason == "plan_attestation_required"
                    else None
                )
            )
            candidate = DraftPlannedItem(
                item_id,
                "eaten",
                None,
                None,
                stated.reference_text,
                original_quantity_text,
                "unresolved",
                reason,
                source_event_id,
            )
        prior = existing.get(item_id)
        if (
            prior is not None
            and prior.quantity_status == "resolved"
            and candidate.unresolved_reason in {
                "reported_quantity_not_grounded",
                "planned_fraction_unresolved",
                "plan_attestation_required",
                "draft_quantity_reference_unresolved",
            }
        ):
            conflicts.append(DraftClarification(
                candidate.unresolved_reason,
                item_id,
                quantity_text=candidate.original_quantity_text,
            ))
            continue
        merged, conflict = _merge_planned_fact(
            prior,
            candidate,
            is_correction=stated.is_correction and _literal_correction_intent(user_text),
        )
        existing[item_id] = merged
        if conflict:
            conflicts.append(
                DraftClarification(
                    "contradictory_planned_report",
                    item_id,
                    quantity_text=stated.quantity_text,
                )
            )
        elif candidate.quantity_status == "resolved" and merged.quantity_status == "resolved":
            settled_plan_ids.add(item_id)

    unplanned = [] if draft is None else list(draft.unplanned_items)
    settled_unplanned_foods: set[str] = set()
    report_corrections = {
        (item.food_text.casefold().strip(), item.quantity_text): (
            item.is_correction and _literal_correction_intent(user_text)
        )
        for item in semantic.additional_foods
    }
    # Grounding may replace a model-selected earlier amount with the user's
    # later correction. In that case retain the correction intent for the
    # uniquely named food while the reconciler supplies the final amount.
    correction_foods = {
        item.food_text.casefold().strip()
        for item in semantic.additional_foods
        if item.is_correction and _literal_correction_intent(user_text)
    }
    ambiguous_correction_foods = {
        name for name in correction_foods
        if sum(item.food_text.casefold().strip() == name for item in semantic.additional_foods) != 1
    }
    for proposed in report.unplanned_items:
        identity_reason = None
        quantity_reason = None
        for clarification in report.clarification_items:
            if (
                clarification.food_text is not None
                and clarification.food_text.casefold().strip()
                == proposed.food_text.casefold().strip()
            ):
                if clarification.reason in {
                    "unplanned_food_ambiguous",
                    "unplanned_food_unresolved",
                }:
                    identity_reason = clarification.reason
                else:
                    quantity_reason = clarification.reason
        candidate = DraftUnplannedItem(
            proposed.food_text,
            proposed.quantity_text,
            proposed.resolved_food,
            proposed.official_servings,
            proposed.quantity_source,
            proposed.confidence,
            identity_status="resolved" if proposed.resolved_food is not None else "unresolved",
            identity_unresolved_reason=identity_reason,
            quantity_status="resolved" if proposed.official_servings is not None else "unresolved",
            quantity_unresolved_reason=(
                None
                if proposed.official_servings is not None
                else quantity_reason or "unplanned_quantity_required"
            ),
            last_source_event_id=source_event_id,
        )
        prior = _matching_unplanned_fact(unplanned, candidate)
        correction = report_corrections.get(
            (proposed.food_text.casefold().strip(), proposed.quantity_text),
            False,
        )
        if not correction and proposed.food_text.casefold().strip() in (
            correction_foods - ambiguous_correction_foods
        ):
            correction = True
        if prior is None:
            unplanned.append(candidate)
            if candidate.quantity_status == "resolved":
                settled_unplanned_foods.add(candidate.food_text.casefold().strip())
            continue
        merged, conflict = _merge_unplanned_fact(prior, candidate, is_correction=correction)
        unplanned[unplanned.index(prior)] = merged
        if conflict:
            conflicts.append(
                DraftClarification(
                    "contradictory_unplanned_report",
                    food_text=proposed.food_text,
                    quantity_text=proposed.quantity_text,
                )
            )
        elif candidate.quantity_status == "resolved" and merged.quantity_status == "resolved":
            settled_unplanned_foods.add(candidate.food_text.casefold().strip())

    questions = list(conflicts)
    if draft is not None:
        questions.extend(
            clarification
            for clarification in draft.clarifications
            if clarification.reason in {
                "reported_quantity_not_grounded",
                "planned_fraction_unresolved",
                "plan_attestation_required",
                "draft_quantity_reference_unresolved",
            }
            and clarification.plan_item_id not in settled_plan_ids
            and not any(
                current.plan_item_id == clarification.plan_item_id
                for current in conflicts
            )
        )
        questions.extend(
            clarification
            for clarification in draft.clarifications
            if clarification.reason == "contradictory_unplanned_report"
            and clarification.food_text is not None
            and clarification.food_text.casefold().strip() not in settled_unplanned_foods
            and not any(
                current.food_text == clarification.food_text
                for current in conflicts
            )
        )
    questions.extend(
        DraftClarification(
            item.unresolved_reason or "planned_quantity_unresolved",
            item.plan_item_id,
            quantity_text=item.original_quantity_text,
        )
        for item in existing.values()
        if item.quantity_status == "unresolved"
    )

    for item in unplanned:
        if item.identity_status == "unresolved":
            questions.append(
                DraftClarification(
                    item.identity_unresolved_reason or "unplanned_food_unresolved",
                    food_text=item.food_text,
                    quantity_text=item.quantity_text,
                )
            )
        elif item.quantity_status == "unresolved":
            questions.append(
                DraftClarification(
                    item.quantity_unresolved_reason or "unplanned_quantity_required",
                    food_text=item.food_text,
                    quantity_text=item.quantity_text,
                )
            )
    questions.extend(
        _draft_clarification_from_report(plan, item)
        for item in report.clarification_items
        if item.plan_item is None and item.food_text is None
    )
    if semantic.report_scope != "complete":
        questions.append(DraftClarification("completion_required"))

    return (
        tuple(sorted(existing.values(), key=lambda item: item.plan_item_id)),
        tuple(unplanned),
        _deduplicate_clarifications(questions),
    )


def _literal_correction_intent(user_text: str) -> bool:
    return bool(re.search(
        r"\b(?:actually|no|rather|instead|correction|make that)\b",
        user_text,
        re.IGNORECASE,
    ))


def _merge_planned_fact(
    prior: DraftPlannedItem | None,
    candidate: DraftPlannedItem,
    *,
    is_correction: bool,
) -> tuple[DraftPlannedItem, bool]:
    if prior is None:
        return candidate, False
    if _same_draft_planned_item(prior, candidate):
        return prior, False
    if is_correction:
        return DraftPlannedItem(
            candidate.plan_item_id,
            candidate.action,
            candidate.official_servings,
            candidate.quantity_source,
            candidate.original_reference_text,
            candidate.original_quantity_text,
            candidate.quantity_status,
            candidate.unresolved_reason,
            candidate.last_source_event_id,
            prior.fact_revision + 1,
        ), False
    if prior.quantity_status == "unresolved":
        return DraftPlannedItem(
            candidate.plan_item_id,
            candidate.action,
            candidate.official_servings,
            candidate.quantity_source,
            candidate.original_reference_text,
            candidate.original_quantity_text,
            candidate.quantity_status,
            candidate.unresolved_reason,
            candidate.last_source_event_id,
            prior.fact_revision,
        ), False
    return prior, True


def _matching_unplanned_fact(
    existing: list[DraftUnplannedItem],
    candidate: DraftUnplannedItem,
) -> DraftUnplannedItem | None:
    for prior in existing:
        if prior.food is not None and candidate.food is not None and (
            prior.food.occurrence.occurrence_id == candidate.food.occurrence.occurrence_id
            and prior.food.nutrition_snapshot_id == candidate.food.nutrition_snapshot_id
        ):
            return prior
        if prior.food_text.casefold().strip() == candidate.food_text.casefold().strip():
            return prior
    return None


def _merge_unplanned_fact(
    prior: DraftUnplannedItem,
    candidate: DraftUnplannedItem,
    *,
    is_correction: bool,
) -> tuple[DraftUnplannedItem, bool]:
    if _same_draft_unplanned_item(prior, candidate):
        return prior, False
    if prior.quantity_status == "resolved" and (
        candidate.identity_status == "unresolved" or candidate.quantity_text is None
    ):
        return prior, True
    if not is_correction and prior.quantity_status == "resolved":
        return prior, True
    return DraftUnplannedItem(
        candidate.food_text,
        candidate.quantity_text,
        candidate.food,
        candidate.official_servings,
        candidate.quantity_source,
        candidate.confidence,
        prior.fact_id,
        candidate.identity_status,
        candidate.identity_unresolved_reason,
        candidate.quantity_status,
        candidate.quantity_unresolved_reason,
        candidate.last_source_event_id,
        prior.fact_revision + (1 if is_correction else 0),
    ), False


def _report_from_draft_facts(
    plan: MealPlan,
    planned_items: tuple[DraftPlannedItem, ...],
    unplanned_items: tuple[DraftUnplannedItem, ...],
) -> ReconciledMealReport:
    by_id = {item.plan_item_id: item for item in planned_items}
    eaten: list[ProposedEatenMealItem] = []
    skipped: list[SkippedMealItem] = []
    unspecified = []
    for item in plan.items:
        draft_item = by_id.get(plan.item_id(item))
        if draft_item is None:
            unspecified.append(item)
        elif draft_item.action == "skipped":
            skipped.append(SkippedMealItem(item, draft_item.original_reference_text))
        elif draft_item.quantity_status == "unresolved":
            unspecified.append(item)
        else:
            assert draft_item.official_servings is not None
            assert draft_item.quantity_source is not None
            eaten.append(
                ProposedEatenMealItem(
                    item,
                    draft_item.official_servings,
                    draft_item.quantity_source,
                    draft_item.original_reference_text,
                )
            )
    unplanned = tuple(
        ProposedUnplannedMealItem(
            item.food_text,
            item.quantity_text,
            item.food,
            item.official_servings,
            item.quantity_source,
            item.confidence,
        )
        for item in unplanned_items
        if item.identity_status == "resolved" and item.quantity_status == "resolved"
    )
    return ReconciledMealReport(
        plan,
        tuple(eaten),
        tuple(skipped),
        tuple(unspecified),
        unplanned,
        (),
    )


def _draft_clarification_from_report(
    plan: MealPlan,
    item: ClarificationItem,
) -> DraftClarification:
    if item.plan_item is not None:
        return DraftClarification(item.reason, plan.item_id(item.plan_item))
    return DraftClarification(item.reason, food_text=item.food_text)


def _deduplicate_clarifications(
    values: list[DraftClarification],
) -> tuple[DraftClarification, ...]:
    seen: set[tuple[str, str | None, str | None, str | None]] = set()
    result: list[DraftClarification] = []
    for value in values:
        key = (value.reason, value.plan_item_id, value.food_text, value.quantity_text)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _same_draft_planned_item(left: DraftPlannedItem, right: DraftPlannedItem) -> bool:
    return (
        left.action == right.action
        and left.quantity_status == right.quantity_status
        and left.official_servings == right.official_servings
        and left.quantity_source == right.quantity_source
        and left.original_quantity_text == right.original_quantity_text
        and left.unresolved_reason == right.unresolved_reason
    )


def _same_draft_unplanned_item(left: DraftUnplannedItem, right: DraftUnplannedItem) -> bool:
    return (
        left.food_text == right.food_text
        and left.quantity_text == right.quantity_text
        and left.identity_status == right.identity_status
        and left.quantity_status == right.quantity_status
        and left.official_servings == right.official_servings
        and left.quantity_source == right.quantity_source
        and left.identity_unresolved_reason == right.identity_unresolved_reason
        and left.quantity_unresolved_reason == right.quantity_unresolved_reason
        and (
            (left.food is None and right.food is None)
            or (
                left.food is not None
                and right.food is not None
                and left.food.occurrence.occurrence_id == right.food.occurrence.occurrence_id
                and left.food.nutrition_snapshot_id == right.food.nutrition_snapshot_id
            )
        )
    )


def _has_report_content(report: ReconciledMealReport) -> bool:
    return bool(report.eaten_items or report.skipped_items or report.unplanned_items)


def _outcome_for_nonreport_intent(
    intent: MealReportIntent,
) -> Literal[
    "location_question",
    "replacement_request",
    "meal_request",
    "unsupported_or_ambiguous",
]:
    if intent == "location_question":
        return "location_question"
    if intent == "replacement_request":
        return "replacement_request"
    if intent == "meal_request":
        return "meal_request"
    return "unsupported_or_ambiguous"


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
