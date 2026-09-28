"""Transport-independent deterministic application commands and workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal
from decimal import InvalidOperation
import os
from typing import Literal

from .application_clock import (
    DEFAULT_NUTRITION_APPLICATION_CLOCK,
    NutritionApplicationClock,
)
from .durable_state import (
    AppliedMealReport,
    DurableMealState,
    MealReportCommitTimeRejection,
    PendingMealRequest,
    PersistedMealPlan,
)
from .fdmealplanner.catalog import OfficialNutritionCatalog
from .food_resolution import ResolvedFood
from .meal_identity import (
    MealSlot,
    canonical_meal_id,
    meal_context_key,
    meal_name_for_display,
    meal_values_equal,
)
from .meal_optimizer import (
    LocalMealOptimizer,
    MealOptimizationPolicy,
    MealRecommendation,
    RecommendedMealItem,
)
from .meal_request import RequestedMealFood, requested_meal_food_from_resolved
from .meal_report import MealReportReconciler, PlannedMealItem, ReconciledMealReport
from .meal_report_rendering import (
    format_meal_report_clarification,
    format_meal_report_confirmation,
)
from .nutrition import (
    DailyBalance,
    DailyLedger,
    DailyMinimums,
    DailyTargets,
    IntakeEntry,
    NutritionRecord,
    add_nutrients,
)
from .phelps_service_calendar import PhelpsServiceCalendar
from .recommendation_rendering import (
    MealRecommendationRendering,
    RecommendedPortionRenderer,
    RecommendedPortionSemanticRenderer,
    persist_rendered_meal_recommendation,
    render_meal_recommendation,
)


__all__ = [
    "DAILY_CALORIES_KCAL_ENV_VAR",
    "DAILY_CARBOHYDRATES_G_ENV_VAR",
    "DAILY_DIETARY_FIBER_MINIMUM_G_ENV_VAR",
    "DAILY_FAT_G_ENV_VAR",
    "DAILY_PROTEIN_G_ENV_VAR",
    "MealRecommendationOrchestrator",
    "MealRecommendationPreparationError",
    "MealRecommendationRequestError",
    "MealRecommendationReplacementUnavailableError",
    "MealRecommendationRequestedFoodError",
    "MealRecommendationServiceUnavailableError",
    "MealRecommendationUnavailableError",
    "MealReportNoActivePlanError",
    "MealReportOrchestrator",
    "MealReportProcessingError",
    "MealReportProcessingResult",
    "MealReportRequestError",
    "PreparedMealRecommendation",
    "PreparedMealRecommendationReplacement",
    "PreparedMealRecommendationRequest",
    "ProductionNutritionConfigurationError",
    "RecordIntakeCommand",
    "RecordIntakeResult",
    "execute_command",
    "load_production_daily_targets",
]


DAILY_CALORIES_KCAL_ENV_VAR = "NUTRITION_OPTIMIZER_DAILY_CALORIES_KCAL"
DAILY_PROTEIN_G_ENV_VAR = "NUTRITION_OPTIMIZER_DAILY_PROTEIN_G"
DAILY_CARBOHYDRATES_G_ENV_VAR = "NUTRITION_OPTIMIZER_DAILY_CARBOHYDRATES_G"
DAILY_FAT_G_ENV_VAR = "NUTRITION_OPTIMIZER_DAILY_FAT_G"
DAILY_DIETARY_FIBER_MINIMUM_G_ENV_VAR = (
    "NUTRITION_OPTIMIZER_DAILY_DIETARY_FIBER_MINIMUM_G"
)


class ProductionNutritionConfigurationError(ValueError):
    """Raised when required production nutrition targets are unavailable or invalid."""


class MealRecommendationPreparationError(RuntimeError):
    """Base error for a recommendation that cannot be safely prepared."""


class MealRecommendationRequestError(MealRecommendationPreparationError, ValueError):
    """Raised when a recommendation request lacks a known FD meal identity."""


class MealRecommendationUnavailableError(MealRecommendationPreparationError):
    """Raised when the optimizer deliberately returns no actionable meal plan."""

    def __init__(self, recommendation: MealRecommendation) -> None:
        if not isinstance(recommendation, MealRecommendation):
            raise TypeError("recommendation must be a MealRecommendation")
        self.recommendation = recommendation
        self.outcome = recommendation.diagnostics.outcome
        meal_name = meal_name_for_display(recommendation.meal)
        if self.outcome == "empty_menu":
            message = (
                "no current local menu is available for "
                f"{meal_name} on {recommendation.service_date.isoformat()}"
            )
        else:
            message = f"no actionable {meal_name} recommendation is available ({self.outcome})"
        super().__init__(message)


class MealRecommendationReplacementUnavailableError(MealRecommendationPreparationError):
    """Raised when no distinct authoritative pre-meal replacement is possible."""

    def __init__(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be non-empty text")
        self.reason = reason
        super().__init__("no valid replacement recommendation is available")


class MealRecommendationRequestedFoodError(MealRecommendationPreparationError):
    """Raised when an authoritative positive food constraint cannot be met."""

    def __init__(self, reason: str, requested_foods: tuple[RequestedMealFood, ...]) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be non-empty text")
        if not isinstance(requested_foods, tuple) or not requested_foods:
            raise ValueError("requested_foods must be a non-empty tuple")
        if not all(isinstance(food, RequestedMealFood) for food in requested_foods):
            raise TypeError("requested_foods must contain RequestedMealFood values")
        self.reason = reason
        self.requested_foods = requested_foods
        super().__init__("requested current-menu food cannot be included safely")


class MealRecommendationServiceUnavailableError(MealRecommendationPreparationError):
    """Raised when the current Phelps service opportunity is unavailable."""

    def __init__(
        self,
        service_date: date,
        meal: str | int,
        local_now: datetime,
    ) -> None:
        _validate_service_date(service_date)
        canonical_meal = _canonical_fd_meal_id(meal)
        if not isinstance(local_now, datetime) or local_now.tzinfo is None:
            raise TypeError("local_now must be timezone-aware")
        self.service_date = service_date
        self.meal = canonical_meal
        self.local_now = local_now
        super().__init__(
            "Phelps is not currently available for "
            f"{meal_name_for_display(canonical_meal)} recommendations"
        )


class MealReportProcessingError(RuntimeError):
    """Raised when an inbound meal report cannot be processed safely."""


class MealReportRequestError(MealReportProcessingError, ValueError):
    """Raised when inbound deterministic service context is invalid."""


class MealReportNoActivePlanError(MealReportProcessingError):
    """Raised when a new inbound report has no active plan for its context."""


@dataclass(frozen=True, slots=True)
class PreparedMealRecommendation:
    """One fully rendered and durably persisted transport-agnostic meal plan.

    A later successful preparation for the same service date and canonical meal
    supersedes this plan.  This object is a preparation result only: it neither
    sends a message nor provides delivery retry or transport idempotency
    semantics.
    """

    current_ledger: DailyLedger
    recommendation: MealRecommendation
    rendering: MealRecommendationRendering
    persisted_plan: PersistedMealPlan

    def __post_init__(self) -> None:
        if not isinstance(self.current_ledger, DailyLedger):
            raise TypeError("current_ledger must be a DailyLedger")
        if not isinstance(self.recommendation, MealRecommendation):
            raise TypeError("recommendation must be a MealRecommendation")
        if not isinstance(self.rendering, MealRecommendationRendering):
            raise TypeError("rendering must be a MealRecommendationRendering")
        if not isinstance(self.persisted_plan, PersistedMealPlan):
            raise TypeError("persisted_plan must be a PersistedMealPlan")
        if self.rendering.recommendation != self.recommendation:
            raise ValueError("rendering must belong to the recommendation")
        if self.persisted_plan.plan != self.rendering.meal_plan:
            raise ValueError("persisted plan must be the rendered meal plan")

    @property
    def plan_id(self) -> str:
        """Return the durable identifier for the prepared plan."""

        return self.persisted_plan.plan_id

    @property
    def message(self) -> str:
        """Return the concise, deterministic text ready for a future sender."""

        return self.rendering.message

    @property
    def projected_balance(self) -> DailyBalance:
        """Return the optimizer's projected balance without recalculating it."""

        return self.recommendation.projected_balance


@dataclass(frozen=True, slots=True)
class PreparedMealRecommendationReplacement:
    """One rendered but not-yet-persisted deterministic replacement plan.

    The optimizer produces the exact new recommendation here; durable state
    later atomically supersedes the prior plan and reserves outbound delivery.
    Keeping that write outside this calculation lets persistence bind the
    replacement to the inbound GUID without creating a second optimizer.
    """

    current_ledger: DailyLedger
    prior_plan: PersistedMealPlan
    rejected_plan_item_ids: tuple[str, ...]
    whole_meal: bool
    strategy: Literal["targeted_preserved", "wider_reoptimization", "whole_meal"]
    recommendation: MealRecommendation
    rendering: MealRecommendationRendering

    def __post_init__(self) -> None:
        if not isinstance(self.current_ledger, DailyLedger):
            raise TypeError("current_ledger must be a DailyLedger")
        if not isinstance(self.prior_plan, PersistedMealPlan):
            raise TypeError("prior_plan must be a PersistedMealPlan")
        if not isinstance(self.rejected_plan_item_ids, tuple) or not self.rejected_plan_item_ids:
            raise ValueError("rejected_plan_item_ids must be non-empty")
        if len(set(self.rejected_plan_item_ids)) != len(self.rejected_plan_item_ids):
            raise ValueError("rejected_plan_item_ids must not repeat")
        for item_id in self.rejected_plan_item_ids:
            if self.prior_plan.plan.item_for_id(item_id) is None:
                raise ValueError("rejected plan item does not belong to prior plan")
        if not isinstance(self.whole_meal, bool):
            raise TypeError("whole_meal must be a bool")
        if self.whole_meal and set(self.rejected_plan_item_ids) != {
            self.prior_plan.plan.item_id(item) for item in self.prior_plan.plan.items
        }:
            raise ValueError("whole-meal replacement must reject every prior plan item")
        if self.strategy not in {
            "targeted_preserved",
            "wider_reoptimization",
            "whole_meal",
        }:
            raise ValueError("replacement strategy is invalid")
        if not isinstance(self.recommendation, MealRecommendation):
            raise TypeError("recommendation must be a MealRecommendation")
        if not isinstance(self.rendering, MealRecommendationRendering):
            raise TypeError("rendering must be a MealRecommendationRendering")
        if self.rendering.recommendation != self.recommendation:
            raise ValueError("rendering must belong to recommendation")
        if (
            self.recommendation.service_date != self.prior_plan.plan.service_date
            or not meal_values_equal(self.recommendation.meal, self.prior_plan.plan.meal)
        ):
            raise ValueError("replacement recommendation must keep prior context")


@dataclass(frozen=True, slots=True)
class PreparedMealRecommendationRequest:
    """One calculated positive-constrained recommendation before persistence.

    The same deterministic optimizer supplies both immediate/new requests and
    active-plan replacements.  Keeping this value transport- and persistence-
    free lets the caller atomically bind it to the appropriate inbound GUID.
    """

    current_ledger: DailyLedger
    requested_foods: tuple[RequestedMealFood, ...]
    whole_meal: bool
    recommendation: MealRecommendation
    rendering: MealRecommendationRendering
    prior_plan: PersistedMealPlan | None = None
    strategy: Literal[
        "requested_new_meal",
        "requested_preserved",
        "requested_reoptimization",
        "requested_whole_meal",
    ] = "requested_new_meal"

    def __post_init__(self) -> None:
        if not isinstance(self.current_ledger, DailyLedger):
            raise TypeError("current_ledger must be a DailyLedger")
        if not isinstance(self.requested_foods, tuple):
            raise TypeError("requested_foods must be a tuple")
        if not all(isinstance(food, RequestedMealFood) for food in self.requested_foods):
            raise TypeError("requested_foods must contain RequestedMealFood values")
        if len({food.identity_with_signature for food in self.requested_foods}) != len(
            self.requested_foods
        ):
            raise ValueError("requested_foods must not repeat an authoritative food")
        if not isinstance(self.whole_meal, bool):
            raise TypeError("whole_meal must be a bool")
        if not self.whole_meal and not self.requested_foods:
            raise ValueError("targeted request needs requested_foods")
        if not isinstance(self.recommendation, MealRecommendation):
            raise TypeError("recommendation must be a MealRecommendation")
        if not isinstance(self.rendering, MealRecommendationRendering):
            raise TypeError("rendering must be a MealRecommendationRendering")
        if self.rendering.recommendation != self.recommendation:
            raise ValueError("rendering must belong to recommendation")
        if self.prior_plan is not None and not isinstance(self.prior_plan, PersistedMealPlan):
            raise TypeError("prior_plan must be a PersistedMealPlan or None")
        if self.strategy not in {
            "requested_new_meal",
            "requested_preserved",
            "requested_reoptimization",
            "requested_whole_meal",
        }:
            raise ValueError("request strategy is invalid")
        if self.prior_plan is None:
            if self.strategy != "requested_new_meal":
                raise ValueError("a new request needs requested_new_meal strategy")
        elif (
            self.recommendation.service_date != self.prior_plan.plan.service_date
            or not meal_values_equal(self.recommendation.meal, self.prior_plan.plan.meal)
        ):
            raise ValueError("requested replacement must keep prior context")
        recommended = {
            (
                item.occurrence.source_identifier.kind,
                item.occurrence.source_identifier.value,
                item.occurrence.content_signature,
            )
            for item in self.recommendation.items
        }
        if not {food.identity_with_signature for food in self.requested_foods} <= recommended:
            raise ValueError("recommendation must retain every requested food")


@dataclass(frozen=True, slots=True)
class MealReportProcessingResult:
    """Transport-neutral outcome for one inbound meal-report event.

    A normal application retains the exact reconciled report and durable
    application result.  A replay deliberately returns only historic accepted
    intake because existing storage does not retain a second conversational
    transcript from which skipped/unspecified assertions could be recreated.
    """

    outcome: Literal["applied", "clarification_required", "replayed"]
    persisted_plan: PersistedMealPlan
    reconciled_report: ReconciledMealReport | None
    application: AppliedMealReport | None
    updated_ledger: DailyLedger | None
    message: str

    def __post_init__(self) -> None:
        if self.outcome not in {"applied", "clarification_required", "replayed"}:
            raise ValueError("outcome is invalid")
        if not isinstance(self.persisted_plan, PersistedMealPlan):
            raise TypeError("persisted_plan must be a PersistedMealPlan")
        if self.reconciled_report is not None:
            if not isinstance(self.reconciled_report, ReconciledMealReport):
                raise TypeError("reconciled_report must be a ReconciledMealReport or None")
            if self.reconciled_report.plan != self.persisted_plan.plan:
                raise ValueError("reconciled_report must belong to persisted_plan")
        if self.application is not None and not isinstance(self.application, AppliedMealReport):
            raise TypeError("application must be an AppliedMealReport or None")
        if self.updated_ledger is not None and not isinstance(self.updated_ledger, DailyLedger):
            raise TypeError("updated_ledger must be a DailyLedger or None")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be non-empty text")

        if self.outcome == "clarification_required":
            if (
                self.reconciled_report is None
                or not self.reconciled_report.clarification_items
                or self.application is not None
                or self.updated_ledger is not None
                or self.persisted_plan.status != "active"
            ):
                raise ValueError("clarification result has invalid durable state")
        elif self.outcome == "applied":
            if (
                self.reconciled_report is None
                or self.reconciled_report.clarification_items
                or self.application is None
                or self.application.plan_id != self.persisted_plan.plan_id
                or self.updated_ledger is None
                or self.persisted_plan.status != "applied"
            ):
                raise ValueError("applied result has invalid durable state")
        elif (
            self.reconciled_report is not None
            or self.application is None
            or not self.application.already_applied
            or self.application.plan_id != self.persisted_plan.plan_id
            or self.updated_ledger is None
            or self.persisted_plan.status != "applied"
        ):
            raise ValueError("replay result has invalid durable state")

    @property
    def clarification_required(self) -> bool:
        """Return whether the plan remains active pending user clarification."""

        return self.outcome == "clarification_required"

    @property
    def already_applied(self) -> bool:
        """Return whether this outcome was loaded from a prior event application."""

        return self.application is not None and self.application.already_applied


class MealRecommendationOrchestrator:
    """Prepare one persisted local-FD recommendation without transport behavior.

    The catalog is the sole menu and nutrition source.  Its matching durable
    state is constructed here so the optimizer's menu occurrences and the
    persisted plan use one SQLite-backed immutable snapshot history.
    """

    def __init__(
        self,
        catalog: OfficialNutritionCatalog,
        renderer: RecommendedPortionRenderer | RecommendedPortionSemanticRenderer | None = None,
        *,
        service_calendar: PhelpsServiceCalendar | None = None,
        clock: NutritionApplicationClock | None = None,
    ) -> None:
        if not isinstance(catalog, OfficialNutritionCatalog):
            raise TypeError("catalog must be an OfficialNutritionCatalog")
        self._catalog = catalog
        self._state = DurableMealState(catalog)
        self._optimizer = LocalMealOptimizer(catalog)
        # The default remains deterministic for transport-independent callers.
        # A production composition site may inject the narrow reverse-direction
        # renderer; RecommendedPortionRenderer retains the canonical physical
        # quantity and fails back to deterministic text when that optional
        # presentation enhancement is unavailable or rejected.
        if renderer is None:
            self._renderer = RecommendedPortionRenderer()
        elif isinstance(renderer, RecommendedPortionRenderer):
            self._renderer = renderer
        elif callable(getattr(renderer, "render", None)):
            self._renderer = RecommendedPortionRenderer(renderer)
        else:
            raise TypeError("renderer must provide render")
        if service_calendar is not None and not isinstance(
            service_calendar, PhelpsServiceCalendar
        ):
            raise TypeError("service_calendar must be a PhelpsServiceCalendar")
        if clock is not None and not isinstance(clock, NutritionApplicationClock):
            raise TypeError("clock must be a NutritionApplicationClock")
        self._service_calendar = service_calendar
        self._clock = clock or DEFAULT_NUTRITION_APPLICATION_CLOCK

    def prepare(
        self,
        service_date: date,
        meal: str | int,
        targets: DailyTargets,
    ) -> PreparedMealRecommendation:
        """Load durable intake, optimize, render, and persist one new meal plan.

        This operation is intentionally not an outbound-send retry mechanism:
        every successful invocation persists a newly prepared active plan and
        atomically supersedes a prior active plan for the same service context.
        """

        return self._prepare(service_date, meal, targets, scheduled=False)

    def prepare_scheduled(
        self,
        service_date: date,
        meal: str | int,
        targets: DailyTargets,
        *,
        meal_slot: MealSlot | None = None,
    ) -> PreparedMealRecommendation:
        """Prepare one scheduler-owned plan without replacement semantics.

        Optimization, rendering, exact quantities, service-calendar checks,
        and the returned result are identical to :meth:`prepare`.  Only the
        final durable write differs: it atomically creates a pending scheduled
        dispatch and refuses both a scheduler race and an active manual plan.
        """

        return self._prepare(
            service_date,
            meal,
            targets,
            scheduled=True,
            meal_slot=meal_slot,
        )

    def prepare_scheduled_pending_request(
        self,
        pending_request: PendingMealRequest,
        targets: DailyTargets,
    ) -> PreparedMealRecommendation:
        """Consume one stored request while creating its scheduler-owned plan.

        The request's food identities were resolved when the inbound message
        arrived.  They are rechecked only against the current local menu by
        the normal optimizer; no network call or model interpretation occurs
        at the scheduler boundary.
        """

        if not isinstance(pending_request, PendingMealRequest):
            raise TypeError("pending_request must be a PendingMealRequest")
        if pending_request.status != "pending":
            raise ValueError("pending_request must still be pending")
        if not isinstance(targets, DailyTargets):
            raise TypeError("targets must be a DailyTargets")
        service_date = pending_request.service_date
        canonical_meal = _canonical_fd_meal_id(pending_request.meal)
        self._retire_stale_plans_and_require_current_eligibility(
            service_date,
            canonical_meal,
        )
        current_ledger = self._state.load_recommendation_ledger(service_date)
        recommendation = self._optimizer.optimize_meal(
            service_date,
            canonical_meal,
            targets,
            current_ledger,
            required_foods=pending_request.requested_foods,
        )
        if pending_request.requested_foods:
            _raise_if_requested_recommendation_unavailable(
                recommendation,
                pending_request.requested_foods,
            )
            _require_requested_foods(recommendation, pending_request.requested_foods)
        elif not recommendation.is_recommendation:
            raise MealRecommendationUnavailableError(recommendation)
        rendering = render_meal_recommendation(recommendation, self._renderer)
        persisted_plan = self._state.save_scheduled_meal_plan(
            rendering.meal_plan,
            meal_slot=pending_request.meal_slot,
            pending_request=pending_request,
        )
        return PreparedMealRecommendation(
            current_ledger=current_ledger,
            recommendation=recommendation,
            rendering=rendering,
            persisted_plan=persisted_plan,
        )

    def prepare_replacement(
        self,
        prior_plan: PersistedMealPlan,
        rejected_plan_item_ids: tuple[str, ...],
        *,
        whole_meal: bool,
        targets: DailyTargets,
    ) -> PreparedMealRecommendationReplacement:
        """Calculate one immutable replacement without changing durable state.

        This is deliberately a sibling of, rather than a shortcut around,
        :meth:`prepare`.  It uses the same local optimizer and renderer but
        first treats retained plan items as exact deterministic contributions
        to the current daily ledger.  A later durable transaction is solely
        responsible for making the calculated plan authoritative.
        """

        if not isinstance(prior_plan, PersistedMealPlan):
            raise TypeError("prior_plan must be a PersistedMealPlan")
        if not isinstance(rejected_plan_item_ids, tuple) or not rejected_plan_item_ids:
            raise ValueError("rejected_plan_item_ids must be a non-empty tuple")
        if len(set(rejected_plan_item_ids)) != len(rejected_plan_item_ids):
            raise ValueError("rejected_plan_item_ids must not repeat")
        if not isinstance(whole_meal, bool):
            raise TypeError("whole_meal must be a bool")
        if not isinstance(targets, DailyTargets):
            raise TypeError("targets must be a DailyTargets")

        current_plan = self._state.load_meal_plan(prior_plan.plan_id)
        if current_plan is None or current_plan.status != "active":
            raise MealRecommendationReplacementUnavailableError(
                "the prior plan is no longer active"
            )
        if current_plan.plan != prior_plan.plan:
            raise MealRecommendationReplacementUnavailableError(
                "the prior plan changed before replacement"
            )
        for item_id in rejected_plan_item_ids:
            if prior_plan.plan.item_for_id(item_id) is None:
                raise ValueError("rejected plan item does not belong to prior plan")

        service_date = prior_plan.plan.service_date
        canonical_meal = _canonical_fd_meal_id(prior_plan.plan.meal)
        self._retire_stale_plans_and_require_current_eligibility(
            service_date,
            canonical_meal,
        )
        refreshed_prior = self._state.load_meal_plan(prior_plan.plan_id)
        if refreshed_prior is None or refreshed_prior.status != "active":
            raise MealRecommendationReplacementUnavailableError(
                "the prior plan is no longer active"
            )

        rejected_ids = (
            tuple(prior_plan.plan.item_id(item) for item in prior_plan.plan.items)
            if whole_meal
            else rejected_plan_item_ids
        )
        rejected_id_set = set(rejected_ids)
        rejected_identities = {
            _planned_item_source_identity(prior_plan.plan.item_for_id(item_id))
            for item_id in rejected_ids
        }
        prior_identities = {
            _planned_item_source_identity(item) for item in prior_plan.plan.items
        }
        retained_items = tuple(
            _recommended_item_from_planned(item)
            for item in prior_plan.plan.items
            if prior_plan.plan.item_id(item) not in rejected_id_set
        )
        current_ledger = self._state.load_recommendation_ledger(service_date)

        if not whole_meal and retained_items:
            residual_ledger = current_ledger
            for item in retained_items:
                residual_ledger = residual_ledger.add_entry(
                    IntakeEntry(item.record, item.official_servings)
                )
            remaining_slots = MealOptimizationPolicy().max_distinct_foods - len(retained_items)
            if remaining_slots > 0:
                residual = self._optimizer.optimize_meal(
                    service_date,
                    canonical_meal,
                    targets,
                    residual_ledger,
                    policy=replace(
                        MealOptimizationPolicy(),
                        max_distinct_foods=remaining_slots,
                    ),
                    # Exclude every old identity while preserving retained
                    # exact items directly.  The changed portion must be a
                    # genuinely new authoritative current-menu food.
                    excluded_source_identities=prior_identities,
                )
                if residual.is_recommendation:
                    recommendation = _combine_replacement_recommendation(
                        service_date,
                        canonical_meal,
                        retained_items,
                        residual,
                    )
                    rendering = render_meal_recommendation(recommendation, self._renderer)
                    return PreparedMealRecommendationReplacement(
                        current_ledger=current_ledger,
                        prior_plan=prior_plan,
                        rejected_plan_item_ids=rejected_ids,
                        whole_meal=False,
                        strategy="targeted_preserved",
                        recommendation=recommendation,
                        rendering=rendering,
                    )

        # If freezing the retained exact quantities cannot create a valid new
        # recommendation, retain no foods and use the same optimizer over the
        # true daily intake.  Only the explicitly rejected provider identities
        # stay excluded; this is a single operation, not a preference profile.
        wider = self._optimizer.optimize_meal(
            service_date,
            canonical_meal,
            targets,
            current_ledger,
            excluded_source_identities=rejected_identities,
        )
        wider_identities = {
            _recommended_item_source_identity(item) for item in wider.items
        }
        if not wider.is_recommendation or wider_identities <= prior_identities:
            raise MealRecommendationReplacementUnavailableError(
                "the current menu has no distinct valid replacement"
            )
        rendering = render_meal_recommendation(wider, self._renderer)
        return PreparedMealRecommendationReplacement(
            current_ledger=current_ledger,
            prior_plan=prior_plan,
            rejected_plan_item_ids=rejected_ids,
            whole_meal=whole_meal,
            strategy="whole_meal" if whole_meal else "wider_reoptimization",
            recommendation=wider,
            rendering=rendering,
        )

    def prepare_requested_replacement(
        self,
        prior_plan: PersistedMealPlan,
        requested_foods: tuple[ResolvedFood, ...],
        *,
        whole_meal: bool,
        targets: DailyTargets,
    ) -> PreparedMealRecommendationRequest:
        """Calculate one active-plan replacement with positive food constraints.

        A requested food is first a uniquely resolved current FD occurrence;
        the optimizer then chooses its exact allowed serving quantity.  For a
        targeted request, non-requested prior items are kept at their existing
        exact quantities when there is room for a valid residual meal.  A
        single wider deterministic pass is used only when that preservation is
        infeasible.
        """

        if not isinstance(prior_plan, PersistedMealPlan):
            raise TypeError("prior_plan must be a PersistedMealPlan")
        requested = _requested_foods_from_resolved(requested_foods)
        if not isinstance(whole_meal, bool):
            raise TypeError("whole_meal must be a bool")
        if not isinstance(targets, DailyTargets):
            raise TypeError("targets must be a DailyTargets")

        current_plan = self._state.load_meal_plan(prior_plan.plan_id)
        if current_plan is None or current_plan.status != "active":
            raise MealRecommendationReplacementUnavailableError(
                "the prior plan is no longer active"
            )
        if current_plan.plan != prior_plan.plan:
            raise MealRecommendationReplacementUnavailableError(
                "the prior plan changed before replacement"
            )
        service_date = prior_plan.plan.service_date
        canonical_meal = _canonical_fd_meal_id(prior_plan.plan.meal)
        self._retire_stale_plans_and_require_current_eligibility(service_date, canonical_meal)
        refreshed_prior = self._state.load_meal_plan(prior_plan.plan_id)
        if refreshed_prior is None or refreshed_prior.status != "active":
            raise MealRecommendationReplacementUnavailableError(
                "the prior plan is no longer active"
            )

        current_ledger = self._state.load_recommendation_ledger(service_date)
        requested_source_identities = {food.source_identity for food in requested}
        prior_identities = {
            _planned_item_source_identity(item) for item in prior_plan.plan.items
        }
        current_occurrences = {
            (
                occurrence.source_identifier.kind,
                occurrence.source_identifier.value,
                occurrence.content_signature,
            ): occurrence
            for occurrence in self._catalog.list_current_meal_occurrences(
                service_date, meal=canonical_meal,
            )
        }
        retained_items = tuple(
            RecommendedMealItem(
                current_occurrences[identity],
                current_occurrences[identity].nutrition_record,
                item.recommended_official_servings,
            )
            for item in prior_plan.plan.items
            if _planned_item_source_identity(item) not in requested_source_identities
            and (identity := (
                item.food.source_identifier.kind,
                item.food.source_identifier.value,
                item.food.content_signature,
            )) in current_occurrences
        ) if not whole_meal else ()

        if not whole_meal and retained_items:
            remaining_slots = MealOptimizationPolicy().max_distinct_foods - len(retained_items)
            if remaining_slots >= len(requested) and remaining_slots > 0:
                residual_ledger = current_ledger
                for item in retained_items:
                    residual_ledger = residual_ledger.add_entry(
                        IntakeEntry(item.record, item.official_servings)
                    )
                residual = self._optimizer.optimize_meal(
                    service_date,
                    canonical_meal,
                    targets,
                    residual_ledger,
                    policy=replace(
                        MealOptimizationPolicy(),
                        max_distinct_foods=remaining_slots,
                    ),
                    # The retained values are contributed directly, so only a
                    # specifically requested old identity may be considered
                    # by the residual optimizer.
                    excluded_source_identities=prior_identities - requested_source_identities,
                    required_foods=requested,
                )
                if residual.is_recommendation:
                    recommendation = _combine_replacement_recommendation(
                        service_date,
                        canonical_meal,
                        retained_items,
                        residual,
                    )
                    _require_requested_foods(recommendation, requested)
                    rendering = render_meal_recommendation(recommendation, self._renderer)
                    return PreparedMealRecommendationRequest(
                        current_ledger=current_ledger,
                        requested_foods=requested,
                        whole_meal=False,
                        recommendation=recommendation,
                        rendering=rendering,
                        prior_plan=prior_plan,
                        strategy="requested_preserved",
                    )

        wider = self._optimizer.optimize_meal(
            service_date,
            canonical_meal,
            targets,
            current_ledger,
            # A whole-meal request retains none of the old menu identities
            # unless the user explicitly asked for that same authoritative
            # food. A targeted request may reuse non-requested food only in
            # this one fallback calculation.
            excluded_source_identities=(
                prior_identities - requested_source_identities if whole_meal else ()
            ),
            required_foods=requested,
        )
        _raise_if_requested_recommendation_unavailable(wider, requested)
        _require_requested_foods(wider, requested)
        rendering = render_meal_recommendation(wider, self._renderer)
        return PreparedMealRecommendationRequest(
            current_ledger=current_ledger,
            requested_foods=requested,
            whole_meal=whole_meal,
            recommendation=wider,
            rendering=rendering,
            prior_plan=prior_plan,
            strategy="requested_whole_meal" if whole_meal else "requested_reoptimization",
        )

    def prepare_requested_meal(
        self,
        service_date: date,
        meal: str | int,
        requested_foods: tuple[ResolvedFood, ...],
        targets: DailyTargets,
        *,
        whole_meal: bool = False,
    ) -> PreparedMealRecommendationRequest:
        """Calculate a new immediate meal request without persisting it.

        The caller owns the single transaction that associates this immutable
        calculation with an inbound GUID and outbound delivery state.
        """

        _validate_service_date(service_date)
        canonical_meal = _canonical_fd_meal_id(meal)
        if not isinstance(whole_meal, bool):
            raise TypeError("whole_meal must be a bool")
        requested = _requested_foods_from_resolved(
            requested_foods,
            allow_empty=whole_meal,
        )
        if not isinstance(targets, DailyTargets):
            raise TypeError("targets must be a DailyTargets")
        self._retire_stale_plans_and_require_current_eligibility(service_date, canonical_meal)
        current_ledger = self._state.load_recommendation_ledger(service_date)
        recommendation = self._optimizer.optimize_meal(
            service_date,
            canonical_meal,
            targets,
            current_ledger,
            required_foods=requested,
        )
        _raise_if_requested_recommendation_unavailable(recommendation, requested)
        _require_requested_foods(recommendation, requested)
        rendering = render_meal_recommendation(recommendation, self._renderer)
        return PreparedMealRecommendationRequest(
            current_ledger=current_ledger,
            requested_foods=requested,
            whole_meal=whole_meal,
            recommendation=recommendation,
            rendering=rendering,
        )

    def _prepare(
        self,
        service_date: date,
        meal: str | int,
        targets: DailyTargets,
        *,
        scheduled: bool,
        meal_slot: MealSlot | None = None,
    ) -> PreparedMealRecommendation:
        _validate_service_date(service_date)
        canonical_meal = _canonical_fd_meal_id(meal)
        if not isinstance(targets, DailyTargets):
            raise TypeError("targets must be a DailyTargets")

        self._retire_stale_plans_and_require_current_eligibility(
            service_date,
            canonical_meal,
        )

        current_ledger = self._state.load_recommendation_ledger(service_date)
        recommendation = self._optimizer.optimize_meal(
            service_date,
            canonical_meal,
            targets,
            current_ledger,
        )
        if not recommendation.is_recommendation:
            raise MealRecommendationUnavailableError(recommendation)

        # Rendering and all quantity-chain validation happen before the single
        # durable write.  DurableMealState owns the transactional save itself.
        rendering = render_meal_recommendation(recommendation, self._renderer)
        persisted_plan = (
            self._state.save_scheduled_meal_plan(
                rendering.meal_plan,
                meal_slot=meal_slot,
            )
            if scheduled
            else persist_rendered_meal_recommendation(rendering, self._state)
        )
        return PreparedMealRecommendation(
            current_ledger=current_ledger,
            recommendation=recommendation,
            rendering=rendering,
            persisted_plan=persisted_plan,
        )

    def _retire_stale_plans_and_require_current_eligibility(
        self,
        service_date: date,
        canonical_meal: int,
    ) -> None:
        """Apply production availability only when a calendar is composed.

        Transport-independent callers retain their explicit service-date
        behavior.  The production composition supplies the Phelps calendar,
        which first performs deterministic stale retirement and then rejects a
        recommendation for *today* outside an eligible published opportunity.
        Explicit past/future service dates are intentionally not rewritten to
        the current date, preserving controlled backfills and deterministic
        test preparation.
        """

        if self._service_calendar is None:
            return
        local_now = self._clock.now()
        self._state.retire_stale_active_meal_plans(
            self._service_calendar,
            evaluated_at=local_now,
        )
        if (
            service_date == local_now.date()
            and not self._service_calendar.is_meal_context_eligible_at(
                service_date,
                canonical_meal,
                local_now,
            )
        ):
            raise MealRecommendationServiceUnavailableError(
                service_date,
                canonical_meal,
                local_now,
            )


class MealReportOrchestrator:
    """Apply one known-plan natural-language report without transport behavior.

    The injected reconciler owns only semantic interpretation, scoped local food
    resolution, and portion interpretation.  This application boundary chooses
    the durable plan by canonical service context, applies exact accepted
    intake, and owns lifecycle/idempotency transitions.
    """

    def __init__(
        self,
        catalog: OfficialNutritionCatalog,
        reconciler: MealReportReconciler,
        *,
        service_calendar: PhelpsServiceCalendar | None = None,
        clock: NutritionApplicationClock | None = None,
    ) -> None:
        if not isinstance(catalog, OfficialNutritionCatalog):
            raise TypeError("catalog must be an OfficialNutritionCatalog")
        if not isinstance(reconciler, MealReportReconciler):
            raise TypeError("reconciler must be a MealReportReconciler")
        if service_calendar is not None and not isinstance(
            service_calendar, PhelpsServiceCalendar
        ):
            raise TypeError("service_calendar must be a PhelpsServiceCalendar")
        if clock is not None and not isinstance(clock, NutritionApplicationClock):
            raise TypeError("clock must be a NutritionApplicationClock")
        self._state = DurableMealState(catalog)
        self._reconciler = reconciler
        self._service_calendar = service_calendar
        self._clock = clock or DEFAULT_NUTRITION_APPLICATION_CLOCK
        self._enforce_commit_reportability = service_calendar is not None or clock is not None

    def process(
        self,
        service_date: date,
        meal: str | int,
        raw_user_text: str,
        source_event_id: str,
    ) -> MealReportProcessingResult:
        """Reconcile and apply one event against its sole authoritative plan."""

        _validate_service_date(service_date)
        try:
            canonical_meal = _canonical_fd_meal_id(meal)
        except MealRecommendationRequestError as exc:
            raise MealReportRequestError(str(exc)) from None
        _required_text(raw_user_text, "raw_user_text")
        _required_text(source_event_id, "source_event_id")

        replayed_plan = self._state.load_meal_plan_for_source_event(source_event_id)
        if replayed_plan is not None:
            if (
                replayed_plan.plan.service_date != service_date
                or meal_context_key(replayed_plan.plan.meal)
                != meal_context_key(canonical_meal)
            ):
                raise MealReportProcessingError(
                    "source_event_id belongs to a different meal service context"
                )
            if replayed_plan.status != "applied":
                raise MealReportProcessingError(
                    "source_event_id references a meal plan with invalid lifecycle state"
                )
            applied = self._state.load_applied_meal_report(source_event_id)
            if applied is None:
                raise MealReportProcessingError("source_event_id application is missing")
            return MealReportProcessingResult(
                "replayed",
                replayed_plan,
                None,
                applied,
                self._state.load_recommendation_ledger(service_date),
                "Got it — that meal report was already logged.",
            )

        reportability_now = self._retire_stale_plans_and_require_current_reportability(
            service_date,
        )

        persisted_plan = self._state.load_active_meal_plan(service_date, canonical_meal)
        if persisted_plan is None:
            raise MealReportNoActivePlanError(
                "no active meal plan exists for that service date and meal"
            )
        if reportability_now is not None:
            assert self._service_calendar is not None
            if not self._state.is_active_meal_plan_reportable(
                persisted_plan,
                self._service_calendar,
                evaluated_at=reportability_now,
            ):
                raise MealReportNoActivePlanError(
                    "no active meal plan exists for the current reportable service"
                )
        report = self._reconciler.reconcile(persisted_plan.plan, raw_user_text)
        if not isinstance(report, ReconciledMealReport):
            raise MealReportProcessingError("meal report reconciler returned invalid output")
        if report.clarification_items:
            return MealReportProcessingResult(
                "clarification_required",
                persisted_plan,
                report,
                None,
                None,
                format_meal_report_clarification(report),
            )

        try:
            application = self._state.apply_reconciled_meal_report(
                persisted_plan,
                report,
                source_event_id=source_event_id,
                application_clock=(
                    self._clock if self._enforce_commit_reportability else None
                ),
                service_calendar=self._service_calendar,
            )
        except MealReportCommitTimeRejection:
            raise MealReportNoActivePlanError(
                "meal plan is no longer reportable at final commit"
            ) from None
        completed_plan = self._state.load_meal_plan(persisted_plan.plan_id)
        if completed_plan is None or completed_plan.status != "applied":
            raise MealReportProcessingError("meal report application did not complete its plan")
        return MealReportProcessingResult(
            "applied",
            completed_plan,
            report,
            application,
            self._state.load_recommendation_ledger(service_date),
            format_meal_report_confirmation(report),
        )

    def _retire_stale_plans_and_require_current_reportability(
        self,
        service_date: date,
    ) -> datetime | None:
        """Defend the application boundary after replay handling.

        The context resolver performs the normal reportability selection.  This
        small second check closes the race between resolution and a potentially
        slow semantic interpretation.  It intentionally distinguishes a
        delivered scheduler plan, which remains reportable after Phelps closes,
        from a plan that is still eligible for a new recommendation. Durable
        replay is returned before this method so an already-applied source
        event remains retryable forever.
        """

        if self._service_calendar is None:
            return None
        local_now = self._clock.now()
        self._state.retire_stale_active_meal_plans(
            self._service_calendar,
            evaluated_at=local_now,
        )
        if service_date != local_now.date():
            raise MealReportNoActivePlanError(
                "no active meal plan exists for the current reportable service"
            )
        return local_now


def load_production_daily_targets(
    environ: Mapping[str, str] | None = None,
) -> DailyTargets:
    """Load all required production targets from explicit environment values.

    No target has a fallback.  Values are parsed directly as finite positive
    :class:`~decimal.Decimal` instances before the existing domain type owns
    the resulting daily target bundle.
    """

    source = os.environ if environ is None else environ
    if not isinstance(source, Mapping):
        raise TypeError("environ must be a mapping")
    return DailyTargets(
        calories_kcal=_required_positive_decimal(source, DAILY_CALORIES_KCAL_ENV_VAR),
        protein_g=_required_positive_decimal(source, DAILY_PROTEIN_G_ENV_VAR),
        carbohydrates_g=_required_positive_decimal(
            source, DAILY_CARBOHYDRATES_G_ENV_VAR
        ),
        fat_g=_required_positive_decimal(source, DAILY_FAT_G_ENV_VAR),
        minimums=DailyMinimums(
            dietary_fiber_g=_required_positive_decimal(
                source, DAILY_DIETARY_FIBER_MINIMUM_G_ENV_VAR
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class RecordIntakeCommand:
    """Record an already-resolved nutrition record for some servings."""

    record: NutritionRecord
    servings: Decimal


@dataclass(frozen=True, slots=True)
class RecordIntakeResult:
    """The immutable ledger produced by a successful intake command."""

    updated_ledger: DailyLedger


def execute_command(
    ledger: DailyLedger,
    command: RecordIntakeCommand,
) -> RecordIntakeResult:
    """Execute one supported structured command deterministically."""

    if not isinstance(ledger, DailyLedger):
        raise TypeError("ledger must be a DailyLedger")
    if not isinstance(command, RecordIntakeCommand):
        raise TypeError("unsupported application command")

    # IntakeEntry owns record and serving validation; DailyLedger owns the
    # immutable append operation.  No nutrition values are resolved here.
    entry = IntakeEntry(command.record, command.servings)
    return RecordIntakeResult(updated_ledger=ledger.add_entry(entry))


def _required_positive_decimal(source: Mapping[str, str], variable_name: str) -> Decimal:
    value = source.get(variable_name)
    if not isinstance(value, str) or not value.strip():
        raise ProductionNutritionConfigurationError(f"{variable_name} must be set")
    try:
        parsed = Decimal(value.strip())
    except (InvalidOperation, ValueError):
        raise ProductionNutritionConfigurationError(
            f"{variable_name} must be a positive finite decimal"
        ) from None
    if not parsed.is_finite() or parsed <= Decimal("0"):
        raise ProductionNutritionConfigurationError(
            f"{variable_name} must be a positive finite decimal"
        )
    return parsed


def _requested_foods_from_resolved(
    foods: tuple[ResolvedFood, ...],
    *,
    allow_empty: bool = False,
) -> tuple[RequestedMealFood, ...]:
    """Convert only uniquely local-resolved foods into optimizer constraints."""

    if not isinstance(foods, tuple):
        raise TypeError("requested_foods must be a tuple")
    if not foods and not allow_empty:
        raise ValueError("requested_foods must be a non-empty tuple")
    if not all(isinstance(food, ResolvedFood) for food in foods):
        raise TypeError("requested_foods must contain ResolvedFood values")
    requested = tuple(requested_meal_food_from_resolved(food) for food in foods)
    if len({food.identity_with_signature for food in requested}) != len(requested):
        raise ValueError("requested_foods must not repeat an authoritative food")
    return requested


def _require_requested_foods(
    recommendation: MealRecommendation,
    requested_foods: tuple[RequestedMealFood, ...],
) -> None:
    """Defend the positive constraint at the application boundary too."""

    if not isinstance(recommendation, MealRecommendation):
        raise TypeError("recommendation must be a MealRecommendation")
    required = {food.identity_with_signature for food in requested_foods}
    actual = {
        (
            item.occurrence.source_identifier.kind,
            item.occurrence.source_identifier.value,
            item.occurrence.content_signature,
        )
        for item in recommendation.items
    }
    if not required <= actual:
        raise MealRecommendationRequestedFoodError(
            "requested_food_not_included",
            requested_foods,
        )


def _raise_if_requested_recommendation_unavailable(
    recommendation: MealRecommendation,
    requested_foods: tuple[RequestedMealFood, ...],
) -> None:
    """Translate a failed required optimization into a precise safe error."""

    if not isinstance(recommendation, MealRecommendation):
        raise TypeError("recommendation must be a MealRecommendation")
    if recommendation.is_recommendation:
        return
    if recommendation.diagnostics.outcome in {
        "required_food_unavailable",
        "required_food_infeasible",
    }:
        raise MealRecommendationRequestedFoodError(
            recommendation.diagnostics.outcome,
            requested_foods,
        )
    raise MealRecommendationUnavailableError(recommendation)


def _planned_item_source_identity(item: PlannedMealItem | None) -> tuple[str, str]:
    """Return one persisted provider identity without using display text."""

    if not isinstance(item, PlannedMealItem):
        raise ValueError("planned replacement item is missing")
    return (item.food.source_identifier.kind, item.food.source_identifier.value)


def _recommended_item_source_identity(item: RecommendedMealItem) -> tuple[str, str]:
    if not isinstance(item, RecommendedMealItem):
        raise TypeError("item must be a RecommendedMealItem")
    return (item.occurrence.source_identifier.kind, item.occurrence.source_identifier.value)


def _recommended_item_from_planned(item: PlannedMealItem) -> RecommendedMealItem:
    """Project an exact persisted plan item into the existing optimizer shape."""

    if not isinstance(item, PlannedMealItem):
        raise TypeError("item must be a PlannedMealItem")
    return RecommendedMealItem(
        item.food.occurrence,
        item.food.nutrition_record,
        item.recommended_official_servings,
    )


def _combine_replacement_recommendation(
    service_date: date,
    canonical_meal: int,
    retained_items: tuple[RecommendedMealItem, ...],
    residual: MealRecommendation,
) -> MealRecommendation:
    """Combine frozen exact plan items with one ordinary optimizer result."""

    if not retained_items:
        raise ValueError("retained_items must not be empty")
    if not isinstance(residual, MealRecommendation) or not residual.is_recommendation:
        raise ValueError("residual must be a recommendation")
    items = retained_items + residual.items
    if len({item.occurrence.occurrence_id for item in items}) != len(items):
        raise MealRecommendationReplacementUnavailableError(
            "the current menu cannot provide a distinct replacement"
        )
    return MealRecommendation(
        service_date=service_date,
        meal=canonical_meal,
        items=items,
        projected_meal_nutrition=add_nutrients(
            *(item.nutrition_contribution for item in items)
        ),
        # ``residual`` optimized against a ledger that already contained the
        # retained exact quantities, so its projected daily state is the final
        # replacement state rather than a separate estimate.
        projected_daily_total=residual.projected_daily_total,
        projected_balance=residual.projected_balance,
        objective_score=residual.objective_score,
        diagnostics=residual.diagnostics,
    )


def _validate_service_date(service_date: date) -> None:
    if not isinstance(service_date, date) or isinstance(service_date, datetime):
        raise TypeError("service_date must be a date")


def _canonical_fd_meal_id(meal: str | int) -> int:
    if isinstance(meal, bool) or not isinstance(meal, (str, int)):
        raise TypeError("meal must be text or an integer")
    canonical = canonical_meal_id(meal)
    if canonical is None:
        raise MealRecommendationRequestError(
            "meal must be a known FD meal name or numeric meal ID"
        )
    return canonical


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MealReportRequestError(f"{name} must be non-empty text")
    return value
