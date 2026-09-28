"""Deterministic, local-cache-only meal recommendation optimization.

This module deliberately contains no persistence, network, messaging, natural
language portion rendering, or AI dependencies.  It selects official-serving
quantities on a unit-aware physical grid: discrete foods use whole item counts,
while continuous foods use the configured serving grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Iterable, Literal, Protocol

from .fdmealplanner.catalog import FDMenuOccurrence
from .meal_request import RequestedMealFood
from .meal_identity import canonical_meal_name, meal_identity_matches
from .nutrition import (
    DailyBalance,
    DailyLedger,
    DailyTargets,
    NutrientProfile,
    NutritionRecord,
    add_nutrients,
    calculate_daily_balance,
    scale_nutrients,
)
from .nutrition.models import Serving
from .physical_quantity import (
    PhysicalRecommendedQuantity,
    classify_serving_unit,
    physical_quantity_for,
    serving_multipliers_for_physical_counts,
)

__all__ = [
    "LocalMealOptimizer",
    "MealOptimizationDiagnostics",
    "MealOptimizationPolicy",
    "MealRecommendation",
    "NutrientObjectiveWeights",
    "RecommendedMealItem",
    "optimize_meal",
]


_ZERO = Decimal("0")
_ONE = Decimal("1")
_REQUIRED_NUTRIENTS = (
    "calories_kcal",
    "protein_g",
    "carbohydrates_g",
    "fat_g",
)
OptimizationOutcome = Literal[
    "recommended",
    "empty_menu",
    "unknown_current_balance",
    "targets_satisfied_or_exceeded",
    "no_eligible_candidates",
    "no_improving_recommendation",
    "required_food_unavailable",
    "required_food_infeasible",
]


def _positive_decimal(value: Decimal, field_name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite() or value <= _ZERO:
        raise ValueError(f"{field_name} must be a positive finite Decimal")


def _non_negative_decimal(value: Decimal, field_name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite() or value < _ZERO:
        raise ValueError(f"{field_name} must be a non-negative finite Decimal")


@dataclass(frozen=True, slots=True)
class NutrientObjectiveWeights:
    """Relative, normalized importance for the four existing daily targets.

    Values are deliberately policy rather than targets.  Since every term is
    divided by its caller-supplied daily target, calories and gram nutrients
    are comparable without treating raw units as interchangeable.
    """

    calories_kcal: Decimal = _ONE
    protein_g: Decimal = Decimal("1.20")
    carbohydrates_g: Decimal = Decimal("0.80")
    fat_g: Decimal = Decimal("0.80")
    dietary_fiber_g: Decimal = Decimal("0.75")

    def __post_init__(self) -> None:
        for field_name in _REQUIRED_NUTRIENTS:
            _positive_decimal(getattr(self, field_name), field_name)
        _positive_decimal(self.dietary_fiber_g, "dietary_fiber_g")


@dataclass(frozen=True, slots=True)
class MealOptimizationPolicy:
    """Mechanical constraints and objective configuration for one optimizer.

    The defaults are generic usability constraints, not personal nutritional
    targets: up to two official servings of a food and at most three distinct
    foods. Continuous serving units use quarter-serving increments. Discrete
    units instead use whole physical item counts whose derived official
    multipliers stay inside these same bounds. Stage fractions say what share
    of *the current remaining deficit* a meal should try to address. Dinner
    is allowed to address all remaining deficits because no future major meal
    is assumed; breakfast is intentionally less aggressive.
    """

    minimum_servings: Decimal = Decimal("0.25")
    maximum_servings_per_food: Decimal = Decimal("2.00")
    serving_increment: Decimal = Decimal("0.25")
    max_distinct_foods: int = 3
    beam_width: int = 1000
    complexity_penalty: Decimal = Decimal("0.015")
    deficit_penalty: Decimal = _ONE
    overshoot_penalty: Decimal = Decimal("1.75")
    breakfast_stage_fraction: Decimal = Decimal("0.35")
    lunch_stage_fraction: Decimal = Decimal("0.60")
    dinner_stage_fraction: Decimal = _ONE
    other_meal_stage_fraction: Decimal = Decimal("0.60")
    weights: NutrientObjectiveWeights = NutrientObjectiveWeights()

    def __post_init__(self) -> None:
        _positive_decimal(self.minimum_servings, "minimum_servings")
        _positive_decimal(self.maximum_servings_per_food, "maximum_servings_per_food")
        _positive_decimal(self.serving_increment, "serving_increment")
        if self.maximum_servings_per_food < self.minimum_servings:
            raise ValueError("maximum_servings_per_food must be at least minimum_servings")
        if self.max_distinct_foods <= 0:
            raise ValueError("max_distinct_foods must be greater than zero")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be greater than zero")
        _non_negative_decimal(self.complexity_penalty, "complexity_penalty")
        _positive_decimal(self.deficit_penalty, "deficit_penalty")
        _positive_decimal(self.overshoot_penalty, "overshoot_penalty")
        for field_name in (
            "breakfast_stage_fraction",
            "lunch_stage_fraction",
            "dinner_stage_fraction",
            "other_meal_stage_fraction",
        ):
            value = getattr(self, field_name)
            _positive_decimal(value, field_name)
            if value > _ONE:
                raise ValueError(f"{field_name} cannot exceed one")
        if not isinstance(self.weights, NutrientObjectiveWeights):
            raise TypeError("weights must be NutrientObjectiveWeights")

    def stage_fraction_for(self, meal: str | int) -> Decimal:
        """Return the configured aggressiveness for a meal label or ID."""

        normalized = canonical_meal_name(meal)
        if normalized == "breakfast":
            return self.breakfast_stage_fraction
        if normalized == "lunch":
            return self.lunch_stage_fraction
        if normalized == "dinner":
            return self.dinner_stage_fraction
        return self.other_meal_stage_fraction

    def serving_multipliers(self) -> tuple[Decimal, ...]:
        """Return the configured grid for continuous official-serving units."""

        values: list[Decimal] = []
        value = self.minimum_servings
        while value <= self.maximum_servings_per_food:
            values.append(value)
            value += self.serving_increment
        return tuple(values)

    def serving_multipliers_for(self, serving: Serving) -> tuple[Decimal, ...]:
        """Return a physically safe official-serving grid for one definition."""

        if not isinstance(serving, Serving):
            raise TypeError("serving must be a Serving")
        category = classify_serving_unit(serving.unit)
        if category == "discrete_count":
            return serving_multipliers_for_physical_counts(
                serving,
                self.minimum_servings,
                self.maximum_servings_per_food,
            )
        if category == "continuous":
            return self.serving_multipliers()
        # An unfamiliar provider unit has no safe physical interpretation.
        return ()


@dataclass(frozen=True, slots=True)
class RecommendedMealItem:
    """One exact current FD occurrence selected at an official quantity."""

    occurrence: FDMenuOccurrence
    record: NutritionRecord
    official_servings: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.occurrence, FDMenuOccurrence):
            raise TypeError("occurrence must be an FDMenuOccurrence")
        if not isinstance(self.record, NutritionRecord):
            raise TypeError("record must be a NutritionRecord")
        if self.occurrence.nutrition_record != self.record:
            raise ValueError("record must be the occurrence's exact nutrition snapshot")
        _positive_decimal(self.official_servings, "official_servings")
        # Enforce the physical invariant at the optimizer-item boundary. A
        # three-each serving cannot be represented by an arbitrary .75
        # multiplier because that would mean 2.25 physical items.
        physical_quantity_for(self.record.serving, self.official_servings)

    @property
    def physical_quantity(self) -> PhysicalRecommendedQuantity:
        """Return the exact physical quantity represented by this item."""

        return physical_quantity_for(self.record.serving, self.official_servings)

    @property
    def nutrition_contribution(self) -> NutrientProfile:
        return scale_nutrients(self.record.nutrients, self.official_servings)


@dataclass(frozen=True, slots=True)
class MealOptimizationDiagnostics:
    """Auditable deterministic search facts, not user-facing prose."""

    candidate_occurrences: int
    eligible_candidates: int
    duplicate_occurrences_collapsed: int
    excluded_wrong_context: int
    excluded_missing_required_nutrition: int
    excluded_invalid_serving: int
    search_states_evaluated: int
    baseline_objective_score: Decimal
    stage_fraction: Decimal
    outcome: OptimizationOutcome

    def __post_init__(self) -> None:
        for field_name in (
            "candidate_occurrences",
            "eligible_candidates",
            "duplicate_occurrences_collapsed",
            "excluded_wrong_context",
            "excluded_missing_required_nutrition",
            "excluded_invalid_serving",
            "search_states_evaluated",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        _non_negative_decimal(self.baseline_objective_score, "baseline_objective_score")
        _positive_decimal(self.stage_fraction, "stage_fraction")


@dataclass(frozen=True, slots=True)
class MealRecommendation:
    """Immutable result of deterministic, non-persisting meal optimization."""

    service_date: date
    meal: str | int
    items: tuple[RecommendedMealItem, ...]
    projected_meal_nutrition: NutrientProfile
    projected_daily_total: NutrientProfile
    projected_balance: DailyBalance
    objective_score: Decimal
    diagnostics: MealOptimizationDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.service_date, date) or isinstance(self.service_date, datetime):
            raise TypeError("service_date must be a date")
        if isinstance(self.meal, bool) or not isinstance(self.meal, (str, int)):
            raise TypeError("meal must be text or an integer")
        if isinstance(self.meal, str) and not self.meal.strip():
            raise ValueError("meal must be non-empty text")
        if not isinstance(self.items, tuple) or not all(
            isinstance(item, RecommendedMealItem) for item in self.items
        ):
            raise TypeError("items must be RecommendedMealItem values")
        if any(item.occurrence.service_date != self.service_date for item in self.items):
            raise ValueError("items must use the recommendation service_date")
        if len({item.occurrence.occurrence_id for item in self.items}) != len(self.items):
            raise ValueError("items must not repeat occurrences")
        if not isinstance(self.projected_meal_nutrition, NutrientProfile):
            raise TypeError("projected_meal_nutrition must be NutrientProfile")
        if not isinstance(self.projected_daily_total, NutrientProfile):
            raise TypeError("projected_daily_total must be NutrientProfile")
        if not isinstance(self.projected_balance, DailyBalance):
            raise TypeError("projected_balance must be DailyBalance")
        _non_negative_decimal(self.objective_score, "objective_score")
        if not isinstance(self.diagnostics, MealOptimizationDiagnostics):
            raise TypeError("diagnostics must be MealOptimizationDiagnostics")

    @property
    def is_recommendation(self) -> bool:
        return bool(self.items)


class _CurrentMealOccurrenceCatalog(Protocol):
    def list_current_meal_occurrences(
        self,
        service_date: date,
        meal: str | int | None = None,
    ) -> tuple[FDMenuOccurrence, ...]: ...


class LocalMealOptimizer:
    """Thin local-catalog adapter around the pure :func:`optimize_meal` API."""

    def __init__(self, catalog: _CurrentMealOccurrenceCatalog) -> None:
        if not hasattr(catalog, "list_current_meal_occurrences"):
            raise TypeError("catalog must provide list_current_meal_occurrences")
        self._catalog = catalog

    def optimize_meal(
        self,
        service_date: date,
        meal: str | int,
        targets: DailyTargets,
        current_ledger: DailyLedger,
        *,
        policy: MealOptimizationPolicy | None = None,
        excluded_source_identities: Iterable[tuple[str, str]] = (),
        required_foods: Iterable[RequestedMealFood] = (),
    ) -> MealRecommendation:
        """Read only the current local FD meal menu and optimize against intake.

        ``excluded_source_identities`` is a narrow authoritative exclusion
        seam used by a one-time replacement operation.  It deliberately takes
        provider-backed source identity pairs rather than display text, and it
        has no durable preference meaning outside the one caller operation.
        ``required_foods`` is the complementary positive constraint boundary:
        each value was already resolved from the current local menu and must
        remain in the recommendation at an optimizer-selected physical serving
        quantity.  It is deliberately not a user preference profile.
        """

        occurrences = self._catalog.list_current_meal_occurrences(service_date, meal=meal)
        excluded = _normalize_excluded_source_identities(excluded_source_identities)
        required = _normalize_required_foods(required_foods)
        if {food.source_identity for food in required} & excluded:
            raise ValueError("required_foods cannot also be excluded")
        if excluded:
            occurrences = tuple(
                occurrence
                for occurrence in occurrences
                if _occurrence_source_identity(occurrence) not in excluded
            )
        return optimize_meal(
            service_date,
            meal,
            targets,
            current_ledger,
            occurrences,
            policy=policy,
            required_foods=required,
        )


def _normalize_excluded_source_identities(
    values: Iterable[tuple[str, str]],
) -> frozenset[tuple[str, str]]:
    """Validate immutable provider identities without accepting name matching."""

    try:
        normalized = frozenset(values)
    except TypeError:
        raise TypeError("excluded_source_identities must be an iterable of pairs") from None
    for value in normalized:
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or not all(isinstance(part, str) and part.strip() for part in value)
        ):
            raise TypeError("excluded_source_identities must contain non-empty text pairs")
    return normalized


def _normalize_required_foods(
    values: Iterable[RequestedMealFood],
) -> tuple[RequestedMealFood, ...]:
    """Validate uniquely resolved positive constraints without name matching."""

    try:
        normalized = tuple(values)
    except TypeError:
        raise TypeError("required_foods must be an iterable of RequestedMealFood") from None
    if not all(isinstance(value, RequestedMealFood) for value in normalized):
        raise TypeError("required_foods must contain RequestedMealFood values")
    identities = tuple(value.identity_with_signature for value in normalized)
    if len(set(identities)) != len(identities):
        raise ValueError("required_foods must not repeat an authoritative food")
    return normalized


def _occurrence_source_identity(occurrence: FDMenuOccurrence) -> tuple[str, str]:
    return (occurrence.source_identifier.kind, occurrence.source_identifier.value)


@dataclass(frozen=True, slots=True)
class _SearchState:
    items: tuple[RecommendedMealItem, ...]
    meal_nutrition: NutrientProfile
    last_candidate_index: int
    score: Decimal


def optimize_meal(
    service_date: date,
    meal: str | int,
    targets: DailyTargets,
    current_ledger: DailyLedger,
    local_menu: Iterable[FDMenuOccurrence],
    *,
    policy: MealOptimizationPolicy | None = None,
    required_foods: Iterable[RequestedMealFood] = (),
) -> MealRecommendation:
    """Optimize one current meal from supplied local FD occurrences.

    The objective is a weighted normalized distance from the meal's configured
    share of the *current remaining daily deficit*.  Deficits and contribution
    beyond that share have distinct configurable penalties.  This makes an
    under-eaten breakfast change lunch input automatically when callers supply
    the durable ledger reconstructed from actual accepted intake.
    """

    _validate_request(service_date, meal, targets, current_ledger)
    active_policy = policy or MealOptimizationPolicy()
    if not isinstance(active_policy, MealOptimizationPolicy):
        raise TypeError("policy must be MealOptimizationPolicy or None")
    occurrences = tuple(local_menu)
    if not all(isinstance(item, FDMenuOccurrence) for item in occurrences):
        raise TypeError("local_menu must contain FDMenuOccurrence values")

    required = _normalize_required_foods(required_foods)
    required_keys = frozenset(food.identity_with_signature for food in required)
    current_total = current_ledger.total_consumed_nutrients
    current_balance = calculate_daily_balance(targets, current_total)
    stage_fraction = active_policy.stage_fraction_for(meal)
    candidates, exclusion_counts = _eligible_candidates(
        occurrences, service_date, meal, targets, active_policy
    )
    candidate_required_keys = {
        _occurrence_required_food_key(candidate) for candidate in candidates
    }
    if required_keys - candidate_required_keys:
        return _no_recommendation(
            service_date,
            meal,
            current_total,
            targets,
            _ZERO,
            stage_fraction,
            0,
            len(occurrences),
            len(candidates),
            exclusion_counts,
            "required_food_unavailable",
        )
    if len(required_keys) > active_policy.max_distinct_foods:
        return _no_recommendation(
            service_date,
            meal,
            current_total,
            targets,
            _ZERO,
            stage_fraction,
            0,
            len(occurrences),
            len(candidates),
            exclusion_counts,
            "required_food_infeasible",
        )
    # An unknown accepted-intake nutrient makes the corresponding daily balance
    # unknowable.  Return explicitly rather than silently scoring it as zero.
    if _balance_has_unknown_required_values(current_balance):
        return _no_recommendation(
            service_date, meal, current_total, targets, _ZERO, stage_fraction, 0,
            len(occurrences), len(candidates), exclusion_counts, "unknown_current_balance",
        )
    baseline = _objective(
        _zero_nutrients(), current_balance, targets, stage_fraction, active_policy, item_count=0
    )

    if not required_keys and _all_target_deficits_are_non_positive(current_balance):
        return _no_recommendation(
            service_date, meal, current_total, targets, baseline, stage_fraction, 0,
            len(occurrences), len(candidates), exclusion_counts, "targets_satisfied_or_exceeded",
        )
    if not occurrences:
        return _no_recommendation(
            service_date, meal, current_total, targets, baseline, stage_fraction, 0,
            0, 0, exclusion_counts, "empty_menu",
        )
    if not candidates:
        return _no_recommendation(
            service_date, meal, current_total, targets, baseline, stage_fraction, 0,
            len(occurrences), 0, exclusion_counts, "no_eligible_candidates",
        )

    if required_keys:
        # Enumerate every required identity before optional foods, using the
        # same quantity grids and bounded beam. Otherwise choosing a requested
        # food with a later menu index can prune both earlier complements and
        # earlier required identities, even though a feasible meal exists.
        by_required_key: dict[tuple[str, str, str], FDMenuOccurrence] = {}
        for candidate in sorted(candidates, key=lambda item: item.occurrence_id):
            key = _occurrence_required_food_key(candidate)
            if key in required_keys:
                by_required_key.setdefault(key, candidate)
        candidates = tuple(by_required_key[key] for key in sorted(required_keys)) + tuple(
            candidate for candidate in candidates
            if _occurrence_required_food_key(candidate) not in required_keys
        )

    states = (
        _SearchState((), _zero_nutrients(), -1, baseline),
    )
    best: _SearchState | None = states[0] if not required_keys else None
    evaluated = 0
    for _ in range(active_policy.max_distinct_foods):
        expanded: list[_SearchState] = []
        for state in states:
            next_index = state.last_candidate_index + 1
            stop_index = next_index + 1 if next_index < len(required_keys) else len(candidates)
            for candidate_index in range(next_index, stop_index):
                occurrence = candidates[candidate_index]
                multipliers = active_policy.serving_multipliers_for(
                    occurrence.nutrition_record.serving
                )
                for servings in multipliers:
                    item = RecommendedMealItem(
                        occurrence, occurrence.nutrition_record, servings
                    )
                    meal_nutrition = add_nutrients(
                        state.meal_nutrition,
                        item.nutrition_contribution,
                    )
                    score = _objective(
                        meal_nutrition,
                        current_balance,
                        targets,
                        stage_fraction,
                        active_policy,
                        item_count=len(state.items) + 1,
                    )
                    evaluated += 1
                    expanded.append(
                        _SearchState(
                            state.items + (item,), meal_nutrition, candidate_index, score
                        )
                    )
        if not expanded:
            break
        expanded.sort(key=lambda state: _state_sort_key(state, required_keys))
        complete_required = tuple(
            state
            for state in expanded
            if _state_contains_required_foods(state, required_keys)
        )
        if complete_required and (
            best is None or complete_required[0].score < best.score
        ):
            best = complete_required[0]
        states = tuple(expanded[: active_policy.beam_width])

    if best is None:
        return _no_recommendation(
            service_date,
            meal,
            current_total,
            targets,
            baseline,
            stage_fraction,
            evaluated,
            len(occurrences),
            len(candidates),
            exclusion_counts,
            "required_food_infeasible",
        )
    outcome: OptimizationOutcome = "recommended" if best.items else "no_improving_recommendation"
    daily_total = add_nutrients(current_total, best.meal_nutrition)
    projected_balance = calculate_daily_balance(targets, daily_total)
    return MealRecommendation(
        service_date=service_date,
        meal=meal,
        items=best.items,
        projected_meal_nutrition=best.meal_nutrition,
        projected_daily_total=daily_total,
        projected_balance=projected_balance,
        objective_score=best.score,
        diagnostics=_diagnostics(
            len(occurrences), len(candidates), exclusion_counts, evaluated,
            baseline, stage_fraction, outcome,
        ),
    )


def _eligible_candidates(
    occurrences: tuple[FDMenuOccurrence, ...],
    service_date: date,
    meal: str | int,
    targets: DailyTargets,
    policy: MealOptimizationPolicy,
) -> tuple[tuple[FDMenuOccurrence, ...], dict[str, int]]:
    counts = {
        "duplicates": 0,
        "wrong_context": 0,
        "missing_nutrition": 0,
        "invalid_serving": 0,
    }
    candidates: list[FDMenuOccurrence] = []
    seen: set[tuple[str, str, int]] = set()
    for occurrence in occurrences:
        if occurrence.service_date != service_date or not _meal_matches(occurrence, meal):
            counts["wrong_context"] += 1
            continue
        record = occurrence.nutrition_record
        if record.serving.quantity is None or record.serving.quantity <= _ZERO:
            counts["invalid_serving"] += 1
            continue
        if not policy.serving_multipliers_for(record.serving):
            counts["invalid_serving"] += 1
            continue
        if any(getattr(record.nutrients, field_name) is None for field_name in _REQUIRED_NUTRIENTS):
            counts["missing_nutrition"] += 1
            continue
        if (
            targets.minimums.dietary_fiber_g is not None
            and record.nutrients.dietary_fiber_g is None
        ):
            counts["missing_nutrition"] += 1
            continue
        identity = (
            record.provenance.provider,
            f"{occurrence.source_identifier.kind}:{occurrence.source_identifier.value}",
            occurrence.nutrition_snapshot_id,
        )
        if identity in seen:
            counts["duplicates"] += 1
            continue
        seen.add(identity)
        candidates.append(occurrence)
    return tuple(candidates), counts


def _objective(
    meal_nutrition: NutrientProfile,
    current_balance: DailyBalance,
    targets: DailyTargets,
    stage_fraction: Decimal,
    policy: MealOptimizationPolicy,
    *,
    item_count: int,
) -> Decimal:
    score = policy.complexity_penalty * Decimal(item_count)
    for field_name in _REQUIRED_NUTRIENTS:
        remaining = getattr(current_balance.remaining, field_name)
        contribution = getattr(meal_nutrition, field_name)
        target = getattr(targets, field_name)
        weight = getattr(policy.weights, field_name)
        assert remaining is not None and contribution is not None
        desired = max(remaining, _ZERO) * stage_fraction
        deficit = max(desired - contribution, _ZERO)
        overshoot = max(contribution - desired, _ZERO)
        normalizer = target if target > _ZERO else _ONE
        score += weight * (
            policy.deficit_penalty * deficit + policy.overshoot_penalty * overshoot
        ) / normalizer
    fiber_minimum = targets.minimums.dietary_fiber_g
    fiber_deficit = current_balance.minimums.dietary_fiber_deficit_g
    if fiber_minimum is not None:
        contribution = meal_nutrition.dietary_fiber_g
        assert fiber_deficit is not None and contribution is not None
        desired = fiber_deficit * stage_fraction
        score += (
            policy.weights.dietary_fiber_g
            * policy.deficit_penalty
            * max(desired - contribution, _ZERO)
            / (fiber_minimum if fiber_minimum > _ZERO else _ONE)
        )
    return score


def _no_recommendation(
    service_date: date,
    meal: str | int,
    current_total: NutrientProfile,
    targets: DailyTargets,
    baseline: Decimal,
    stage_fraction: Decimal,
    evaluated: int,
    occurrences: int,
    candidates: int,
    exclusion_counts: dict[str, int],
    outcome: OptimizationOutcome,
) -> MealRecommendation:
    return MealRecommendation(
        service_date=service_date,
        meal=meal,
        items=(),
        projected_meal_nutrition=_zero_nutrients(),
        projected_daily_total=current_total,
        projected_balance=calculate_daily_balance(targets, current_total),
        objective_score=baseline,
        diagnostics=_diagnostics(
            occurrences, candidates, exclusion_counts, evaluated, baseline, stage_fraction, outcome
        ),
    )


def _diagnostics(
    occurrences: int,
    candidates: int,
    counts: dict[str, int],
    evaluated: int,
    baseline: Decimal,
    stage_fraction: Decimal,
    outcome: OptimizationOutcome,
) -> MealOptimizationDiagnostics:
    return MealOptimizationDiagnostics(
        candidate_occurrences=occurrences,
        eligible_candidates=candidates,
        duplicate_occurrences_collapsed=counts["duplicates"],
        excluded_wrong_context=counts["wrong_context"],
        excluded_missing_required_nutrition=counts["missing_nutrition"],
        excluded_invalid_serving=counts["invalid_serving"],
        search_states_evaluated=evaluated,
        baseline_objective_score=baseline,
        stage_fraction=stage_fraction,
        outcome=outcome,
    )


def _state_sort_key(
    state: _SearchState,
    required_keys: frozenset[tuple[str, str, str]] = frozenset(),
) -> tuple[int, Decimal, tuple[tuple[int, Decimal], ...]]:
    """Sort deterministic beam states while retaining positive constraints.

    Missing a requested current-menu food is a hard feasibility concern, not
    an objective preference.  Ordering those partial states first prevents a
    nutritionally worse requested food from being discarded by the beam before
    a complete feasible meal can be considered.
    """

    missing = len(required_keys - _state_required_food_keys(state))
    return (
        missing,
        state.score,
        tuple((item.occurrence.occurrence_id, item.official_servings) for item in state.items),
    )


def _occurrence_required_food_key(occurrence: FDMenuOccurrence) -> tuple[str, str, str]:
    return (
        occurrence.source_identifier.kind,
        occurrence.source_identifier.value,
        occurrence.content_signature,
    )


def _state_required_food_keys(state: _SearchState) -> frozenset[tuple[str, str, str]]:
    return frozenset(_occurrence_required_food_key(item.occurrence) for item in state.items)


def _state_contains_required_foods(
    state: _SearchState,
    required_keys: frozenset[tuple[str, str, str]],
) -> bool:
    return required_keys <= _state_required_food_keys(state)


def _zero_nutrients() -> NutrientProfile:
    return NutrientProfile(_ZERO, _ZERO, _ZERO, _ZERO, _ZERO, _ZERO)


def _balance_has_unknown_required_values(balance: DailyBalance) -> bool:
    return any(getattr(balance.remaining, field_name) is None for field_name in _REQUIRED_NUTRIENTS) or (
        balance.targets.minimums.dietary_fiber_g is not None
        and balance.minimums.dietary_fiber_deficit_g is None
    )


def _all_target_deficits_are_non_positive(balance: DailyBalance) -> bool:
    return (
        all(getattr(balance.remaining, field_name) <= _ZERO for field_name in _REQUIRED_NUTRIENTS)
        and balance.minimums.dietary_fiber_deficit_g in (None, _ZERO)
    )


def _meal_matches(occurrence: FDMenuOccurrence, meal: str | int) -> bool:
    if meal_identity_matches(meal, occurrence.meal_period_id) or meal_identity_matches(
        meal, occurrence.meal_period_name
    ):
        return True
    requested = str(meal).strip()
    return str(occurrence.meal_period_id) == requested or occurrence.meal_period_name.casefold() == requested.casefold()


def _validate_request(
    service_date: date,
    meal: str | int,
    targets: DailyTargets,
    current_ledger: DailyLedger,
) -> None:
    if not isinstance(service_date, date) or isinstance(service_date, datetime):
        raise TypeError("service_date must be a date")
    if isinstance(meal, bool) or not isinstance(meal, (str, int)):
        raise TypeError("meal must be text or an integer")
    if isinstance(meal, str) and not meal.strip():
        raise ValueError("meal must be non-empty text")
    if not isinstance(targets, DailyTargets):
        raise TypeError("targets must be DailyTargets")
    if not isinstance(current_ledger, DailyLedger):
        raise TypeError("current_ledger must be DailyLedger")
