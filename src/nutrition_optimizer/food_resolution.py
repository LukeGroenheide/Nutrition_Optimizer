"""Fail-closed resolution of natural food text to current local FD foods.

This module owns food identity only.  It deliberately has no concept of an
eaten quantity, nutrition arithmetic, intake persistence, messaging, or a
live FDMealPlanner client.  A later portion interpreter can pair a
``ResolvedFood`` with an independently interpreted quantity expressed in the
official serving units carried by its exact ``NutritionRecord``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
import re
from typing import Literal, Protocol, TypeAlias
import unicodedata

from .fdmealplanner.catalog import (
    FDMenuOccurrence,
    NutritionCatalogError,
    OfficialNutritionCatalog,
)
from .meal_identity import meal_identity_matches
from .nutrition.models import NutritionRecord, SourceIdentifier


__all__ = [
    "AmbiguousFood",
    "FoodResolutionCandidate",
    "FoodResolutionError",
    "FoodResolutionMethod",
    "FoodResolutionRequest",
    "FoodResolutionResult",
    "FoodSemanticGatewayUnavailableError",
    "FoodSemanticDecision",
    "FoodSemanticMatcher",
    "FoodSemanticOutputError",
    "FoodSemanticRuntimeUnavailableError",
    "FoodSemanticTimeoutError",
    "FoodSemanticTransportError",
    "LocalFDFoodResolver",
    "ResolvedFood",
    "UnresolvedFood",
]


FoodResolutionMethod = Literal[
    "exact_name",
    "normalized_name",
    "station_context_name",
    "semantic",
]
FoodSemanticDecisionKind = Literal["match", "ambiguous", "no_match"]


class FoodResolutionError(RuntimeError):
    """Raised when the local food-resolution boundary cannot operate safely."""


class FoodSemanticTransportError(FoodResolutionError):
    """A sanitized failure while invoking an optional semantic matcher."""


class FoodSemanticTimeoutError(FoodSemanticTransportError):
    """The optional semantic matcher exceeded its finite timeout."""


class FoodSemanticGatewayUnavailableError(FoodSemanticTransportError):
    """The optional local semantic gateway was clearly unavailable."""


class FoodSemanticRuntimeUnavailableError(FoodResolutionError):
    """The configured optional semantic runtime could not be started."""


class FoodSemanticOutputError(FoodResolutionError):
    """The optional semantic matcher returned unusable structured output."""


@dataclass(frozen=True, slots=True)
class FoodResolutionRequest:
    """One date-scoped request to resolve a user's food wording.

    ``meal`` and ``station`` are optional hard constraints when the caller has
    that context.  They may be the display name or the provider identifier
    accepted by :meth:`OfficialNutritionCatalog.list_current_meal_occurrences`.
    """

    food_text: str
    service_date: date
    meal: str | int | None = None
    station: str | int | None = None

    def __post_init__(self) -> None:
        _require_text(self.food_text, "food_text")
        if not isinstance(self.service_date, date) or isinstance(self.service_date, datetime):
            raise TypeError("service_date must be a date")
        _validate_scope(self.meal, "meal")
        _validate_scope(self.station, "station")


@dataclass(frozen=True, slots=True)
class FoodSemanticDecision:
    """One strictly validated semantic decision over an explicit candidate set."""

    decision: FoodSemanticDecisionKind
    component_identity: str | None
    reason: str

    def __post_init__(self) -> None:
        if self.decision not in {"match", "ambiguous", "no_match"}:
            raise ValueError("decision must be match, ambiguous, or no_match")
        _require_text(self.reason, "reason")
        if self.decision == "match":
            _require_text(self.component_identity, "component_identity")
        elif self.component_identity is not None:
            raise ValueError("component_identity must be null unless decision is match")


class FoodSemanticMatcher(Protocol):
    """Narrow model boundary that can only choose from local candidates."""

    def decide(
        self,
        request: FoodResolutionRequest,
        candidates: tuple["FoodResolutionCandidate", ...],
    ) -> FoodSemanticDecision:
        """Return one decision over exactly the supplied local candidates."""


@dataclass(frozen=True, slots=True)
class FoodResolutionCandidate:
    """One nutrition-distinct current FD food candidate.

    All ``equivalent_occurrences`` share the same stable component identity and
    exact nutrition snapshot.  They therefore do not introduce a nutrition
    ambiguity when a food appears at multiple current stations or meals.
    """

    occurrence: FDMenuOccurrence
    equivalent_occurrences: tuple[FDMenuOccurrence, ...]
    source_identifier: SourceIdentifier
    content_signature: str
    nutrition_snapshot_id: int
    official_display_name: str
    official_serving_description: str | None
    ingredients: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.occurrence, FDMenuOccurrence):
            raise TypeError("occurrence must be an FDMenuOccurrence")
        if not isinstance(self.equivalent_occurrences, tuple) or not self.equivalent_occurrences:
            raise ValueError("equivalent_occurrences must be a non-empty tuple")
        if not isinstance(self.source_identifier, SourceIdentifier):
            raise TypeError("source_identifier must be a SourceIdentifier")
        _require_text(self.content_signature, "content_signature")
        if (
            isinstance(self.nutrition_snapshot_id, bool)
            or not isinstance(self.nutrition_snapshot_id, int)
            or self.nutrition_snapshot_id <= 0
        ):
            raise ValueError("nutrition_snapshot_id must be a positive integer")
        _require_text(self.official_display_name, "official_display_name")
        if self.official_serving_description is not None:
            _require_text(self.official_serving_description, "official_serving_description")
        if not isinstance(self.ingredients, tuple) or not all(
            isinstance(value, str) and value.strip() for value in self.ingredients
        ):
            raise ValueError("ingredients must contain non-empty text")

        occurrence_ids: set[int] = set()
        for occurrence in self.equivalent_occurrences:
            if not isinstance(occurrence, FDMenuOccurrence):
                raise TypeError("equivalent_occurrences must contain FDMenuOccurrence values")
            if occurrence.occurrence_id in occurrence_ids:
                raise ValueError("equivalent_occurrences must not repeat an occurrence")
            occurrence_ids.add(occurrence.occurrence_id)
            if (
                occurrence.source_identifier != self.source_identifier
                or occurrence.content_signature != self.content_signature
                or occurrence.nutrition_snapshot_id != self.nutrition_snapshot_id
                or occurrence.nutrition_record.name != self.official_display_name
                or occurrence.nutrition_record != self.occurrence.nutrition_record
            ):
                raise ValueError("equivalent occurrences must share one exact nutrition snapshot")
        if self.occurrence not in self.equivalent_occurrences:
            raise ValueError("occurrence must be present in equivalent_occurrences")

    @property
    def nutrition_record(self) -> NutritionRecord:
        """Return the exact immutable record linked by the current occurrence."""

        return self.occurrence.nutrition_record

    @property
    def record(self) -> NutritionRecord:
        """Compatibility-style short alias for later application code."""

        return self.nutrition_record

    def semantic_payload(self) -> dict[str, object]:
        """Return compact, local-only context for a semantic matcher.

        Nutrition values are intentionally absent: the model decides identity,
        while deterministic Python retains the exact record for later use.
        """

        contexts = tuple(
            {
                "meal": occurrence.meal_period_name,
                "station": occurrence.station_name,
            }
            for occurrence in self.equivalent_occurrences
        )
        payload: dict[str, object] = {
            "component_identity": self.source_identifier.value,
            "official_display_name": self.official_display_name,
            "meal": self.occurrence.meal_period_name,
            "station": self.occurrence.station_name,
            "serving": self.official_serving_description,
        }
        if len(contexts) > 1:
            payload["menu_contexts"] = contexts
        ingredient_text = _compact_ingredients(self.ingredients)
        if ingredient_text is not None:
            payload["ingredients"] = ingredient_text
        return payload


@dataclass(frozen=True, slots=True)
class ResolvedFood:
    """A unique local current-menu food with its exact nutrition snapshot."""

    original_food_text: str
    occurrence: FDMenuOccurrence
    equivalent_occurrences: tuple[FDMenuOccurrence, ...]
    source_identifier: SourceIdentifier
    content_signature: str
    nutrition_snapshot_id: int
    nutrition_record: NutritionRecord
    resolution_method: FoodResolutionMethod

    def __post_init__(self) -> None:
        _require_text(self.original_food_text, "original_food_text")
        if not isinstance(self.occurrence, FDMenuOccurrence):
            raise TypeError("occurrence must be an FDMenuOccurrence")
        if not isinstance(self.source_identifier, SourceIdentifier):
            raise TypeError("source_identifier must be a SourceIdentifier")
        _require_text(self.content_signature, "content_signature")
        if (
            isinstance(self.nutrition_snapshot_id, bool)
            or not isinstance(self.nutrition_snapshot_id, int)
            or self.nutrition_snapshot_id <= 0
        ):
            raise ValueError("nutrition_snapshot_id must be a positive integer")
        if not isinstance(self.nutrition_record, NutritionRecord):
            raise TypeError("nutrition_record must be a NutritionRecord")
        if self.resolution_method not in {
            "exact_name",
            "normalized_name",
            "station_context_name",
            "semantic",
        }:
            raise ValueError("resolution_method is invalid")
        if (
            self.occurrence.source_identifier != self.source_identifier
            or self.occurrence.content_signature != self.content_signature
            or self.occurrence.nutrition_snapshot_id != self.nutrition_snapshot_id
            or self.occurrence.nutrition_record != self.nutrition_record
        ):
            raise ValueError("resolved food must retain the occurrence-linked snapshot")
        if not isinstance(self.equivalent_occurrences, tuple) or self.occurrence not in self.equivalent_occurrences:
            raise ValueError("equivalent_occurrences must include occurrence")

    @property
    def record(self) -> NutritionRecord:
        """Short alias for the exact official nutrition record."""

        return self.nutrition_record


@dataclass(frozen=True, slots=True)
class AmbiguousFood:
    """A request that safely maps to multiple nutrition-distinct candidates."""

    original_food_text: str
    candidates: tuple[FoodResolutionCandidate, ...]
    reason: str

    def __post_init__(self) -> None:
        _require_text(self.original_food_text, "original_food_text")
        if not isinstance(self.candidates, tuple) or not self.candidates:
            raise ValueError("candidates must be a non-empty tuple")
        if not all(isinstance(candidate, FoodResolutionCandidate) for candidate in self.candidates):
            raise TypeError("candidates must contain FoodResolutionCandidate values")
        _require_text(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class UnresolvedFood:
    """A fail-closed outcome that selects no official food."""

    original_food_text: str
    reason: str
    candidates: tuple[FoodResolutionCandidate, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.original_food_text, "original_food_text")
        _require_text(self.reason, "reason")
        if not isinstance(self.candidates, tuple) or not all(
            isinstance(candidate, FoodResolutionCandidate) for candidate in self.candidates
        ):
            raise TypeError("candidates must contain FoodResolutionCandidate values")

    @property
    def candidate_count(self) -> int:
        """Return the count of current local candidates considered."""

        return len(self.candidates)


FoodResolutionResult: TypeAlias = ResolvedFood | AmbiguousFood | UnresolvedFood


class LocalFDFoodResolver:
    """Resolve food identity from current SQLite-backed FD menu occurrences.

    Candidate generation is always local and date scoped.  A semantic matcher,
    when explicitly supplied, receives only those candidates after conservative
    deterministic matching cannot choose a unique nutrition-distinct result.
    """

    def __init__(
        self,
        catalog: OfficialNutritionCatalog,
        *,
        semantic_matcher: FoodSemanticMatcher | None = None,
    ) -> None:
        if not isinstance(catalog, OfficialNutritionCatalog):
            raise TypeError("catalog must be an OfficialNutritionCatalog")
        if semantic_matcher is not None and not callable(
            getattr(semantic_matcher, "decide", None)
        ):
            raise TypeError("semantic_matcher must provide decide")
        self._catalog = catalog
        self._semantic_matcher = semantic_matcher

    def list_current_candidates(
        self,
        request: FoodResolutionRequest,
    ) -> tuple[FoodResolutionCandidate, ...]:
        """Build the complete admissible candidate universe from local SQLite.

        This method performs no model or network call.  It intentionally asks
        the catalog for only the request date, with optional meal and station
        constraints applied before grouping duplicate occurrences.
        """

        if not isinstance(request, FoodResolutionRequest):
            raise TypeError("request must be a FoodResolutionRequest")
        try:
            occurrences = self._catalog.list_current_meal_occurrences(
                request.service_date,
                meal=request.meal,
                station=request.station,
            )
        except (NutritionCatalogError, TypeError, ValueError) as exc:
            raise FoodResolutionError("current local FD menu could not be queried") from exc
        return _candidates_from_occurrences(request, occurrences)

    def resolve(self, request: FoodResolutionRequest) -> FoodResolutionResult:
        """Resolve one food identity without estimating a serving quantity."""

        candidates = self.list_current_candidates(request)
        if not candidates:
            return UnresolvedFood(
                original_food_text=request.food_text,
                reason="no_current_local_candidates",
            )

        for method, matches in _deterministic_match_stages(request.food_text, candidates):
            if len(matches) == 1:
                return _resolved_food(request.food_text, matches[0], method)
            if len(matches) > 1:
                return AmbiguousFood(
                    original_food_text=request.food_text,
                    candidates=matches,
                    reason="multiple_current_foods_match_deterministically",
                )

        if self._semantic_matcher is None:
            return UnresolvedFood(
                original_food_text=request.food_text,
                reason="semantic_matcher_unavailable",
                candidates=candidates,
            )

        return self._resolve_semantically(request, candidates)

    def _resolve_semantically(
        self,
        request: FoodResolutionRequest,
        candidates: tuple[FoodResolutionCandidate, ...],
    ) -> FoodResolutionResult:
        try:
            decision = self._semantic_matcher.decide(request, candidates)
        except FoodSemanticTimeoutError:
            return UnresolvedFood(
                original_food_text=request.food_text,
                reason="semantic_timeout",
                candidates=candidates,
            )
        except FoodSemanticGatewayUnavailableError:
            return UnresolvedFood(
                original_food_text=request.food_text,
                reason="semantic_gateway_unavailable",
                candidates=candidates,
            )
        except FoodSemanticOutputError:
            return UnresolvedFood(
                original_food_text=request.food_text,
                reason="semantic_output_invalid",
                candidates=candidates,
            )
        except FoodSemanticRuntimeUnavailableError:
            return UnresolvedFood(
                original_food_text=request.food_text,
                reason="semantic_runtime_unavailable",
                candidates=candidates,
            )
        except FoodSemanticTransportError:
            return UnresolvedFood(
                original_food_text=request.food_text,
                reason="semantic_transport_failed",
                candidates=candidates,
            )
        except Exception:
            # A model timeout, invalid transport result, or unavailable local
            # semantic runtime must never pick a food indirectly.
            return UnresolvedFood(
                original_food_text=request.food_text,
                reason="semantic_resolution_failed",
                candidates=candidates,
            )

        if not isinstance(decision, FoodSemanticDecision):
            return UnresolvedFood(
                original_food_text=request.food_text,
                reason="semantic_response_invalid",
                candidates=candidates,
            )
        if decision.decision == "no_match":
            return UnresolvedFood(
                original_food_text=request.food_text,
                reason="semantic_no_match",
                candidates=candidates,
            )
        if decision.decision == "ambiguous":
            return AmbiguousFood(
                original_food_text=request.food_text,
                candidates=candidates,
                reason="semantic_ambiguous",
            )

        identity = decision.component_identity
        matching_candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.source_identifier.value == identity
            and _candidate_respects_request(candidate, request)
        )
        if not matching_candidates:
            return UnresolvedFood(
                original_food_text=request.food_text,
                reason="semantic_candidate_not_admissible",
                candidates=candidates,
            )
        if len(matching_candidates) > 1:
            # The model schema deliberately selects a stable component ID, not
            # a snapshot.  Do not invent one when that ID has multiple current
            # nutrition versions in the requested scope.
            return AmbiguousFood(
                original_food_text=request.food_text,
                candidates=matching_candidates,
                reason="semantic_identity_has_multiple_current_snapshots",
            )
        return _resolved_food(request.food_text, matching_candidates[0], "semantic")


def _candidates_from_occurrences(
    request: FoodResolutionRequest,
    occurrences: Iterable[FDMenuOccurrence],
) -> tuple[FoodResolutionCandidate, ...]:
    grouped: dict[tuple[str, str, str], list[FDMenuOccurrence]] = {}
    for occurrence in occurrences:
        if not _occurrence_is_admissible(occurrence, request):
            continue
        key = (
            occurrence.source_identifier.kind,
            occurrence.source_identifier.value,
            occurrence.content_signature,
        )
        grouped.setdefault(key, []).append(occurrence)

    candidates: list[FoodResolutionCandidate] = []
    for key in sorted(grouped):
        equivalents = tuple(sorted(grouped[key], key=_occurrence_sort_key))
        occurrence = equivalents[0]
        record = occurrence.nutrition_record
        candidates.append(
            FoodResolutionCandidate(
                occurrence=occurrence,
                equivalent_occurrences=equivalents,
                source_identifier=occurrence.source_identifier,
                content_signature=occurrence.content_signature,
                nutrition_snapshot_id=occurrence.nutrition_snapshot_id,
                official_display_name=record.name,
                official_serving_description=_official_serving_description(record),
                ingredients=record.ingredients,
            )
        )
    return tuple(candidates)


def _deterministic_match_stages(
    food_text: str,
    candidates: tuple[FoodResolutionCandidate, ...],
) -> tuple[tuple[FoodResolutionMethod, tuple[FoodResolutionCandidate, ...]], ...]:
    exact_input = unicodedata.normalize("NFKC", food_text).strip()
    exact_matches = tuple(
        candidate
        for candidate in candidates
        if unicodedata.normalize("NFKC", candidate.official_display_name).strip()
        == exact_input
    )
    normalized_input = _normalized_food_name(food_text)
    normalized_matches = tuple(
        candidate
        for candidate in candidates
        if _normalized_food_name(candidate.official_display_name) == normalized_input
    )
    context_matches = tuple(
        candidate
        for candidate in candidates
        if _normalized_food_name(_without_verified_station_prefix(candidate))
        == normalized_input
    )
    return (
        ("exact_name", exact_matches),
        ("normalized_name", normalized_matches),
        ("station_context_name", context_matches),
    )


def _resolved_food(
    original_food_text: str,
    candidate: FoodResolutionCandidate,
    method: FoodResolutionMethod,
) -> ResolvedFood:
    return ResolvedFood(
        original_food_text=original_food_text,
        occurrence=candidate.occurrence,
        equivalent_occurrences=candidate.equivalent_occurrences,
        source_identifier=candidate.source_identifier,
        content_signature=candidate.content_signature,
        nutrition_snapshot_id=candidate.nutrition_snapshot_id,
        nutrition_record=candidate.nutrition_record,
        resolution_method=method,
    )


def _occurrence_is_admissible(
    occurrence: object,
    request: FoodResolutionRequest,
) -> bool:
    if not isinstance(occurrence, FDMenuOccurrence):
        return False
    if occurrence.service_date != request.service_date:
        return False
    if not _meal_scope_matches(
        request.meal,
        occurrence.meal_period_id,
        occurrence.meal_period_name,
    ):
        return False
    if not _scope_matches(
        request.station,
        occurrence.station_concept_id,
        occurrence.station_name,
    ):
        return False
    if (
        not occurrence.content_signature.strip()
        or occurrence.nutrition_snapshot_id <= 0
        or occurrence.source_identifier not in occurrence.nutrition_record.provenance.identifiers
    ):
        return False
    return True


def _candidate_respects_request(
    candidate: FoodResolutionCandidate,
    request: FoodResolutionRequest,
) -> bool:
    return (
        _occurrence_is_admissible(candidate.occurrence, request)
        and candidate.occurrence.source_identifier == candidate.source_identifier
        and candidate.occurrence.content_signature == candidate.content_signature
        and candidate.occurrence.nutrition_snapshot_id == candidate.nutrition_snapshot_id
        and candidate.occurrence.nutrition_record == candidate.nutrition_record
    )


def _scope_matches(
    requested: str | int | None,
    identifier: str | int | None,
    display_name: str | None,
) -> bool:
    if requested is None:
        return True
    requested_text = _scope_text(requested)
    return requested_text == _scope_text(identifier) or requested_text == _scope_text(display_name)


def _meal_scope_matches(
    requested: str | int | None,
    identifier: str | int | None,
    display_name: str | None,
) -> bool:
    if requested is None:
        return True
    return (
        meal_identity_matches(requested, identifier)
        or meal_identity_matches(requested, display_name)
        or _scope_matches(requested, identifier, display_name)
    )


def _scope_text(value: str | int | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip().casefold()


def _without_verified_station_prefix(candidate: FoodResolutionCandidate) -> str:
    """Remove only the verified ``Zone`` display marker when context permits."""

    if not any(_scope_text(occurrence.station_name) == "zone" for occurrence in candidate.equivalent_occurrences):
        return candidate.official_display_name
    normalized = unicodedata.normalize("NFKC", candidate.official_display_name).strip()
    return re.sub(r"^zone\s+", "", normalized, count=1, flags=re.IGNORECASE)


_TRAILING_MENU_MARKER = re.compile(
    r"(?:\s*[*†‡]+|\s*\((?:v|ve|vg|vegetarian|vegan|gf|df)\))\s*$",
    re.IGNORECASE,
)


def _normalized_food_name(value: str) -> str:
    """Apply only harmless local spelling and punctuation normalization."""

    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = normalized.replace("’", "'").replace("‘", "'").replace("`", "'")
    previous = None
    while normalized != previous:
        previous = normalized
        normalized = _TRAILING_MENU_MARKER.sub("", normalized).strip()
    normalized = re.sub(r"\s*&\s*", " and ", normalized)
    normalized = re.sub(r"[-–—/]+", " ", normalized)
    normalized = re.sub(r"[\"'.,:;!?()\[\]{}]+", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _official_serving_description(record: NutritionRecord) -> str | None:
    serving = record.serving
    if serving.text is not None:
        return serving.text
    if serving.quantity is not None and serving.unit is not None:
        quantity = format(serving.quantity.normalize(), "f")
        return f"{quantity} {serving.unit}"
    return serving.unit


def _compact_ingredients(ingredients: tuple[str, ...]) -> str | None:
    if not ingredients:
        return None
    combined = "; ".join(value.strip() for value in ingredients if value.strip())
    if not combined:
        return None
    maximum_length = 320
    return combined if len(combined) <= maximum_length else f"{combined[:maximum_length - 1].rstrip()}…"


def _occurrence_sort_key(occurrence: FDMenuOccurrence) -> tuple[str, str, str, str, int]:
    return (
        _scope_text(occurrence.meal_period_name),
        _scope_text(occurrence.station_name),
        _scope_text(occurrence.station_concept_id),
        occurrence.occurrence_key,
        occurrence.occurrence_id,
    )


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")


def _validate_scope(value: object, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"{field_name} must be text or an integer when provided")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{field_name} must be non-empty when provided")
