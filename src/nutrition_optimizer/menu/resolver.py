"""Fail-closed deterministic DiningBucket to FD occurrence resolution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
import re
from typing import Literal
import unicodedata

from nutrition_optimizer.fdmealplanner.mapper import FDMappingResult, is_student_visible_component
from nutrition_optimizer.fdmealplanner.models import FDMealOccurrence
from nutrition_optimizer.meal_identity import MEAL_PERIOD_IDS, canonical_meal_name
from nutrition_optimizer.nutrition.models import NutritionRecord, SourceIdentifier

FDStationKey = Literal[
    "global bowls",
    "trattoria",
    "zone",
    "american grille",
    "homestyle",
    "deli",
    "plant forward",
]

MatchRule = Literal[
    "exact_display_name",
    "normalized_display_name",
    "station_context_normalization",
    "conservative_lexical_normalization",
]

NoMatchReason = Literal[
    "unknown_station_context",
    "known_context_but_fd_empty",
    "no_name_match",
]

FD_STATION_CONCEPT_IDS: dict[FDStationKey, int] = {
    "global bowls": 203,
    "trattoria": 114,
    "zone": 121,
    "american grille": 48,
    "homestyle": 81,
    "deli": 65,
    "plant forward": 156,
}

_STATIC_STATION_ALIASES: dict[str, FDStationKey] = {
    "global 1": "global bowls",
    "global bowls": "global bowls",
    "trattoria": "trattoria",
    "trattoria 1": "trattoria",
    "trattoria 2": "trattoria",
    "zone": "zone",
    "zone 1": "zone",
    "zone 2": "zone",
    "the zone": "zone",
    "american grille": "american grille",
    "grille 1": "american grille",
    "grille 2": "american grille",
    "homestyle": "homestyle",
    "homestyle 2": "homestyle",
    "deli": "deli",
    "plant forward": "plant forward",
}


@dataclass(frozen=True, slots=True)
class DiningBucketOccurrence:
    """One logical DiningBucket food line with explicit service context."""

    service_date: date
    meal: str
    station_name: str
    name: str
    service_start: datetime | None = None
    service_end: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.service_date, date) or isinstance(self.service_date, datetime):
            raise TypeError("service_date must be a date")
        if _meal_name(self.meal) is None:
            raise ValueError("meal must be breakfast, lunch, or dinner")
        _require_text(self.station_name, "station_name")
        _require_text(self.name, "name")
        if self.service_start is not None:
            _validate_datetime(self.service_start, "service_start")
        if self.service_end is not None:
            _validate_datetime(self.service_end, "service_end")
        if self.service_start is not None and self.service_end is not None:
            if self.service_start > self.service_end:
                raise ValueError("service_start must not be after service_end")

    @classmethod
    def from_menu_entry(
        cls,
        *,
        service_date: date,
        meal: str,
        station_name: str,
        name: str,
        service_start: datetime | None = None,
        service_end: datetime | None = None,
    ) -> "DiningBucketOccurrence":
        """Construct a resolver occurrence without changing menu ingestion models."""

        return cls(
            service_date=service_date,
            meal=meal,
            station_name=station_name,
            name=name,
            service_start=service_start,
            service_end=service_end,
        )


@dataclass(frozen=True, slots=True)
class FDStationScope:
    """The already-resolved FD concept used as a hard candidate boundary."""

    key: FDStationKey
    concept_id: int


@dataclass(frozen=True, slots=True)
class ResolvedFDMatch:
    """One deterministic, unique DiningBucket to FD match."""

    source: DiningBucketOccurrence
    fd_result: FDMappingResult
    source_identifier: SourceIdentifier
    rule: MatchRule

    @property
    def fd_occurrence(self) -> FDMealOccurrence:
        occurrence = self.fd_result.occurrence
        if occurrence is None:  # pragma: no cover - guarded by resolver
            raise RuntimeError("resolved FD result has no occurrence")
        return occurrence

    @property
    def record(self) -> NutritionRecord:
        record = self.fd_result.record
        if record is None:  # pragma: no cover - guarded by resolver
            raise RuntimeError("resolved FD result has no NutritionRecord")
        return record


@dataclass(frozen=True, slots=True)
class AmbiguousFDMatch:
    """More than one stable FD component identity survived one rule."""

    source: DiningBucketOccurrence
    candidates: tuple[FDMappingResult, ...]
    source_identifiers: tuple[SourceIdentifier, ...]
    rule: MatchRule


@dataclass(frozen=True, slots=True)
class NoFDMatch:
    """A fail-closed resolution result with a concise machine-readable reason."""

    source: DiningBucketOccurrence
    reason: NoMatchReason
    station_scope: FDStationScope | None = None
    eligible_candidate_count: int = 0


FDMatchResult = ResolvedFDMatch | AmbiguousFDMatch | NoFDMatch


def resolve_station_scope(
    station_name: str,
    *,
    service_date: date,
    meal: str,
) -> FDStationScope | None:
    """Resolve a Hope station to an FD concept using authoritative context rules."""

    if not isinstance(service_date, date) or isinstance(service_date, datetime):
        raise TypeError("service_date must be a date")
    meal_name = _meal_name(meal)
    if meal_name is None:
        raise ValueError("meal must be breakfast, lunch, or dinner")
    normalized = _normalize_station_label(station_name)
    if normalized == "comfort corner":
        if meal_name == "dinner":
            key: FDStationKey = "homestyle"
        elif meal_name == "lunch":
            key = "american grille" if service_date.weekday() < 5 else "homestyle"
        else:
            return None
    else:
        key = _STATIC_STATION_ALIASES.get(normalized)
        if key is None:
            return None
    return FDStationScope(key=key, concept_id=FD_STATION_CONCEPT_IDS[key])


def resolve_occurrence(
    source: DiningBucketOccurrence,
    fd_candidates: Iterable[FDMappingResult],
) -> FDMatchResult:
    """Resolve one DiningBucket occurrence against explicitly supplied FD results.

    The resolver performs no network access and only compares candidates that
    already contain a valid mapped ``NutritionRecord``.  Date, meal, station,
    visibility, and recipe-type checks are all hard filters.
    """

    if not isinstance(source, DiningBucketOccurrence):
        raise TypeError("source must be a DiningBucketOccurrence")
    candidates = tuple(fd_candidates)
    if not all(isinstance(candidate, FDMappingResult) for candidate in candidates):
        raise TypeError("fd_candidates must contain FDMappingResult values")

    meal_name = _meal_name(source.meal)
    assert meal_name is not None  # validated by DiningBucketOccurrence
    scope = resolve_station_scope(
        source.station_name,
        service_date=source.service_date,
        meal=meal_name,
    )
    if scope is None:
        return NoFDMatch(source=source, reason="unknown_station_context")

    eligible = _eligible_candidates(source, meal_name, scope, candidates)
    if not eligible:
        return NoFDMatch(
            source=source,
            reason="known_context_but_fd_empty",
            station_scope=scope,
        )

    stages: tuple[tuple[MatchRule, str], ...] = (
        ("exact_display_name", "exact"),
        ("normalized_display_name", "normalized"),
        ("station_context_normalization", "context"),
        ("conservative_lexical_normalization", "lexical"),
    )
    for rule, stage in stages:
        matches = _stage_matches(source.name, scope, eligible, stage)
        unique = _unique_candidates(matches)
        if len(unique) == 1:
            candidate, identifier = unique[0]
            return ResolvedFDMatch(
                source=source,
                fd_result=candidate,
                source_identifier=identifier,
                rule=rule,
            )
        if len(unique) > 1:
            return AmbiguousFDMatch(
                source=source,
                candidates=tuple(candidate for candidate, _ in unique),
                source_identifiers=tuple(identifier for _, identifier in unique),
                rule=rule,
            )
    return NoFDMatch(
        source=source,
        reason="no_name_match",
        station_scope=scope,
        eligible_candidate_count=len(eligible),
    )


def _eligible_candidates(
    source: DiningBucketOccurrence,
    meal_name: str,
    scope: FDStationScope,
    candidates: tuple[FDMappingResult, ...],
) -> tuple[FDMappingResult, ...]:
    meal_period_id = MEAL_PERIOD_IDS[meal_name]
    eligible: list[FDMappingResult] = []
    seen: set[SourceIdentifier] = set()
    for candidate in candidates:
        record = candidate.record
        occurrence = candidate.occurrence
        if record is None or occurrence is None:
            continue
        component = occurrence.component
        if not isinstance(component, Mapping):
            continue
        if _parse_date(occurrence.menu_date) != source.service_date:
            continue
        if _parse_positive_int(occurrence.meal_period_id) != meal_period_id:
            continue
        if _parse_positive_int(occurrence.station_concept_id) != scope.concept_id:
            continue
        if _parse_positive_int(component.get("componentTypeId")) != 181:
            continue
        if not is_student_visible_component(component):
            continue
        identifier = _component_identifier(record)
        if identifier is None or identifier in seen:
            continue
        seen.add(identifier)
        eligible.append(candidate)
    return tuple(eligible)


def _stage_matches(
    source_name: str,
    scope: FDStationScope,
    candidates: tuple[FDMappingResult, ...],
    stage: str,
) -> tuple[FDMappingResult, ...]:
    source_key = _name_key(source_name, ampersand_as_and=stage == "lexical")
    matches: list[FDMappingResult] = []
    for candidate in candidates:
        assert candidate.record is not None
        candidate_name = candidate.record.name
        if stage in {"context", "lexical"}:
            candidate_name = _remove_approved_context_prefix(candidate_name, scope)
        candidate_key = _name_key(candidate_name, ampersand_as_and=stage == "lexical")
        if stage == "exact":
            if source_name == candidate.record.name:
                matches.append(candidate)
        elif candidate_key == source_key:
            matches.append(candidate)
    return tuple(matches)


def _unique_candidates(
    candidates: Iterable[FDMappingResult],
) -> tuple[tuple[FDMappingResult, SourceIdentifier], ...]:
    by_identifier: dict[SourceIdentifier, FDMappingResult] = {}
    for candidate in candidates:
        if candidate.record is None:
            continue
        identifier = _component_identifier(candidate.record)
        if identifier is not None:
            by_identifier.setdefault(identifier, candidate)
    return tuple((by_identifier[key], key) for key in sorted(by_identifier, key=lambda value: (value.kind, value.value)))


def _component_identifier(record: NutritionRecord) -> SourceIdentifier | None:
    identifiers = tuple(
        identifier
        for identifier in record.provenance.identifiers
        if identifier.kind == "component"
    )
    return identifiers[0] if len(identifiers) == 1 else None


def _remove_approved_context_prefix(name: str, scope: FDStationScope) -> str:
    """Remove only the verified ZONE prefix from an FD display name."""

    if scope.key != "zone":
        return name
    normalized = unicodedata.normalize("NFKC", name).strip()
    if normalized.casefold().startswith("zone "):
        return normalized[5:].lstrip()
    return normalized


def _name_key(value: str, *, ampersand_as_and: bool) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = normalized.replace("’", "'").replace("‘", "'").replace("`", "'")
    normalized = re.sub(r"\*+\s*$", "", normalized).strip()
    if ampersand_as_and:
        normalized = re.sub(r"\s*&\s*", " and ", normalized)
    else:
        normalized = re.sub(r"\s*&\s*", " & ", normalized)
    normalized = re.sub(r"[-–—/]+", " ", normalized)
    normalized = re.sub(r"[\"'.,:;!?()\[\]{}]+", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_station_label(value: str) -> str:
    _require_text(value, "station_name")
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def _meal_name(value: str | int) -> str | None:
    return canonical_meal_name(value)


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _parse_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


def _validate_datetime(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")


__all__ = [
    "AmbiguousFDMatch",
    "DiningBucketOccurrence",
    "FDMatchResult",
    "FDStationScope",
    "FD_STATION_CONCEPT_IDS",
    "MEAL_PERIOD_IDS",
    "MatchRule",
    "NoFDMatch",
    "NoMatchReason",
    "ResolvedFDMatch",
    "resolve_occurrence",
    "resolve_station_scope",
]
