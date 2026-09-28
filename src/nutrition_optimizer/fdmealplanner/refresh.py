"""Manual, bounded FDMealPlanner nutrition-and-menu refresh workflow."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import sqlite3
import sys
from urllib.parse import quote

from .catalog import (
    DEFAULT_CATALOG_PATH,
    OfficialNutritionCatalog,
    NutritionCatalogError,
    occurrence_key_for_fd_result,
)
from .client import (
    DEFAULT_TENANT_ID,
    DINING_PERIOD_IDS,
    FDMealPlannerClient,
    FDMealPlannerError,
)
from .mapper import (
    FDMappingBatch,
    FDMappingResult,
    content_signature,
    is_student_visible_component,
    map_meals_payload,
)
from .models import FDMealOccurrence, FDMealsPayload
from ..meal_identity import canonical_meal_id


class FDRefreshValidationError(NutritionCatalogError):
    """Fetched FD data is not safe to use as an authoritative replacement."""


@dataclass(frozen=True, slots=True)
class FDRefreshReport:
    """Operator-facing counts for one bounded refresh attempt."""

    requested_start: date
    requested_end: date
    catalog_path: Path
    populated_start: date | None
    populated_end: date | None
    meals: tuple[str, ...]
    meal_request_count: int
    fd_occurrence_count: int
    valid_mapped_occurrences: int
    rejected_occurrences: int
    rejected_by_reason: tuple[tuple[str, int], ...]
    unique_component_identities: int
    catalog_snapshots_inserted: int
    catalog_snapshots_unchanged: int
    catalog_new_versions: int
    catalog_new_version_identities: tuple[str, ...]
    final_catalog_snapshot_count: int
    menu_occurrences_observed: int
    current_logical_occurrences: int
    occurrence_additions: int
    occurrence_changes: int
    occurrence_removals: int
    final_current_occurrence_count: int
    validated_complete_snapshot: bool
    dry_run: bool

    @property
    def populated_date_range(self) -> tuple[date, date] | None:
        if self.populated_start is None or self.populated_end is None:
            return None
        return self.populated_start, self.populated_end


@dataclass(frozen=True, slots=True)
class _FetchedFDData:
    valid_results: tuple[FDMappingResult, ...]
    catalog_results: tuple[FDMappingResult, ...]
    requested_start: date
    requested_end: date
    populated_start: date | None
    populated_end: date | None
    meal_request_count: int
    fd_occurrence_count: int
    valid_mapped_occurrences: int
    rejected_occurrences: int
    rejected_by_reason: tuple[tuple[str, int], ...]
    unique_component_identities: int
    meal_periods: tuple["_FetchedMealPeriod", ...]


@dataclass(frozen=True, slots=True)
class _FetchedMealPeriod:
    """One scoped upstream response retained until snapshot validation passes."""

    requested_start: date
    requested_end: date
    meal_name: str
    meal_period_id: int
    payload: FDMealsPayload
    mapped: FDMappingBatch


@dataclass(frozen=True, slots=True)
class _CatalogState:
    identities: frozenset[tuple[str, str, str]]
    snapshots: frozenset[tuple[str, str, str, str]]


def _validate_date_range(start_date: date, end_date: date) -> None:
    if not isinstance(start_date, date) or isinstance(start_date, datetime):
        raise TypeError("start_date must be a date")
    if not isinstance(end_date, date) or isinstance(end_date, datetime):
        raise TypeError("end_date must be a date")
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")


def _iter_month_bounds(start_date: date, end_date: date) -> Iterable[tuple[date, date]]:
    current = date(start_date.year, start_date.month, 1)
    while current <= end_date:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        month_end = date.fromordinal(next_month.toordinal() - 1)
        yield max(current, start_date), min(month_end, end_date)
        current = next_month


def _occurrence_date(occurrence: FDMealOccurrence) -> date | None:
    value = occurrence.menu_date
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _component_identity(result: FDMappingResult) -> tuple[str, str, str] | None:
    record = result.record
    if record is None:
        return None
    identifiers = tuple(
        identifier
        for identifier in record.provenance.identifiers
        if identifier.kind == "component"
    )
    if len(identifiers) != 1:
        return None
    identifier = identifiers[0]
    return record.provenance.provider, identifier.kind, identifier.value


def _result_signature(result: FDMappingResult) -> str | None:
    signature = result.content_signature
    return signature.strip() if isinstance(signature, str) and signature.strip() else None


def _deduplicate_results(results: Iterable[FDMappingResult]) -> tuple[FDMappingResult, ...]:
    """Avoid repeated writes for repeated menu occurrences of one snapshot."""

    unique: dict[tuple[tuple[str, str, str], str], FDMappingResult] = {}
    for result in results:
        identity = _component_identity(result)
        signature = _result_signature(result)
        if identity is None or signature is None:
            continue
        unique.setdefault((identity, signature), result)
    return tuple(unique[key] for key in sorted(unique))


def _fetch_and_map(
    client: FDMealPlannerClient,
    start_date: date,
    end_date: date,
    *,
    observed_at: datetime,
    time_offset: int,
) -> _FetchedFDData:
    valid_results: list[FDMappingResult] = []
    meal_periods: list[_FetchedMealPeriod] = []
    rejected_reasons: Counter[str] = Counter()
    populated_dates: list[date] = []
    occurrence_count = 0
    valid_count = 0
    request_count = 0

    for month_start, month_end in _iter_month_bounds(start_date, end_date):
        for meal_name, meal_period_id in DINING_PERIOD_IDS.items():
            request_count += 1
            payload = client.fetch_phelps_month(
                year=month_start.year,
                month=month_start.month,
                meal_period_id=meal_period_id,
                start_date=month_start,
                end_date=month_end,
                time_offset=time_offset,
            )
            if not isinstance(payload, FDMealsPayload):
                raise FDRefreshValidationError(
                    "FDMealPlanner client returned an invalid meals payload"
                )
            mapped = map_meals_payload(
                payload,
                tenant_id=DEFAULT_TENANT_ID,
                retrieved_at=observed_at,
                default_meal_period_id=meal_period_id,
                default_meal_period_name=meal_name,
            )
            meal_periods.append(
                _FetchedMealPeriod(
                    requested_start=month_start,
                    requested_end=month_end,
                    meal_name=meal_name,
                    meal_period_id=meal_period_id,
                    payload=payload,
                    mapped=mapped,
                )
            )
            for result in mapped.results:
                occurrence_count += 1
                if result.record is None:
                    reason = result.diagnostic.reason if result.diagnostic is not None else "invalid_mapping"
                    rejected_reasons[reason] += 1
                    continue
                occurrence = result.occurrence
                if occurrence is None:
                    rejected_reasons["missing_occurrence"] += 1
                    continue
                if not is_student_visible_component(occurrence.component):
                    rejected_reasons["not_student_visible"] += 1
                    continue
                occurrence_day = _occurrence_date(occurrence)
                if occurrence_day is None:
                    rejected_reasons["invalid_occurrence_date"] += 1
                    continue
                if not start_date <= occurrence_day <= end_date:
                    rejected_reasons["outside_requested_range"] += 1
                    continue
                valid_count += 1
                populated_dates.append(occurrence_day)
                valid_results.append(result)

    unique_identities = {
        identity
        for result in valid_results
        if (identity := _component_identity(result)) is not None
    }
    return _FetchedFDData(
        valid_results=tuple(valid_results),
        catalog_results=_deduplicate_results(valid_results),
        requested_start=start_date,
        requested_end=end_date,
        populated_start=min(populated_dates) if populated_dates else None,
        populated_end=max(populated_dates) if populated_dates else None,
        meal_request_count=request_count,
        fd_occurrence_count=occurrence_count,
        valid_mapped_occurrences=valid_count,
        rejected_occurrences=sum(rejected_reasons.values()),
        rejected_by_reason=tuple(sorted(rejected_reasons.items())),
        unique_component_identities=len(unique_identities),
        meal_periods=tuple(meal_periods),
    )


def _default_expected_meal_ids(
    start_date: date,
    end_date: date,
) -> dict[date, tuple[int, ...]]:
    """Reuse the product scheduler's Phelps opportunity configuration.

    The FD endpoint has no independent local weekday map.  Keeping this
    deferred import here makes the existing bounded refresh callable without
    adding an import-time dependency from the catalog package to application
    composition, while still ensuring Sunday brunch and other future schedule
    overrides are evaluated from the same source as recommendation delivery.
    """

    from ..recommendation_scheduler import DEFAULT_RECOMMENDATION_DISPATCH_SCHEDULE

    return {
        current: tuple(
            opportunity.meal_id
            for opportunity in DEFAULT_RECOMMENDATION_DISPATCH_SCHEDULE.opportunities_for(
                current
            )
        )
        for current in _iter_dates(start_date, end_date)
    }


def _iter_dates(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current = date.fromordinal(current.toordinal() + 1)


def _normalize_expected_meal_ids(
    start_date: date,
    end_date: date,
    expected_meal_ids_by_date: Mapping[date, Iterable[int | str]] | None,
) -> dict[date, frozenset[int]]:
    """Normalize a Phelps opportunity map without inventing meal contexts."""

    source: Mapping[date, Iterable[int | str]]
    if expected_meal_ids_by_date is None:
        source = _default_expected_meal_ids(start_date, end_date)
    else:
        if not isinstance(expected_meal_ids_by_date, Mapping):
            raise TypeError("expected_meal_ids_by_date must be a mapping")
        source = expected_meal_ids_by_date

    requested_dates = set(_iter_dates(start_date, end_date))
    unexpected_dates = set(source) - requested_dates
    if unexpected_dates:
        raise ValueError("expected meal coverage contains a date outside the refresh range")

    normalized: dict[date, frozenset[int]] = {}
    for service_date in _iter_dates(start_date, end_date):
        values = source.get(service_date, ())
        if isinstance(values, (str, bytes)):
            raise TypeError("expected meal IDs must be an iterable of meal values")
        try:
            raw_values = tuple(values)
        except TypeError as exc:
            raise TypeError("expected meal IDs must be an iterable of meal values") from exc
        meal_ids: list[int] = []
        for value in raw_values:
            meal_id = canonical_meal_id(value)
            if meal_id is None:
                raise ValueError("expected meal IDs must be known FD meal periods")
            meal_ids.append(meal_id)
        if len(set(meal_ids)) != len(meal_ids):
            raise ValueError("a service date cannot expect the same FD meal twice")
        normalized[service_date] = frozenset(meal_ids)
    return normalized


def _payload_day_date(day: Mapping[str, object]) -> date:
    raw_date = day.get("strMenuForDate") or day.get("menuForDate")
    if not isinstance(raw_date, str) or len(raw_date) < 10:
        raise FDRefreshValidationError("FD meals payload has a day without a valid menu date")
    try:
        return date.fromisoformat(raw_date[:10])
    except ValueError as exc:
        raise FDRefreshValidationError("FD meals payload has a day with an invalid menu date") from exc


def _is_fd_absent_station_placeholder(value: object) -> bool:
    """Recognize FD's explicit ``conceptData: null`` station-container form.

    The public endpoint currently returns this exact form for otherwise complete
    Phelps recipe days.  It is distinct from a malformed list row: the
    established mapper already represents the absence of station metadata as
    ``None`` without inventing a station identity.
    """

    return value is None


def _is_well_formed_station_container(value: object) -> bool:
    """Accept only FD's absent marker or an object-only station list."""

    return _is_fd_absent_station_placeholder(value) or (
        isinstance(value, list)
        and all(isinstance(concept, Mapping) for concept in value)
    )


def _is_fd_unserved_meal_placeholder(
    *,
    station_data: object,
    recipes: object,
    meal_period_id: int,
    expected_meal_ids: frozenset[int],
) -> bool:
    """Recognize FD's recipe-empty row for a calendar-unserved meal context.

    This is deliberately narrower than accepting arbitrary empty or malformed
    responses. FD has returned both ``conceptData: null`` and a structurally
    valid, apparently reused station list alongside ``allMenuRecipes: null``
    for Phelps Sunday breakfast. Station metadata alone contains no component,
    visibility, or nutrition data. The absence is safe only when the shared
    recommendation schedule does not expect that meal on that date.
    """

    return (
        recipes is None
        and meal_period_id not in expected_meal_ids
        and _is_well_formed_station_container(station_data)
    )


def _validate_fetched_snapshot(
    fetched: _FetchedFDData,
    *,
    expected_meal_ids_by_date: Mapping[date, Iterable[int | str]] | None,
) -> None:
    """Fail closed before an authoritative current-menu replacement.

    ``fd_refresh_runs`` intentionally model a complete source snapshot, so a
    partial response must never be allowed to become the latest completed run.
    Network calls and all validation happen before opening SQLite for writing.
    Product-like components and explicitly non-student-visible components are
    intentionally excluded by the established mapper/filtering rules; a
    malformed *visible* recipe is not safe to silently omit from a replacement
    snapshot.
    """

    expected = _normalize_expected_meal_ids(
        fetched.requested_start,
        fetched.requested_end,
        expected_meal_ids_by_date,
    )
    visible_counts: Counter[tuple[date, int]] = Counter()

    for response in fetched.meal_periods:
        for day in response.payload.days:
            service_date = _payload_day_date(day)
            if not response.requested_start <= service_date <= response.requested_end:
                # The client already rejects these occurrences from persistence.
                # They do not prove or disprove completeness for this bounded
                # request, so keep the existing safe range behavior.
                continue
            raw_meal_id = day.get("mealPeriodId")
            if raw_meal_id is not None and canonical_meal_id(raw_meal_id) != response.meal_period_id:
                raise FDRefreshValidationError(
                    "FD meals payload contains a day for the wrong meal period"
                )
            concepts = day.get("conceptData")
            recipes = day.get("allMenuRecipes")
            if _is_fd_unserved_meal_placeholder(
                station_data=concepts,
                recipes=recipes,
                meal_period_id=response.meal_period_id,
                expected_meal_ids=expected[service_date],
            ):
                continue
            if not _is_well_formed_station_container(concepts):
                # ``iter_meal_occurrences`` can technically continue without
                # station data for compatibility with historical inputs, but
                # a fresh authoritative replacement must not silently discard
                # malformed published station rows.  The sole exception is the
                # official endpoint's explicit ``conceptData: null`` absence.
                raise FDRefreshValidationError(
                    "FD meals payload has malformed station rows"
                )
            if not isinstance(recipes, list) or not all(
                isinstance(recipe, Mapping) for recipe in recipes
            ):
                raise FDRefreshValidationError(
                    "FD meals payload has malformed recipe rows"
                )

        for result in response.mapped.results:
            occurrence = result.occurrence
            if occurrence is None:
                raise FDRefreshValidationError(
                    "FD meals payload produced an occurrence without menu context"
                )
            service_date = _occurrence_date(occurrence)
            if service_date is None:
                raise FDRefreshValidationError(
                    "FD meals payload produced an occurrence with an invalid menu date"
                )
            if not response.requested_start <= service_date <= response.requested_end:
                continue
            if canonical_meal_id(occurrence.meal_period_id) != response.meal_period_id:
                raise FDRefreshValidationError(
                    "FD meals payload produced an occurrence for the wrong meal period"
                )
            if result.record is None:
                if not is_student_visible_component(occurrence.component):
                    continue
                diagnostic = result.diagnostic
                if diagnostic is not None and diagnostic.reason == "unsupported_component_type":
                    # This is an explicit, established non-recipe exclusion;
                    # retaining it avoids weakening the current component-type
                    # boundary while still allowing an otherwise complete menu.
                    continue
                reason = diagnostic.reason if diagnostic is not None else "invalid_mapping"
                raise FDRefreshValidationError(
                    f"FD meals payload contains an unsafe visible recipe ({reason})"
                )
            if is_student_visible_component(occurrence.component):
                visible_counts[(service_date, response.meal_period_id)] += 1

    missing = tuple(
        (service_date, meal_id)
        for service_date in _iter_dates(fetched.requested_start, fetched.requested_end)
        for meal_id in sorted(expected[service_date])
        if visible_counts[(service_date, meal_id)] == 0
    )
    if missing:
        rendered = ", ".join(
            f"{service_date.isoformat()} fd:{meal_id}"
            for service_date, meal_id in missing
        )
        raise FDRefreshValidationError(
            f"FD meals payload is incomplete for expected menu coverage: {rendered}"
        )


def _read_catalog_state(path: Path) -> _CatalogState:
    """Read existing catalog identity metadata without opening it for writes."""

    path = path.expanduser()
    if not path.exists():
        return _CatalogState(frozenset(), frozenset())
    if not path.is_file():
        raise NutritionCatalogError(f"catalog path is not a file: {path}")
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        try:
            rows = connection.execute(
                """
                SELECT provider, source_kind, source_value, content_signature
                FROM nutrition_snapshots
                """
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise NutritionCatalogError("unable to inspect the existing nutrition catalog") from exc
    snapshots = frozenset(tuple(str(value) for value in row) for row in rows)
    identities = frozenset(snapshot[:3] for snapshot in snapshots)
    return _CatalogState(identities, snapshots)


def _preview_catalog(
    path: Path,
    results: Iterable[FDMappingResult],
) -> tuple[int, int, int, int, tuple[str, ...]]:
    state = _read_catalog_state(path)
    identities = set(state.identities)
    snapshots = set(state.snapshots)
    inserted = unchanged = new_versions = 0
    new_version_identities: set[str] = set()
    for result in results:
        identity = _component_identity(result)
        signature = _result_signature(result)
        if identity is None or signature is None:
            continue
        snapshot_key = (*identity, signature)
        if snapshot_key in snapshots:
            unchanged += 1
        elif identity in identities:
            new_versions += 1
            new_version_identities.add(identity[2])
            snapshots.add(snapshot_key)
        else:
            inserted += 1
            identities.add(identity)
            snapshots.add(snapshot_key)
    return (
        inserted,
        unchanged,
        new_versions,
        len(snapshots),
        tuple(sorted(new_version_identities)),
    )


def _read_current_occurrence_state(
    path: Path,
    start_date: date,
    end_date: date,
) -> dict[str, str]:
    """Inspect current occurrence membership without changing a dry-run DB."""

    if not path.exists() or not path.is_file():
        return {}
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        try:
            rows = connection.execute(
                """
                SELECT o.occurrence_key, o.content_signature
                FROM fd_menu_occurrences AS o
                JOIN fd_refresh_occurrences AS membership
                  ON membership.occurrence_id = o.occurrence_id
                JOIN fd_refresh_runs AS run
                  ON run.refresh_id = membership.refresh_id
                WHERE o.service_date BETWEEN ? AND ?
                  AND run.status = 'complete'
                  AND run.refresh_id = (
                      SELECT MAX(latest.refresh_id)
                      FROM fd_refresh_runs AS latest
                      WHERE latest.provider = o.provider
                        AND latest.status = 'complete'
                        AND latest.requested_start_date <= o.service_date
                        AND latest.requested_end_date >= o.service_date
                  )
                ORDER BY o.occurrence_key, o.content_signature, o.occurrence_id
                """,
                (start_date.isoformat(), end_date.isoformat()),
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).casefold():
            return {}
        raise NutritionCatalogError("unable to inspect cached FD occurrences") from exc
    except sqlite3.Error as exc:
        raise NutritionCatalogError("unable to inspect cached FD occurrences") from exc
    state: dict[str, str] = {}
    for key, signature in rows:
        previous = state.setdefault(str(key), str(signature))
        if previous != str(signature):
            raise NutritionCatalogError(
                "current FD cache contains conflicting nutrition versions for one occurrence"
            )
    return state


def _preview_occurrences(
    path: Path,
    results: Iterable[FDMappingResult],
    *,
    start_date: date,
    end_date: date,
) -> tuple[int, int, int, int]:
    previous = _read_current_occurrence_state(path, start_date, end_date)
    current: dict[str, str] = {}
    for result in results:
        key = occurrence_key_for_fd_result(result)
        signature = result.content_signature
        if not isinstance(signature, str) or not signature.strip():
            occurrence = result.occurrence
            if occurrence is None:
                continue
            signature = content_signature(occurrence.component)
        signature = signature.strip()
        prior = current.setdefault(key, signature)
        if prior != signature:
            raise NutritionCatalogError(
                "one FD refresh contains conflicting nutrition versions for one occurrence"
            )
    additions = set(current) - set(previous)
    removals = set(previous) - set(current)
    changes = {
        key
        for key in set(previous) & set(current)
        if previous[key] != current[key]
    }
    return len(current), len(additions), len(changes), len(removals)


def refresh_phelps_catalog(
    start_date: date,
    end_date: date,
    *,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    dry_run: bool = False,
    client: FDMealPlannerClient | None = None,
    observed_at: datetime | None = None,
    timeout: float = 30.0,
    time_offset: int = 0,
    expected_meal_ids_by_date: Mapping[date, Iterable[int | str]] | None = None,
) -> FDRefreshReport:
    """Fetch one explicit Phelps window and safely persist a complete snapshot.

    The local current-menu query deliberately follows the latest completed
    snapshot for a requested date.  Therefore this boundary validates every
    expected Phelps meal opportunity before starting its short SQLite write
    transaction.  A failed request, malformed visible recipe, or coverage gap
    leaves the prior completed snapshot untouched.
    """

    _validate_date_range(start_date, end_date)
    timestamp = observed_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    active_client = client or FDMealPlannerClient(timeout=timeout)
    fetched = _fetch_and_map(
        active_client,
        start_date,
        end_date,
        observed_at=timestamp,
        time_offset=time_offset,
    )
    _validate_fetched_snapshot(
        fetched,
        expected_meal_ids_by_date=expected_meal_ids_by_date,
    )
    catalog_path = Path(catalog_path).expanduser()
    unique_results = fetched.catalog_results
    if dry_run:
        inserted, unchanged, new_versions, final_count, new_version_identities = _preview_catalog(
            catalog_path,
            unique_results,
        )
        current_occurrence_count, occurrence_additions, occurrence_changes, occurrence_removals = (
            _preview_occurrences(
                catalog_path,
                fetched.valid_results,
                start_date=start_date,
                end_date=end_date,
            )
        )
    else:
        with OfficialNutritionCatalog(catalog_path) as catalog:
            refreshed = catalog.synchronize_fd_refresh(
                fetched.valid_results,
                requested_start=start_date,
                requested_end=end_date,
                observed_at=timestamp,
            )
            inserted = refreshed.catalog.inserted
            unchanged = refreshed.catalog.unchanged
            new_versions = refreshed.catalog.new_versions
            final_count = catalog.snapshot_count
            new_version_identities = tuple(
                sorted(
                    {
                        write.snapshot.source_identifier.value
                        for write in refreshed.catalog.writes
                        if write.outcome == "new_version"
                    }
                )
            )
            current_occurrence_count = refreshed.current_logical_occurrences
            occurrence_additions = refreshed.occurrence_additions
            occurrence_changes = refreshed.occurrence_changes
            occurrence_removals = refreshed.occurrence_removals
    return FDRefreshReport(
        requested_start=fetched.requested_start,
        requested_end=fetched.requested_end,
        catalog_path=catalog_path,
        populated_start=fetched.populated_start,
        populated_end=fetched.populated_end,
        meals=tuple(DINING_PERIOD_IDS),
        meal_request_count=fetched.meal_request_count,
        fd_occurrence_count=fetched.fd_occurrence_count,
        valid_mapped_occurrences=fetched.valid_mapped_occurrences,
        rejected_occurrences=fetched.rejected_occurrences,
        rejected_by_reason=fetched.rejected_by_reason,
        unique_component_identities=fetched.unique_component_identities,
        catalog_snapshots_inserted=inserted,
        catalog_snapshots_unchanged=unchanged,
        catalog_new_versions=new_versions,
        catalog_new_version_identities=new_version_identities,
        final_catalog_snapshot_count=final_count,
        menu_occurrences_observed=len(fetched.valid_results),
        current_logical_occurrences=current_occurrence_count,
        occurrence_additions=occurrence_additions,
        occurrence_changes=occurrence_changes,
        occurrence_removals=occurrence_removals,
        final_current_occurrence_count=current_occurrence_count,
        validated_complete_snapshot=True,
        dry_run=dry_run,
    )


def format_refresh_report(report: FDRefreshReport) -> str:
    """Render a stable concise report for a human operator."""

    populated = (
        f"{report.populated_start.isoformat()}..{report.populated_end.isoformat()}"
        if report.populated_date_range is not None
        else "none"
    )
    rejected = ", ".join(
        f"{reason}={count}" for reason, count in report.rejected_by_reason
    ) or "none"
    mode = "DRY RUN" if report.dry_run else "REFRESH"
    return "\n".join(
        (
            f"mode: {mode}",
            f"requested date range: {report.requested_start.isoformat()}..{report.requested_end.isoformat()}",
            f"catalog path: {report.catalog_path}",
            f"actual populated date range observed: {populated}",
            f"meals fetched: {', '.join(report.meals)} ({report.meal_request_count} bounded requests)",
            f"authoritative snapshot validation: {'passed' if report.validated_complete_snapshot else 'not run'}",
            f"FD occurrences: {report.fd_occurrence_count}",
            f"valid mapped student-visible occurrences: {report.valid_mapped_occurrences}",
            f"rejected occurrences: {report.rejected_occurrences} ({rejected})",
            f"unique stable component identities: {report.unique_component_identities}",
            f"catalog snapshots inserted{' (would be)' if report.dry_run else ''}: {report.catalog_snapshots_inserted}",
            f"catalog snapshots unchanged{' (would be)' if report.dry_run else ''}: {report.catalog_snapshots_unchanged}",
            f"catalog new versions{' (would be)' if report.dry_run else ''}: {report.catalog_new_versions}",
            f"identities with new versions: {', '.join(report.catalog_new_version_identities) or 'none'}",
            f"final catalog snapshot count{' (predicted)' if report.dry_run else ''}: {report.final_catalog_snapshot_count}",
            f"menu occurrences observed: {report.menu_occurrences_observed}",
            f"current logical occurrences{' (predicted)' if report.dry_run else ''}: {report.current_logical_occurrences}",
            f"occurrence additions/changes/removals{' (predicted)' if report.dry_run else ''}: "
            f"{report.occurrence_additions}/{report.occurrence_changes}/{report.occurrence_removals}",
            f"final current occurrence count{' (predicted)' if report.dry_run else ''}: "
            f"{report.final_current_occurrence_count}",
        )
    )


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manually refresh bounded public Phelps FD nutrition into the local catalog"
    )
    parser.add_argument("--start-date", required=True, type=_parse_date)
    parser.add_argument("--end-date", required=True, type=_parse_date)
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--time-offset", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        report = refresh_phelps_catalog(
            args.start_date,
            args.end_date,
            catalog_path=args.catalog_path,
            dry_run=args.dry_run,
            timeout=args.timeout,
            time_offset=args.time_offset,
        )
    except (FDMealPlannerError, NutritionCatalogError, TypeError, ValueError) as exc:
        print(f"FD nutrition refresh failed: {exc}", file=sys.stderr)
        return 2
    print(format_refresh_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FDRefreshReport",
    "FDRefreshValidationError",
    "format_refresh_report",
    "main",
    "refresh_phelps_catalog",
]
