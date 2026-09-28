"""Detroit-local rolling maintenance for the authoritative FD menu cache.

This module deliberately sits between the public FDMealPlanner client/ingestion
path and the local SQLite catalog.  It never creates a BlueBubbles client,
prepares a recommendation, or changes meal-plan/intake state.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import logging
from pathlib import Path
import sys
from typing import Literal

from ..application_clock import (
    DEFAULT_NUTRITION_APPLICATION_CLOCK,
    NutritionApplicationClock,
)
from ..meal_identity import canonical_meal_name
from ..recommendation_scheduler import (
    DEFAULT_RECOMMENDATION_DISPATCH_SCHEDULE,
    RecommendationDispatchSchedule,
    ScheduledMealOpportunity,
)
from .catalog import DEFAULT_CATALOG_PATH, NutritionCatalogError, OfficialNutritionCatalog
from .client import DINING_PERIOD_IDS, FDMealPlannerClient, FDMealPlannerError
from .refresh import FDRefreshReport, format_refresh_report, refresh_phelps_catalog


__all__ = [
    "DEFAULT_MENU_CACHE_HORIZON_DAYS",
    "MenuCacheRefreshReport",
    "MenuCoverageStatus",
    "MenuCoverageStatusItem",
    "expected_menu_opportunities",
    "format_menu_coverage_status",
    "main",
    "menu_cache_date_range",
    "menu_coverage_status",
    "refresh_menu_cache",
]


_LOGGER = logging.getLogger(__name__)

# The production service intentionally asks for a small rolling window rather
# than the whole six-week FD cycle.  The CLI's ``--days`` option is the single
# controlled operator override for testing/backfill.
DEFAULT_MENU_CACHE_HORIZON_DAYS = 7


CoverageState = Literal["present", "missing", "not_expected"]


@dataclass(frozen=True, slots=True)
class MenuCoverageStatusItem:
    """Read-only coverage result for one FD meal context on one service date."""

    service_date: date
    meal_id: int
    meal_context: str
    expected: bool
    occurrence_count: int
    coverage: CoverageState
    latest_observed_at: datetime | None

    def __post_init__(self) -> None:
        if not isinstance(self.service_date, date) or isinstance(self.service_date, datetime):
            raise TypeError("service_date must be a date")
        if self.meal_id not in set(DINING_PERIOD_IDS.values()):
            raise ValueError("meal_id must be a known FD meal period")
        if not isinstance(self.meal_context, str) or not self.meal_context.strip():
            raise ValueError("meal_context must be non-empty text")
        if not isinstance(self.expected, bool):
            raise TypeError("expected must be a bool")
        if isinstance(self.occurrence_count, bool) or not isinstance(self.occurrence_count, int):
            raise TypeError("occurrence_count must be an integer")
        if self.occurrence_count < 0:
            raise ValueError("occurrence_count must not be negative")
        if self.coverage not in {"present", "missing", "not_expected"}:
            raise ValueError("coverage is invalid")
        if self.expected != (self.coverage != "not_expected"):
            raise ValueError("coverage expected flag is inconsistent")
        if self.expected and self.coverage == "present" and self.occurrence_count == 0:
            raise ValueError("present coverage requires at least one occurrence")
        if self.expected and self.coverage == "missing" and self.occurrence_count != 0:
            raise ValueError("missing coverage requires zero occurrences")
        if self.latest_observed_at is not None and (
            not isinstance(self.latest_observed_at, datetime)
            or self.latest_observed_at.tzinfo is None
            or self.latest_observed_at.utcoffset() is None
        ):
            raise ValueError("latest_observed_at must be timezone-aware when supplied")


@dataclass(frozen=True, slots=True)
class MenuCoverageStatus:
    """A compact rolling local-cache diagnostic with no upstream access."""

    requested_start: date
    requested_end: date
    catalog_path: Path
    items: tuple[MenuCoverageStatusItem, ...]

    def __post_init__(self) -> None:
        _validate_date_range(self.requested_start, self.requested_end)
        if not isinstance(self.catalog_path, Path):
            raise TypeError("catalog_path must be a Path")
        if not isinstance(self.items, tuple) or not all(
            isinstance(item, MenuCoverageStatusItem) for item in self.items
        ):
            raise TypeError("items must be MenuCoverageStatusItem values")

    @property
    def missing_expected(self) -> tuple[MenuCoverageStatusItem, ...]:
        return tuple(item for item in self.items if item.coverage == "missing")


@dataclass(frozen=True, slots=True)
class MenuCacheRefreshReport:
    """One production refresh application result plus its local coverage read."""

    refresh: FDRefreshReport
    coverage: MenuCoverageStatus

    def __post_init__(self) -> None:
        if not isinstance(self.refresh, FDRefreshReport):
            raise TypeError("refresh must be an FDRefreshReport")
        if not isinstance(self.coverage, MenuCoverageStatus):
            raise TypeError("coverage must be a MenuCoverageStatus")
        if (
            self.refresh.requested_start != self.coverage.requested_start
            or self.refresh.requested_end != self.coverage.requested_end
        ):
            raise ValueError("refresh and coverage ranges must match")


def _validate_date_range(start_date: date, end_date: date) -> None:
    if not isinstance(start_date, date) or isinstance(start_date, datetime):
        raise TypeError("start_date must be a date")
    if not isinstance(end_date, date) or isinstance(end_date, datetime):
        raise TypeError("end_date must be a date")
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")


def _validate_days(days: int) -> int:
    if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
        raise ValueError("days must be a positive integer")
    return days


def _iter_dates(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def menu_cache_date_range(
    *,
    start_date: date | None = None,
    days: int = DEFAULT_MENU_CACHE_HORIZON_DAYS,
    clock: NutritionApplicationClock | None = None,
) -> tuple[date, date]:
    """Return the Detroit-local rolling range used by refresh and diagnostics."""

    _validate_days(days)
    if start_date is not None and (
        not isinstance(start_date, date) or isinstance(start_date, datetime)
    ):
        raise TypeError("start_date must be a date when supplied")
    service_start = start_date or (clock or DEFAULT_NUTRITION_APPLICATION_CLOCK).service_date()
    return service_start, service_start + timedelta(days=days - 1)


def expected_menu_opportunities(
    start_date: date,
    end_date: date,
    *,
    schedule: RecommendationDispatchSchedule = DEFAULT_RECOMMENDATION_DISPATCH_SCHEDULE,
) -> Mapping[date, tuple[ScheduledMealOpportunity, ...]]:
    """Return exact expected FD contexts from the existing product schedule."""

    _validate_date_range(start_date, end_date)
    if not isinstance(schedule, RecommendationDispatchSchedule):
        raise TypeError("schedule must be a RecommendationDispatchSchedule")
    return {
        service_date: schedule.opportunities_for(service_date)
        for service_date in _iter_dates(start_date, end_date)
    }


def _coverage_status(
    catalog: OfficialNutritionCatalog | None,
    *,
    catalog_path: Path,
    start_date: date,
    end_date: date,
    schedule: RecommendationDispatchSchedule,
) -> MenuCoverageStatus:
    opportunities = expected_menu_opportunities(start_date, end_date, schedule=schedule)
    items: list[MenuCoverageStatusItem] = []
    for service_date in _iter_dates(start_date, end_date):
        by_meal_id = {opportunity.meal_id: opportunity for opportunity in opportunities[service_date]}
        for meal_id in DINING_PERIOD_IDS.values():
            opportunity = by_meal_id.get(meal_id)
            occurrences = () if catalog is None else catalog.list_current_meal_occurrences(
                service_date,
                meal=meal_id,
            )
            occurrence_count = len(occurrences)
            latest_observed_at = max(
                (occurrence.last_observed_at for occurrence in occurrences),
                default=None,
            )
            expected = opportunity is not None
            coverage: CoverageState
            if not expected:
                coverage = "not_expected"
            elif occurrence_count:
                coverage = "present"
            else:
                coverage = "missing"
            items.append(
                MenuCoverageStatusItem(
                    service_date=service_date,
                    meal_id=meal_id,
                    meal_context=(
                        opportunity.label
                        if opportunity is not None
                        else (canonical_meal_name(meal_id) or f"fd:{meal_id}")
                    ),
                    expected=expected,
                    occurrence_count=occurrence_count,
                    coverage=coverage,
                    latest_observed_at=latest_observed_at,
                )
            )
    return MenuCoverageStatus(start_date, end_date, catalog_path, tuple(items))


def menu_coverage_status(
    *,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    start_date: date | None = None,
    days: int = DEFAULT_MENU_CACHE_HORIZON_DAYS,
    clock: NutritionApplicationClock | None = None,
    schedule: RecommendationDispatchSchedule = DEFAULT_RECOMMENDATION_DISPATCH_SCHEDULE,
) -> MenuCoverageStatus:
    """Inspect local expected-menu coverage without migrations or network calls."""

    requested_start, requested_end = menu_cache_date_range(
        start_date=start_date,
        days=days,
        clock=clock,
    )
    path = Path(catalog_path).expanduser()
    if not path.exists():
        return _coverage_status(
            None,
            catalog_path=path,
            start_date=requested_start,
            end_date=requested_end,
            schedule=schedule,
        )
    if not path.is_file():
        raise NutritionCatalogError(f"catalog path is not a file: {path}")
    # Read-only mode rejects stale schemas rather than silently migrating one
    # during an operator diagnostic.
    with OfficialNutritionCatalog(path, read_only=True) as catalog:
        return _coverage_status(
            catalog,
            catalog_path=path,
            start_date=requested_start,
            end_date=requested_end,
            schedule=schedule,
        )


def refresh_menu_cache(
    *,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    start_date: date | None = None,
    days: int = DEFAULT_MENU_CACHE_HORIZON_DAYS,
    clock: NutritionApplicationClock | None = None,
    schedule: RecommendationDispatchSchedule = DEFAULT_RECOMMENDATION_DISPATCH_SCHEDULE,
    client: FDMealPlannerClient | None = None,
    observed_at: datetime | None = None,
    timeout: float = 30.0,
    dry_run: bool = False,
) -> MenuCacheRefreshReport:
    """Refresh the rolling cache through the existing FD ingestion boundary.

    Upstream fetch/mapping/complete-snapshot validation remain entirely inside
    :func:`refresh_phelps_catalog`; it opens SQLite only after they pass.
    This wrapper solely supplies the Detroit-local rolling horizon and reuses
    recommendation opportunities as its expected-coverage policy.
    """

    requested_start, requested_end = menu_cache_date_range(
        start_date=start_date,
        days=days,
        clock=clock,
    )
    opportunities = expected_menu_opportunities(
        requested_start,
        requested_end,
        schedule=schedule,
    )
    expected_meal_ids = {
        service_date: tuple(opportunity.meal_id for opportunity in values)
        for service_date, values in opportunities.items()
    }
    report = refresh_phelps_catalog(
        requested_start,
        requested_end,
        catalog_path=catalog_path,
        dry_run=dry_run,
        client=client,
        observed_at=observed_at,
        timeout=timeout,
        expected_meal_ids_by_date=expected_meal_ids,
    )
    coverage = menu_coverage_status(
        catalog_path=catalog_path,
        start_date=requested_start,
        days=days,
        schedule=schedule,
    )
    result = MenuCacheRefreshReport(report, coverage)
    if result.coverage.missing_expected:
        missing = ", ".join(
            f"{item.service_date.isoformat()} fd:{item.meal_id} {item.meal_context}"
            for item in result.coverage.missing_expected
        )
        _LOGGER.warning("FD menu-cache refresh completed with missing expected coverage: %s", missing)
    return result


def format_menu_coverage_status(status: MenuCoverageStatus) -> str:
    """Render concise deterministic cache coverage without listing foods."""

    lines = [
        f"catalog path: {status.catalog_path}",
        f"requested date range: {status.requested_start.isoformat()}..{status.requested_end.isoformat()}",
    ]
    for item in status.items:
        latest = item.latest_observed_at.isoformat() if item.latest_observed_at else "none"
        expectation = "expected" if item.expected else "not expected"
        lines.append(
            f"{item.service_date.isoformat()} fd:{item.meal_id} {item.meal_context}: "
            f"{expectation}; occurrences={item.occurrence_count}; "
            f"coverage={item.coverage}; latest_observed={latest}"
        )
    return "\n".join(lines)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def _parse_days(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("days must be a positive integer") from exc
    try:
        return _validate_days(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _add_range_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-date", type=_parse_date)
    parser.add_argument("--days", type=_parse_days, default=DEFAULT_MENU_CACHE_HORIZON_DAYS)
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)


def main(argv: Sequence[str] | None = None) -> int:
    """Run an explicit rolling refresh or a strictly local coverage diagnostic."""

    parser = argparse.ArgumentParser(
        description="Maintain or inspect the local authoritative FD menu cache"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    refresh_parser = subparsers.add_parser(
        "refresh-menu-cache",
        help="refresh the Detroit-local rolling FD cache; no recommendations or messages",
    )
    _add_range_arguments(refresh_parser)
    refresh_parser.add_argument("--timeout", type=float, default=30.0)
    refresh_parser.add_argument("--dry-run", action="store_true")
    coverage_parser = subparsers.add_parser(
        "menu-coverage-status",
        help="inspect rolling local FD coverage without network or writes",
    )
    _add_range_arguments(coverage_parser)
    args = parser.parse_args(argv)
    try:
        if args.command == "menu-coverage-status":
            print(
                format_menu_coverage_status(
                    menu_coverage_status(
                        catalog_path=args.catalog_path,
                        start_date=args.start_date,
                        days=args.days,
                    )
                )
            )
            return 0
        result = refresh_menu_cache(
            catalog_path=args.catalog_path,
            start_date=args.start_date,
            days=args.days,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )
    except (FDMealPlannerError, NutritionCatalogError, TypeError, ValueError) as exc:
        print(f"FD menu-cache refresh failed: {exc}", file=sys.stderr)
        return 2
    print(format_refresh_report(result.refresh))
    print(format_menu_coverage_status(result.coverage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
