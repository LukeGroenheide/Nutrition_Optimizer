"""Bounded recommendation-rendering benchmark over the local Phelps cache.

Run from the repository root after the local OpenClaw gateway is healthy:

    cp data/state/nutrition.sqlite3 /tmp/nutrition-render.sqlite3
    .venv/bin/python benchmarks/recommendation_render_benchmark.py \
        --database /tmp/nutrition-render.sqlite3

The benchmark reads real locally cached FDMealPlanner occurrences, uses only
the optimizer's supported quarter-serving grid, and makes no FD refresh,
SQLite application-state write, intake write, message send, or benchmark-file
write. It requires a copied SQLite path because opening the catalog may perform
its normal schema-initialization bookkeeping. Every semantic result, its exact
canonical amount, and its final user-facing instruction are printed for manual
physical-honesty review.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
import argparse
from pathlib import Path
import statistics
import time

from nutrition_optimizer.fdmealplanner import FDMenuOccurrence, OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import ResolvedFood
from nutrition_optimizer.meal_report import MealPlan, PlannedMealItem
from nutrition_optimizer.openclaw_recommendation_rendering import (
    OpenClawRecommendedPortionRenderer,
)
from nutrition_optimizer.recommendation_rendering import (
    RecommendedPortionRenderRequest,
    RecommendedPortionRenderer,
    RecommendedPortionRendering,
    RecommendedPortionRenderingError,
    format_meal_message,
)
from nutrition_optimizer.physical_quantity import (
    DISCRETE_COUNT_UNIT_CATEGORY,
    classify_serving_unit,
    physical_quantity_for,
    serving_multipliers_for_physical_counts,
)


DEFAULT_SERVICE_DATE = date(2026, 8, 25)
MAX_CASES = 36
GRID = tuple(Decimal(index) / Decimal("4") for index in range(1, 9))


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    occurrence: FDMenuOccurrence
    recommended_official_servings: Decimal

    @property
    def serving_unit(self) -> str:
        return self.occurrence.nutrition_record.serving.unit or "<unspecified>"

    @property
    def expected_path(self) -> str:
        quantity = physical_quantity_for(
            self.occurrence.nutrition_record.serving,
            self.recommended_official_servings,
        )
        return "luna" if quantity.is_weight else "deterministic"


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurement:
    case: BenchmarkCase
    latency_seconds: float
    rendering: RecommendedPortionRendering | None
    error: str | None

    @property
    def path(self) -> str:
        if self.rendering is not None:
            return self.rendering.rendering_method
        return self.case.expected_path

    @property
    def final_user_text(self) -> str | None:
        if self.rendering is None:
            return None
        return _final_user_text(self.case, self.rendering)


def main() -> int:
    arguments = _arguments()
    catalog = OfficialNutritionCatalog(arguments.database)
    try:
        occurrences = _load_cached_occurrences(
            catalog,
            arguments.service_date,
            required_food_names=tuple(food_name for food_name, _ in arguments.includes),
        )
        cases = _build_cases(occurrences, arguments.count, arguments.includes)
        if arguments.semantic_only:
            cases = tuple(case for case in cases if case.expected_path == "luna")
            if not cases:
                raise RuntimeError("the selected cached dates have no remaining semantic weight cases")
        semantic_renderer = OpenClawRecommendedPortionRenderer.from_env()
        renderer = RecommendedPortionRenderer(semantic_renderer)
        measurements = tuple(_measure(renderer, case) for case in cases)
        _print_report(arguments.service_date, measurements)
        return 0 if not any(measurement.error for measurement in measurements) else 1
    finally:
        catalog.close()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        required=True,
        help="path to an isolated copy of the local SQLite catalog",
    )
    parser.add_argument(
        "--date",
        dest="service_date",
        type=date.fromisoformat,
        default=DEFAULT_SERVICE_DATE,
        help=f"cached Phelps service date (default: {DEFAULT_SERVICE_DATE.isoformat()})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=MAX_CASES,
        help=f"maximum rendered items, no more than {MAX_CASES}",
    )
    parser.add_argument(
        "--semantic-only",
        action="store_true",
        help="render only cases that still use the semantic weight-description path",
    )
    parser.add_argument(
        "--include",
        dest="includes",
        action="append",
        type=_parse_include_case,
        default=[],
        metavar="FOOD=OFFICIAL_SERVINGS",
        help=(
            "include one exact cached food at the supplied official-serving "
            "multiplier before representative cases; repeat as needed"
        ),
    )
    arguments = parser.parse_args()
    if arguments.count <= 0 or arguments.count > MAX_CASES:
        parser.error(f"--count must be between 1 and {MAX_CASES}")
    if len(arguments.includes) > arguments.count:
        parser.error("--count must be at least the number of --include cases")
    if Path(arguments.database).expanduser().resolve() == Path(
        "data/state/nutrition.sqlite3"
    ).resolve():
        parser.error("--database must name an isolated SQLite copy, not production")
    return arguments


def _build_cases(
    occurrences: tuple[FDMenuOccurrence, ...],
    requested_count: int,
    includes: list[tuple[str, Decimal]] | None = None,
) -> tuple[BenchmarkCase, ...]:
    if not occurrences:
        raise RuntimeError("the selected service date has no locally cached Phelps occurrences")
    ordered = tuple(sorted(occurrences, key=_occurrence_sort_key))
    included_cases = _included_cases(occurrences, includes or ())
    included_occurrence_ids = {
        case.occurrence.occurrence_id for case in included_cases
    }
    deterministic = tuple(
        occurrence
        for occurrence in ordered
        if (
            occurrence.occurrence_id not in included_occurrence_ids
            and _is_known_unit(occurrence)
            and not _is_weight_unit(occurrence)
        )
    )
    semantic = tuple(
        occurrence
        for occurrence in ordered
        if occurrence.occurrence_id not in included_occurrence_ids
        and _is_weight_unit(occurrence)
    )
    # Keep both paths represented whenever the requested date has both kinds
    # of official serving definitions. The local cache currently has many
    # more semantic foods than the bounded audit needs.
    remaining_count = requested_count - len(included_cases)
    if remaining_count:
        deterministic_count = min(len(deterministic), max(1, remaining_count // 3))
        semantic_count = min(len(semantic), remaining_count - deterministic_count)
        if deterministic_count == 0 or semantic_count == 0:
            raise RuntimeError(
                "the selected cached date does not contain both count and semantic serving types"
            )
        selected_deterministic = _select_representative_count_cases(
            deterministic,
            deterministic_count,
        )
        selected_semantic = _select_representative_semantic_cases(
            semantic,
            semantic_count,
        )
    else:
        selected_deterministic = ()
        selected_semantic = ()
    selected = selected_deterministic + selected_semantic
    representative_cases = tuple(
        BenchmarkCase(
            occurrence=occurrence,
            recommended_official_servings=_case_multiplier(occurrence, index),
        )
        for index, occurrence in enumerate(selected)
    )
    return included_cases + representative_cases


def _included_cases(
    occurrences: tuple[FDMenuOccurrence, ...],
    includes: tuple[tuple[str, Decimal], ...] | list[tuple[str, Decimal]],
) -> tuple[BenchmarkCase, ...]:
    """Locate explicit audit examples without hard-coding foods into rendering."""

    selected: list[BenchmarkCase] = []
    used_occurrence_ids: set[int] = set()
    for food_name, official_servings in includes:
        matches = tuple(
            item
            for item in occurrences
            if item.occurrence_id not in used_occurrence_ids
            and item.nutrition_record.name.casefold() == food_name.casefold()
            and _is_known_unit(item)
        )
        # The product audit is usually run for a lunch serving line. Prefer a
        # cached lunch occurrence when the same FD component appears at more
        # than one meal; otherwise retain the current cache-window order.
        occurrence = next(
            (item for item in matches if item.meal_period_name.casefold() == "lunch"),
            matches[0] if matches else None,
        )
        if occurrence is None:
            raise RuntimeError(f"requested benchmark food is not cached: {food_name}")
        try:
            physical_quantity_for(occurrence.nutrition_record.serving, official_servings)
        except Exception as error:
            raise RuntimeError(
                f"requested benchmark multiplier is not physically safe for {food_name}"
            ) from error
        selected.append(BenchmarkCase(occurrence, official_servings))
        used_occurrence_ids.add(occurrence.occurrence_id)
    return tuple(selected)


def _measure(
    renderer: RecommendedPortionRenderer,
    case: BenchmarkCase,
) -> BenchmarkMeasurement:
    record = case.occurrence.nutrition_record
    physical_quantity = physical_quantity_for(
        record.serving,
        case.recommended_official_servings,
    )
    request = RecommendedPortionRenderRequest(
        official_food_display_name=record.name,
        physical_quantity=physical_quantity,
        station_name=case.occurrence.station_name,
    )
    started = time.monotonic()
    try:
        rendering = renderer.render(request)
    except RecommendedPortionRenderingError as error:
        return BenchmarkMeasurement(
            case=case,
            latency_seconds=time.monotonic() - started,
            rendering=None,
            error=type(error).__name__,
        )
    return BenchmarkMeasurement(
        case=case,
        latency_seconds=time.monotonic() - started,
        rendering=rendering,
        error=None,
    )


def _print_report(
    service_date: date,
    measurements: tuple[BenchmarkMeasurement, ...],
) -> None:
    renderings = tuple(
        measurement.rendering
        for measurement in measurements
        if measurement.rendering is not None
    )
    latencies = [measurement.latency_seconds for measurement in measurements]
    method_counts = Counter(measurement.path for measurement in measurements)
    descriptor_dispositions = Counter(
        rendering.semantic_descriptor_disposition for rendering in renderings
    )
    confidence_counts = Counter(rendering.confidence for rendering in renderings)
    practicality_counts = Counter(
        rendering.presentation_practicality for rendering in renderings
    )
    serving_counts = Counter(
        _serving_definition_text(measurement.case.occurrence)
        for measurement in measurements
    )
    source_date_counts = Counter(
        measurement.case.occurrence.service_date.isoformat()
        for measurement in measurements
    )
    failures = Counter(measurement.error for measurement in measurements if measurement.error)
    print(f"service_date: {service_date.isoformat()}")
    print(f"items: {len(measurements)}")
    print(f"serving_definition_distribution: {dict(sorted(serving_counts.items()))}")
    print(f"source_service_date_distribution: {dict(sorted(source_date_counts.items()))}")
    print(f"deterministic_render_count: {method_counts['deterministic']}")
    print(f"luna_render_count: {method_counts['luna']}")
    print(
        "semantic_call_count: "
        f"{sum(measurement.case.expected_path == 'luna' for measurement in measurements)}"
    )
    print(
        "semantic_descriptor_dispositions: "
        f"{dict(sorted(descriptor_dispositions.items()))}"
    )
    print(f"confidence_counts: {dict(sorted(confidence_counts.items()))}")
    print(f"practicality_counts: {dict(sorted(practicality_counts.items()))}")
    print(f"schema_runtime_failures: {dict(sorted(failures.items()))}")
    print(
        "latency_seconds: "
        f"median={statistics.median(latencies):.3f} "
        f"p95={_percentile(latencies, 0.95):.3f} "
        f"max={max(latencies):.3f}"
    )
    print("semantic_rendering_audit:")
    for index, measurement in enumerate(measurements, start=1):
        occurrence = measurement.case.occurrence
        rendering = measurement.rendering
        if rendering is None:
            print(
                f"{index:02d} | {measurement.path} | {occurrence.nutrition_record.name} | "
                f"{measurement.case.serving_unit} x {measurement.case.recommended_official_servings} | "
                f"ERROR {measurement.error}"
            )
            continue
        print(
            f"{index:02d} | {rendering.rendering_method} | "
            f"{occurrence.nutrition_record.name} | "
            f"{measurement.case.serving_unit} x "
            f"{measurement.case.recommended_official_servings} | "
            f"physical={rendering.physical_quantity.amount} "
            f"{rendering.physical_quantity.unit if rendering.physical_quantity else '<missing>'} | "
            f"canonical={rendering.canonical_quantity_text} | "
            f"final={measurement.final_user_text} | "
            f"confidence={rendering.confidence} | "
            f"practicality={rendering.presentation_practicality} | "
            f"semantic_disposition={rendering.semantic_descriptor_disposition} | "
            f"provisional_audit={_provisional_audit(rendering)} | "
            f"latency={measurement.latency_seconds:.3f}s"
        )


def _occurrence_sort_key(occurrence: FDMenuOccurrence) -> tuple[str, str, str, int]:
    return (
        occurrence.nutrition_record.serving.unit or "",
        occurrence.nutrition_record.name.casefold(),
        occurrence.station_name or "",
        occurrence.occurrence_id,
    )


def _load_cached_occurrences(
    catalog: OfficialNutritionCatalog,
    requested_date: date,
    *,
    required_food_names: tuple[str, ...] = (),
) -> tuple[FDMenuOccurrence, ...]:
    """Read the requested cache date plus nearby cached dates for diversity."""

    values: list[FDMenuOccurrence] = []
    # The benchmark is intentionally allowed to use a short local cache
    # window: serving definitions are food properties, and this makes the
    # audit include rare cached definitions such as two sticks and cooked
    # weight without refreshing or contacting FDMealPlanner.
    for offset in range(15):
        service_date = requested_date - timedelta(days=offset)
        values.extend(catalog.list_current_meal_occurrences(service_date))
        if _has_required_serving_diversity(values) and _has_required_food_names(
            values,
            required_food_names,
        ):
            break
    return tuple(values)


def _has_required_serving_diversity(
    occurrences: list[FDMenuOccurrence],
) -> bool:
    definitions = {
        _serving_definition_text(occurrence)
        for occurrence in occurrences
    }
    return {
        "1 Each",
        "3 Each",
        "2 Sticks",
        "1 Slice",
        "1 Cup",
        "3 Ounce",
        "6 Ounce Cooked Weight",
    } <= definitions


def _has_required_food_names(
    occurrences: list[FDMenuOccurrence],
    required_food_names: tuple[str, ...],
) -> bool:
    available = {occurrence.nutrition_record.name.casefold() for occurrence in occurrences}
    return all(food_name.casefold() in available for food_name in required_food_names)


def _select_representative_count_cases(
    occurrences: tuple[FDMenuOccurrence, ...],
    count: int,
) -> tuple[FDMenuOccurrence, ...]:
    required = ("1 Each", "3 Each", "2 Sticks", "1 Slice")
    selected: list[FDMenuOccurrence] = []
    used: set[int] = set()
    for definition in required:
        match = next(
            (
                occurrence
                for occurrence in occurrences
                if occurrence.occurrence_id not in used
                and _serving_definition_text(occurrence) == definition
            ),
            None,
        )
        if match is not None:
            selected.append(match)
            used.add(match.occurrence_id)
    selected.extend(
        occurrence
        for occurrence in occurrences
        if occurrence.occurrence_id not in used
    )
    return tuple(selected[:count])


def _select_representative_semantic_cases(
    occurrences: tuple[FDMenuOccurrence, ...],
    count: int,
) -> tuple[FDMenuOccurrence, ...]:
    required = ("3 Ounce", "6 Ounce Cooked Weight", "4 Ounce", "2 Ounce")
    selected: list[FDMenuOccurrence] = []
    used: set[int] = set()
    for definition in required:
        match = next(
            (
                occurrence
                for occurrence in occurrences
                if occurrence.occurrence_id not in used
                and _serving_definition_text(occurrence) == definition
            ),
            None,
        )
        if match is not None:
            selected.append(match)
            used.add(match.occurrence_id)
    selected.extend(
        occurrence
        for occurrence in occurrences
        if occurrence.occurrence_id not in used
    )
    return tuple(selected[:count])


def _serving_definition_text(occurrence: FDMenuOccurrence) -> str:
    serving = occurrence.nutrition_record.serving
    if serving.text is not None:
        return serving.text
    if serving.quantity is not None and serving.unit is not None:
        return f"{serving.quantity} {serving.unit}"
    return serving.unit or "<unspecified>"


def _is_known_unit(occurrence: FDMenuOccurrence) -> bool:
    return classify_serving_unit(occurrence.nutrition_record.serving.unit) != "unknown"


def _is_weight_unit(occurrence: FDMenuOccurrence) -> bool:
    serving = occurrence.nutrition_record.serving
    if not _is_known_unit(occurrence) or serving.quantity is None:
        return False
    # The physical quantity is built from a valid representative multiplier;
    # this property is independent of the selected grid value.
    return physical_quantity_for(serving, Decimal("1")).is_weight


def _case_multiplier(occurrence: FDMenuOccurrence, index: int) -> Decimal:
    serving = occurrence.nutrition_record.serving
    if classify_serving_unit(serving.unit) == DISCRETE_COUNT_UNIT_CATEGORY:
        grid = serving_multipliers_for_physical_counts(
            serving,
            Decimal("0.25"),
            Decimal("2"),
        )
        if not grid:
            raise RuntimeError("count serving has no safe benchmark grid")
        return grid[index % len(grid)]
    return GRID[index % len(GRID)]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _provisional_audit(rendering: RecommendedPortionRendering) -> str:
    """Bucket output for the required manual semantic-output audit."""

    if rendering.presentation_practicality == "unclear":
        return "E"
    if rendering.presentation_practicality == "awkward":
        return "C"
    if rendering.confidence == "low":
        return "B"
    return "A"


def _final_user_text(
    case: BenchmarkCase,
    rendering: RecommendedPortionRendering,
) -> str:
    """Exercise the production formatter without persisting a plan or intake."""

    occurrence = case.occurrence
    record = occurrence.nutrition_record
    food = ResolvedFood(
        original_food_text=record.name,
        occurrence=occurrence,
        equivalent_occurrences=(occurrence,),
        source_identifier=occurrence.source_identifier,
        content_signature=occurrence.content_signature,
        nutrition_snapshot_id=occurrence.nutrition_snapshot_id,
        nutrition_record=record,
        resolution_method="exact_name",
    )
    plan = MealPlan(
        occurrence.service_date,
        occurrence.meal_period_id,
        (
            PlannedMealItem(
                food,
                case.recommended_official_servings,
                rendering.natural_quantity_text,
            ),
        ),
    )
    return format_meal_message(plan)


def _parse_include_case(value: str) -> tuple[str, Decimal]:
    if not isinstance(value, str) or "=" not in value:
        raise argparse.ArgumentTypeError(
            "--include must use FOOD=OFFICIAL_SERVINGS"
        )
    food_name, raw_servings = value.rsplit("=", 1)
    food_name = food_name.strip()
    try:
        official_servings = Decimal(raw_servings.strip())
    except Exception:
        raise argparse.ArgumentTypeError(
            "--include official servings must be a decimal"
        ) from None
    if not food_name or not official_servings.is_finite() or official_servings <= 0:
        raise argparse.ArgumentTypeError(
            "--include requires a food name and a positive finite decimal"
        )
    return food_name, official_servings


if __name__ == "__main__":
    raise SystemExit(main())
