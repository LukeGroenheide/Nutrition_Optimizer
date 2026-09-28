"""Read-only serving-unit and optimizer-grid audit over the local FD cache.

Run from the repository root:

    .venv/bin/python benchmarks/physical_quantity_audit.py

The script asks ``OfficialNutritionCatalog`` for current occurrences only. It
does not refresh FDMealPlanner data, write recommendations, record intake, or
send messages. Counts are occurrence counts because a serving definition may
appear at several stations or meals.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from decimal import Decimal
import argparse

from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.meal_optimizer import MealOptimizationPolicy
from nutrition_optimizer.physical_quantity import (
    classify_serving_unit,
)


DEFAULT_DATABASE = "data/state/nutrition.sqlite3"


def main() -> int:
    arguments = _arguments()
    catalog = OfficialNutritionCatalog(arguments.database)
    try:
        dates = _cached_dates(catalog)
        occurrences = tuple(
            occurrence
            for service_date in dates
            for occurrence in catalog.list_current_meal_occurrences(service_date)
        )
        _print_report(occurrences)
        return 0
    finally:
        catalog.close()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE,
        help="local SQLite catalog path",
    )
    return parser.parse_args()


def _cached_dates(catalog: OfficialNutritionCatalog) -> tuple[date, ...]:
    rows = catalog._connection.execute(  # noqa: SLF001 - read-only audit query
        "SELECT DISTINCT service_date FROM fd_menu_occurrences ORDER BY service_date"
    ).fetchall()
    return tuple(date.fromisoformat(str(row[0])) for row in rows)


def _print_report(occurrences: tuple[object, ...]) -> None:
    policy = MealOptimizationPolicy()
    category_counts: Counter[str] = Counter()
    unit_counts: Counter[str] = Counter()
    discrete_units: Counter[str] = Counter()
    continuous_units: Counter[str] = Counter()
    unknown_units: Counter[str] = Counter()
    changed_grid_occurrences = 0
    unknown_candidates = 0
    invalid_candidates = 0

    for occurrence in occurrences:
        serving = occurrence.nutrition_record.serving
        unit = serving.unit or "<unspecified>"
        category = classify_serving_unit(serving.unit)
        category_counts[category] += 1
        unit_counts[unit] += 1
        if category == "discrete_count":
            discrete_units[unit] += 1
        elif category == "continuous":
            continuous_units[unit] += 1
        else:
            unknown_units[unit] += 1

        if category == "unknown":
            unknown_candidates += 1
            continue
        try:
            physical_grid = policy.serving_multipliers_for(serving)
        except (TypeError, ValueError):
            invalid_candidates += 1
            continue
        if not physical_grid:
            invalid_candidates += 1
        elif category == "discrete_count" and physical_grid != policy.serving_multipliers():
            changed_grid_occurrences += 1

    print(f"occurrences: {len(occurrences)}")
    print(f"discrete_count_occurrences: {category_counts['discrete_count']}")
    print(f"continuous_occurrences: {category_counts['continuous']}")
    print(f"unknown_occurrences: {category_counts['unknown']}")
    print(f"discrete_unit_frequencies: {dict(sorted(discrete_units.items()))}")
    print(f"continuous_unit_frequencies: {dict(sorted(continuous_units.items()))}")
    print(f"unknown_unit_frequencies: {dict(sorted(unknown_units.items()))}")
    print(f"all_unit_frequencies: {dict(sorted(unit_counts.items()))}")
    print(f"count_candidates_with_changed_grid: {changed_grid_occurrences}")
    print(f"unknown_candidates_conservatively_excluded: {unknown_candidates}")
    print(f"invalid_candidates: {invalid_candidates}")


if __name__ == "__main__":
    raise SystemExit(main())
