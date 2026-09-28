"""Bounded, read-only Luna benchmark for known-meal report reconciliation.

Run from the repository root after the local OpenClaw gateway is healthy:

    .venv/bin/python benchmarks/meal_report_benchmark.py

It performs exactly 25 meal-report parses against synthetic ``MealPlan``
objects assembled from current local FD occurrences.  It neither refreshes FD
data nor writes intake, ledger, benchmark logs, or database state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
import argparse
import statistics
import time

from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import FoodResolutionRequest, LocalFDFoodResolver, ResolvedFood
from nutrition_optimizer.meal_report import MealPlan, MealReportReconciler, PlannedMealItem
from nutrition_optimizer.openclaw_meal_report import OpenClawMealReportInterpreter
from nutrition_optimizer.portion_interpretation import NaturalPortionInterpreter


SERVICE_DATE = date(2026, 8, 25)
MEAL = "lunch"
CASE_COUNT = 25


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    text: str
    eaten_ids: frozenset[str] = frozenset()
    skipped_ids: frozenset[str] = frozenset()
    inherited_ids: frozenset[str] = frozenset()
    override_ids: frozenset[str] = frozenset()
    unplanned_foods: frozenset[str] = frozenset()
    clarification_expected: bool = False


def main() -> int:
    arguments = _arguments()
    catalog = OfficialNutritionCatalog(Path("data/state/nutrition.sqlite3"))
    try:
        plan = _plan_from_local_cache(catalog)
        all_cases = _cases()
        assert len(all_cases) == CASE_COUNT
        cases = all_cases[arguments.start : arguments.start + arguments.count]
        if not cases:
            raise ValueError("the selected benchmark range is empty")
        parser = OpenClawMealReportInterpreter()
        # Deliberately no semantic portion matcher: this benchmark makes one
        # Luna call per case, only for the meal-report boundary. Explicit
        # portion estimates remain a separate existing benchmark concern.
        measurements = []
        for index, case in enumerate(cases, start=arguments.start + 1):
            started = time.monotonic()
            semantic = parser.interpret(plan, case.text)
            parse_latency = time.monotonic() - started
            # Reconciliation is deterministic (and uses no downstream Luna
            # matcher); call it through a tiny result-returning adapter.
            outcome = MealReportReconciler(
                _FixedSemanticInterpreter(semantic),
                LocalFDFoodResolver(catalog),
                NaturalPortionInterpreter(),
            ).reconcile(plan, case.text)
            measurements.append(_measure(case, plan, semantic, outcome, parse_latency))
            print(f"completed_case: {index}/{CASE_COUNT}", flush=True)
        _print_report(measurements)
        return 0
    finally:
        catalog.close()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0, help="zero-based case offset")
    parser.add_argument(
        "--count",
        type=int,
        default=CASE_COUNT,
        help="number of cases to run (default: all 25)",
    )
    arguments = parser.parse_args()
    if arguments.start < 0 or arguments.count <= 0 or arguments.start + arguments.count > CASE_COUNT:
        parser.error(f"--start/--count must select within 0..{CASE_COUNT}")
    return arguments


@dataclass(frozen=True, slots=True)
class _FixedSemanticInterpreter:
    result: object

    def interpret(self, plan: MealPlan, user_text: str) -> object:
        return self.result


def _plan_from_local_cache(catalog: OfficialNutritionCatalog) -> MealPlan:
    resolver = LocalFDFoodResolver(catalog)
    requested = (
        ("Chicken Tenders", "AMERICAN GRILLE", "3 chicken tenders", Decimal("1")),
        ("Broccoli", None, "some broccoli", Decimal("1")),
        ("Garlic Bread", None, "one garlic bread", Decimal("1")),
    )
    items = []
    for name, station, display_quantity, servings in requested:
        food = resolver.resolve(
            FoodResolutionRequest(name, SERVICE_DATE, meal=MEAL, station=station)
        )
        if not isinstance(food, ResolvedFood):
            raise RuntimeError(f"benchmark local food is unavailable or ambiguous: {name}")
        items.append(PlannedMealItem(food, servings, display_quantity))
    return MealPlan(SERVICE_DATE, MEAL, tuple(items))


def _cases() -> tuple[BenchmarkCase, ...]:
    """25 examples: plan actions, aliases, overrides, additions, and failures."""

    return (
        BenchmarkCase("I ate everything.", frozenset({"item_1", "item_2", "item_3"}), inherited_ids=frozenset({"item_1", "item_2", "item_3"})),
        BenchmarkCase("I ate all of it.", frozenset({"item_1", "item_2", "item_3"}), inherited_ids=frozenset({"item_1", "item_2", "item_3"})),
        BenchmarkCase("I ate everything except the broccoli.", frozenset({"item_1", "item_3"}), frozenset({"item_2"}), frozenset({"item_1", "item_3"})),
        BenchmarkCase("I ate the chicken tenders and broccoli.", frozenset({"item_1", "item_2"}), inherited_ids=frozenset({"item_1", "item_2"})),
        BenchmarkCase("I skipped the garlic bread.", skipped_ids=frozenset({"item_3"})),
        BenchmarkCase("I only ate one tender.", frozenset({"item_1"}), override_ids=frozenset({"item_1"}), clarification_expected=True),
        BenchmarkCase("I had two servings of broccoli instead.", frozenset({"item_2"}), override_ids=frozenset({"item_2"})),
        BenchmarkCase("I ate the chicken and grabbed baked potatoes too.", frozenset({"item_1"}), inherited_ids=frozenset({"item_1"}), unplanned_foods=frozenset({"baked potatoes"}), clarification_expected=True),
        BenchmarkCase("I ate the tenders.", frozenset({"item_1"}), inherited_ids=frozenset({"item_1"})),
        BenchmarkCase("I ate broccoli but skipped the garlic bread.", frozenset({"item_2"}), frozenset({"item_3"}), frozenset({"item_2"})),
        BenchmarkCase("I ate the food.", clarification_expected=True),
        BenchmarkCase("I ate the broccoli but I skipped the broccoli.", clarification_expected=True),
        BenchmarkCase("I ate the tenders and grabbed some banana.", frozenset({"item_1"}), inherited_ids=frozenset({"item_1"}), unplanned_foods=frozenset({"banana"}), clarification_expected=True),
        BenchmarkCase("I had all the chicken tenders.", frozenset({"item_1"}), inherited_ids=frozenset({"item_1"})),
        BenchmarkCase("The broccoli was skipped, but I ate the bread.", frozenset({"item_3"}), frozenset({"item_2"}), frozenset({"item_3"})),
        BenchmarkCase("I ate all of it except the tenders.", frozenset({"item_2", "item_3"}), frozenset({"item_1"}), frozenset({"item_2", "item_3"})),
        BenchmarkCase("I ate one garlic bread.", frozenset({"item_3"}), inherited_ids=frozenset({"item_3"})),
        BenchmarkCase("I ate the tenders and broccoli, not the garlic bread.", frozenset({"item_1", "item_2"}), frozenset({"item_3"}), frozenset({"item_1", "item_2"})),
        BenchmarkCase("I skipped everything.", skipped_ids=frozenset({"item_1", "item_2", "item_3"})),
        BenchmarkCase("I ate the broccoli and also got baked potatoes.", frozenset({"item_2"}), inherited_ids=frozenset({"item_2"}), unplanned_foods=frozenset({"baked potatoes"}), clarification_expected=True),
        BenchmarkCase("I had half a serving of broccoli.", frozenset({"item_2"}), override_ids=frozenset({"item_2"})),
        BenchmarkCase("I ate the chicken, skipped broccoli, and ate the bread.", frozenset({"item_1", "item_3"}), frozenset({"item_2"}), frozenset({"item_1", "item_3"})),
        BenchmarkCase("I ate none of the broccoli.", skipped_ids=frozenset({"item_2"})),
        BenchmarkCase("I ate everything and grabbed a banana.", frozenset({"item_1", "item_2", "item_3"}), inherited_ids=frozenset({"item_1", "item_2", "item_3"}), unplanned_foods=frozenset({"banana"}), clarification_expected=True),
        BenchmarkCase("I just ate the tenders, nothing else.", frozenset({"item_1"}), frozenset({"item_2", "item_3"}), frozenset({"item_1"})),
    )


def _measure(case, plan, semantic, outcome, latency):
    # Action classification belongs to the meal-report semantic boundary.  An
    # explicit quantity can correctly identify an eaten item yet remain
    # unresolved later when this benchmark deliberately disables the distinct
    # portion semantic matcher.
    returned_eaten = frozenset(
        item.plan_item_id for item in semantic.planned_items if item.action == "eaten"
    )
    returned_skipped = frozenset(
        item.plan_item_id for item in semantic.planned_items if item.action == "skipped"
    )
    returned_inherited = frozenset(
        plan.item_id(item.plan_item)
        for item in outcome.eaten_items
        if item.quantity_source == "planned_quantity"
    )
    returned_overrides = frozenset(
        item.plan_item_id
        for item in semantic.planned_items
        if item.action == "eaten" and item.quantity_text is not None
    )
    returned_unplanned = frozenset(item.food_text.casefold() for item in outcome.unplanned_items)
    return {
        "eaten": returned_eaten == case.eaten_ids,
        "skipped": returned_skipped == case.skipped_ids,
        "inheritance": returned_inherited == case.inherited_ids,
        "overrides": returned_overrides == case.override_ids,
        "unplanned": returned_unplanned == case.unplanned_foods,
        "clarification": bool(outcome.clarification_items) == case.clarification_expected,
        "false_positive": bool((returned_eaten - case.eaten_ids) or (returned_skipped - case.skipped_ids)),
        "schema_runtime_failure": False,
        "latency": latency,
        "inherited_count": len(returned_inherited),
        "eaten_count": len(returned_eaten),
    }


def _print_report(rows) -> None:
    count = len(rows)
    def rate(field): return sum(row[field] for row in rows), count
    latencies = sorted(row["latency"] for row in rows)
    inherited = sum(row["inherited_count"] for row in rows)
    eaten = sum(row["eaten_count"] for row in rows)
    print(f"cases: {count}")
    for field in ("eaten", "skipped", "inheritance", "overrides", "unplanned", "clarification"):
        passed, total = rate(field)
        print(f"{field}_accuracy: {passed}/{total} ({passed / total:.1%})")
    false_positives, total = rate("false_positive")
    failures, _ = rate("schema_runtime_failure")
    print(f"false_positive_rate: {false_positives}/{total} ({false_positives / total:.1%})")
    print(f"schema_runtime_failures: {failures}/{total}")
    print(f"planned_quantity_inheritance: {inherited}/{eaten}")
    print(f"portion_model_calls_avoided: {inherited}")
    print(f"latency_seconds: median={statistics.median(latencies):.3f} p95={latencies[max(0, int(.95 * count) - 1)]:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
