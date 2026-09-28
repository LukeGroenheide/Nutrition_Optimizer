"""Fiber-only domain, immutable sidecar, and optimizer coverage."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from nutrition_optimizer.fdmealplanner import (
    CATALOG_SCHEMA_VERSION,
    CatalogObservation,
    OfficialNutritionCatalog,
    content_signature,
    map_component,
)
from nutrition_optimizer.fdmealplanner.catalog import NutritionCatalogValidationError
from nutrition_optimizer.meal_optimizer import (
    MealOptimizationPolicy,
    NutrientObjectiveWeights,
    optimize_meal,
)
from nutrition_optimizer.nutrition import (
    DailyLedger,
    DailyMinimums,
    DailyTargets,
    IntakeEntry,
    NutritionProvenance,
    NutritionRecord,
    NutrientProfile,
    Serving,
    SourceIdentifier,
    add_nutrients,
    calculate_daily_balance,
    scale_nutrients,
)


T0 = datetime(2026, 8, 26, tzinfo=timezone.utc)
DAY = date(2026, 8, 26)


def record(name: str, *, fiber: Decimal | None = Decimal("3"), component_id: int = 1) -> NutritionRecord:
    return NutritionRecord(
        name=name,
        # Optimizer fixtures use a recognized continuous unit; unknown
        # provider units are intentionally excluded from recommendation grids.
        serving=Serving(Decimal("1"), "Cup", "1 Cup"),
        nutrients=NutrientProfile(
            calories_kcal=Decimal("100"), protein_g=Decimal("10"),
            carbohydrates_g=Decimal("10"), fat_g=Decimal("3"), sodium_mg=Decimal("100"),
            dietary_fiber_g=fiber,
        ),
        provenance=NutritionProvenance(
            "FDMealPlanner", T0, identifiers=(SourceIdentifier("component", f"7:181:{component_id}"),)
        ),
    )


def cached_occurrence(identifier: int, food: NutritionRecord, *, meal: str = "lunch"):
    from nutrition_optimizer.fdmealplanner.catalog import FDMenuOccurrence
    return FDMenuOccurrence(
        occurrence_id=identifier, occurrence_key=f"key-{identifier}", service_date=DAY,
        meal_period_id=meal, meal_period_name=meal, station_concept_id="test", station_name="Test",
        source_identifier=food.provenance.identifiers[0], content_signature=f"sig-{identifier}",
        nutrition_snapshot_id=identifier, nutrition_record=food, menu_detail_id=str(identifier), menu_id="test",
        first_observed_at=T0, last_observed_at=T0,
    )


def component(*, fiber: object = "3", fiber_uom: object = "g") -> dict[str, object]:
    return {
        "componentId": 1, "componentTypeId": 181, "englishAlternateName": "Fiber food",
        "componentName": "Fiber food", "recipePortionSize": "1", "recipePortionSizeUnit": "Serving",
        "calories": "100", "caloriesUOM": "kcal", "protein": "10", "proteinUOM": "g",
        "carbohydrates": "10", "carbohydratesUOM": "g", "fat": "3", "fatUOM": "g",
        "sodium": "100", "sodiumUOM": "mg", "dietaryFiber": fiber,
        "dietaryFiberUOM": fiber_uom,
    }


def targets(*, fiber: Decimal | None = None) -> DailyTargets:
    return DailyTargets(
        Decimal("1000"), Decimal("100"), Decimal("100"), Decimal("50"),
        DailyMinimums(fiber),
    )


class FiberDomainTests(unittest.TestCase):
    def test_zero_none_negative_and_nonfinite_follow_domain_rules(self) -> None:
        self.assertEqual(NutrientProfile(dietary_fiber_g=Decimal("0")).dietary_fiber_g, Decimal("0"))
        self.assertIsNone(NutrientProfile().dietary_fiber_g)
        for value in (Decimal("-0.1"), Decimal("NaN"), Decimal("Infinity")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    NutrientProfile(dietary_fiber_g=value)

    def test_scaling_and_summing_preserve_decimal_and_unknown(self) -> None:
        self.assertEqual(scale_nutrients(NutrientProfile(dietary_fiber_g=Decimal("3")), Decimal("0.5")).dietary_fiber_g, Decimal("1.5"))
        self.assertEqual(add_nutrients(NutrientProfile(dietary_fiber_g=Decimal("2")), NutrientProfile(dietary_fiber_g=Decimal("3"))).dietary_fiber_g, Decimal("5"))
        self.assertIsNone(add_nutrients(NutrientProfile(dietary_fiber_g=Decimal("2")), NutrientProfile()).dietary_fiber_g)

    def test_minimum_balance_is_compatible_and_never_negative(self) -> None:
        inactive = calculate_daily_balance(targets(), NutrientProfile(dietary_fiber_g=Decimal("0")))
        self.assertIsNone(inactive.minimums.dietary_fiber_deficit_g)
        deficit = calculate_daily_balance(targets(fiber=Decimal("10")), NutrientProfile(dietary_fiber_g=Decimal("3")))
        met = calculate_daily_balance(targets(fiber=Decimal("10")), NutrientProfile(dietary_fiber_g=Decimal("10")))
        exceeded = calculate_daily_balance(targets(fiber=Decimal("10")), NutrientProfile(dietary_fiber_g=Decimal("12")))
        unknown = calculate_daily_balance(targets(fiber=Decimal("10")), NutrientProfile())
        self.assertEqual(deficit.minimums.dietary_fiber_deficit_g, Decimal("7"))
        self.assertEqual(met.minimums.dietary_fiber_deficit_g, Decimal("0"))
        self.assertEqual(exceeded.minimums.dietary_fiber_deficit_g, Decimal("0"))
        self.assertIsNone(unknown.minimums.dietary_fiber_deficit_g)


class FiberMapperTests(unittest.TestCase):
    def test_maps_zero_and_rejects_bad_uom(self) -> None:
        mapped = map_component(component(fiber="0"), tenant_id=7)
        self.assertEqual(mapped.record.nutrients.dietary_fiber_g, Decimal("0"))
        bad = map_component(component(fiber_uom="mg"), tenant_id=7)
        self.assertEqual(bad.diagnostic.reason, "unexpected_uom")

    def test_existing_signature_algorithm_is_stable(self) -> None:
        self.assertEqual(
            content_signature(component()),
            "bdaa9bd9cf83445e909b80633532d5c1d74eb9e5e340ebd9df121d3f4805de43",
        )


class FiberCatalogTests(unittest.TestCase):
    def open(self):
        directory = TemporaryDirectory(); self.addCleanup(directory.cleanup)
        catalog = OfficialNutritionCatalog(Path(directory.name) / "nutrition.sqlite3")
        self.addCleanup(catalog.close)
        return catalog

    def test_v3_to_current_sidecar_preserves_snapshots_and_hydrates_fiber(self) -> None:
        catalog = self.open()
        stored = catalog.store(record("Fiber", fiber=Decimal("4")), content_signature="fiber-sig", observed_at=T0)
        snapshot_json = catalog._connection.execute("SELECT snapshot_json FROM nutrition_snapshots WHERE snapshot_id = ?", (stored.snapshot.snapshot_id,)).fetchone()[0]
        self.assertNotIn("dietary_fiber_g", snapshot_json)
        self.assertEqual(catalog.dietary_fiber_extension_count, 1)
        snapshot_count = catalog.snapshot_count
        occurrence_count = catalog._connection.execute(
            "SELECT COUNT(*) FROM fd_menu_occurrences"
        ).fetchone()[0]
        catalog._connection.execute("DROP TABLE nutrition_snapshot_dietary_fiber")
        catalog._connection.execute("PRAGMA user_version = 3")
        catalog._connection.commit(); catalog.close()
        catalog = OfficialNutritionCatalog(catalog.path); self.addCleanup(catalog.close)
        self.assertEqual(CATALOG_SCHEMA_VERSION, 14)
        self.assertEqual(catalog.snapshot_count, snapshot_count)
        self.assertEqual(
            catalog._connection.execute("SELECT COUNT(*) FROM fd_menu_occurrences").fetchone()[0],
            occurrence_count,
        )
        self.assertIsNone(catalog.get_snapshot_by_id(stored.snapshot.snapshot_id).record.nutrients.dietary_fiber_g)

    def test_exact_snapshot_extensions_backfill_idempotently_without_guessing(self) -> None:
        catalog = self.open()
        old = catalog.store(record("Old", fiber=None, component_id=8), content_signature="old", observed_at=T0).snapshot
        new = catalog.store(record("New", fiber=None, component_id=8), content_signature="new", observed_at=T0).snapshot
        observations = (
            CatalogObservation(record("Old", fiber=Decimal("2"), component_id=8), "old", T0),
            CatalogObservation(record("New", fiber=Decimal("7"), component_id=8), "new", T0),
            CatalogObservation(record("Missing", fiber=Decimal("9"), component_id=9), "missing", T0),
        )
        first = catalog.backfill_dietary_fiber(observations)
        second = catalog.backfill_dietary_fiber(observations)
        self.assertEqual((first.matched_snapshots, first.inserted, len(first.unmatched)), (2, 2, 1))
        self.assertEqual((second.inserted, second.unchanged), (0, 2))
        self.assertEqual(catalog.get_snapshot_by_id(old.snapshot_id).record.nutrients.dietary_fiber_g, Decimal("2"))
        self.assertEqual(catalog.get_snapshot_by_id(new.snapshot_id).record.nutrients.dietary_fiber_g, Decimal("7"))

    def test_conflicting_backfill_rolls_back_without_partial_extension(self) -> None:
        catalog = self.open()
        catalog.store(record("A", fiber=None, component_id=1), content_signature="a", observed_at=T0)
        catalog.store(record("B", fiber=Decimal("1"), component_id=2), content_signature="b", observed_at=T0)
        with self.assertRaises(NutritionCatalogValidationError):
            catalog.backfill_dietary_fiber((
                CatalogObservation(record("A", fiber=Decimal("2"), component_id=1), "a", T0),
                CatalogObservation(record("B", fiber=Decimal("3"), component_id=2), "b", T0),
            ))
        self.assertEqual(catalog.dietary_fiber_extension_count, 1)
        self.assertIsNone(catalog.get_snapshot_by_id(1).record.nutrients.dietary_fiber_g)


class FiberOptimizerTests(unittest.TestCase):
    def test_skipped_earlier_fiber_increases_later_pressure(self) -> None:
        earlier = record("Fiber earlier", fiber=Decimal("10"))
        low = cached_occurrence(1, record("Low fiber", fiber=Decimal("1")))
        high = cached_occurrence(2, record("High fiber", fiber=Decimal("8"), component_id=2))
        policy = MealOptimizationPolicy(minimum_servings=Decimal("1"), maximum_servings_per_food=Decimal("1"), serving_increment=Decimal("1"), max_distinct_foods=1, breakfast_stage_fraction=Decimal("1"), lunch_stage_fraction=Decimal("1"), dinner_stage_fraction=Decimal("1"))
        eaten = optimize_meal(DAY, "lunch", targets(fiber=Decimal("10")), DailyLedger((IntakeEntry(earlier, Decimal("1")),)), (low, high), policy=policy)
        skipped = optimize_meal(DAY, "lunch", targets(fiber=Decimal("10")), DailyLedger(), (low, high), policy=policy)
        self.assertEqual(eaten.items[0].record.name, "Low fiber")
        self.assertEqual(skipped.items[0].record.name, "High fiber")

    def test_fiber_uses_the_configured_meal_stage_fraction(self) -> None:
        low = cached_occurrence(1, record("Low fiber", fiber=Decimal("3")), meal="breakfast")
        high = cached_occurrence(2, record("High fiber", fiber=Decimal("8"), component_id=2), meal="breakfast")
        policy = MealOptimizationPolicy(
            minimum_servings=Decimal("1"), maximum_servings_per_food=Decimal("1"),
            serving_increment=Decimal("1"), max_distinct_foods=1,
            breakfast_stage_fraction=Decimal("0.25"), dinner_stage_fraction=Decimal("1"),
            weights=NutrientObjectiveWeights(
                calories_kcal=Decimal("0.0001"), protein_g=Decimal("0.0001"),
                carbohydrates_g=Decimal("0.0001"), fat_g=Decimal("0.0001"),
                dietary_fiber_g=Decimal("5"),
            ),
        )
        zero_macro_targets = DailyTargets(
            Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), DailyMinimums(Decimal("10")),
        )
        breakfast = optimize_meal(DAY, "breakfast", zero_macro_targets, DailyLedger(), (low, high), policy=policy)
        dinner_menu = (
            cached_occurrence(1, low.record, meal="dinner"),
            cached_occurrence(2, high.record, meal="dinner"),
        )
        dinner = optimize_meal(DAY, "dinner", zero_macro_targets, DailyLedger(), dinner_menu, policy=policy)
        self.assertEqual(breakfast.items[0].record.name, "Low fiber")
        self.assertEqual(dinner.items[0].record.name, "High fiber")

    def test_fiber_has_no_reward_beyond_minimum_and_respects_calorie_conflict(self) -> None:
        existing = record("Near calorie target", fiber=Decimal("0"))
        existing = replace(existing, nutrients=replace(existing.nutrients, calories_kcal=Decimal("950"), protein_g=Decimal("100"), carbohydrates_g=Decimal("100"), fat_g=Decimal("50")))
        sensible = cached_occurrence(1, record("Sensible fiber", fiber=Decimal("5")), meal="dinner")
        excessive_record = record("Excessive fiber", fiber=Decimal("5"), component_id=2)
        excessive_record = replace(excessive_record, nutrients=replace(excessive_record.nutrients, calories_kcal=Decimal("700"), fat_g=Decimal("40")))
        excessive = cached_occurrence(2, excessive_record, meal="dinner")
        policy = MealOptimizationPolicy(minimum_servings=Decimal("1"), maximum_servings_per_food=Decimal("1"), serving_increment=Decimal("1"), max_distinct_foods=1)
        conflict_targets = DailyTargets(
            Decimal("1000"), Decimal("120"), Decimal("120"), Decimal("55"),
            DailyMinimums(Decimal("5")),
        )
        result = optimize_meal(DAY, "dinner", conflict_targets, DailyLedger((IntakeEntry(existing, Decimal("1")),)), (sensible, excessive), policy=policy)
        self.assertEqual(result.items[0].record.name, "Sensible fiber")
        self.assertEqual(result.projected_balance.minimums.dietary_fiber_deficit_g, Decimal("0"))


if __name__ == "__main__":
    unittest.main()
