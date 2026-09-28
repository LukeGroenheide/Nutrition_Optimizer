"""Focused offline tests for the durable FD menu occurrence cache."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal
import inspect
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from nutrition_optimizer.fdmealplanner import (
    CATALOG_SCHEMA_VERSION,
    OfficialNutritionCatalog,
    map_meals_payload,
    serialize_nutrition_record,
)
from nutrition_optimizer.nutrition import (
    NutritionProvenance,
    NutritionRecord,
    NutrientProfile,
    Serving,
    SourceIdentifier,
)


T1 = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
DAY1 = date(2026, 8, 25)
DAY2 = date(2026, 8, 26)


def component(
    component_id: int,
    name: str,
    *,
    protein: str = "10",
    visible: bool = True,
    component_type_id: int = 181,
    menu_detail_id: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "componentId": component_id,
        "componentTypeId": component_type_id,
        "englishAlternateName": name,
        "componentName": f"{name}-master",
        "recipePortionSize": "1",
        "recipePortionSizeUnit": "Serving",
        "calories": "100",
        "caloriesUOM": "kcal",
        "protein": protein,
        "proteinUOM": "g",
        "carbohydrates": "0",
        "carbohydratesUOM": "g",
        "fat": "2",
        "fatUOM": "g",
        "sodium": "50",
        "sodiumUOM": "mg",
        "isShowOnMenu": "1" if visible else "0",
        "isFoodBar": "0",
    }
    if menu_detail_id is not None:
        value["MenuDetailId"] = menu_detail_id
    return value


def menu_day(
    service_date: date,
    meal_id: int,
    meal_name: str,
    recipes: tuple[tuple[dict[str, object], str], ...],
    *,
    menu_id: int = 0,
) -> dict[str, object]:
    concepts: list[dict[str, object]] = []
    all_recipes: list[dict[str, object]] = []
    for index, (recipe, station_name) in enumerate(recipes):
        row_id = f"row-{index}"
        concepts.append(
            {"rowId": row_id, "conceptId": 40 + index, "conceptName": station_name}
        )
        all_recipes.append({**recipe, "rowId": row_id})
    return {
        "menuForDate": service_date.isoformat(),
        "strMenuForDate": service_date.isoformat(),
        "menuId": menu_id,
        "mealPeriodId": meal_id,
        "mealPeriodName": meal_name,
        "conceptData": concepts,
        "allMenuRecipes": all_recipes,
    }


def mapped(*days: dict[str, object]):
    return map_meals_payload(
        {"result": list(days)},
        tenant_id=7,
        retrieved_at=T1,
    ).results


def old_record() -> NutritionRecord:
    return NutritionRecord(
        name="Migrated food",
        serving=Serving(quantity=Decimal("1"), unit="Serving", text="1 Serving"),
        nutrients=NutrientProfile(calories_kcal=Decimal("100")),
        provenance=NutritionProvenance(
            provider="FDMealPlanner",
            retrieved_at=T1,
            identifiers=(SourceIdentifier(kind="component", value="7:181:999"),),
        ),
    )


class FDOccurrenceCacheTests(unittest.TestCase):
    def path(self, directory: TemporaryDirectory[str]) -> Path:
        return Path(directory.name) / "state" / "nutrition.sqlite3"

    def sync(
        self,
        catalog: OfficialNutritionCatalog,
        results,
        *,
        start: date = DAY1,
        end: date = DAY2,
        observed_at: datetime = T1,
    ):
        return catalog.synchronize_fd_refresh(
            results,
            requested_start=start,
            requested_end=end,
            observed_at=observed_at,
        )

    def test_migration_preserves_existing_nutrition_snapshots(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = self.path(directory)
        path.parent.mkdir(parents=True)
        record = old_record()
        connection = sqlite3.connect(path)
        connection.execute(
            """
            CREATE TABLE nutrition_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                source_value TEXT NOT NULL,
                content_signature TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                first_observed_at TEXT NOT NULL,
                last_observed_at TEXT NOT NULL,
                UNIQUE(provider, source_kind, source_value, content_signature)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX nutrition_snapshots_current_idx
            ON nutrition_snapshots(provider, source_kind, source_value, last_observed_at DESC, snapshot_id DESC)
            """
        )
        connection.execute(
            """
            INSERT INTO nutrition_snapshots
            (provider, source_kind, source_value, content_signature, snapshot_json,
             first_observed_at, last_observed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "FDMealPlanner",
                "component",
                "7:181:999",
                "legacy-signature",
                serialize_nutrition_record(record),
                T1.isoformat(),
                T1.isoformat(),
            ),
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
        connection.close()

        with OfficialNutritionCatalog(path) as catalog:
            self.assertEqual(catalog.snapshot_count, 1)
            self.assertEqual(catalog.get_history(SourceIdentifier("component", "7:181:999"))[0].record, record)
            self.assertEqual(
                catalog._connection.execute("PRAGMA user_version").fetchone()[0],
                CATALOG_SCHEMA_VERSION,
            )
        connection = sqlite3.connect(path)
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        finally:
            connection.close()
        self.assertTrue({"fd_refresh_runs", "fd_menu_occurrences", "fd_refresh_occurrences"} <= tables)

    def test_occurrence_insert_succeeds_and_links_exact_snapshot(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        with OfficialNutritionCatalog(self.path(directory)) as catalog:
            result = self.sync(
                catalog,
                mapped(
                    menu_day(
                        DAY1,
                        3,
                        "dinner",
                        ((component(1, "Chicken"), "Grill"),),
                    )
                ),
                start=DAY1,
                end=DAY1,
            )
            occurrences = catalog.list_current_meal_occurrences(DAY1)
            self.assertEqual(result.current_logical_occurrences, 1)
            self.assertEqual(len(occurrences), 1)
            self.assertEqual(occurrences[0].source_identifier.value, "7:181:1")
            self.assertEqual(occurrences[0].nutrition_snapshot_id, 1)
            exact_snapshot = catalog.get_snapshot(
                occurrences[0].source_identifier,
                content_signature=occurrences[0].content_signature,
            )
            self.assertIsNotNone(exact_snapshot)
            assert exact_snapshot is not None
            self.assertEqual(exact_snapshot.snapshot_id, occurrences[0].nutrition_snapshot_id)
            self.assertEqual(occurrences[0].nutrition_record.name, "Chicken")

    def test_repeated_identical_refresh_is_idempotent(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        with OfficialNutritionCatalog(self.path(directory)) as catalog:
            results = mapped(
                menu_day(DAY1, 3, "dinner", ((component(1, "Chicken"), "Grill"),))
            )
            first = self.sync(catalog, results, start=DAY1, end=DAY1, observed_at=T1)
            second = self.sync(catalog, results, start=DAY1, end=DAY1, observed_at=T2)
            self.assertEqual(first.occurrence_additions, 1)
            self.assertEqual((second.occurrence_additions, second.occurrence_changes, second.occurrence_removals), (0, 0, 0))
            self.assertEqual(catalog.snapshot_count, 1)
            self.assertEqual(catalog._connection.execute("SELECT COUNT(*) FROM fd_menu_occurrences").fetchone()[0], 1)
            self.assertEqual(
                [item.occurrence_key for item in catalog.list_current_meal_occurrences(DAY1)],
                [item.occurrence_key for item in catalog.list_current_meal_occurrences(DAY1)],
            )

    def test_same_recipe_can_occur_on_multiple_dates(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        results = mapped(
            menu_day(DAY1, 3, "dinner", ((component(1, "Chicken"), "Grill"),)),
            menu_day(DAY2, 3, "dinner", ((component(1, "Chicken"), "Grill"),)),
        )
        with OfficialNutritionCatalog(self.path(directory)) as catalog:
            self.sync(catalog, results)
            self.assertEqual(len(catalog.list_current_meal_occurrences(DAY1)), 1)
            self.assertEqual(len(catalog.list_current_meal_occurrences(DAY2)), 1)
            self.assertEqual(
                catalog.list_current_meal_occurrences(DAY1)[0].source_identifier,
                catalog.list_current_meal_occurrences(DAY2)[0].source_identifier,
            )

    def test_same_recipe_can_occur_in_multiple_meals_and_stations(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        results = mapped(
            menu_day(DAY1, 1, "breakfast", ((component(1, "Chicken"), "Grill"),)),
            menu_day(DAY1, 3, "dinner", ((component(1, "Chicken"), "Wok"),)),
        )
        with OfficialNutritionCatalog(self.path(directory)) as catalog:
            self.sync(catalog, results, start=DAY1, end=DAY1)
            self.assertEqual(len(catalog.list_current_meal_occurrences(DAY1)), 2)
            self.assertEqual(len(catalog.list_current_meal_occurrences(DAY1, meal="dinner")), 1)
            dinner_wok = catalog.list_current_meal_occurrences(DAY1, "dinner", "Wok")
            self.assertEqual(len(dinner_wok), 1)
            self.assertEqual(dinner_wok[0].station, "Wok")

    def test_changed_nutrition_version_is_linked_to_later_occurrence(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        first_results = mapped(
            menu_day(DAY1, 3, "dinner", ((component(1, "Chicken", protein="10"), "Grill"),))
        )
        second_results = mapped(
            menu_day(DAY2, 3, "dinner", ((component(1, "Chicken", protein="11"), "Grill"),))
        )
        with OfficialNutritionCatalog(self.path(directory)) as catalog:
            self.sync(catalog, first_results, start=DAY1, end=DAY1, observed_at=T1)
            self.sync(catalog, second_results, start=DAY2, end=DAY2, observed_at=T2)
            first = catalog.list_current_meal_occurrences(DAY1)[0]
            second = catalog.list_current_meal_occurrences(DAY2)[0]
            self.assertNotEqual(first.content_signature, second.content_signature)
            self.assertNotEqual(first.nutrition_snapshot_id, second.nutrition_snapshot_id)
            self.assertEqual(first.nutrition_record.nutrients.protein_g, Decimal("10"))
            self.assertEqual(second.nutrition_record.nutrients.protein_g, Decimal("11"))
            self.assertEqual(len(catalog.get_history(first.source_identifier)), 2)

    def test_removed_item_is_absent_from_current_query_after_completed_refresh(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        initial = mapped(
            menu_day(
                DAY1,
                3,
                "dinner",
                ((component(1, "Chicken"), "Grill"), (component(2, "Rice"), "Grill")),
            )
        )
        later = mapped(
            menu_day(DAY1, 3, "dinner", ((component(1, "Chicken"), "Grill"),))
        )
        with OfficialNutritionCatalog(self.path(directory)) as catalog:
            self.sync(catalog, initial, start=DAY1, end=DAY1, observed_at=T1)
            changed = self.sync(catalog, later, start=DAY1, end=DAY1, observed_at=T2)
            self.assertEqual(changed.occurrence_removals, 1)
            self.assertEqual([item.nutrition_record.name for item in catalog.list_current_meal_occurrences(DAY1)], ["Chicken"])
            self.assertEqual(catalog.snapshot_count, 2)
            self.assertEqual(catalog._connection.execute("SELECT COUNT(*) FROM fd_menu_occurrences").fetchone()[0], 2)

    def test_failed_refresh_does_not_replace_current_authoritative_state(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        initial = mapped(menu_day(DAY1, 3, "dinner", ((component(1, "Chicken"), "Grill"),)))
        later = mapped(menu_day(DAY1, 3, "dinner", ((component(2, "Rice"), "Grill"),)))
        with OfficialNutritionCatalog(self.path(directory)) as catalog:
            self.sync(catalog, initial, start=DAY1, end=DAY1, observed_at=T1)
            with patch.object(catalog, "_upsert_occurrence", side_effect=RuntimeError("fail")):
                with self.assertRaises(RuntimeError):
                    self.sync(catalog, later, start=DAY1, end=DAY1, observed_at=T2)
            self.assertEqual([item.nutrition_record.name for item in catalog.list_current_meal_occurrences(DAY1)], ["Chicken"])
            self.assertEqual(catalog.snapshot_count, 1)
            self.assertEqual(catalog._connection.execute("SELECT COUNT(*) FROM fd_refresh_runs").fetchone()[0], 1)

    def test_hidden_and_product_components_are_not_persisted_current(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        results = mapped(
            menu_day(
                DAY1,
                3,
                "dinner",
                (
                    (component(1, "Hidden", visible=False), "Grill"),
                    (component(2, "Product", component_type_id=180), "Grill"),
                ),
            )
        )
        with OfficialNutritionCatalog(self.path(directory)) as catalog:
            self.sync(catalog, results, start=DAY1, end=DAY1)
            self.assertEqual(catalog.snapshot_count, 0)
            self.assertEqual(catalog.list_current_meal_occurrences(DAY1), ())

    def test_query_by_date_meal_and_station_is_local_and_deterministic(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        results = mapped(
            menu_day(DAY1, 1, "breakfast", ((component(1, "Eggs"), "Grill"),)),
            menu_day(
                DAY1,
                3,
                "dinner",
                ((component(2, "Rice"), "Grill"), (component(3, "Tofu"), "Wok")),
            ),
        )
        with OfficialNutritionCatalog(self.path(directory)) as catalog:
            self.sync(catalog, results, start=DAY1, end=DAY1)
            self.assertEqual(len(catalog.list_current_meal_occurrences(DAY1)), 3)
            self.assertEqual(
                [item.nutrition_record.name for item in catalog.list_current_meal_occurrences(DAY1, meal="dinner")],
                ["Rice", "Tofu"],
            )
            self.assertEqual(
                [item.nutrition_record.name for item in catalog.list_current_meal_occurrences(DAY1, "dinner", "Wok")],
                ["Tofu"],
            )
            self.assertEqual(catalog.list_current_meal_occurrences(DAY2), ())

    def test_reopened_catalog_query_requires_no_fd_client(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = self.path(directory)
        with OfficialNutritionCatalog(path) as catalog:
            self.sync(
                catalog,
                mapped(menu_day(DAY1, 3, "dinner", ((component(1, "Chicken"), "Grill"),))),
                start=DAY1,
                end=DAY1,
            )
        with OfficialNutritionCatalog(path) as reopened:
            result = reopened.list_current_meal_occurrences(DAY1)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].nutrition_record.name, "Chicken")

    def test_mapper_and_occurrence_cache_do_not_mutate_source_objects(self) -> None:
        source = component(1, "Chicken")
        before = deepcopy(source)
        result = mapped(menu_day(DAY1, 3, "dinner", ((source, "Grill"),)))[0]
        self.assertEqual(source, before)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        with OfficialNutritionCatalog(self.path(directory)) as catalog:
            self.sync(catalog, (result,), start=DAY1, end=DAY1)
        self.assertEqual(source, before)

    def test_production_database_path_is_ignored(self) -> None:
        ignore_text = Path(".gitignore").read_text(encoding="utf-8")
        self.assertIn("data/state/", ignore_text)
        self.assertIn("*.sqlite3", ignore_text)

    def test_occurrence_query_path_has_no_secondary_or_ai_dependency(self) -> None:
        from nutrition_optimizer.fdmealplanner import catalog as catalog_module

        source = inspect.getsource(catalog_module)
        self.assertNotIn("DiningBucket", source)
        self.assertNotIn("OpenAI", source)
        self.assertNotIn("OpenClaw", source)

    def test_historical_nutrition_snapshots_remain_intact(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        with OfficialNutritionCatalog(self.path(directory)) as catalog:
            first = mapped(menu_day(DAY1, 3, "dinner", ((component(1, "Chicken", protein="10"), "Grill"),)))
            second = mapped(menu_day(DAY1, 3, "dinner", ((component(1, "Chicken", protein="11"), "Grill"),)))
            self.sync(catalog, first, start=DAY1, end=DAY1, observed_at=T1)
            self.sync(catalog, second, start=DAY1, end=DAY1, observed_at=T2)
            history = catalog.get_history(SourceIdentifier("component", "7:181:1"))
            self.assertEqual(len(history), 2)
            self.assertEqual(
                {snapshot.record.nutrients.protein_g for snapshot in history},
                {Decimal("10"), Decimal("11")},
            )


if __name__ == "__main__":
    unittest.main()
