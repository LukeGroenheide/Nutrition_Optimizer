"""Offline tests for the bounded manual FD nutrition refresh workflow."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from nutrition_optimizer.fdmealplanner import (
    FDMealsPayload,
    OfficialNutritionCatalog,
)
from nutrition_optimizer.fdmealplanner.catalog import CatalogObservation
from nutrition_optimizer.fdmealplanner.refresh import format_refresh_report, refresh_phelps_catalog
from nutrition_optimizer.nutrition import SourceIdentifier


T1 = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
START = date(2026, 8, 24)
END = date(2026, 8, 28)


def component(
    component_id: int,
    name: str,
    *,
    protein: str = "10",
    component_type_id: int = 181,
    visible: bool = True,
) -> dict[str, object]:
    return {
        "componentId": component_id,
        "componentTypeId": component_type_id,
        "englishAlternateName": name,
        "componentName": name,
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


def payload_for(
    meal_period_id: int,
    start_date: date,
    end_date: date,
    *,
    changed: bool = False,
) -> FDMealsPayload:
    valid = component(
        70000 + meal_period_id,
        f"Meal {meal_period_id}",
        protein="11" if changed and meal_period_id == 1 else "10",
    )
    hidden = component(71000 + meal_period_id, f"Hidden {meal_period_id}", visible=False)
    product = component(
        72000 + meal_period_id,
        f"Product {meal_period_id}",
        component_type_id=180,
    )
    result: list[dict[str, object]] = []
    current = start_date
    while current <= end_date:
        result.append(
            {
                "menuForDate": current.isoformat(),
                "strMenuForDate": current.isoformat(),
                "mealPeriodId": meal_period_id,
                "conceptData": [
                    {"rowId": "row", "conceptId": 48, "conceptName": "AMERICAN GRILLE"}
                ],
                "allMenuRecipes": [
                    {**valid, "rowId": "row"},
                    {**hidden, "rowId": "row"},
                    {**product, "rowId": "row"},
                ],
            }
        )
        current = date.fromordinal(current.toordinal() + 1)
    return FDMealsPayload.from_payload({"result": result})


class FakeFDClient:
    def __init__(self, *, changed: bool = False) -> None:
        self.changed = changed
        self.calls: list[dict[str, object]] = []

    def fetch_phelps_month(self, **kwargs: object) -> FDMealsPayload:
        self.calls.append(kwargs)
        meal_period_id = int(kwargs["meal_period_id"])
        start_date = kwargs["start_date"]
        end_date = kwargs["end_date"]
        assert isinstance(start_date, date)
        assert isinstance(end_date, date)
        return payload_for(meal_period_id, start_date, end_date, changed=self.changed)


def catalog_path(directory: str) -> Path:
    return Path(directory) / "state" / "nutrition.sqlite3"


class RefreshWorkflowTests(unittest.TestCase):
    def test_dry_run_writes_nothing_and_processes_all_meals(self) -> None:
        client = FakeFDClient()
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = catalog_path(directory.name)

        report = refresh_phelps_catalog(
            START,
            END,
            catalog_path=path,
            dry_run=True,
            client=client,
            observed_at=T1,
        )

        self.assertFalse(path.exists())
        self.assertEqual([call["meal_period_id"] for call in client.calls], [1, 2, 3])
        self.assertEqual(report.fd_occurrence_count, 45)
        self.assertEqual(report.valid_mapped_occurrences, 15)
        self.assertEqual(report.rejected_occurrences, 30)
        self.assertEqual(report.unique_component_identities, 3)
        self.assertEqual(report.catalog_snapshots_inserted, 3)
        self.assertEqual(report.catalog_new_version_identities, ())
        self.assertEqual(report.final_catalog_snapshot_count, 3)
        self.assertTrue(report.validated_complete_snapshot)
        self.assertIn("mode: DRY RUN", format_refresh_report(report))

    def test_valid_records_are_stored_and_rejected_records_are_skipped(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = catalog_path(directory.name)
        report = refresh_phelps_catalog(
            START,
            END,
            catalog_path=path,
            client=FakeFDClient(),
            observed_at=T1,
        )

        self.assertEqual(report.catalog_snapshots_inserted, 3)
        self.assertEqual(report.catalog_new_versions, 0)
        self.assertEqual(report.menu_occurrences_observed, 15)
        self.assertEqual(report.current_logical_occurrences, 15)
        self.assertEqual(report.final_current_occurrence_count, 15)
        with OfficialNutritionCatalog(path) as catalog:
            self.assertEqual(catalog.snapshot_count, 3)
            for component_id in (70001, 70002, 70003):
                identifier = SourceIdentifier(kind="component", value=f"7:181:{component_id}")
                self.assertIsNotNone(catalog.get_current_record(identifier))

    def test_unchanged_refresh_is_idempotent(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = catalog_path(directory.name)
        first = refresh_phelps_catalog(START, END, catalog_path=path, client=FakeFDClient(), observed_at=T1)
        second = refresh_phelps_catalog(START, END, catalog_path=path, client=FakeFDClient(), observed_at=T2)

        self.assertEqual(first.catalog_snapshots_inserted, 3)
        self.assertEqual(second.catalog_snapshots_inserted, 0)
        self.assertEqual(second.catalog_snapshots_unchanged, 3)
        self.assertEqual(second.catalog_new_versions, 0)
        self.assertEqual(second.occurrence_additions, 0)
        self.assertEqual(second.occurrence_changes, 0)
        self.assertEqual(second.occurrence_removals, 0)
        with OfficialNutritionCatalog(path) as catalog:
            self.assertEqual(catalog.snapshot_count, 3)

    def test_changed_content_creates_a_new_version_and_preserves_history(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = catalog_path(directory.name)
        refresh_phelps_catalog(START, END, catalog_path=path, client=FakeFDClient(), observed_at=T1)
        changed = refresh_phelps_catalog(
            START,
            END,
            catalog_path=path,
            client=FakeFDClient(changed=True),
            observed_at=T2,
        )

        self.assertEqual(changed.catalog_snapshots_inserted, 0)
        self.assertEqual(changed.catalog_snapshots_unchanged, 2)
        self.assertEqual(changed.catalog_new_versions, 1)
        self.assertEqual(changed.catalog_new_version_identities, ("7:181:70001",))
        with OfficialNutritionCatalog(path) as catalog:
            self.assertEqual(catalog.snapshot_count, 4)
            identifier = SourceIdentifier(kind="component", value="7:181:70001")
            self.assertEqual(len(catalog.get_history(identifier)), 2)

    def test_refresh_failure_rolls_back_the_catalog_batch(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = catalog_path(directory.name)

        class FailingCatalog(OfficialNutritionCatalog):
            calls = 0

            def _upsert_observation(self, observation: CatalogObservation):  # type: ignore[no-untyped-def]
                self.calls += 1
                result = super()._upsert_observation(observation)
                if self.calls == 2:
                    raise RuntimeError("intentional refresh failure")
                return result

        with patch("nutrition_optimizer.fdmealplanner.refresh.OfficialNutritionCatalog", FailingCatalog):
            with self.assertRaises(RuntimeError):
                refresh_phelps_catalog(START, END, catalog_path=path, client=FakeFDClient(), observed_at=T1)

        with OfficialNutritionCatalog(path) as catalog:
            self.assertEqual(catalog.snapshot_count, 0)

    def test_refresh_uses_only_public_client_boundary(self) -> None:
        client = FakeFDClient()
        report = refresh_phelps_catalog(START, END, catalog_path=":memory:", dry_run=True, client=client)
        self.assertEqual(report.meals, ("breakfast", "lunch", "dinner"))
        self.assertTrue(all("token" not in repr(call).casefold() for call in client.calls))


if __name__ == "__main__":
    unittest.main()
