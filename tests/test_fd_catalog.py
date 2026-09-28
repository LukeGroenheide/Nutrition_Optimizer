"""Offline tests for the persistent official FD nutrition catalog."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from nutrition_optimizer.fdmealplanner import (
    OfficialNutritionCatalog,
    deserialize_nutrition_record,
    map_component,
    map_meals_payload,
    refresh_from_fd_results,
    serialize_nutrition_record,
)
from nutrition_optimizer.fdmealplanner.catalog import (
    CatalogObservation,
    NutritionCatalogSerializationError,
)
from nutrition_optimizer.nutrition import (
    NutritionProvenance,
    NutritionRecord,
    NutrientProfile,
    Serving,
    SourceIdentifier,
)


T0 = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)


def make_record(
    *,
    component_id: int = 69534,
    protein: Decimal | None = Decimal("27"),
    carbohydrates: Decimal | None = Decimal("0"),
) -> NutritionRecord:
    return NutritionRecord(
        name="Herb Roasted Chicken" if component_id == 69534 else f"Food {component_id}",
        serving=Serving(
            quantity=Decimal("4.0"),
            unit="Ounce Cooked Weight",
            text="4 Ounce Cooked Weight",
        ),
        nutrients=NutrientProfile(
            calories_kcal=Decimal("175"),
            protein_g=protein,
            carbohydrates_g=carbohydrates,
            fat_g=Decimal("4"),
            sodium_mg=Decimal("303"),
        ),
        provenance=NutritionProvenance(
            provider="FDMealPlanner",
            retrieved_at=T0,
            record_type="recipe",
            source_reference="https://apiservicelocatorstenantcds.fdmealplanner.com",
            identifiers=(
                SourceIdentifier(kind="component", value=f"7:181:{component_id}"),
            ),
        ),
        ingredients=("Chicken, oil, salt",),
        allergens=("MILK",),
    )


def valid_component(**overrides: object) -> dict[str, object]:
    component: dict[str, object] = {
        "componentId": 69534,
        "componentTypeId": 181,
        "englishAlternateName": "Herb Roasted Chicken",
        "componentName": "Herb Roasted Chicken-Master",
        "recipePortionSize": 4.0,
        "recipePortionSizeUnit": "Ounce Cooked Weight",
        "calories": "175",
        "caloriesUOM": "kcal",
        "protein": "27",
        "proteinUOM": "g",
        "carbohydrates": "0",
        "carbohydratesUOM": "g",
        "fat": "4",
        "fatUOM": "g",
        "sodium": "303",
        "sodiumUOM": "mg",
        "ingredientStatement": "Chicken, oil, salt",
        "allergenName": "MILK",
    }
    component.update(overrides)
    return component


def catalog_path(directory: str) -> Path:
    return Path(directory) / "state" / "nutrition.sqlite3"


class CatalogPersistenceTests(unittest.TestCase):
    def open_catalog(self) -> tuple[TemporaryDirectory[str], OfficialNutritionCatalog]:
        directory = TemporaryDirectory()
        return directory, OfficialNutritionCatalog(catalog_path(directory.name))

    def test_record_round_trip_preserves_domain_values_and_metadata(self) -> None:
        directory, catalog = self.open_catalog()
        self.addCleanup(directory.cleanup)
        self.addCleanup(catalog.close)
        record = make_record()
        written = catalog.store(record, content_signature="sig-a", observed_at=T0)
        loaded = catalog.get_current_record(record.provenance.identifiers[0])
        self.assertEqual(written.outcome, "inserted")
        self.assertEqual(loaded, record)
        self.assertEqual(catalog.get_current_snapshot(record.provenance.identifiers[0]).content_signature, "sig-a")
        self.assertEqual(serialize_nutrition_record(record), serialize_nutrition_record(loaded))
        self.assertEqual(deserialize_nutrition_record(serialize_nutrition_record(record)), record)
        catalog.close()
        reopened = OfficialNutritionCatalog(catalog.path)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.get_current_record(record.provenance.identifiers[0]), record)

    def test_decimal_none_and_zero_round_trip_distinctly(self) -> None:
        directory, catalog = self.open_catalog()
        self.addCleanup(directory.cleanup)
        self.addCleanup(catalog.close)
        record = make_record(protein=None, carbohydrates=Decimal("0"))
        loaded = catalog.store(record, content_signature="sig-none-zero", observed_at=T0).snapshot.record
        self.assertIsNone(loaded.nutrients.protein_g)
        self.assertEqual(loaded.nutrients.carbohydrates_g, Decimal("0"))
        self.assertIsInstance(loaded.nutrients.carbohydrates_g, Decimal)

    def test_same_identity_and_signature_is_idempotent_with_observation_metadata(self) -> None:
        directory, catalog = self.open_catalog()
        self.addCleanup(directory.cleanup)
        self.addCleanup(catalog.close)
        record = make_record()
        first = catalog.store(record, content_signature="sig-a", observed_at=T1)
        second = catalog.store(record, content_signature="sig-a", observed_at=T2)
        self.assertEqual(first.outcome, "inserted")
        self.assertEqual(second.outcome, "unchanged")
        self.assertEqual(catalog.snapshot_count, 1)
        snapshot = catalog.get_current_snapshot(record.provenance.identifiers[0])
        self.assertEqual(snapshot.first_observed_at, T1)
        self.assertEqual(snapshot.last_observed_at, T2)

    def test_changed_signature_preserves_history_and_latest_is_observed_newest(self) -> None:
        directory, catalog = self.open_catalog()
        self.addCleanup(directory.cleanup)
        self.addCleanup(catalog.close)
        record = make_record()
        changed = replace(
            record,
            nutrients=replace(record.nutrients, protein_g=Decimal("28")),
        )
        catalog.store(record, content_signature="sig-old", observed_at=T1)
        version = catalog.store(changed, content_signature="sig-new", observed_at=T2)
        self.assertEqual(version.outcome, "new_version")
        history = catalog.get_history(record.provenance.identifiers[0])
        self.assertEqual([item.content_signature for item in history], ["sig-new", "sig-old"])
        self.assertEqual(catalog.get_current_record(record.provenance.identifiers[0]), changed)

        # Insertion order does not override observation time.
        older_observation = replace(changed, nutrients=replace(changed.nutrients, protein_g=Decimal("29")))
        catalog.store(older_observation, content_signature="sig-older", observed_at=T0)
        self.assertEqual(
            catalog.get_current_snapshot(record.provenance.identifiers[0]).content_signature,
            "sig-new",
        )

    def test_different_component_ids_are_independent(self) -> None:
        directory, catalog = self.open_catalog()
        self.addCleanup(directory.cleanup)
        self.addCleanup(catalog.close)
        first = make_record(component_id=69534)
        second = make_record(component_id=69535)
        catalog.store(first, content_signature="sig-a", observed_at=T0)
        catalog.store(second, content_signature="sig-b", observed_at=T0)
        self.assertEqual(catalog.snapshot_count, 2)
        self.assertEqual(catalog.get_current_record(first.provenance.identifiers[0]), first)
        self.assertEqual(catalog.get_current_record(second.provenance.identifiers[0]), second)

    def test_corrupt_snapshot_fails_without_guessing(self) -> None:
        directory, catalog = self.open_catalog()
        self.addCleanup(directory.cleanup)
        self.addCleanup(catalog.close)
        record = make_record()
        catalog.store(record, content_signature="sig-a", observed_at=T0)
        connection = sqlite3.connect(catalog.path)
        try:
            connection.execute(
                "UPDATE nutrition_snapshots SET snapshot_json = ?",
                ("not-json",),
            )
            connection.commit()
        finally:
            connection.close()
        with self.assertRaises(NutritionCatalogSerializationError):
            catalog.get_current_record(record.provenance.identifiers[0])

    def test_batch_failure_rolls_back_all_prior_writes(self) -> None:
        class FailingCatalog(OfficialNutritionCatalog):
            calls = 0

            def _upsert_observation(self, observation: CatalogObservation):  # type: ignore[no-untyped-def]
                self.calls += 1
                result = super()._upsert_observation(observation)
                if self.calls == 2:
                    raise RuntimeError("intentional test failure")
                return result

        directory = TemporaryDirectory()
        catalog = FailingCatalog(catalog_path(directory.name))
        self.addCleanup(directory.cleanup)
        self.addCleanup(catalog.close)
        first = make_record(component_id=69534)
        second = make_record(component_id=69535)
        with self.assertRaises(RuntimeError):
            catalog.store_many(
                (
                    CatalogObservation(first, "sig-a", T0),
                    CatalogObservation(second, "sig-b", T0),
                )
            )
        self.assertEqual(catalog.snapshot_count, 0)


class FDRefreshTests(unittest.TestCase):
    def test_occurrence_context_does_not_create_catalog_versions(self) -> None:
        directory = TemporaryDirectory()
        catalog = OfficialNutritionCatalog(catalog_path(directory.name))
        self.addCleanup(directory.cleanup)
        self.addCleanup(catalog.close)
        component = valid_component()
        payload = {
            "result": [
                {
                    "menuForDate": "2026-08-20",
                    "mealPeriodId": 2,
                    "conceptData": [{"rowId": 1, "conceptId": 10, "conceptName": "Grill"}],
                    "allMenuRecipes": [{**component, "rowId": 1}],
                },
                {
                    "menuForDate": "2026-08-21",
                    "mealPeriodId": 3,
                    "conceptData": [{"rowId": 2, "conceptId": 20, "conceptName": "Wok"}],
                    "allMenuRecipes": [{**component, "rowId": 2}],
                },
            ]
        }
        batch = map_meals_payload(payload, tenant_id=7)
        refreshed = refresh_from_fd_results(catalog, batch, observed_at=T1)
        self.assertEqual((refreshed.inserted, refreshed.unchanged, refreshed.new_versions), (1, 1, 0))
        self.assertEqual(catalog.snapshot_count, 1)

    def test_rejected_fd_results_are_not_catalog_records(self) -> None:
        directory = TemporaryDirectory()
        catalog = OfficialNutritionCatalog(catalog_path(directory.name))
        self.addCleanup(directory.cleanup)
        self.addCleanup(catalog.close)
        valid = map_component(valid_component(), tenant_id=7, retrieved_at=T0)
        rejected = map_component(
            valid_component(componentTypeId=180, recipePortionSize=0, recipePortionSizeUnit=""),
            tenant_id=7,
            retrieved_at=T0,
        )
        refreshed = refresh_from_fd_results(catalog, (valid, rejected), observed_at=T1)
        self.assertEqual(refreshed.inserted, 1)
        self.assertEqual(refreshed.rejected, 1)
        self.assertEqual(catalog.snapshot_count, 1)


if __name__ == "__main__":
    unittest.main()
