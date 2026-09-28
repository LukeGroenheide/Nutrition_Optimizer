from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from decimal import Decimal
import unittest

from nutrition_optimizer.nutrition.models import (
    NutritionProvenance,
    NutritionRecord,
    NutrientProfile,
    Serving,
    SourceIdentifier,
)


def observed_record() -> NutritionRecord:
    """Build a contract record from the observed historical Phelps values."""

    return NutritionRecord(
        name="Herb Roasted Chicken",
        serving=Serving(
            quantity=Decimal("4"),
            unit="ounce cooked",
            text="4 Ounce Cooked",
        ),
        nutrients=NutrientProfile(
            calories_kcal=Decimal("175"),
            protein_g=Decimal("27"),
            carbohydrates_g=Decimal("0"),
            fat_g=Decimal("4"),
            sodium_mg=Decimal("303"),
        ),
        provenance=NutritionProvenance(
            provider="FD MealPlanner",
            retrieved_at=datetime(2026, 7, 1, 12, tzinfo=timezone.utc),
        ),
    )


class NutritionRecordTests(unittest.TestCase):
    def test_complete_observed_record(self) -> None:
        record = observed_record()

        self.assertEqual(record.name, "Herb Roasted Chicken")
        self.assertEqual(record.serving.quantity, Decimal("4"))
        self.assertEqual(record.serving.unit, "ounce cooked")
        self.assertEqual(record.serving.text, "4 Ounce Cooked")
        self.assertEqual(record.nutrients.calories_kcal, Decimal("175"))
        self.assertEqual(record.nutrients.protein_g, Decimal("27"))
        self.assertEqual(record.nutrients.carbohydrates_g, Decimal("0"))
        self.assertEqual(record.nutrients.fat_g, Decimal("4"))
        self.assertEqual(record.nutrients.sodium_mg, Decimal("303"))
        self.assertEqual(record.provenance.provider, "FD MealPlanner")
        self.assertEqual(record.provenance.retrieved_at.tzinfo, timezone.utc)

    def test_missing_nutrients_remain_none(self) -> None:
        nutrients = NutrientProfile(
            calories_kcal=Decimal("100"),
            protein_g=None,
            carbohydrates_g=None,
            fat_g=None,
            sodium_mg=None,
        )

        self.assertIsNone(nutrients.protein_g)
        self.assertIsNone(nutrients.carbohydrates_g)
        self.assertIsNone(nutrients.fat_g)
        self.assertIsNone(nutrients.sodium_mg)

    def test_decimal_nutrient_values_are_preserved(self) -> None:
        nutrients = NutrientProfile(
            calories_kcal=Decimal("175.25"),
            protein_g=Decimal("27.125"),
            carbohydrates_g=Decimal("0.50"),
            fat_g=Decimal("4.75"),
            sodium_mg=Decimal("303.5"),
        )

        self.assertEqual(nutrients.calories_kcal, Decimal("175.25"))
        self.assertEqual(nutrients.protein_g, Decimal("27.125"))
        self.assertEqual(nutrients.carbohydrates_g, Decimal("0.50"))
        self.assertEqual(nutrients.fat_g, Decimal("4.75"))
        self.assertEqual(nutrients.sodium_mg, Decimal("303.5"))

    def test_source_identifiers_and_provenance_are_preserved(self) -> None:
        provenance = NutritionProvenance(
            provider="test official source",
            record_type="component",
            source_reference="fixture-record-1",
            retrieved_at=datetime(2026, 7, 1, 8, tzinfo=timezone.utc),
            identifiers=(
                SourceIdentifier(kind="recipe_id", value="recipe-1"),
                SourceIdentifier(kind="component_id", value="component-1"),
            ),
        )

        self.assertEqual(provenance.record_type, "component")
        self.assertEqual(provenance.source_reference, "fixture-record-1")
        self.assertEqual(
            [(item.kind, item.value) for item in provenance.identifiers],
            [("recipe_id", "recipe-1"), ("component_id", "component-1")],
        )

    def test_retrieval_timestamp_must_be_timezone_aware(self) -> None:
        with self.assertRaises(ValueError):
            NutritionProvenance(
                provider="test source",
                retrieved_at=datetime(2026, 7, 1, 8),
            )

        provenance = NutritionProvenance(
            provider="test source",
            retrieved_at=datetime(2026, 7, 1, 8, tzinfo=timezone.utc),
        )
        self.assertIsNotNone(provenance.retrieved_at.utcoffset())

    def test_negative_nutrients_are_rejected(self) -> None:
        for field_name in (
            "calories_kcal",
            "protein_g",
            "carbohydrates_g",
            "fat_g",
            "sodium_mg",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    NutrientProfile(**{field_name: Decimal("-0.01")})

    def test_invalid_serving_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Serving(quantity=Decimal("0"), unit="gram", text="0 gram")
        with self.assertRaises(ValueError):
            Serving(quantity=Decimal("-1"), unit="gram", text="-1 gram")
        with self.assertRaises(ValueError):
            Serving()

    def test_non_decimal_nutrient_values_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            NutrientProfile(calories_kcal=175)  # type: ignore[arg-type]

    def test_records_are_immutable(self) -> None:
        record = observed_record()

        with self.assertRaises(FrozenInstanceError):
            record.name = "Changed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
