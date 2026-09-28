from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from decimal import Decimal
import unittest

from nutrition_optimizer.nutrition.balance import DailyTargets
from nutrition_optimizer.nutrition.ledger import DailyLedger, IntakeEntry
from nutrition_optimizer.nutrition.models import (
    NutritionProvenance,
    NutritionRecord,
    NutrientProfile,
    Serving,
)


def record(
    name: str,
    *,
    calories: Decimal | None = Decimal("175"),
    protein: Decimal | None = Decimal("27"),
    carbohydrates: Decimal | None = Decimal("0"),
    fat: Decimal | None = Decimal("4"),
    sodium: Decimal | None = Decimal("303"),
) -> NutritionRecord:
    return NutritionRecord(
        name=name,
        serving=Serving(quantity=Decimal("1"), unit="serving"),
        nutrients=NutrientProfile(
            calories_kcal=calories,
            protein_g=protein,
            carbohydrates_g=carbohydrates,
            fat_g=fat,
            sodium_mg=sodium,
        ),
        provenance=NutritionProvenance(
            provider="test official source",
            retrieved_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
        ),
    )


def targets() -> DailyTargets:
    return DailyTargets(
        calories_kcal=Decimal("2000"),
        protein_g=Decimal("150"),
        carbohydrates_g=Decimal("250"),
        fat_g=Decimal("65"),
    )


class IntakeEntryTests(unittest.TestCase):
    def test_contribution_is_derived_from_record_and_servings(self) -> None:
        entry = IntakeEntry(record("Chicken"), Decimal("1.5"))

        self.assertEqual(entry.consumed_nutrients.calories_kcal, Decimal("262.5"))
        self.assertEqual(entry.consumed_nutrients.protein_g, Decimal("40.5"))
        self.assertEqual(entry.consumed_nutrients.carbohydrates_g, Decimal("0.0"))
        self.assertEqual(entry.consumed_nutrients.fat_g, Decimal("6.0"))
        self.assertEqual(entry.consumed_nutrients.sodium_mg, Decimal("454.5"))

    def test_invalid_serving_multipliers_are_rejected(self) -> None:
        for value in (
            Decimal("0"),
            Decimal("-0.5"),
            Decimal("NaN"),
            Decimal("Infinity"),
            Decimal("-Infinity"),
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    IntakeEntry(record("Food"), value)

        with self.assertRaises(TypeError):
            IntakeEntry(record("Food"), 1)  # type: ignore[arg-type]

    def test_entry_is_immutable(self) -> None:
        entry = IntakeEntry(record("Food"), Decimal("1"))

        with self.assertRaises(FrozenInstanceError):
            entry.servings = Decimal("2")  # type: ignore[misc]


class DailyLedgerTests(unittest.TestCase):
    def test_empty_ledger_has_zero_total(self) -> None:
        total = DailyLedger().total_consumed_nutrients
        zero = Decimal("0")

        self.assertEqual(total, NutrientProfile(
            calories_kcal=zero,
            protein_g=zero,
            carbohydrates_g=zero,
            fat_g=zero,
            sodium_mg=zero,
            dietary_fiber_g=zero,
        ))

    def test_one_consumed_food_is_totaled(self) -> None:
        entry = IntakeEntry(record("Chicken"), Decimal("1"))
        ledger = DailyLedger().add_entry(entry)

        self.assertEqual(ledger.entries, (entry,))
        self.assertEqual(ledger.total_consumed_nutrients, entry.consumed_nutrients)

    def test_multiple_foods_are_added(self) -> None:
        first = IntakeEntry(record("Chicken"), Decimal("1"))
        second = IntakeEntry(
            record(
                "Rice",
                calories=Decimal("200"),
                protein=Decimal("4"),
                carbohydrates=Decimal("45"),
                fat=Decimal("1"),
                sodium=Decimal("10"),
            ),
            Decimal("2"),
        )

        total = DailyLedger().add_entry(first).add_entry(second).total_consumed_nutrients

        self.assertEqual(total.calories_kcal, Decimal("575"))
        self.assertEqual(total.protein_g, Decimal("35"))
        self.assertEqual(total.carbohydrates_g, Decimal("90"))
        self.assertEqual(total.fat_g, Decimal("6"))
        self.assertEqual(total.sodium_mg, Decimal("323"))

    def test_fractional_servings_are_scaled(self) -> None:
        ledger = DailyLedger().add_entry(
            IntakeEntry(record("Chicken"), Decimal("0.5"))
        )

        self.assertEqual(ledger.total_consumed_nutrients.calories_kcal, Decimal("87.5"))
        self.assertEqual(ledger.total_consumed_nutrients.protein_g, Decimal("13.5"))

    def test_unknown_nutrients_propagate(self) -> None:
        ledger = DailyLedger().add_entry(
            IntakeEntry(record("Unknown protein", protein=None), Decimal("1"))
        )

        total = ledger.total_consumed_nutrients

        self.assertEqual(total.calories_kcal, Decimal("175"))
        self.assertIsNone(total.protein_g)
        self.assertEqual(total.carbohydrates_g, Decimal("0"))

    def test_balance_uses_ledger_total(self) -> None:
        ledger = DailyLedger().add_entry(
            IntakeEntry(record("Chicken"), Decimal("1"))
        )

        balance = ledger.balance_against(targets())

        self.assertEqual(balance.consumed, ledger.total_consumed_nutrients)
        self.assertEqual(balance.remaining.calories_kcal, Decimal("1825"))
        self.assertEqual(balance.remaining.protein_g, Decimal("123"))
        self.assertEqual(balance.remaining.carbohydrates_g, Decimal("250"))
        self.assertEqual(balance.remaining.fat_g, Decimal("61"))

    def test_over_target_balance_is_negative(self) -> None:
        ledger = DailyLedger().add_entry(
            IntakeEntry(record("Chicken"), Decimal("20"))
        )

        balance = ledger.balance_against(targets())

        self.assertEqual(balance.remaining.calories_kcal, Decimal("-1500"))
        self.assertEqual(balance.remaining.protein_g, Decimal("-390"))
        self.assertEqual(balance.remaining.carbohydrates_g, Decimal("250"))
        self.assertEqual(balance.remaining.fat_g, Decimal("-15"))

    def test_ledger_is_immutable_and_previous_state_is_unchanged(self) -> None:
        entry = IntakeEntry(record("Food"), Decimal("1"))
        empty = DailyLedger()
        populated = empty.add_entry(entry)

        self.assertEqual(empty.entries, ())
        self.assertEqual(populated.entries, (entry,))
        with self.assertRaises(FrozenInstanceError):
            populated.entries = ()  # type: ignore[misc]

    def test_nutrition_record_is_not_mutated(self) -> None:
        nutrition_record = record("Food")
        original_nutrients = nutrition_record.nutrients
        entry = IntakeEntry(nutrition_record, Decimal("1.5"))

        _ = entry.consumed_nutrients

        self.assertEqual(nutrition_record.nutrients, original_nutrients)


if __name__ == "__main__":
    unittest.main()
