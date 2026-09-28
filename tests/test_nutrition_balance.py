from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
import unittest

from nutrition_optimizer.nutrition.balance import (
    DailyRemainder,
    DailyTargets,
    calculate_daily_balance,
)
from nutrition_optimizer.nutrition.models import NutrientProfile


def targets() -> DailyTargets:
    return DailyTargets(
        calories_kcal=Decimal("2000"),
        protein_g=Decimal("150"),
        carbohydrates_g=Decimal("250"),
        fat_g=Decimal("65"),
    )


class DailyBalanceTests(unittest.TestCase):
    def test_zero_consumed_returns_exact_targets(self) -> None:
        target_profile = targets()
        consumed = NutrientProfile(
            calories_kcal=Decimal("0"),
            protein_g=Decimal("0"),
            carbohydrates_g=Decimal("0"),
            fat_g=Decimal("0"),
        )

        balance = calculate_daily_balance(target_profile, consumed)

        self.assertEqual(balance.targets, target_profile)
        self.assertEqual(balance.consumed, consumed)
        self.assertEqual(balance.remaining.calories_kcal, Decimal("2000"))
        self.assertEqual(balance.remaining.protein_g, Decimal("150"))
        self.assertEqual(balance.remaining.carbohydrates_g, Decimal("250"))
        self.assertEqual(balance.remaining.fat_g, Decimal("65"))

    def test_partial_consumption_is_subtracted(self) -> None:
        balance = calculate_daily_balance(
            targets(),
            NutrientProfile(
                calories_kcal=Decimal("500"),
                protein_g=Decimal("30"),
                carbohydrates_g=Decimal("75"),
                fat_g=Decimal("20"),
            ),
        )

        self.assertEqual(balance.remaining.calories_kcal, Decimal("1500"))
        self.assertEqual(balance.remaining.protein_g, Decimal("120"))
        self.assertEqual(balance.remaining.carbohydrates_g, Decimal("175"))
        self.assertEqual(balance.remaining.fat_g, Decimal("45"))

    def test_exact_target_has_zero_remaining(self) -> None:
        balance = calculate_daily_balance(
            targets(),
            NutrientProfile(
                calories_kcal=Decimal("2000"),
                protein_g=Decimal("150"),
                carbohydrates_g=Decimal("250"),
                fat_g=Decimal("65"),
            ),
        )

        self.assertEqual(balance.remaining, DailyRemainder(
            calories_kcal=Decimal("0"),
            protein_g=Decimal("0"),
            carbohydrates_g=Decimal("0"),
            fat_g=Decimal("0"),
        ))

    def test_over_target_remaining_can_be_negative(self) -> None:
        balance = calculate_daily_balance(
            targets(),
            NutrientProfile(
                calories_kcal=Decimal("2100"),
                protein_g=Decimal("175"),
                carbohydrates_g=Decimal("300"),
                fat_g=Decimal("70"),
            ),
        )

        self.assertEqual(balance.remaining.calories_kcal, Decimal("-100"))
        self.assertEqual(balance.remaining.protein_g, Decimal("-25"))
        self.assertEqual(balance.remaining.carbohydrates_g, Decimal("-50"))
        self.assertEqual(balance.remaining.fat_g, Decimal("-5"))

    def test_decimal_quantities_are_preserved(self) -> None:
        balance = calculate_daily_balance(
            DailyTargets(
                calories_kcal=Decimal("2000.25"),
                protein_g=Decimal("150.125"),
                carbohydrates_g=Decimal("250.50"),
                fat_g=Decimal("65.75"),
            ),
            NutrientProfile(
                calories_kcal=Decimal("100.10"),
                protein_g=Decimal("25.025"),
                carbohydrates_g=Decimal("50.25"),
                fat_g=Decimal("10.50"),
            ),
        )

        self.assertEqual(balance.remaining.calories_kcal, Decimal("1900.15"))
        self.assertEqual(balance.remaining.protein_g, Decimal("125.100"))
        self.assertEqual(balance.remaining.carbohydrates_g, Decimal("200.25"))
        self.assertEqual(balance.remaining.fat_g, Decimal("55.25"))

    def test_unknown_consumed_values_remain_unknown(self) -> None:
        balance = calculate_daily_balance(
            targets(),
            NutrientProfile(
                calories_kcal=Decimal("500"),
                protein_g=None,
                carbohydrates_g=Decimal("75"),
                fat_g=None,
                sodium_mg=Decimal("300"),
            ),
        )

        self.assertEqual(balance.remaining.calories_kcal, Decimal("1500"))
        self.assertIsNone(balance.remaining.protein_g)
        self.assertEqual(balance.remaining.carbohydrates_g, Decimal("175"))
        self.assertIsNone(balance.remaining.fat_g)

    def test_negative_targets_are_rejected(self) -> None:
        for field_name in (
            "calories_kcal",
            "protein_g",
            "carbohydrates_g",
            "fat_g",
        ):
            with self.subTest(field_name=field_name):
                values = {
                    "calories_kcal": Decimal("1"),
                    "protein_g": Decimal("1"),
                    "carbohydrates_g": Decimal("1"),
                    "fat_g": Decimal("1"),
                }
                values[field_name] = Decimal("-0.01")
                with self.assertRaises(ValueError):
                    DailyTargets(**values)

    def test_non_finite_targets_are_rejected(self) -> None:
        for value in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    DailyTargets(value, Decimal("1"), Decimal("1"), Decimal("1"))

    def test_non_decimal_targets_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            DailyTargets(2000, Decimal("1"), Decimal("1"), Decimal("1"))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            DailyTargets(Decimal("1"), None, Decimal("1"), Decimal("1"))  # type: ignore[arg-type]

    def test_models_are_immutable(self) -> None:
        target_profile = targets()
        balance = calculate_daily_balance(
            target_profile,
            NutrientProfile(
                calories_kcal=Decimal("500"),
                protein_g=Decimal("30"),
                carbohydrates_g=Decimal("75"),
                fat_g=Decimal("20"),
            ),
        )

        with self.assertRaises(FrozenInstanceError):
            target_profile.calories_kcal = Decimal("1")  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            balance.remaining.calories_kcal = Decimal("1")  # type: ignore[misc]

    def test_inputs_remain_unchanged(self) -> None:
        target_profile = targets()
        consumed = NutrientProfile(
            calories_kcal=Decimal("500"),
            protein_g=Decimal("30"),
            carbohydrates_g=Decimal("75"),
            fat_g=Decimal("20"),
        )
        target_snapshot = target_profile
        consumed_snapshot = consumed

        calculate_daily_balance(target_profile, consumed)

        self.assertEqual(target_profile, target_snapshot)
        self.assertEqual(consumed, consumed_snapshot)


if __name__ == "__main__":
    unittest.main()
