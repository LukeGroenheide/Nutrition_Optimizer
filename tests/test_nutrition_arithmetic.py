from __future__ import annotations

from decimal import Decimal
import unittest

from nutrition_optimizer.nutrition.arithmetic import add_nutrients, scale_nutrients
from nutrition_optimizer.nutrition.models import NutrientProfile


def observed_profile() -> NutrientProfile:
    return NutrientProfile(
        calories_kcal=Decimal("175"),
        protein_g=Decimal("27"),
        carbohydrates_g=Decimal("0"),
        fat_g=Decimal("4"),
        sodium_mg=Decimal("303"),
    )


class NutrientArithmeticTests(unittest.TestCase):
    def test_one_serving_preserves_values(self) -> None:
        profile = observed_profile()

        self.assertEqual(scale_nutrients(profile, Decimal("1")), profile)

    def test_fractional_servings_use_decimal_arithmetic(self) -> None:
        scaled = scale_nutrients(observed_profile(), Decimal("1.5"))

        self.assertEqual(scaled.calories_kcal, Decimal("262.5"))
        self.assertEqual(scaled.protein_g, Decimal("40.5"))
        self.assertEqual(scaled.carbohydrates_g, Decimal("0.0"))
        self.assertEqual(scaled.fat_g, Decimal("6.0"))
        self.assertEqual(scaled.sodium_mg, Decimal("454.5"))

    def test_zero_servings_zeroes_all_values(self) -> None:
        profile = NutrientProfile(
            calories_kcal=Decimal("10"),
            protein_g=None,
            carbohydrates_g=Decimal("2"),
            fat_g=None,
            sodium_mg=None,
        )

        scaled = scale_nutrients(profile, Decimal("0"))

        for field in (
            "calories_kcal",
            "protein_g",
            "carbohydrates_g",
            "fat_g",
            "sodium_mg",
        ):
            with self.subTest(field=field):
                self.assertEqual(getattr(scaled, field), Decimal("0"))

    def test_negative_servings_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            scale_nutrients(observed_profile(), Decimal("-0.5"))

    def test_non_finite_servings_are_rejected(self) -> None:
        for value in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    scale_nutrients(observed_profile(), value)

    def test_non_decimal_servings_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            scale_nutrients(observed_profile(), 1.5)  # type: ignore[arg-type]

    def test_adds_multiple_known_profiles(self) -> None:
        first = observed_profile()
        second = NutrientProfile(
            calories_kcal=Decimal("25.25"),
            protein_g=Decimal("3.125"),
            carbohydrates_g=Decimal("4.50"),
            fat_g=Decimal("1.75"),
            sodium_mg=Decimal("10.5"),
        )

        total = add_nutrients(first, second)

        self.assertEqual(total.calories_kcal, Decimal("200.25"))
        self.assertEqual(total.protein_g, Decimal("30.125"))
        self.assertEqual(total.carbohydrates_g, Decimal("4.50"))
        self.assertEqual(total.fat_g, Decimal("5.75"))
        self.assertEqual(total.sodium_mg, Decimal("313.5"))

    def test_unknown_values_propagate_per_nutrient(self) -> None:
        first = NutrientProfile(
            calories_kcal=Decimal("100"),
            protein_g=Decimal("10"),
            carbohydrates_g=None,
            fat_g=Decimal("2"),
            sodium_mg=None,
        )
        second = NutrientProfile(
            calories_kcal=Decimal("50"),
            protein_g=None,
            carbohydrates_g=Decimal("5"),
            fat_g=Decimal("1"),
            sodium_mg=Decimal("20"),
        )

        total = add_nutrients(first, second)

        self.assertEqual(total.calories_kcal, Decimal("150"))
        self.assertIsNone(total.protein_g)
        self.assertIsNone(total.carbohydrates_g)
        self.assertEqual(total.fat_g, Decimal("3"))
        self.assertIsNone(total.sodium_mg)

    def test_optional_sodium_can_be_added_when_known(self) -> None:
        total = add_nutrients(
            NutrientProfile(sodium_mg=Decimal("100")),
            NutrientProfile(sodium_mg=Decimal("25.5")),
        )

        self.assertEqual(total.sodium_mg, Decimal("125.5"))

    def test_empty_addition_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            add_nutrients()

    def test_inputs_remain_unchanged(self) -> None:
        original = observed_profile()
        snapshot = original

        scaled = scale_nutrients(original, Decimal("1.5"))
        total = add_nutrients(original, observed_profile())

        self.assertEqual(original, snapshot)
        self.assertEqual(original, observed_profile())
        self.assertNotEqual(scaled, original)
        self.assertNotEqual(total, original)


if __name__ == "__main__":
    unittest.main()
