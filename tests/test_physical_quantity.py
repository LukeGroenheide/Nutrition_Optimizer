"""Offline tests for deterministic physical recommendation quantities."""

from __future__ import annotations

from decimal import Decimal
import unittest

from nutrition_optimizer.nutrition import Serving
from nutrition_optimizer.physical_quantity import (
    CONTINUOUS_UNIT_CATEGORY,
    DISCRETE_COUNT_UNIT_CATEGORY,
    UNKNOWN_UNIT_CATEGORY,
    PhysicalQuantityError,
    classify_serving_unit,
    format_physical_amount,
    official_servings_for_physical_count,
    physical_quantity_for,
    serving_multipliers_for_physical_counts,
)


class PhysicalQuantityTests(unittest.TestCase):
    def test_serving_units_are_classified_conservatively(self) -> None:
        for unit in ("Each", "Count", "Piece", "Slice", "Sticks", "Roll", "Patty"):
            with self.subTest(unit=unit):
                self.assertEqual(classify_serving_unit(unit), DISCRETE_COUNT_UNIT_CATEGORY)
        for unit in (
            "Cup",
            "Cup (fl)",
            "Fluid Ounce",
            "Ounce",
            "Ounce Cooked Weight",
            "Tablespoon",
            "Teaspoon",
        ):
            with self.subTest(unit=unit):
                self.assertEqual(classify_serving_unit(unit), CONTINUOUS_UNIT_CATEGORY)
        for unit in ("Portion", "Serving", "Mystery Unit"):
            with self.subTest(unit=unit):
                self.assertEqual(classify_serving_unit(unit), UNKNOWN_UNIT_CATEGORY)

    def test_one_third_and_two_thirds_use_fixed_decimal_ratio_precision(self) -> None:
        serving = Serving(Decimal("3"), "Each", "3 Each")
        one = official_servings_for_physical_count(serving, Decimal("1"))
        two = official_servings_for_physical_count(serving, Decimal("2"))

        self.assertIsInstance(one, Decimal)
        self.assertEqual(one, Decimal("0.333333333333333333333333333333333333"))
        self.assertEqual(two, Decimal("0.666666666666666666666666666666666667"))
        self.assertEqual(physical_quantity_for(serving, one).amount, Decimal("1"))
        self.assertEqual(physical_quantity_for(serving, two).amount, Decimal("2"))

    def test_fractional_discrete_quantity_is_rejected_without_rounding(self) -> None:
        with self.assertRaises(PhysicalQuantityError):
            physical_quantity_for(
                Serving(Decimal("3"), "Each", "3 Each"),
                Decimal("0.75"),
            )

    def test_continuous_amount_and_common_fraction_formatting_are_exact(self) -> None:
        cases = (
            ("0.5", "1.25", "⅝ cup", Decimal("0.625")),
            ("1", "0.5", "½ cup", Decimal("0.5")),
            ("1", "0.75", "¾ cup", Decimal("0.75")),
            ("1", "1.25", "1¼ cups", Decimal("1.25")),
        )
        for serving_amount, multiplier, expected_text, expected_amount in cases:
            with self.subTest(multiplier=multiplier):
                quantity = physical_quantity_for(
                    Serving(Decimal(serving_amount), "Cup", f"{serving_amount} Cup"),
                    Decimal(multiplier),
                )
                self.assertEqual(quantity.amount, expected_amount)
                self.assertEqual(format_physical_amount(quantity), expected_text)

    def test_weight_and_cooked_weight_formatting_is_honest(self) -> None:
        ounces = physical_quantity_for(
            Serving(Decimal("4"), "Ounce", "4 Ounce"),
            Decimal("1.25"),
        )
        cooked = physical_quantity_for(
            Serving(Decimal("4"), "Ounce Cooked Weight", "4 Ounce Cooked Weight"),
            Decimal("1.25"),
        )

        self.assertEqual(format_physical_amount(ounces), "about 5 oz")
        self.assertEqual(format_physical_amount(cooked), "about 5 oz cooked")
        self.assertTrue(cooked.is_weight)

    def test_fluid_ounce_is_deterministic_continuous_measurement(self) -> None:
        quantity = physical_quantity_for(
            Serving(Decimal("4"), "Fluid Ounce", "4 Fluid Ounce"),
            Decimal("0.75"),
        )

        self.assertEqual(quantity.amount, Decimal("3.00"))
        self.assertEqual(format_physical_amount(quantity), "3 fluid ounces")
        self.assertFalse(quantity.is_weight)

    def test_count_grid_respects_existing_official_bounds_and_maximum(self) -> None:
        serving = Serving(Decimal("3"), "Each", "3 Each")

        grid = serving_multipliers_for_physical_counts(
            serving,
            Decimal("0.25"),
            Decimal("2"),
        )

        self.assertEqual(len(grid), 6)
        self.assertEqual(physical_quantity_for(serving, grid[0]).amount, Decimal("1"))
        self.assertEqual(physical_quantity_for(serving, grid[-1]).amount, Decimal("6"))
        self.assertEqual(
            serving_multipliers_for_physical_counts(
                Serving(Decimal("10000"), "Each", "10000 Each"),
                Decimal("0.25"),
                Decimal("2"),
            ),
            (),
        )

    def test_unknown_units_have_no_safe_count_grid(self) -> None:
        self.assertEqual(
            serving_multipliers_for_physical_counts(
                Serving(Decimal("1"), "Portion", "1 Portion"),
                Decimal("0.25"),
                Decimal("2"),
            ),
            (),
        )
        with self.assertRaises(PhysicalQuantityError):
            format_physical_amount(
                physical_quantity_for(
                    Serving(Decimal("1"), "Portion", "1 Portion"),
                    Decimal("1"),
                )
            )


if __name__ == "__main__":
    unittest.main()
