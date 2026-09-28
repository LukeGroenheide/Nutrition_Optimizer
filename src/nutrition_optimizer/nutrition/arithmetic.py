"""Deterministic Decimal arithmetic for official nutrient profiles."""

from __future__ import annotations

from decimal import Decimal
from typing import Iterable

from .models import NutrientProfile


def scale_nutrients(profile: NutrientProfile, servings: Decimal) -> NutrientProfile:
    """Scale nutrients per official serving by a non-negative multiplier.

    Zero servings contribute zero for every nutrient, including nutrients that
    are unknown per serving. For positive multipliers, unknown nutrient values
    remain unknown. No rounding is performed; callers retain the exact Decimal
    result.
    """

    _validate_profile(profile)
    _validate_multiplier(servings)
    if servings == Decimal("0"):
        return NutrientProfile(
            calories_kcal=Decimal("0"),
            protein_g=Decimal("0"),
            carbohydrates_g=Decimal("0"),
            fat_g=Decimal("0"),
            sodium_mg=Decimal("0"),
            dietary_fiber_g=Decimal("0"),
        )
    return NutrientProfile(
        calories_kcal=_scale_value(profile.calories_kcal, servings),
        protein_g=_scale_value(profile.protein_g, servings),
        carbohydrates_g=_scale_value(profile.carbohydrates_g, servings),
        fat_g=_scale_value(profile.fat_g, servings),
        sodium_mg=_scale_value(profile.sodium_mg, servings),
        dietary_fiber_g=_scale_value(profile.dietary_fiber_g, servings),
    )


def add_nutrients(*profiles: NutrientProfile) -> NutrientProfile:
    """Add one or more nutrient profiles conservatively.

    A nutrient is summed only when it is known in every input profile. If any
    input is ``None`` for that nutrient, the total remains ``None`` rather than
    implying a complete value.
    """

    if not profiles:
        raise ValueError("at least one nutrient profile is required")
    for profile in profiles:
        _validate_profile(profile)

    return NutrientProfile(
        calories_kcal=_sum_values(profile.calories_kcal for profile in profiles),
        protein_g=_sum_values(profile.protein_g for profile in profiles),
        carbohydrates_g=_sum_values(profile.carbohydrates_g for profile in profiles),
        fat_g=_sum_values(profile.fat_g for profile in profiles),
        sodium_mg=_sum_values(profile.sodium_mg for profile in profiles),
        dietary_fiber_g=_sum_values(profile.dietary_fiber_g for profile in profiles),
    )


def _scale_value(value: Decimal | None, servings: Decimal) -> Decimal | None:
    return None if value is None else value * servings


def _sum_values(values: Iterable[Decimal | None]) -> Decimal | None:
    total = Decimal("0")
    for value in values:
        if value is None:
            return None
        total += value
    return total


def _validate_profile(profile: NutrientProfile) -> None:
    if not isinstance(profile, NutrientProfile):
        raise TypeError("profile must be a NutrientProfile")


def _validate_multiplier(servings: Decimal) -> None:
    if not isinstance(servings, Decimal):
        raise TypeError("servings must be a Decimal")
    if not servings.is_finite():
        raise ValueError("servings must be finite")
    if servings < 0:
        raise ValueError("servings cannot be negative")
