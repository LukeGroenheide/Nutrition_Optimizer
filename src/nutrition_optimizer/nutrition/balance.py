"""Deterministic daily nutrition target and balance calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from .models import NutrientProfile


def _validate_target(value: Decimal, field_name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative")


def _validate_remainder(value: Decimal | None, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal or None")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")


@dataclass(frozen=True, slots=True)
class DailyMinimums:
    """Optional minimum-oriented goals, separate from centered macro targets."""

    dietary_fiber_g: Decimal | None = None

    def __post_init__(self) -> None:
        if self.dietary_fiber_g is not None:
            _validate_target(self.dietary_fiber_g, "dietary_fiber_g")


@dataclass(frozen=True, slots=True)
class DailyTargets:
    """Required non-negative macro targets plus optional minimum goals."""

    calories_kcal: Decimal
    protein_g: Decimal
    carbohydrates_g: Decimal
    fat_g: Decimal
    minimums: DailyMinimums = field(default_factory=DailyMinimums)

    def __post_init__(self) -> None:
        _validate_target(self.calories_kcal, "calories_kcal")
        _validate_target(self.protein_g, "protein_g")
        _validate_target(self.carbohydrates_g, "carbohydrates_g")
        _validate_target(self.fat_g, "fat_g")
        if not isinstance(self.minimums, DailyMinimums):
            raise TypeError("minimums must be DailyMinimums")


@dataclass(frozen=True, slots=True)
class DailyRemainder:
    """Remaining daily amounts, allowing negative and unknown values."""

    calories_kcal: Decimal | None
    protein_g: Decimal | None
    carbohydrates_g: Decimal | None
    fat_g: Decimal | None

    def __post_init__(self) -> None:
        _validate_remainder(self.calories_kcal, "calories_kcal")
        _validate_remainder(self.protein_g, "protein_g")
        _validate_remainder(self.carbohydrates_g, "carbohydrates_g")
        _validate_remainder(self.fat_g, "fat_g")


@dataclass(frozen=True, slots=True)
class DailyMinimumBalance:
    """Deficits for optional minimum goals; ``None`` is inactive or unknown."""

    dietary_fiber_deficit_g: Decimal | None = None

    def __post_init__(self) -> None:
        _validate_remainder(self.dietary_fiber_deficit_g, "dietary_fiber_deficit_g")


@dataclass(frozen=True, slots=True)
class DailyBalance:
    """Targets, consumed totals, and deterministic remaining amounts."""

    targets: DailyTargets
    consumed: NutrientProfile
    remaining: DailyRemainder
    minimums: DailyMinimumBalance = field(default_factory=DailyMinimumBalance)

    def __post_init__(self) -> None:
        if not isinstance(self.targets, DailyTargets):
            raise TypeError("targets must be DailyTargets")
        if not isinstance(self.consumed, NutrientProfile):
            raise TypeError("consumed must be a NutrientProfile")
        if not isinstance(self.remaining, DailyRemainder):
            raise TypeError("remaining must be DailyRemainder")
        if not isinstance(self.minimums, DailyMinimumBalance):
            raise TypeError("minimums must be DailyMinimumBalance")


def calculate_daily_balance(
    targets: DailyTargets,
    consumed: NutrientProfile,
) -> DailyBalance:
    """Calculate target minus consumed for each tracked nutrient.

    A missing consumed value remains missing in the corresponding remainder.
    Sodium, when present on ``consumed``, is intentionally not part of the
    daily target balance.
    """

    if not isinstance(targets, DailyTargets):
        raise TypeError("targets must be DailyTargets")
    if not isinstance(consumed, NutrientProfile):
        raise TypeError("consumed must be a NutrientProfile")

    return DailyBalance(
        targets=targets,
        consumed=consumed,
        remaining=DailyRemainder(
            calories_kcal=_subtract(targets.calories_kcal, consumed.calories_kcal),
            protein_g=_subtract(targets.protein_g, consumed.protein_g),
            carbohydrates_g=_subtract(
                targets.carbohydrates_g,
                consumed.carbohydrates_g,
            ),
            fat_g=_subtract(targets.fat_g, consumed.fat_g),
        ),
        minimums=DailyMinimumBalance(
            dietary_fiber_deficit_g=_minimum_deficit(
                targets.minimums.dietary_fiber_g,
                consumed.dietary_fiber_g,
            ),
        ),
    )


def _subtract(target: Decimal, consumed: Decimal | None) -> Decimal | None:
    return None if consumed is None else target - consumed


def _minimum_deficit(minimum: Decimal | None, consumed: Decimal | None) -> Decimal | None:
    if minimum is None or consumed is None:
        return None
    return max(Decimal("0"), minimum - consumed)
