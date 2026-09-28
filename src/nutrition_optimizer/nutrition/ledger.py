"""Immutable daily intake ledger domain models."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .arithmetic import add_nutrients, scale_nutrients
from .balance import DailyBalance, DailyTargets, calculate_daily_balance
from .models import NutritionRecord, NutrientProfile


@dataclass(frozen=True, slots=True)
class IntakeEntry:
    """One positive-serving consumption of an official nutrition record."""

    record: NutritionRecord
    servings: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.record, NutritionRecord):
            raise TypeError("record must be a NutritionRecord")
        _validate_servings(self.servings)

    @property
    def consumed_nutrients(self) -> NutrientProfile:
        """Return this entry's nutrients derived from its serving multiplier."""

        return scale_nutrients(self.record.nutrients, self.servings)


@dataclass(frozen=True, slots=True)
class DailyLedger:
    """An immutable collection of intake entries for one logical day."""

    entries: tuple[IntakeEntry, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple):
            raise TypeError("entries must be a tuple")
        if not all(isinstance(entry, IntakeEntry) for entry in self.entries):
            raise TypeError("entries must contain IntakeEntry values")

    def add_entry(self, entry: IntakeEntry) -> DailyLedger:
        """Return a new ledger with one additional intake entry."""

        if not isinstance(entry, IntakeEntry):
            raise TypeError("entry must be an IntakeEntry")
        return DailyLedger(entries=self.entries + (entry,))

    @property
    def total_consumed_nutrients(self) -> NutrientProfile:
        """Return the deterministic total for all entries in this ledger."""

        if not self.entries:
            return _zero_nutrients()
        return add_nutrients(*(entry.consumed_nutrients for entry in self.entries))

    def balance_against(self, targets: DailyTargets) -> DailyBalance:
        """Return the current daily target balance for this ledger."""

        return calculate_daily_balance(targets, self.total_consumed_nutrients)


def _zero_nutrients() -> NutrientProfile:
    zero = Decimal("0")
    return NutrientProfile(
        calories_kcal=zero,
        protein_g=zero,
        carbohydrates_g=zero,
        fat_g=zero,
        sodium_mg=zero,
        dietary_fiber_g=zero,
    )


def _validate_servings(value: Decimal) -> None:
    if not isinstance(value, Decimal):
        raise TypeError("servings must be a Decimal")
    if not value.is_finite():
        raise ValueError("servings must be finite")
    if value <= 0:
        raise ValueError("servings must be greater than zero")
