"""Immutable source-independent models for official nutrition data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


def _validate_optional_text(value: str | None, field_name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{field_name} must be non-empty when provided")


def _validate_required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")


def _validate_decimal(
    value: Decimal | None,
    field_name: str,
    *,
    require_positive: bool = False,
) -> None:
    if value is None:
        return
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal or None")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if require_positive and value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    if not require_positive and value < 0:
        raise ValueError(f"{field_name} cannot be negative")


def _validate_text_tuple(values: tuple[str, ...], field_name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name}[{index}] must be non-empty text")


@dataclass(frozen=True, slots=True)
class Serving:
    """The official serving definition attached to a nutrition record.

    Quantity is represented independently from the original source text so a
    later adapter can preserve both parsed and unparsed serving information.
    """

    quantity: Decimal | None = None
    unit: str | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        _validate_decimal(self.quantity, "serving quantity", require_positive=True)
        _validate_optional_text(self.unit, "serving unit")
        _validate_optional_text(self.text, "serving text")
        if self.quantity is None and self.unit is None and self.text is None:
            raise ValueError("serving must include quantity, unit, or text")


@dataclass(frozen=True, slots=True)
class NutrientProfile:
    """Source-reported nutrients per official serving.

    Calories are kcal, protein/carbohydrates/fat are grams, and sodium is
    milligrams. Missing source values remain ``None``.
    """

    calories_kcal: Decimal | None = None
    protein_g: Decimal | None = None
    carbohydrates_g: Decimal | None = None
    fat_g: Decimal | None = None
    sodium_mg: Decimal | None = None
    dietary_fiber_g: Decimal | None = None

    def __post_init__(self) -> None:
        _validate_decimal(self.calories_kcal, "calories_kcal")
        _validate_decimal(self.protein_g, "protein_g")
        _validate_decimal(self.carbohydrates_g, "carbohydrates_g")
        _validate_decimal(self.fat_g, "fat_g")
        _validate_decimal(self.sodium_mg, "sodium_mg")
        _validate_decimal(self.dietary_fiber_g, "dietary_fiber_g")


@dataclass(frozen=True, slots=True)
class SourceIdentifier:
    """A source-specific identifier such as a recipe or component ID."""

    kind: str
    value: str

    def __post_init__(self) -> None:
        _validate_required_text(self.kind, "identifier kind")
        _validate_required_text(self.value, "identifier value")


@dataclass(frozen=True, slots=True)
class NutritionProvenance:
    """Origin and retrieval metadata for an official nutrition record."""

    provider: str
    retrieved_at: datetime
    record_type: str | None = None
    source_reference: str | None = None
    identifiers: tuple[SourceIdentifier, ...] = ()

    def __post_init__(self) -> None:
        _validate_required_text(self.provider, "provenance provider")
        _validate_optional_text(self.record_type, "provenance record type")
        _validate_optional_text(self.source_reference, "provenance source reference")
        if not isinstance(self.retrieved_at, datetime):
            raise TypeError("provenance retrieved_at must be a datetime")
        if self.retrieved_at.tzinfo is None or self.retrieved_at.utcoffset() is None:
            raise ValueError("provenance retrieved_at must be timezone-aware")
        if not isinstance(self.identifiers, tuple):
            raise TypeError("provenance identifiers must be a tuple")
        if not all(isinstance(identifier, SourceIdentifier) for identifier in self.identifiers):
            raise TypeError("provenance identifiers must contain SourceIdentifier values")


@dataclass(frozen=True, slots=True)
class NutritionRecord:
    """Source nutrition per serving; provenance distinguishes official and derived."""

    name: str
    serving: Serving
    nutrients: NutrientProfile
    provenance: NutritionProvenance
    ingredients: tuple[str, ...] = ()
    allergens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_required_text(self.name, "nutrition record name")
        if not isinstance(self.serving, Serving):
            raise TypeError("serving must be a Serving")
        if not isinstance(self.nutrients, NutrientProfile):
            raise TypeError("nutrients must be a NutrientProfile")
        if not isinstance(self.provenance, NutritionProvenance):
            raise TypeError("provenance must be a NutritionProvenance")
        _validate_text_tuple(self.ingredients, "ingredients")
        _validate_text_tuple(self.allergens, "allergens")
