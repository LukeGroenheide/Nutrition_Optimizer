"""Deterministic physical quantities for recommended official servings.

The optimizer stores official-serving multipliers because that is the unit of
the nutrition records.  This module is the small, deterministic bridge to the
physical quantity a person can actually take.  It deliberately has no food
selection, nutrition scoring, persistence, AI, or messaging dependency.

Discrete serving units are converted to whole physical item counts.  Their
official-serving ratios use Decimal division at a fixed 36-significant-digit
precision; binary floating point is never used.  Continuous units retain the
caller-configured official-serving grid and use Decimal multiplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import (
    Decimal,
    InvalidOperation,
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    localcontext,
)
from fractions import Fraction
import re
from typing import Literal

from .nutrition.models import Serving


__all__ = [
    "CONTINUOUS_UNIT_CATEGORY",
    "DISCRETE_COUNT_UNIT_CATEGORY",
    "MAX_DISCRETE_GRID_CHOICES",
    "PHYSICAL_RATIO_DECIMAL_PRECISION",
    "UNKNOWN_UNIT_CATEGORY",
    "PhysicalQuantityError",
    "PhysicalRecommendedQuantity",
    "PhysicalUnitCategory",
    "classify_serving_unit",
    "format_physical_amount",
    "physical_quantity_for",
    "serving_multipliers_for_physical_counts",
    "official_servings_for_physical_count",
]


PhysicalUnitCategory = Literal["discrete_count", "continuous", "unknown"]
DISCRETE_COUNT_UNIT_CATEGORY: Literal["discrete_count"] = "discrete_count"
CONTINUOUS_UNIT_CATEGORY: Literal["continuous"] = "continuous"
UNKNOWN_UNIT_CATEGORY: Literal["unknown"] = "unknown"

# This is intentionally independent of the Decimal context used by unrelated
# application code.  A 36-digit ratio makes 1/3 and 2/3 stable while keeping
# multiplication error far below the precision of source nutrition values.
PHYSICAL_RATIO_DECIMAL_PRECISION = 36

# Prevent a malformed/bulk provider serving definition from creating an
# unbounded optimizer search.  The normal two-serving policy produces far
# fewer choices; exceeding this threshold is conservatively treated as unsafe.
MAX_DISCRETE_GRID_CHOICES = 1000

_COUNT_UNIT_ALIASES: dict[str, str] = {
    "each": "each",
    "ea": "each",
    "count": "count",
    "piece": "piece",
    "pieces": "piece",
    "item": "piece",
    "items": "piece",
    "slice": "slice",
    "slices": "slice",
    "stick": "stick",
    "sticks": "stick",
    "roll": "roll",
    "rolls": "roll",
    "patty": "patty",
    "patties": "patty",
}

_CONTINUOUS_UNITS = frozenset(
    {
        "cup",
        "cups",
        "cup (fl)",
        "fluid ounce",
        "fluid ounces",
        "ounce",
        "ounces",
        "ounce cooked weight",
        "ounces cooked weight",
        "tablespoon",
        "tablespoons",
        "teaspoon",
        "teaspoons",
        "gram",
        "grams",
        "kilogram",
        "kilograms",
        "milligram",
        "milligrams",
        "milliliter",
        "milliliters",
        "millilitre",
        "millilitres",
        "liter",
        "liters",
        "litre",
        "litres",
        "pound",
        "pounds",
    }
)

_WEIGHT_UNITS = frozenset(
    {
        "ounce",
        "ounces",
        "ounce cooked weight",
        "ounces cooked weight",
        "gram",
        "grams",
        "kilogram",
        "kilograms",
        "milligram",
        "milligrams",
        "pound",
        "pounds",
    }
)

_FRACTION_GLYPHS = {
    (1, 2): "½",
    (1, 3): "⅓",
    (2, 3): "⅔",
    (1, 4): "¼",
    (3, 4): "¾",
    (1, 5): "⅕",
    (2, 5): "⅖",
    (3, 5): "⅗",
    (4, 5): "⅘",
    (1, 6): "⅙",
    (5, 6): "⅚",
    (1, 8): "⅛",
    (3, 8): "⅜",
    (5, 8): "⅝",
    (7, 8): "⅞",
}
_WORD_UNIT = re.compile(r"\s+")


class PhysicalQuantityError(ValueError):
    """Raised when an official quantity cannot be made physically safe."""


@dataclass(frozen=True, slots=True)
class PhysicalRecommendedQuantity:
    """One exact physical amount linked to one official serving definition."""

    amount: Decimal
    unit: str
    category: PhysicalUnitCategory
    official_servings: Decimal
    official_serving: Serving
    count_unit: str | None = None

    def __post_init__(self) -> None:
        _positive_decimal(self.amount, "amount")
        _text(self.unit, "unit")
        if self.category not in {
            DISCRETE_COUNT_UNIT_CATEGORY,
            CONTINUOUS_UNIT_CATEGORY,
            UNKNOWN_UNIT_CATEGORY,
        }:
            raise ValueError("category is invalid")
        _positive_decimal(self.official_servings, "official_servings")
        if not isinstance(self.official_serving, Serving):
            raise TypeError("official_serving must be a Serving")
        if self.official_serving.unit != self.unit:
            raise ValueError("unit must match official_serving.unit")
        if self.category == DISCRETE_COUNT_UNIT_CATEGORY:
            if self.count_unit is None:
                raise ValueError("discrete quantities require count_unit")
            if self.amount != self.amount.to_integral_value():
                raise ValueError("discrete physical amount must be a whole count")
        elif self.count_unit is not None:
            raise ValueError("continuous and unknown quantities cannot have count_unit")

    @property
    def is_discrete_count(self) -> bool:
        return self.category == DISCRETE_COUNT_UNIT_CATEGORY

    @property
    def is_continuous(self) -> bool:
        return self.category == CONTINUOUS_UNIT_CATEGORY

    @property
    def is_unknown(self) -> bool:
        return self.category == UNKNOWN_UNIT_CATEGORY

    @property
    def is_weight(self) -> bool:
        """Whether this known continuous amount is expressed by weight."""

        return _normalized_unit(self.unit) in _WEIGHT_UNITS


def classify_serving_unit(unit: str | None) -> PhysicalUnitCategory:
    """Classify only known provider units; unfamiliar units remain unknown."""

    normalized = _normalized_unit(unit)
    if normalized in _COUNT_UNIT_ALIASES:
        return DISCRETE_COUNT_UNIT_CATEGORY
    if normalized in _CONTINUOUS_UNITS:
        return CONTINUOUS_UNIT_CATEGORY
    return UNKNOWN_UNIT_CATEGORY


def physical_quantity_for(
    serving: Serving,
    official_servings: Decimal,
) -> PhysicalRecommendedQuantity:
    """Compute the exact physical amount for an official multiplier.

    A discrete amount must already be a whole count.  This deliberately raises
    instead of rounding a fractional item count, which prevents a rendering
    such as "2 tenders" from silently retaining a 2.25-item authoritative
    quantity.
    """

    _validate_serving_for_quantity(serving)
    _positive_decimal(official_servings, "official_servings")
    assert serving.quantity is not None
    assert serving.unit is not None
    category = classify_serving_unit(serving.unit)
    amount = _decimal_product(serving.quantity, official_servings)
    count_unit = _COUNT_UNIT_ALIASES.get(_normalized_unit(serving.unit))
    if category == DISCRETE_COUNT_UNIT_CATEGORY:
        integral = amount.to_integral_value()
        if abs(amount - integral) > _count_rounding_tolerance():
            raise PhysicalQuantityError(
                "discrete official servings do not represent a whole physical count"
            )
        amount = integral
    return PhysicalRecommendedQuantity(
        amount=amount,
        unit=serving.unit,
        category=category,
        official_servings=official_servings,
        official_serving=serving,
        count_unit=count_unit,
    )


def official_servings_for_physical_count(
    serving: Serving,
    physical_count: Decimal,
) -> Decimal:
    """Derive a deterministic official multiplier for a whole item count."""

    _validate_serving_for_quantity(serving)
    if classify_serving_unit(serving.unit) != DISCRETE_COUNT_UNIT_CATEGORY:
        raise PhysicalQuantityError("physical count conversion requires a count unit")
    _positive_decimal(physical_count, "physical_count")
    if physical_count != physical_count.to_integral_value():
        raise PhysicalQuantityError("physical_count must be a whole count")
    assert serving.quantity is not None
    with localcontext() as context:
        context.prec = PHYSICAL_RATIO_DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        return physical_count / serving.quantity


def serving_multipliers_for_physical_counts(
    serving: Serving,
    minimum_servings: Decimal,
    maximum_servings: Decimal,
) -> tuple[Decimal, ...]:
    """Return whole-count ratios inside existing official-serving limits.

    The bounds are derived as ``ceil(minimum × official_count)`` and
    ``floor(maximum × official_count)``.  Thus a three-each serving with the
    existing .25–2.00 policy yields whole counts 1 through 6, represented as
    Decimal ratios such as 1/3 and 2/3.  Unknown units return no safe grid.
    """

    _validate_serving_for_quantity(serving)
    _positive_decimal(minimum_servings, "minimum_servings")
    _positive_decimal(maximum_servings, "maximum_servings")
    if maximum_servings < minimum_servings:
        raise ValueError("maximum_servings must be at least minimum_servings")
    if classify_serving_unit(serving.unit) != DISCRETE_COUNT_UNIT_CATEGORY:
        return ()
    assert serving.quantity is not None
    with localcontext() as context:
        context.prec = PHYSICAL_RATIO_DECIMAL_PRECISION
        lower = (minimum_servings * serving.quantity).to_integral_value(
            rounding=ROUND_CEILING
        )
        upper = (maximum_servings * serving.quantity).to_integral_value(
            rounding=ROUND_FLOOR
        )
    lower = max(lower, Decimal("1"))
    if upper < lower:
        return ()
    choice_count = int(upper - lower + Decimal("1"))
    if choice_count > MAX_DISCRETE_GRID_CHOICES:
        return ()
    return tuple(
        official_servings_for_physical_count(serving, Decimal(count))
        for count in range(int(lower), int(upper) + 1)
    )


def format_physical_amount(quantity: PhysicalRecommendedQuantity) -> str:
    """Format a canonical physical amount without naming the food."""

    if not isinstance(quantity, PhysicalRecommendedQuantity):
        raise TypeError("quantity must be a PhysicalRecommendedQuantity")
    if quantity.is_unknown:
        raise PhysicalQuantityError("unknown serving units cannot be formatted safely")
    if quantity.is_discrete_count:
        assert quantity.count_unit is not None
        number = _format_decimal(quantity.amount)
        if quantity.count_unit in {"each", "count"}:
            return number
        return f"{number} {_plural_count_unit(quantity.count_unit, quantity.amount)}"

    normalized = _normalized_unit(quantity.unit)
    amount = _format_common_fraction(quantity.amount)
    if normalized in {"ounce", "ounces"}:
        return f"about {amount} oz"
    if normalized in {"ounce cooked weight", "ounces cooked weight"}:
        return f"about {amount} oz cooked"
    unit = _display_unit(normalized, quantity.amount)
    return f"{amount} {unit}"


def _validate_serving_for_quantity(serving: Serving) -> None:
    if not isinstance(serving, Serving):
        raise TypeError("serving must be a Serving")
    if serving.quantity is None or serving.unit is None:
        raise PhysicalQuantityError(
            "serving must have numeric quantity and known unit for physical conversion"
        )


def _decimal_product(left: Decimal, right: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = PHYSICAL_RATIO_DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        return left * right


def _count_rounding_tolerance() -> Decimal:
    return Decimal("1e-32")


def _normalized_unit(unit: str | None) -> str:
    if unit is None:
        return ""
    return _WORD_UNIT.sub(" ", unit.strip().casefold())


def _display_unit(normalized: str, amount: Decimal) -> str:
    if normalized in {"cup", "cups", "cup (fl)"}:
        return "cup" if amount <= Decimal("1") else "cups"
    if normalized in {"fluid ounce", "fluid ounces"}:
        return "fluid ounce" if amount <= Decimal("1") else "fluid ounces"
    if normalized in {"tablespoon", "tablespoons"}:
        return "tablespoon" if amount <= Decimal("1") else "tablespoons"
    if normalized in {"teaspoon", "teaspoons"}:
        return "teaspoon" if amount <= Decimal("1") else "teaspoons"
    if normalized in {"gram", "grams"}:
        return "gram" if amount <= Decimal("1") else "grams"
    if normalized in {"kilogram", "kilograms"}:
        return "kilogram" if amount <= Decimal("1") else "kilograms"
    if normalized in {"milligram", "milligrams"}:
        return "milligram" if amount <= Decimal("1") else "milligrams"
    if normalized in {"milliliter", "milliliters", "millilitre", "millilitres"}:
        return "milliliter" if amount <= Decimal("1") else "milliliters"
    if normalized in {"liter", "liters", "litre", "litres"}:
        return "liter" if amount <= Decimal("1") else "liters"
    if normalized in {"pound", "pounds"}:
        return "pound" if amount <= Decimal("1") else "pounds"
    return normalized


def _plural_count_unit(unit: str, amount: Decimal) -> str:
    if amount == Decimal("1"):
        return unit
    if unit == "patty":
        return "patties"
    return f"{unit}s"


def _format_common_fraction(value: Decimal) -> str:
    """Use a common fraction only when it closely matches the Decimal."""

    integral = value.to_integral_value()
    fraction = value - integral
    if fraction == 0:
        return _format_decimal(value)
    try:
        exact = Fraction(value)
        limited = exact.limit_denominator(8)
        with localcontext() as context:
            context.prec = PHYSICAL_RATIO_DECIMAL_PRECISION
            context.rounding = ROUND_HALF_EVEN
            candidate = Decimal(limited.numerator) / Decimal(limited.denominator)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return _format_decimal(value)
    if abs(candidate - value) > Decimal("1e-12"):
        return _format_decimal(value)
    numerator = limited.numerator
    denominator = limited.denominator
    whole = numerator // denominator
    remainder = numerator % denominator
    if remainder == 0:
        return str(whole)
    glyph = _FRACTION_GLYPHS.get((remainder, denominator))
    if glyph is None:
        return _format_decimal(value)
    return f"{whole}{glyph}" if whole else glyph


def _format_decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _positive_decimal(value: Decimal, field_name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{field_name} must be a positive finite Decimal")


def _text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
