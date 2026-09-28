"""Operator-maintained food-specific serving-line presentation calibrations.

Official FD servings remain the only source for optimizer and intake
arithmetic.  This module contains narrowly scoped, observed relationships for
how one *specific* official serving is physically presented at Phelps.  A
calibration is guarded by both the stable provider identity and the exact
official serving definition, so a renamed food, a similarly named food, or a
provider serving-size change cannot silently inherit it.

The checked-in values are intentionally small MVP configuration.  Update a
value only after an operator has observed a new serving-line presentation;
this module does not infer or learn physical conversions from user reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import re

from .nutrition.models import Serving, SourceIdentifier


__all__ = [
    "DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS",
    "CalibratedServingPresentation",
    "PresentationCalibrationRegistry",
    "ServingPresentationCalibration",
]


_DECIMAL_PRECISION = 36
_SPACE = re.compile(r"\s+")
_NUMBER_WORDS: dict[str, Decimal] = {
    "a": Decimal("1"),
    "an": Decimal("1"),
    "one": Decimal("1"),
    "two": Decimal("2"),
    "three": Decimal("3"),
    "four": Decimal("4"),
    "five": Decimal("5"),
    "six": Decimal("6"),
    "seven": Decimal("7"),
    "eight": Decimal("8"),
    "nine": Decimal("9"),
    "ten": Decimal("10"),
    "eleven": Decimal("11"),
    "twelve": Decimal("12"),
    "half": Decimal("0.5"),
    "a half": Decimal("0.5"),
    "one half": Decimal("0.5"),
    "quarter": Decimal("0.25"),
    "a quarter": Decimal("0.25"),
    "one quarter": Decimal("0.25"),
}
_UNICODE_FRACTIONS = {
    "¼": Decimal("0.25"),
    "½": Decimal("0.5"),
    "¾": Decimal("0.75"),
}
_NUMBER_EXPRESSION = (
    r"(?:\d+(?:\.\d+)?|\.\d+|\d+\s*/\s*\d+|"
    r"a\s+half|one\s+half|a\s+quarter|one\s+quarter|"
    r"half|quarter|an?|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|¼|½|¾)"
)


def _normalized(value: str) -> str:
    return _SPACE.sub(" ", value.strip().casefold())


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{field_name} must be non-empty one-line text")


def _positive_decimal(value: Decimal, field_name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{field_name} must be a positive finite Decimal")


def _format_decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _divide(left: Decimal, right: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        return left / right


def _multiply(left: Decimal, right: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        return left * right


def _parse_amount(value: str) -> Decimal | None:
    normalized = _normalized(value)
    if normalized in _NUMBER_WORDS:
        return _NUMBER_WORDS[normalized]
    if normalized in _UNICODE_FRACTIONS:
        return _UNICODE_FRACTIONS[normalized]
    if "/" in normalized:
        numerator_text, separator, denominator_text = normalized.partition("/")
        if not separator:
            return None
        try:
            numerator = Decimal(numerator_text.strip())
            denominator = Decimal(denominator_text.strip())
        except (InvalidOperation, ValueError):
            return None
        if (
            not numerator.is_finite()
            or not denominator.is_finite()
            or numerator <= 0
            or denominator <= 0
        ):
            return None
        return _divide(numerator, denominator)
    try:
        amount = Decimal(normalized)
    except (InvalidOperation, ValueError):
        return None
    if not amount.is_finite() or amount <= 0:
        return None
    return amount


def _normalized_aliases(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    aliases: list[str] = []
    for value in values:
        _require_text(value, field_name)
        normalized = _normalized(value)
        if normalized in aliases:
            raise ValueError(f"{field_name} must not repeat aliases")
        aliases.append(normalized)
    return tuple(aliases)


def _amount_before_alias(text: str, aliases: tuple[str, ...]) -> Decimal | None:
    """Parse one configured physical phrase, never a name substring.

    Aliases are complete, operator-supplied phrases for one calibration.  The
    caller has already identified the food by stable FD identity, so accepting
    a shorthand such as ``quarter piece`` remains scoped to that one food.
    """

    normalized = _normalized(text)
    for alias in sorted(aliases, key=len, reverse=True):
        match = re.fullmatch(
            rf"(?P<amount>{_NUMBER_EXPRESSION})(?:\s+of)?(?:\s+(?:a|an))?\s+{re.escape(alias)}",
            normalized,
        )
        if match is not None:
            return _parse_amount(match.group("amount"))
    return None


@dataclass(frozen=True, slots=True)
class CalibratedServingPresentation:
    """Exact display facts derived from one food-specific calibration."""

    source_identifier: SourceIdentifier
    official_servings: Decimal
    presentation_amount: Decimal
    natural_quantity_text: str
    display_food_name: str
    whole_item_amount: Decimal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_identifier, SourceIdentifier):
            raise TypeError("source_identifier must be a SourceIdentifier")
        _positive_decimal(self.official_servings, "official_servings")
        _positive_decimal(self.presentation_amount, "presentation_amount")
        _require_text(self.natural_quantity_text, "natural_quantity_text")
        _require_text(self.display_food_name, "display_food_name")
        if self.whole_item_amount is not None:
            _positive_decimal(self.whole_item_amount, "whole_item_amount")


@dataclass(frozen=True, slots=True)
class ServingPresentationCalibration:
    """One exact official-serving-to-serving-line relationship.

    ``presentation_units_per_official_serving`` is an observed ratio for this
    one provider component, not a conversion for the serving unit in general.
    The optional whole-item relation only adds a human-friendly parenthetical;
    it never changes the official multiplier.
    """

    calibration_id: str
    calibration_version: int
    provenance: str
    source_identifier: SourceIdentifier
    expected_serving_quantity: Decimal
    expected_serving_unit: str
    display_food_name: str
    presentation_units_per_official_serving: Decimal
    presentation_unit_singular: str
    presentation_unit_plural: str
    presentation_aliases: tuple[str, ...]
    whole_items_per_official_serving: Decimal | None = None
    whole_item_singular: str | None = None
    whole_item_plural: str | None = None
    whole_item_aliases: tuple[str, ...] = ()
    whole_equivalent_minimum: Decimal = Decimal("1")
    display_rounding_increment: Decimal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_identifier, SourceIdentifier):
            raise TypeError("source_identifier must be a SourceIdentifier")
        _require_text(self.calibration_id, "calibration_id")
        _require_text(self.provenance, "provenance")
        if type(self.calibration_version) is not int or self.calibration_version < 1:
            raise ValueError("invalid calibration version")
        _positive_decimal(self.expected_serving_quantity, "expected_serving_quantity")
        _require_text(self.expected_serving_unit, "expected_serving_unit")
        _require_text(self.display_food_name, "display_food_name")
        _positive_decimal(
            self.presentation_units_per_official_serving,
            "presentation_units_per_official_serving",
        )
        _require_text(self.presentation_unit_singular, "presentation_unit_singular")
        _require_text(self.presentation_unit_plural, "presentation_unit_plural")
        _normalized_aliases(self.presentation_aliases, "presentation_aliases")
        _positive_decimal(self.whole_equivalent_minimum, "whole_equivalent_minimum")
        if self.display_rounding_increment is not None:
            _positive_decimal(
                self.display_rounding_increment,
                "display_rounding_increment",
            )
        whole_fields = (
            self.whole_items_per_official_serving,
            self.whole_item_singular,
            self.whole_item_plural,
        )
        if any(value is not None for value in whole_fields):
            if any(value is None for value in whole_fields):
                raise ValueError("whole-item presentation fields must be supplied together")
            assert self.whole_items_per_official_serving is not None
            assert self.whole_item_singular is not None
            assert self.whole_item_plural is not None
            _positive_decimal(
                self.whole_items_per_official_serving,
                "whole_items_per_official_serving",
            )
            _require_text(self.whole_item_singular, "whole_item_singular")
            _require_text(self.whole_item_plural, "whole_item_plural")
            _normalized_aliases(self.whole_item_aliases, "whole_item_aliases")
        elif self.whole_item_aliases:
            raise ValueError("whole_item_aliases require a whole-item presentation")

    def matches(self, source_identifier: SourceIdentifier, serving: Serving) -> bool:
        """Require both stable identity and the observed official serving."""

        if source_identifier != self.source_identifier or not isinstance(serving, Serving):
            return False
        return (
            serving.quantity == self.expected_serving_quantity
            and serving.unit is not None
            and _normalized(serving.unit) == _normalized(self.expected_serving_unit)
        )

    def render(self, official_servings: Decimal) -> CalibratedServingPresentation:
        """Render exact calibrated serving-line wording for an official amount."""

        _positive_decimal(official_servings, "official_servings")
        presentation_amount = _multiply(
            official_servings,
            self.presentation_units_per_official_serving,
        )
        displayed_amount = presentation_amount
        approximate = False
        if self.display_rounding_increment is not None:
            with localcontext() as context:
                context.prec = _DECIMAL_PRECISION
                context.rounding = ROUND_HALF_EVEN
                displayed_amount = (
                    presentation_amount / self.display_rounding_increment
                ).to_integral_value() * self.display_rounding_increment
            if displayed_amount <= 0:
                displayed_amount = presentation_amount
            approximate = displayed_amount != presentation_amount
        presentation_unit = (
            self.presentation_unit_singular
            if displayed_amount == Decimal("1")
            else self.presentation_unit_plural
        )
        prefix = "about " if approximate else ""
        natural_quantity_text = (
            f"{prefix}{_format_decimal(displayed_amount)} {presentation_unit}"
        )
        whole_item_amount: Decimal | None = None
        if self.whole_items_per_official_serving is not None:
            whole_item_amount = _multiply(
                official_servings,
                self.whole_items_per_official_serving,
            )
            if whole_item_amount >= self.whole_equivalent_minimum:
                assert self.whole_item_singular is not None
                assert self.whole_item_plural is not None
                whole_item = (
                    self.whole_item_singular
                    if whole_item_amount == Decimal("1")
                    else self.whole_item_plural
                )
                natural_quantity_text += (
                    f" ({_format_decimal(whole_item_amount)} {whole_item})"
                )
        return CalibratedServingPresentation(
            source_identifier=self.source_identifier,
            official_servings=official_servings,
            presentation_amount=presentation_amount,
            natural_quantity_text=natural_quantity_text,
            display_food_name=self.display_food_name,
            whole_item_amount=whole_item_amount,
        )

    def official_servings_for_phrase(self, phrase: str) -> Decimal | None:
        """Map only a configured presentation phrase back to official servings."""

        if not isinstance(phrase, str) or not phrase.strip():
            return None
        presentation_amount = _amount_before_alias(
            phrase,
            _normalized_aliases(self.presentation_aliases, "presentation_aliases"),
        )
        if presentation_amount is not None:
            return _divide(
                presentation_amount,
                self.presentation_units_per_official_serving,
            )
        if self.whole_items_per_official_serving is None:
            return None
        whole_item_amount = _amount_before_alias(
            phrase,
            _normalized_aliases(self.whole_item_aliases, "whole_item_aliases"),
        )
        if whole_item_amount is None:
            return None
        return _divide(whole_item_amount, self.whole_items_per_official_serving)


@dataclass(frozen=True, slots=True)
class PresentationCalibrationRegistry:
    """Immutable lookup of calibrated Phelps food presentations."""

    calibrations: tuple[ServingPresentationCalibration, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.calibrations, tuple):
            raise TypeError("calibrations must be a tuple")
        if not all(
            isinstance(value, ServingPresentationCalibration)
            for value in self.calibrations
        ):
            raise TypeError("calibrations must contain ServingPresentationCalibration values")
        versions = [(value.calibration_id, value.calibration_version) for value in self.calibrations]
        if len(set(versions)) != len(versions):
            raise ValueError("calibrations must not repeat a versioned calibration identity")
        identifiers = [value.source_identifier for value in self.calibrations]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("calibrations must not repeat a source identity")

    def find(
        self,
        source_identifier: SourceIdentifier | None,
        serving: Serving,
    ) -> ServingPresentationCalibration | None:
        if source_identifier is None or not isinstance(source_identifier, SourceIdentifier):
            return None
        if not isinstance(serving, Serving):
            raise TypeError("serving must be a Serving")
        for calibration in self.calibrations:
            if calibration.matches(source_identifier, serving):
                return calibration
        return None


# Operator-maintained Phelps serving-line observation, 2026-09-03:
# the FD component is an official 1 Each serving, physically offered as four
# quarter-potato pieces.  This is intentionally keyed by the component ID and
# guarded by its serving definition rather than by the display name.
DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS = PresentationCalibrationRegistry(
    (
        ServingPresentationCalibration(
            calibration_id="phelps-baked-sweet-potato-quarter",
            calibration_version=1,
            provenance="Operator observation at Phelps, 2026-09-03: 1 Each = 4 quarter pieces = 1 whole sweet potato",
            source_identifier=SourceIdentifier("component", "7:181:62889"),
            expected_serving_quantity=Decimal("1"),
            expected_serving_unit="Each",
            display_food_name="Baked Sweet Potatoes",
            presentation_units_per_official_serving=Decimal("4"),
            presentation_unit_singular="sweet potato quarter piece",
            presentation_unit_plural="sweet potato quarter pieces",
            presentation_aliases=(
                "quarter piece",
                "quarter pieces",
                "sweet potato quarter piece",
                "sweet potato quarter pieces",
            ),
            whole_items_per_official_serving=Decimal("1"),
            whole_item_singular="whole sweet potato",
            whole_item_plural="whole sweet potatoes",
            whole_item_aliases=(
                "sweet potato",
                "sweet potatoes",
                "whole sweet potato",
                "whole sweet potatoes",
            ),
        ),
    )
)
