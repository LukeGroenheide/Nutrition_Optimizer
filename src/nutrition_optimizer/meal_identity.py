"""Canonical meal identity shared by deterministic application boundaries."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal


MealSlot = Literal["breakfast", "lunch", "dinner", "brunch"]
MEAL_SLOTS: tuple[MealSlot, ...] = ("breakfast", "lunch", "dinner", "brunch")

MEAL_PERIOD_IDS: dict[str, int] = {
    "breakfast": 1,
    "lunch": 2,
    "dinner": 3,
}


def canonical_meal_slot(value: object) -> MealSlot | None:
    """Return one exact product meal slot without consulting provider identity."""

    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    return normalized if normalized in MEAL_SLOTS else None  # type: ignore[return-value]


def require_meal_slot(value: object) -> MealSlot:
    """Validate and return one canonical persisted product meal slot."""

    slot = canonical_meal_slot(value)
    if slot is None:
        raise ValueError("meal_slot must be breakfast, lunch, dinner, or brunch")
    return slot


def meal_slot_for_provider_meal(value: object) -> MealSlot:
    """Return the ordinary product slot for a known provider meal period.

    Provider period 2 means lunch at this boundary.  Callers with independent
    product knowledge, notably the Sunday scheduler opportunity, must supply
    ``brunch`` explicitly rather than asking provider identity to infer it.
    """

    name = canonical_meal_name(value)
    if name is None:
        raise ValueError("provider meal does not have a default product meal slot")
    return require_meal_slot(name)


def legacy_meal_slot(service_date: date, provider_meal: object) -> MealSlot:
    """Deterministically classify a pre-v10 persisted plan or request.

    The legacy schema retained the Detroit service date and provider meal but
    not the product slot.  The product has no Sunday lunch opportunity: its
    Sunday provider-period-2 opportunity is brunch.  This date/period mapping
    is therefore stable historical product identity, not a service-window or
    current-availability inference.
    """

    if not isinstance(service_date, date) or isinstance(service_date, datetime):
        raise TypeError("service_date must be a date")
    explicit_slot = canonical_meal_slot(provider_meal)
    if explicit_slot == "brunch":
        return explicit_slot
    meal_id = canonical_meal_id(provider_meal)
    if meal_id == 2 and service_date.weekday() == 6:
        return "brunch"
    return meal_slot_for_provider_meal(provider_meal)

_MEAL_NAMES_BY_TOKEN = {
    **{name: name for name in MEAL_PERIOD_IDS},
    **{str(period_id): name for name, period_id in MEAL_PERIOD_IDS.items()},
}


def canonical_meal_name(value: object) -> str | None:
    """Return the semantic name for a known meal name or FD period ID."""

    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    return _MEAL_NAMES_BY_TOKEN.get(str(value).strip().casefold())


def canonical_meal_id(value: object) -> int | None:
    """Return the known FD period ID for a meal value, if one exists."""

    name = canonical_meal_name(value)
    return MEAL_PERIOD_IDS.get(name) if name is not None else None


def meal_identity_matches(left: object, right: object) -> bool:
    """Compare two values only when both resolve to a known meal identity."""

    left_name = canonical_meal_name(left)
    return left_name is not None and left_name == canonical_meal_name(right)


def meal_values_equal(left: object, right: object) -> bool:
    """Compare exact values or their known canonical meal identities."""

    return left == right or meal_identity_matches(left, right)


def meal_context_key(value: str | int) -> str:
    """Return a durable, deterministic key for a meal-plan service context.

    Known FD meal names and period IDs deliberately share one key, so
    ``"Breakfast"`` and ``1`` cannot become independently answerable plans.
    The fallback preserves type for the broader existing ``MealPlan`` domain,
    whose callers may retain an unknown provider-specific meal value.
    """

    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError("meal must be text or an integer")
    canonical = canonical_meal_id(value)
    if canonical is not None:
        return f"fd:{canonical}"
    if isinstance(value, int):
        return f"integer:{value}"
    normalized = value.strip().casefold()
    if not normalized:
        raise ValueError("meal must be non-empty text")
    return f"text:{normalized}"


def meal_name_for_display(value: str | int) -> str:
    """Return a human-readable known meal name, preserving unknown fallback text."""

    return canonical_meal_name(value) or str(value).strip().casefold()


__all__ = [
    "MEAL_SLOTS",
    "MEAL_PERIOD_IDS",
    "MealSlot",
    "canonical_meal_id",
    "canonical_meal_name",
    "canonical_meal_slot",
    "legacy_meal_slot",
    "meal_context_key",
    "meal_identity_matches",
    "meal_name_for_display",
    "meal_slot_for_provider_meal",
    "meal_values_equal",
    "require_meal_slot",
]
