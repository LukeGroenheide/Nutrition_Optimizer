"""Deterministic representations for one user-initiated meal request.

The semantic layer may preserve a user's food wording, but this module holds
only the authoritative local-FD identity selected by Python.  It intentionally
contains no nutrition values, quantities, preference profile, or model output.
"""

from __future__ import annotations

from dataclasses import dataclass

from .food_resolution import ResolvedFood


__all__ = [
    "RequestedMealFood",
    "requested_meal_food_from_resolved",
]


@dataclass(frozen=True, slots=True)
class RequestedMealFood:
    """One uniquely resolved current-menu food requested for one meal only.

    ``content_signature`` is retained alongside the stable provider component
    identity.  A later menu refresh therefore cannot silently turn a pending
    request into a changed nutrition version of a similarly named food.
    """

    food_text: str
    source_kind: str
    source_value: str
    content_signature: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.food_text, "food_text"),
            (self.source_kind, "source_kind"),
            (self.source_value, "source_value"),
            (self.content_signature, "content_signature"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")

    @property
    def source_identity(self) -> tuple[str, str]:
        """Return the stable provider-backed identity used by exclusions."""

        return (self.source_kind, self.source_value)

    @property
    def identity_with_signature(self) -> tuple[str, str, str]:
        """Return the exact current-menu identity required by the optimizer."""

        return (self.source_kind, self.source_value, self.content_signature)


def requested_meal_food_from_resolved(food: ResolvedFood) -> RequestedMealFood:
    """Project a locally resolved food into durable request-safe identity data."""

    if not isinstance(food, ResolvedFood):
        raise TypeError("food must be a ResolvedFood")
    return RequestedMealFood(
        food.original_food_text,
        food.source_identifier.kind,
        food.source_identifier.value,
        food.content_signature,
    )
