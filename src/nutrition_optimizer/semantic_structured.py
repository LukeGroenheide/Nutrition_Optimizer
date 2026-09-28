"""Shared structured semantic validation for model-backed interpreters."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, ValidationError

from .semantic import (
    RecordIntakeIntent,
    SemanticInterpretation,
    UnsupportedIntent,
)


__all__ = [
    "SEMANTIC_SYSTEM_PROMPT",
    "SemanticStructuredValidationError",
    "StructuredSemanticResult",
    "interpret_structured_result",
]


SEMANTIC_SYSTEM_PROMPT = (
    "You are a narrow semantic interpreter for a nutrition application. "
    "Return only the structured result for one of two intents: "
    "record_intake or unsupported. Use record_intake only when the user "
    "states that they ate, had, or consumed a food. Copy or lightly "
    "normalize the food reference without resolving it. Set servings to a "
    "decimal numeral string only when a quantity is explicitly stated or "
    "clearly expressed by the user; normalize written numbers such as "
    "'two' to '2'. Otherwise set servings to null and "
    "clarification_required to true. Never infer a default quantity. Do not "
    "infer nutrition facts, calories, macros, menu items, or any other "
    "unstated information. Use unsupported for unrelated or unsupported "
    "requests, and leave unrelated fields null."
)


class SemanticStructuredValidationError(ValueError):
    """Raised when structured model output cannot become a semantic result."""


class StructuredSemanticResult(BaseModel):
    """Strict structured result shared by model-backed interpreters."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal["record_intake", "unsupported"]
    food_text: StrictStr | None
    servings: StrictStr | None
    clarification_required: StrictBool
    reason: StrictStr | None


def interpret_structured_result(payload: object) -> SemanticInterpretation:
    """Validate model fields and convert them to the public semantic types."""

    try:
        structured = (
            payload
            if isinstance(payload, StructuredSemanticResult)
            else StructuredSemanticResult.model_validate(payload)
        )
    except (TypeError, ValidationError):
        raise SemanticStructuredValidationError(
            "returned unusable structured output"
        ) from None

    if structured.intent == "unsupported":
        if (
            not structured.reason
            or not structured.reason.strip()
            or structured.food_text is not None
            or structured.servings is not None
            or structured.clarification_required
        ):
            raise SemanticStructuredValidationError(
                "returned an invalid unsupported result"
            )
        return UnsupportedIntent(reason=structured.reason.strip())

    if (
        not structured.food_text
        or structured.reason is not None
        or not structured.food_text.strip()
    ):
        raise SemanticStructuredValidationError("returned an invalid intake result")

    food_text = structured.food_text.strip()
    if structured.servings is None:
        if not structured.clarification_required:
            raise SemanticStructuredValidationError(
                "left intake quantity unresolved without clarification"
            )
        return RecordIntakeIntent(
            food_text=food_text,
            servings=None,
            clarification_required="servings_required",
        )

    if structured.clarification_required:
        raise SemanticStructuredValidationError(
            "marked a resolved intake quantity for clarification"
        )

    servings = _decimal_servings(structured.servings)
    try:
        return RecordIntakeIntent(food_text=food_text, servings=servings)
    except (TypeError, ValueError):
        raise SemanticStructuredValidationError(
            "returned an invalid intake result"
        ) from None


def _decimal_servings(value: object) -> Decimal:
    """Convert only exact, finite textual or integer quantities to Decimal."""

    if isinstance(value, bool):
        raise SemanticStructuredValidationError("returned an invalid quantity")
    if isinstance(value, Decimal):
        servings = value
    elif isinstance(value, int):
        servings = Decimal(value)
    elif isinstance(value, str):
        quantity_text = value.strip()
        if not quantity_text:
            raise SemanticStructuredValidationError("returned an invalid quantity")
        try:
            servings = Decimal(quantity_text)
        except InvalidOperation:
            raise SemanticStructuredValidationError(
                "returned an invalid quantity"
            ) from None
    else:
        raise SemanticStructuredValidationError("returned an invalid quantity")

    if not servings.is_finite() or servings <= 0:
        raise SemanticStructuredValidationError("returned an invalid quantity")
    return servings
