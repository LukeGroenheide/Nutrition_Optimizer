"""Strict structured-output contract for official-serving portion estimates."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

from .portion_interpretation import (
    PortionSemanticDecision,
    parse_official_serving_multiplier,
)


__all__ = [
    "PORTION_INTERPRETATION_SYSTEM_PROMPT",
    "PORTION_INTERPRETATION_STRUCTURED_SCHEMA",
    "PortionInterpretationStructuredValidationError",
    "StructuredPortionDecision",
    "interpret_portion_structured_result",
]


PORTION_INTERPRETATION_SYSTEM_PROMPT = (
    "You estimate only the physical amount described by a user's portion phrase "
    "relative to one already-resolved official FDMealPlanner serving. The supplied "
    "food and serving definition are fixed: never change the food, component "
    "identity, serving definition, or serving unit. Express an estimate only as a "
    "multiplier of that exact official serving. Do not estimate nutrition, calories, "
    "macros, daily targets, intake totals, or recommendations. Prefer ambiguous or "
    "no_estimate over unjustified precision. Ordinary container or utensil wording "
    "can receive a medium- or low-confidence estimate when the official serving "
    "has a comparable physical unit; uncertainty alone does not require ambiguity. "
    "Use ambiguity when no reasonable physical comparison is available. Confidence "
    "must reflect physical uncertainty. Do not explain private reasoning; reason must be a brief "
    "user-facing label."
)


PORTION_INTERPRETATION_STRUCTURED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["estimate", "ambiguous", "no_estimate"],
        },
        "estimated_official_servings": {"type": ["string", "null"]},
        "confidence": {
            "type": ["string", "null"],
            "enum": ["high", "medium", "low", None],
        },
        "reason": {"type": "string"},
    },
    "required": [
        "decision",
        "estimated_official_servings",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}


class PortionInterpretationStructuredValidationError(ValueError):
    """Raised when model output cannot safely become a portion decision."""


class StructuredPortionDecision(BaseModel):
    """The exact JSON shape accepted from the OpenClaw structured task."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["estimate", "ambiguous", "no_estimate"]
    estimated_official_servings: StrictStr | None
    confidence: Literal["high", "medium", "low"] | None
    reason: StrictStr


def interpret_portion_structured_result(payload: object) -> PortionSemanticDecision:
    """Validate and Decimal-parse model output without doing nutrition math."""

    try:
        structured = (
            payload
            if isinstance(payload, StructuredPortionDecision)
            else StructuredPortionDecision.model_validate(payload)
        )
    except (TypeError, ValidationError):
        raise PortionInterpretationStructuredValidationError(
            "returned unusable structured output"
        ) from None

    reason = structured.reason.strip()
    if not reason:
        raise PortionInterpretationStructuredValidationError("returned an empty reason")
    if structured.decision == "estimate":
        if structured.estimated_official_servings is None:
            raise PortionInterpretationStructuredValidationError(
                "returned an estimate without official servings"
            )
        if structured.confidence is None:
            raise PortionInterpretationStructuredValidationError(
                "returned an estimate without confidence"
            )
        try:
            multiplier = parse_official_serving_multiplier(
                structured.estimated_official_servings
            )
        except (TypeError, ValueError):
            raise PortionInterpretationStructuredValidationError(
                "returned an invalid official-serving estimate"
            ) from None
        return PortionSemanticDecision(
            decision="estimate",
            estimated_official_servings=multiplier,
            confidence=structured.confidence,
            reason=reason,
        )

    if (
        structured.estimated_official_servings is not None
        or structured.confidence is not None
    ):
        raise PortionInterpretationStructuredValidationError(
            "returned a quantity for a non-estimate decision"
        )
    return PortionSemanticDecision(
        decision=structured.decision,
        estimated_official_servings=None,
        confidence=None,
        reason=reason,
    )
