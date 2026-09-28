"""Strict structured-output contract for recommendation presentation text."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

from .recommendation_rendering import RecommendedPortionSemanticResult


__all__ = [
    "RECOMMENDED_PORTION_RENDERING_SCHEMA",
    "RECOMMENDED_PORTION_RENDERING_SYSTEM_PROMPT",
    "RecommendedPortionRenderingStructuredValidationError",
    "StructuredRecommendedPortionRendering",
    "interpret_recommended_portion_structured_result",
]


RECOMMENDED_PORTION_RENDERING_SYSTEM_PROMPT = (
    "You select an optional practical serving-line recognition phrase for one "
    "already-computed physical quantity from a nutrition optimizer. Python has "
    "already calculated the canonical amount, retains it as authoritative, and "
    "will add the official food name itself. Python also supplies zero or more "
    "approved_descriptive_options. Return natural_descriptor as exactly one supplied "
    "option, unchanged, or return null. Never invent an option. Do not calculate, "
    "round, convert, combine, replace, or restate the canonical amount. Do not name "
    "or rename the food. An approved option is visual guidance, not an asserted "
    "conversion. Do not put a food name, "
    "a full sentence, canonical weight/volume measurement, official serving, "
    "nutrition, target, score, ID, or any claim of exact equivalence in the "
    "descriptor. If no supplied option is useful, return null. "
    "Mark presentation_practicality practical only for a descriptor that improves "
    "real serving-line actionability; otherwise use awkward or unclear."
)


RECOMMENDED_PORTION_RENDERING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "natural_descriptor": {"type": ["string", "null"], "maxLength": 96},
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "presentation_practicality": {
            "type": "string",
            "enum": ["practical", "awkward", "unclear"],
        },
    },
    "required": [
        "natural_descriptor",
        "confidence",
        "presentation_practicality",
    ],
    "additionalProperties": False,
}


class RecommendedPortionRenderingStructuredValidationError(ValueError):
    """Raised when model output cannot become safe presentation text."""


class StructuredRecommendedPortionRendering(BaseModel):
    """The exact JSON shape accepted from the OpenClaw structured task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    natural_descriptor: StrictStr | None
    confidence: Literal["high", "medium", "low"]
    presentation_practicality: Literal["practical", "awkward", "unclear"]


def interpret_recommended_portion_structured_result(
    payload: object,
) -> RecommendedPortionSemanticResult:
    """Validate model fields without introducing any authoritative quantity."""

    try:
        structured = (
            payload
            if isinstance(payload, StructuredRecommendedPortionRendering)
            else StructuredRecommendedPortionRendering.model_validate(payload)
        )
    except (TypeError, ValidationError):
        raise RecommendedPortionRenderingStructuredValidationError(
            "returned unusable structured output"
        ) from None
    natural_descriptor = (
        structured.natural_descriptor.strip()
        if structured.natural_descriptor is not None
        else None
    )
    if structured.natural_descriptor is not None and not natural_descriptor:
        raise RecommendedPortionRenderingStructuredValidationError(
            "returned an empty natural descriptor"
        )
    try:
        return RecommendedPortionSemanticResult(
            natural_descriptor=natural_descriptor,
            confidence=structured.confidence,
            presentation_practicality=structured.presentation_practicality,
        )
    except (TypeError, ValueError):
        raise RecommendedPortionRenderingStructuredValidationError(
            "returned unusable structured output"
        ) from None
