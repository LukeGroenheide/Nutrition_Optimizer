"""Transport-independent semantic interpretation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import re
from typing import Protocol


__all__ = [
    "DeterministicSemanticInterpreter",
    "RecordIntakeIntent",
    "SemanticInterpretation",
    "SemanticInterpretationError",
    "SemanticInterpreter",
    "UnsupportedIntent",
    "interpret_text",
]


@dataclass(frozen=True, slots=True)
class RecordIntakeIntent:
    """A semantic request to record a food reference and known quantity."""

    food_text: str
    servings: Decimal | None = None
    clarification_required: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.food_text, str) or not self.food_text.strip():
            raise ValueError("food_text must be non-empty text")
        if self.servings is not None:
            if not isinstance(self.servings, Decimal):
                raise TypeError("servings must be a Decimal or None")
            if not self.servings.is_finite() or self.servings <= 0:
                raise ValueError("servings must be finite and greater than zero")
        if self.clarification_required is not None and (
            not isinstance(self.clarification_required, str)
            or not self.clarification_required.strip()
        ):
            raise ValueError("clarification_required must be non-empty when provided")
        if self.servings is not None and self.clarification_required is not None:
            raise ValueError(
                "clarification_required cannot be set when servings are resolved"
            )


@dataclass(frozen=True, slots=True)
class UnsupportedIntent:
    """An explicit result for a request outside the supported intent set."""

    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be non-empty text")


SemanticInterpretation = RecordIntakeIntent | UnsupportedIntent


class SemanticInterpreter(Protocol):
    """Interface implemented by deterministic or model-backed interpreters."""

    def interpret(self, user_text: str) -> SemanticInterpretation:
        """Translate user text into a validated structured interpretation."""


class SemanticInterpretationError(RuntimeError):
    """Raised when an interpreter is unavailable or returns an invalid result."""


def interpret_text(
    interpreter: SemanticInterpreter,
    user_text: str,
) -> SemanticInterpretation:
    """Safely invoke an interpreter and validate its structured result."""

    if not isinstance(user_text, str) or not user_text.strip():
        raise SemanticInterpretationError("user text must be non-empty")

    interpret = getattr(interpreter, "interpret", None)
    if not callable(interpret):
        raise SemanticInterpretationError("semantic interpreter is invalid")

    try:
        result = interpret(user_text)
    except Exception:
        # Do not expose model errors, credentials, or user text to callers.
        raise SemanticInterpretationError("semantic interpretation failed") from None

    if not isinstance(result, (RecordIntakeIntent, UnsupportedIntent)):
        raise SemanticInterpretationError("semantic interpreter returned invalid result")
    return result


class DeterministicSemanticInterpreter:
    """A deliberately narrow, non-nutrition-aware interpreter for smoke tests."""

    _INTAKE_PREFIX = re.compile(
        r"^\s*(?:i\s+)?(?:ate|had|consumed)\s+(?P<body>.+?)\s*[.!?]?\s*$",
        re.IGNORECASE,
    )
    _EXPLICIT_QUANTITY = re.compile(
        r"^(?P<quantity>\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|"
        r"eight|nine|ten)\s+servings?\s+(?:of\s+)?(?P<food>.+)$",
        re.IGNORECASE,
    )
    _WORD_QUANTITIES = {
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
    }

    def interpret(self, user_text: str) -> SemanticInterpretation:
        """Interpret only explicit first-person intake statements."""

        if not isinstance(user_text, str) or not user_text.strip():
            return UnsupportedIntent(reason="empty_request")

        prefix_match = self._INTAKE_PREFIX.match(user_text)
        if prefix_match is None:
            return UnsupportedIntent(reason="unsupported_request")

        body = _clean_food_text(prefix_match.group("body"))
        if not body:
            return UnsupportedIntent(reason="missing_food_reference")

        quantity_match = self._EXPLICIT_QUANTITY.match(body)
        if quantity_match is None:
            return RecordIntakeIntent(
                food_text=body,
                servings=None,
                clarification_required="servings_required",
            )

        quantity_text = quantity_match.group("quantity").lower()
        servings = self._WORD_QUANTITIES.get(quantity_text)
        if servings is None:
            servings = Decimal(quantity_text)
        food_text = _clean_food_text(quantity_match.group("food"))
        if not food_text:
            return UnsupportedIntent(reason="missing_food_reference")
        return RecordIntakeIntent(food_text=food_text, servings=servings)


def _clean_food_text(value: str) -> str:
    return value.strip().rstrip(".!?").strip()
