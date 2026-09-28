"""Fail-closed interpretation of natural portions into official FD servings.

This module owns quantity semantics only.  It receives one already-resolved
official food and its immutable serving definition, but it never selects a
food, calculates nutrition, records intake, persists a result, or contacts
FDMealPlanner.  A later application boundary can deliberately compose a
``ResolvedFood`` with an ``InterpretedPortion`` before any deterministic
nutrition arithmetic occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
import re
from typing import Literal, Protocol, TypeAlias

from .food_resolution import ResolvedFood
from .presentation_binding import PresentationBinding


__all__ = [
    "AmbiguousPortion",
    "InterpretedPortion",
    "MAX_OFFICIAL_SERVING_MULTIPLIER",
    "NaturalPortionInterpreter",
    "PortionInterpretationError",
    "PortionInterpretationRequest",
    "PortionInterpretationResult",
    "PortionSemanticDecision",
    "PortionSemanticGatewayUnavailableError",
    "PortionSemanticMatcher",
    "PortionSemanticOutputError",
    "PortionSemanticRuntimeUnavailableError",
    "PortionSemanticTimeoutError",
    "PortionSemanticTransportError",
    "UnresolvedPortion",
    "parse_official_serving_multiplier",
]


# A phrase describing one eaten portion should never silently create an
# unbounded intake. Twelve official servings is deliberately generous (for
# example, 12 cups or 12 each) while treating larger claims as clarification
# territory rather than a plausible automatic estimate.
MAX_OFFICIAL_SERVING_MULTIPLIER = Decimal("12")

PortionConfidence = Literal["high", "medium", "low"]
PortionInterpretationMethod = Literal[
    "calibrated_presentation",
    "explicit_official_servings",
    "count_based_each",
    "semantic",
]
PortionSemanticDecisionKind = Literal["estimate", "ambiguous", "no_estimate"]


class PortionInterpretationError(RuntimeError):
    """Raised when the independent portion boundary cannot operate safely."""


class PortionSemanticTransportError(PortionInterpretationError):
    """A sanitized failure while invoking an optional portion matcher."""


class PortionSemanticTimeoutError(PortionSemanticTransportError):
    """The optional portion matcher exceeded its finite timeout."""


class PortionSemanticGatewayUnavailableError(PortionSemanticTransportError):
    """The optional local semantic gateway was clearly unavailable."""


class PortionSemanticRuntimeUnavailableError(PortionInterpretationError):
    """The configured optional portion runtime could not be started."""


class PortionSemanticOutputError(PortionInterpretationError):
    """The optional portion matcher returned unusable structured output."""


@dataclass(frozen=True, slots=True)
class PortionInterpretationRequest:
    """One natural quantity phrase applied to one exact resolved food."""

    resolved_food: ResolvedFood
    original_quantity_text: str
    presentation_binding: PresentationBinding | None = None
    planned_official_servings: Decimal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.resolved_food, ResolvedFood):
            raise TypeError("resolved_food must be a ResolvedFood")
        _require_text(self.original_quantity_text, "original_quantity_text")


@dataclass(frozen=True, slots=True)
class PortionSemanticDecision:
    """A strict semantic decision about quantity, never food identity."""

    decision: PortionSemanticDecisionKind
    estimated_official_servings: Decimal | None
    confidence: PortionConfidence | None
    reason: str

    def __post_init__(self) -> None:
        if self.decision not in {"estimate", "ambiguous", "no_estimate"}:
            raise ValueError("decision must be estimate, ambiguous, or no_estimate")
        _require_text(self.reason, "reason")
        if self.decision == "estimate":
            _validate_multiplier(self.estimated_official_servings)
            if self.confidence not in {"high", "medium", "low"}:
                raise ValueError("estimate confidence must be high, medium, or low")
        elif self.estimated_official_servings is not None or self.confidence is not None:
            raise ValueError("non-estimate decisions cannot include an estimate or confidence")


class PortionSemanticMatcher(Protocol):
    """Narrow semantic boundary for one immutable official serving context."""

    def decide(self, request: PortionInterpretationRequest) -> PortionSemanticDecision:
        """Return a quantity decision without changing the resolved food."""


@dataclass(frozen=True, slots=True)
class InterpretedPortion:
    """An estimated multiplier of the resolved food's official serving."""

    original_quantity_text: str
    estimated_official_servings: Decimal
    confidence: PortionConfidence
    interpretation_method: PortionInterpretationMethod
    reason: str

    def __post_init__(self) -> None:
        _require_text(self.original_quantity_text, "original_quantity_text")
        _validate_multiplier(self.estimated_official_servings)
        if self.confidence not in {"high", "medium", "low"}:
            raise ValueError("confidence must be high, medium, or low")
        if self.interpretation_method not in {
            "calibrated_presentation",
            "explicit_official_servings",
            "count_based_each",
            "semantic",
        }:
            raise ValueError("interpretation_method is invalid")
        _require_text(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class AmbiguousPortion:
    """A quantity phrase with multiple materially plausible interpretations."""

    original_quantity_text: str
    reason: str

    def __post_init__(self) -> None:
        _require_text(self.original_quantity_text, "original_quantity_text")
        _require_text(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class UnresolvedPortion:
    """A fail-closed quantity outcome that contains no serving multiplier."""

    original_quantity_text: str
    reason: str

    def __post_init__(self) -> None:
        _require_text(self.original_quantity_text, "original_quantity_text")
        _require_text(self.reason, "reason")


PortionInterpretationResult: TypeAlias = (
    InterpretedPortion | AmbiguousPortion | UnresolvedPortion
)


class NaturalPortionInterpreter:
    """Interpret natural quantity language relative to one official serving.

    Safe direct statements are resolved locally.  All physical-size-dependent
    wording (bowls, scoops, pieces against weights, and vague amounts) remains
    optional semantic work and fails closed if that runtime is unavailable.
    """

    def __init__(self, *, semantic_matcher: PortionSemanticMatcher | None = None) -> None:
        if semantic_matcher is not None and not callable(
            getattr(semantic_matcher, "decide", None)
        ):
            raise TypeError("semantic_matcher must provide decide")
        self._semantic_matcher = semantic_matcher

    def interpret(
        self,
        request: PortionInterpretationRequest,
    ) -> PortionInterpretationResult:
        """Estimate only official-serving quantity; never nutrition or intake."""

        if not isinstance(request, PortionInterpretationRequest):
            raise TypeError("request must be a PortionInterpretationRequest")

        deterministic = _deterministic_portion(request)
        if deterministic is not None:
            return deterministic

        if self._semantic_matcher is None:
            return UnresolvedPortion(
                original_quantity_text=request.original_quantity_text,
                reason="portion_semantic_matcher_unavailable",
            )
        return self._interpret_semantically(request)

    def _interpret_semantically(
        self,
        request: PortionInterpretationRequest,
    ) -> PortionInterpretationResult:
        try:
            decision = self._semantic_matcher.decide(request)
        except PortionSemanticTimeoutError:
            return _unresolved(request, "portion_semantic_timeout")
        except PortionSemanticGatewayUnavailableError:
            return _unresolved(request, "portion_semantic_gateway_unavailable")
        except PortionSemanticOutputError:
            return _unresolved(request, "portion_semantic_output_invalid")
        except PortionSemanticRuntimeUnavailableError:
            return _unresolved(request, "portion_semantic_runtime_unavailable")
        except PortionSemanticTransportError:
            return _unresolved(request, "portion_semantic_transport_failed")
        except Exception:
            return _unresolved(request, "portion_semantic_failed")

        if not isinstance(decision, PortionSemanticDecision):
            return _unresolved(request, "portion_semantic_response_invalid")
        if decision.decision == "no_estimate":
            return _unresolved(request, "portion_semantic_no_estimate")
        if decision.decision == "ambiguous":
            return AmbiguousPortion(
                original_quantity_text=request.original_quantity_text,
                reason="portion_semantic_ambiguous",
            )

        return InterpretedPortion(
            original_quantity_text=request.original_quantity_text,
            estimated_official_servings=decision.estimated_official_servings,
            confidence=decision.confidence,
            interpretation_method="semantic",
            reason=decision.reason,
        )


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
}
_DECIMAL_TEXT = re.compile(r"^\d+(?:\.\d+)?$")
_EXPLICIT_SERVINGS = re.compile(
    r"^(?P<quantity>[a-z]+|\d+(?:\.\d+)?)\s+(?:official\s+)?servings?$",
    re.IGNORECASE,
)
_HALF_SERVING = re.compile(
    r"^(?:a\s+)?half(?:\s+of)?\s+(?:a\s+|one\s+)?(?:official\s+)?serving$",
    re.IGNORECASE,
)
_COUNT_BASED_EACH = re.compile(
    r"^(?P<quantity>[a-z]+|\d+(?:\.\d+)?)(?:\s+(?:each|ea|pieces?|items?))?$",
    re.IGNORECASE,
)
_EXPLICIT_PHYSICAL_PREFIX = re.compile(
    r"^(?P<quantity>(?:\d+(?:\.\d+)?)|(?:\.\d+))\s*"
    r"(?P<unit>oz\.?|ounces?|ounce(?:\s+cooked(?:\s+weight)?)?)\b",
    re.IGNORECASE,
)


def parse_official_serving_multiplier(value: str) -> Decimal:
    """Parse one model decimal string into a bounded authoritative Decimal.

    Scientific notation, non-finite values, signs, and imprecise Python float
    values are intentionally not accepted.  The returned Decimal is a serving
    multiplier only; this function performs no nutrition arithmetic.
    """

    if not isinstance(value, str):
        raise ValueError("estimated official servings must be a decimal string")
    normalized = value.strip()
    if not _DECIMAL_TEXT.fullmatch(normalized):
        raise ValueError("estimated official servings must be a plain decimal string")
    try:
        multiplier = Decimal(normalized)
    except (InvalidOperation, ValueError):
        raise ValueError("estimated official servings must be a valid decimal") from None
    _validate_multiplier(multiplier)
    return multiplier


def _deterministic_portion(
    request: PortionInterpretationRequest,
) -> PortionInterpretationResult | None:
    normalized = _normalized_quantity(request.original_quantity_text)
    explicit = _EXPLICIT_SERVINGS.fullmatch(normalized)
    if explicit is not None:
        return _deterministic_multiplier_result(
            request,
            _number_token_to_decimal(explicit.group("quantity")),
            method="explicit_official_servings",
            reason="explicit_official_servings",
        )
    if _HALF_SERVING.fullmatch(normalized) is not None:
        return _deterministic_multiplier_result(
            request,
            Decimal("0.5"),
            method="explicit_official_servings",
            reason="explicit_half_official_serving",
        )

    # Only the exact frozen plan item may authorize a visual conversion.
    if request.presentation_binding is not None:
        try:
            calibrated_multiplier = request.presentation_binding.official_servings_for_phrase(
                request.resolved_food, request.planned_official_servings, normalized)
        except (TypeError, ValueError):
            return _unresolved(request, "invalid_presentation_binding")
        if calibrated_multiplier is not None:
            return _deterministic_multiplier_result(request, calibrated_multiplier,
                method="calibrated_presentation", reason="frozen_presentation_binding")

    physical = _explicit_matching_physical_amount(request, normalized)
    if physical is not None:
        return physical

    if not _is_one_each_serving(request.resolved_food):
        return None
    if normalized == "half of one":
        return _deterministic_multiplier_result(
            request,
            Decimal("0.5"),
            method="count_based_each",
            reason="explicit_half_of_one_each",
        )
    count = _COUNT_BASED_EACH.fullmatch(normalized)
    if count is None:
        return None
    return _deterministic_multiplier_result(
        request,
        _number_token_to_decimal(count.group("quantity")),
        method="count_based_each",
        reason="explicit_count_for_one_each_serving",
    )


def _deterministic_multiplier_result(
    request: PortionInterpretationRequest,
    multiplier: Decimal | None,
    *,
    method: PortionInterpretationMethod,
    reason: str,
) -> PortionInterpretationResult:
    if multiplier is None:
        return _unresolved(request, "invalid_explicit_official_servings")
    try:
        return InterpretedPortion(
            original_quantity_text=request.original_quantity_text,
            estimated_official_servings=multiplier,
            confidence="high",
            interpretation_method=method,
            reason=reason,
        )
    except (TypeError, ValueError):
        return _unresolved(request, "invalid_explicit_official_servings")


def _number_token_to_decimal(value: str) -> Decimal | None:
    normalized = value.casefold()
    if normalized in _NUMBER_WORDS:
        return _NUMBER_WORDS[normalized]
    if not _DECIMAL_TEXT.fullmatch(normalized):
        return None
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def _is_one_each_serving(resolved_food: ResolvedFood) -> bool:
    serving = resolved_food.nutrition_record.serving
    if serving.quantity != Decimal("1"):
        return False
    return serving.unit is not None and serving.unit.strip().casefold() in {"each", "ea"}


def _explicit_matching_physical_amount(
    request: PortionInterpretationRequest,
    normalized: str,
) -> PortionInterpretationResult | None:
    """Resolve an explicitly stated amount in the food's exact official unit.

    This is intentionally much narrower than physical-serving calibration.  It
    accepts only a standard ounce spelling attached to a food whose provider
    serving is already an ounce (including ``ounce cooked weight``), then uses
    exact Decimal division against that one authoritative serving definition.
    No bowl, scoop, ladle, spoon, or cross-unit conversion is introduced.
    """

    match = _EXPLICIT_PHYSICAL_PREFIX.match(normalized)
    if match is None:
        return None
    serving = request.resolved_food.nutrition_record.serving
    if serving.quantity is None or serving.unit is None:
        return None
    if _canonical_ounce_unit(serving.unit) is None:
        return None
    try:
        amount = Decimal(match.group("quantity"))
        if not amount.is_finite() or amount <= 0:
            return _unresolved(request, "invalid_explicit_official_servings")
        with localcontext() as context:
            context.prec = 36
            multiplier = amount / serving.quantity
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return _unresolved(request, "invalid_explicit_official_servings")
    return _deterministic_multiplier_result(
        request,
        multiplier,
        method="explicit_official_servings",
        reason="explicit_matching_official_ounce_unit",
    )


def _canonical_ounce_unit(value: str) -> str | None:
    normalized = _normalized_quantity(value)
    if normalized in {"ounce", "ounces", "oz", "oz."}:
        return "ounce"
    if normalized in {
        "ounce cooked weight",
        "ounces cooked weight",
        "ounce cooked",
        "ounces cooked",
        "oz cooked",
        "oz cooked weight",
    }:
        return "ounce_cooked_weight"
    return None


def _normalized_quantity(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _validate_multiplier(value: Decimal | None) -> None:
    if not isinstance(value, Decimal):
        raise TypeError("estimated_official_servings must be a Decimal")
    if not value.is_finite() or value <= 0:
        raise ValueError("estimated_official_servings must be finite and greater than zero")
    if value > MAX_OFFICIAL_SERVING_MULTIPLIER:
        raise ValueError(
            "estimated_official_servings exceeds the conservative automatic limit"
        )


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")


def _unresolved(
    request: PortionInterpretationRequest,
    reason: str,
) -> UnresolvedPortion:
    return UnresolvedPortion(
        original_quantity_text=request.original_quantity_text,
        reason=reason,
    )
