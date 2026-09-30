"""Render exact meal recommendations into natural dining-hall language.

The optimizer and this module intentionally have different responsibilities.
``MealRecommendation`` remains the source of truth for the selected FD
occurrence and its exact ``Decimal`` official-serving multiplier.  This module
only creates presentation text and adapts that immutable result to the
existing ``MealPlan`` persistence/reconciliation boundary.

In particular, a renderer never returns a replacement serving multiplier.  A
rendering can say "2 chicken tenders" for the deterministic ratio representing
two items from a three-each official serving, while the resulting
``PlannedMealItem`` retains that same exact ratio.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import re
from typing import Literal, Protocol, TYPE_CHECKING

from .food_resolution import ResolvedFood
from .meal_optimizer import MealRecommendation, RecommendedMealItem
from .meal_identity import meal_name_for_display, meal_values_equal
from .meal_report import MealPlan, PlannedMealItem
from .nutrition.models import Serving, SourceIdentifier
from .physical_quantity import (
    PhysicalRecommendedQuantity,
    format_physical_amount,
    physical_quantity_for,
)
from .serving_presentation import DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS
from .presentation_binding import PresentationBinding, PresentationKind, freeze_presentation

if TYPE_CHECKING:
    from .durable_state import PersistedMealPlan


__all__ = [
    "MealRecommendationRendering",
    "MealRecommendationRenderingError",
    "MealRecommendationRenderingUnavailableError",
    "PlannedMealItemRendering",
    "PresentationPracticality",
    "SemanticDescriptorDisposition",
    "RecommendedPortionRenderRequest",
    "RecommendedPortionRenderer",
    "RecommendedPortionSemanticResult",
    "RecommendedPortionRendering",
    "RecommendedPortionRenderingError",
    "RecommendedPortionRenderingOutputError",
    "RecommendedPortionRenderingRuntimeUnavailableError",
    "RecommendedPortionRenderingTransportError",
    "RecommendedPortionSemanticRenderer",
    "format_meal_message",
    "meal_plan_from_recommendation",
    "persist_meal_recommendation",
    "persist_rendered_meal_recommendation",
    "render_meal_recommendation",
]


PortionConfidence = Literal["high", "medium", "low"]
PresentationPracticality = Literal["practical", "awkward", "unclear"]
RenderingMethod = Literal["deterministic", "luna"]
SemanticDescriptorDisposition = Literal[
    "not_requested",
    "deterministic_visual",
    "accepted",
    "not_useful",
    "rejected",
    "unavailable",
]


class RecommendedPortionRenderingError(RuntimeError):
    """Base error for the independent recommendation presentation boundary."""


class RecommendedPortionRenderingTransportError(RecommendedPortionRenderingError):
    """A sanitized failure while invoking an optional semantic renderer."""


class RecommendedPortionRenderingRuntimeUnavailableError(RecommendedPortionRenderingError):
    """The configured optional semantic runtime cannot be started."""


class RecommendedPortionRenderingOutputError(RecommendedPortionRenderingError):
    """The optional semantic renderer returned unusable presentation output."""


class RecommendedPortionRenderingUnavailableError(RecommendedPortionRenderingError):
    """No deterministic rendering exists and no semantic renderer was supplied."""


# A shorter name is useful at application call sites and keeps the public
# error vocabulary parallel with the existing portion boundary.
MealRecommendationRenderingError = RecommendedPortionRenderingError


@dataclass(frozen=True, slots=True)
class RecommendedPortionRenderRequest:
    """Narrow physical context for rendering one exact recommendation item.

    The canonical physical quantity is calculated before this request reaches
    any renderer.  The custom initializer accepts the former
    ``Serving + multiplier`` shape for compatibility, but immediately derives
    and stores the canonical quantity; semantic transport never receives the
    former pair.
    """

    official_food_display_name: str
    physical_quantity: PhysicalRecommendedQuantity
    station_name: str | None = None
    source_identifier: SourceIdentifier | None = None
    approved_descriptive_options: tuple[str, ...] = ()

    def __init__(
        self,
        official_food_display_name: str,
        physical_quantity: PhysicalRecommendedQuantity | Serving | None = None,
        recommended_official_servings: Decimal | None = None,
        station_name: str | None = None,
        source_identifier: SourceIdentifier | None = None,
        approved_descriptive_options: tuple[str, ...] = (),
        *,
        official_serving: Serving | None = None,
    ) -> None:
        if official_serving is not None:
            if physical_quantity is not None:
                raise TypeError("provide physical_quantity or official_serving, not both")
            physical_quantity = official_serving
        if isinstance(physical_quantity, Serving):
            if recommended_official_servings is None:
                raise TypeError(
                    "recommended_official_servings is required with official_serving"
                )
            physical_quantity = physical_quantity_for(
                physical_quantity,
                recommended_official_servings,
            )
        elif isinstance(physical_quantity, PhysicalRecommendedQuantity):
            if recommended_official_servings is not None:
                raise TypeError(
                    "recommended_official_servings is not accepted with physical_quantity"
                )
        else:
            raise TypeError("physical_quantity must be a PhysicalRecommendedQuantity")
        object.__setattr__(self, "official_food_display_name", official_food_display_name)
        object.__setattr__(self, "physical_quantity", physical_quantity)
        object.__setattr__(self, "station_name", station_name)
        object.__setattr__(self, "source_identifier", source_identifier)
        object.__setattr__(
            self,
            "approved_descriptive_options",
            approved_descriptive_options,
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        _text(self.official_food_display_name, "official_food_display_name")
        if not isinstance(self.physical_quantity, PhysicalRecommendedQuantity):
            raise TypeError("physical_quantity must be a PhysicalRecommendedQuantity")
        if self.station_name is not None:
            _text(self.station_name, "station_name")
        if self.source_identifier is not None and not isinstance(
            self.source_identifier,
            SourceIdentifier,
        ):
            raise TypeError("source_identifier must be a SourceIdentifier or None")
        if not isinstance(self.approved_descriptive_options, tuple):
            raise TypeError("approved_descriptive_options must be a tuple")
        for option in self.approved_descriptive_options:
            _text(option, "approved descriptive option")
        normalized_options = tuple(
            " ".join(option.casefold().split())
            for option in self.approved_descriptive_options
        )
        if len(set(normalized_options)) != len(normalized_options):
            raise ValueError("approved descriptive options must not repeat")

    @property
    def food_name(self) -> str:
        """Compatibility-friendly alias for the official display name."""

        return self.official_food_display_name

    @property
    def official_serving(self) -> Serving:
        """Compatibility view of the immutable source serving definition."""

        return self.physical_quantity.official_serving

    @property
    def recommended_official_servings(self) -> Decimal:
        """Compatibility view of the optimizer-owned multiplier."""

        return self.physical_quantity.official_servings

    @property
    def serving(self) -> Serving:
        """Compatibility-friendly alias for the immutable serving definition."""

        return self.physical_quantity.official_serving


@dataclass(frozen=True, slots=True)
class RecommendedPortionSemanticResult:
    """Presentation-only practical wording returned by a semantic renderer.

    ``natural_descriptor`` deliberately contains only the quantity-recognition
    phrase, never a food name or canonical measurement.  Python owns both of
    those fields and combines them only when it formats the final message.
    """

    natural_descriptor: str | None = None
    confidence: PortionConfidence = "medium"
    presentation_practicality: PresentationPracticality = "practical"
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.natural_descriptor is not None:
            _text(self.natural_descriptor, "natural_descriptor")
        if self.confidence not in {"high", "medium", "low"}:
            raise ValueError("confidence is invalid")
        if self.presentation_practicality not in {"practical", "awkward", "unclear"}:
            raise ValueError("presentation_practicality is invalid")
        if self.reason is not None:
            _text(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class RecommendedPortionRendering:
    """Final wording paired with the canonical physical quantity."""

    natural_quantity_text: str
    natural_display_name: str | None = None
    confidence: PortionConfidence = "high"
    presentation_practicality: PresentationPracticality = "practical"
    rendering_method: RenderingMethod = "deterministic"
    reason: str | None = None
    physical_quantity: PhysicalRecommendedQuantity | None = None
    canonical_quantity_text: str | None = None
    semantic_descriptor: str | None = None
    semantic_descriptor_disposition: SemanticDescriptorDisposition = "not_requested"

    def __post_init__(self) -> None:
        _text(self.natural_quantity_text, "natural_quantity_text")
        if self.natural_display_name is not None:
            _text(self.natural_display_name, "natural_display_name")
        if self.confidence not in {"high", "medium", "low"}:
            raise ValueError("confidence is invalid")
        if self.presentation_practicality not in {"practical", "awkward", "unclear"}:
            raise ValueError("presentation_practicality is invalid")
        if self.rendering_method not in {"deterministic", "luna"}:
            raise ValueError("rendering_method is invalid")
        if self.reason is not None:
            _text(self.reason, "reason")
        if self.physical_quantity is not None and not isinstance(
            self.physical_quantity, PhysicalRecommendedQuantity
        ):
            raise TypeError("physical_quantity must be a PhysicalRecommendedQuantity")
        if self.canonical_quantity_text is not None:
            _text(self.canonical_quantity_text, "canonical_quantity_text")
        if self.semantic_descriptor is not None:
            _text(self.semantic_descriptor, "semantic_descriptor")
        if self.semantic_descriptor_disposition not in {
            "not_requested",
            "deterministic_visual",
            "accepted",
            "not_useful",
            "rejected",
            "unavailable",
        }:
            raise ValueError("semantic_descriptor_disposition is invalid")


class RecommendedPortionSemanticRenderer(Protocol):
    """Narrow reverse-direction semantic boundary.

    This is deliberately not ``NaturalPortionInterpreter`` or its matcher:
    those components interpret a human phrase into official servings.  This
    protocol goes in the opposite direction and returns presentation fields
    only.
    """

    def render(
        self,
        request: RecommendedPortionRenderRequest,
    ) -> RecommendedPortionSemanticResult:
        """Render the fixed request without changing its authoritative values."""


class RecommendedPortionRenderer:
    """Use safe deterministic rules, then an optional semantic renderer."""

    def __init__(
        self,
        semantic_renderer: RecommendedPortionSemanticRenderer | None = None,
        *,
        semantic_matcher: RecommendedPortionSemanticRenderer | None = None,
    ) -> None:
        if semantic_renderer is not None and semantic_matcher is not None:
            raise TypeError("provide semantic_renderer or semantic_matcher, not both")
        selected = semantic_renderer if semantic_renderer is not None else semantic_matcher
        if selected is not None and not callable(getattr(selected, "render", None)):
            raise TypeError("semantic_renderer must provide render")
        self._semantic_renderer = selected

    def render(
        self,
        request: RecommendedPortionRenderRequest,
    ) -> RecommendedPortionRendering:
        """Render one request, failing closed when the context is insufficient."""

        if not isinstance(request, RecommendedPortionRenderRequest):
            raise TypeError("request must be a RecommendedPortionRenderRequest")

        deterministic = _deterministic_render(request)
        if deterministic is None:
            raise RecommendedPortionRenderingUnavailableError(
                "no safe deterministic recommendation rendering is available"
            )
        # A calibrated serving-line relationship is an observed, exact Python
        # fact for this one food identity. It is more practical than the
        # generic official unit and deliberately bypasses optional AI wording.
        if deterministic.natural_display_name is not None:
            return deterministic
        visual = _descriptive_visual_for(request)
        if visual is not None:
            return replace(
                deterministic,
                natural_quantity_text=visual,
                confidence="medium",
                reason="food_aware_visual_approximation",
                semantic_descriptor_disposition="deterministic_visual",
            )
        # Count and volume units already have a reliable, concise deterministic
        # representation. Semantic wording remains a bounded enhancement for
        # weight quantities, where a dining hall may expose a useful serving-
        # line descriptor without ever changing the exact physical quantity.
        if (
            self._semantic_renderer is None
            or not request.physical_quantity.is_weight
            or not request.approved_descriptive_options
        ):
            return deterministic

        try:
            semantic_result = _coerce_semantic_result(
                self._semantic_renderer.render(request)
            )
            _validate_semantic_rendering(semantic_result, request)
        except RecommendedPortionRenderingOutputError:
            return _semantic_fallback(
                deterministic,
                disposition="rejected",
                reason="semantic_descriptor_rejected",
            )
        except RecommendedPortionRenderingError:
            return _semantic_fallback(
                deterministic,
                disposition="unavailable",
                reason="semantic_descriptor_unavailable",
            )
        except Exception:
            return _semantic_fallback(
                deterministic,
                disposition="unavailable",
                reason="semantic_descriptor_unavailable",
            )
        if semantic_result.natural_descriptor is None:
            return replace(
                deterministic,
                confidence=semantic_result.confidence,
                presentation_practicality=semantic_result.presentation_practicality,
                reason=semantic_result.reason or "semantic_descriptor_not_useful",
                semantic_descriptor_disposition="not_useful",
            )
        return RecommendedPortionRendering(
            # The descriptor is intentionally the user-facing quantity. The
            # deterministic canonical amount remains below for inspection and
            # validation, never as secondary arithmetic the user must parse.
            natural_quantity_text=semantic_result.natural_descriptor,
            natural_display_name=None,
            confidence=semantic_result.confidence,
            presentation_practicality=semantic_result.presentation_practicality,
            rendering_method="luna",
            reason=semantic_result.reason or "accepted_practical_descriptor",
            physical_quantity=request.physical_quantity,
            canonical_quantity_text=deterministic.natural_quantity_text,
            semantic_descriptor=semantic_result.natural_descriptor,
            semantic_descriptor_disposition="accepted",
        )


@dataclass(frozen=True, slots=True)
class PlannedMealItemRendering:
    """One recommendation item paired with its presentation-only rendering."""

    recommendation_item: RecommendedMealItem
    rendering: RecommendedPortionRendering

    def __post_init__(self) -> None:
        if not isinstance(self.recommendation_item, RecommendedMealItem):
            raise TypeError("recommendation_item must be a RecommendedMealItem")
        if not isinstance(self.rendering, RecommendedPortionRendering):
            raise TypeError("rendering must be a RecommendedPortionRendering")

    @property
    def physical_quantity(self) -> PhysicalRecommendedQuantity:
        """Return the optimizer quantity associated with this rendering."""

        return self.recommendation_item.physical_quantity


@dataclass(frozen=True, slots=True)
class MealRecommendationRendering:
    """A rendered recommendation, its durable plan, and future message text."""

    recommendation: MealRecommendation
    rendered_items: tuple[PlannedMealItemRendering, ...]
    meal_plan: MealPlan
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.recommendation, MealRecommendation):
            raise TypeError("recommendation must be a MealRecommendation")
        if not isinstance(self.rendered_items, tuple) or not all(
            isinstance(item, PlannedMealItemRendering) for item in self.rendered_items
        ):
            raise TypeError("rendered_items must contain PlannedMealItemRendering values")
        if len(self.rendered_items) != len(self.recommendation.items):
            raise ValueError("rendered_items must match recommendation items")
        if not isinstance(self.meal_plan, MealPlan):
            raise TypeError("meal_plan must be a MealPlan")
        if (
            self.meal_plan.service_date != self.recommendation.service_date
            or not meal_values_equal(self.meal_plan.meal, self.recommendation.meal)
            or len(self.meal_plan.items) != len(self.recommendation.items)
        ):
            raise ValueError("meal_plan must represent the recommendation")
        for rendered_item, recommendation_item, planned_item in zip(
            self.rendered_items,
            self.recommendation.items,
            self.meal_plan.items,
            strict=True,
        ):
            if (
                planned_item.food.occurrence != recommendation_item.occurrence
                or planned_item.record != recommendation_item.record
                or planned_item.recommended_official_servings
                != recommendation_item.official_servings
                or rendered_item.rendering.physical_quantity
                != recommendation_item.physical_quantity
            ):
                raise ValueError("meal_plan changed authoritative recommendation data")
            _validate_final_rendering(
                rendered_item.rendering,
                _render_request_for_item(recommendation_item),
            )
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be non-empty text")

    @property
    def plan(self) -> MealPlan:
        """Short alias for the plan suitable for persistence."""

        return self.meal_plan

    @property
    def renderings(self) -> tuple[RecommendedPortionRendering, ...]:
        """Return renderings in the exact optimizer item order."""

        return tuple(item.rendering for item in self.rendered_items)

    def practicality_counts(self) -> dict[PresentationPracticality, int]:
        """Return presentation-quality counts for benchmark/audit reporting."""

        counts: dict[PresentationPracticality, int] = {
            "practical": 0,
            "awkward": 0,
            "unclear": 0,
        }
        for rendering in self.renderings:
            counts[rendering.presentation_practicality] += 1
        return counts


def render_meal_recommendation(
    recommendation: MealRecommendation,
    renderer: RecommendedPortionRenderer | RecommendedPortionSemanticRenderer,
) -> MealRecommendationRendering:
    """Render every optimizer item and construct a deterministic ``MealPlan``."""

    if not isinstance(recommendation, MealRecommendation):
        raise TypeError("recommendation must be a MealRecommendation")
    if not recommendation.items:
        raise MealRecommendationRenderingError(
            "a recommendation without selected items cannot become a MealPlan"
        )
    if not callable(getattr(renderer, "render", None)):
        raise TypeError("renderer must provide render")

    active_renderer = (
        renderer
        if isinstance(renderer, RecommendedPortionRenderer)
        else RecommendedPortionRenderer(renderer)
    )
    rendered: list[PlannedMealItemRendering] = []
    for recommendation_item in recommendation.items:
        request = _render_request_for_item(recommendation_item)
        try:
            portion_rendering = active_renderer.render(request)
        except RecommendedPortionRenderingError:
            raise
        except Exception:
            raise RecommendedPortionRenderingTransportError(
                "recommendation item rendering failed"
            ) from None
        if not isinstance(portion_rendering, RecommendedPortionRendering):
            raise RecommendedPortionRenderingOutputError(
                "recommendation renderer returned invalid output"
            )
        rendered.append(PlannedMealItemRendering(recommendation_item, portion_rendering))

    rendered_items = tuple(rendered)
    plan = meal_plan_from_recommendation(
        recommendation,
        tuple(item.rendering for item in rendered_items),
    )
    return MealRecommendationRendering(
        recommendation=recommendation,
        rendered_items=rendered_items,
        meal_plan=plan,
        message=format_meal_message(plan),
    )


def meal_plan_from_recommendation(
    recommendation: MealRecommendation,
    renderings: Iterable[RecommendedPortionRendering],
) -> MealPlan:
    """Adapt ordered renderings to the existing exact ``MealPlan`` boundary.

    The two sequences are positional by design: renderings are produced from
    the recommendation's items in order.  This function performs no nutrition
    arithmetic and never reads a quantity from presentation output.
    """

    if not isinstance(recommendation, MealRecommendation):
        raise TypeError("recommendation must be a MealRecommendation")
    rendering_values = tuple(renderings)
    if not recommendation.items:
        raise MealRecommendationRenderingError(
            "a recommendation without selected items cannot become a MealPlan"
        )
    if len(rendering_values) != len(recommendation.items):
        raise ValueError("renderings must contain one value per recommendation item")
    if not all(isinstance(value, RecommendedPortionRendering) for value in rendering_values):
        raise TypeError("renderings must contain RecommendedPortionRendering values")
    for recommendation_item, rendering in zip(
        recommendation.items,
        rendering_values,
        strict=True,
    ):
        request = _render_request_for_item(recommendation_item)
        _validate_final_rendering(rendering, request)

    planned_items = tuple(
        PlannedMealItem(
            food=_resolved_food_for_item(recommendation_item),
            # This is the only quantity copied into the plan.  It is the
            # optimizer Decimal, never a value returned by a renderer.
            recommended_official_servings=recommendation_item.official_servings,
            natural_quantity_text=rendering.natural_quantity_text,
            display_food_name=rendering.natural_display_name,
            presentation_binding=_freeze_rendering(recommendation_item, rendering),
        )
        for recommendation_item, rendering in zip(
            recommendation.items,
            rendering_values,
            strict=True,
        )
    )
    return MealPlan(recommendation.service_date, recommendation.meal, planned_items)


def _freeze_rendering(
    item: RecommendedMealItem,
    rendering: RecommendedPortionRendering,
) -> PresentationBinding:
    food = _resolved_food_for_item(item)
    calibration = DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS.find(
        food.source_identifier, food.nutrition_record.serving)
    if rendering.semantic_descriptor_disposition in {"accepted", "deterministic_visual"}:
        kind = PresentationKind.DESCRIPTIVE_ONLY
        calibration = None
    elif calibration is not None:
        # Refuse any registry change between selecting and freezing a render.
        selected = calibration.render(item.official_servings)
        if (selected.natural_quantity_text != rendering.natural_quantity_text
                or selected.display_food_name != rendering.natural_display_name):
            raise RecommendedPortionRenderingOutputError("calibration changed before presentation freeze")
        kind = PresentationKind.REVERSIBLE_CALIBRATED
    else:
        kind = PresentationKind.EXACT_AUTHORITATIVE
    return freeze_presentation(food, item.official_servings,
        rendering.natural_quantity_text, kind, calibration)


def format_meal_message(plan: MealPlan) -> str:
    """Format a concise future message without nutrition or optimizer details."""

    if not isinstance(plan, MealPlan):
        raise TypeError("plan must be a MealPlan")
    meal_name = meal_name_for_display(plan.meal)
    lines = [f"For {meal_name}, get:"]
    for item in plan.items:
        quantity = item.natural_quantity_text.strip()
        # Accepted practical descriptors are intentionally food-free (for
        # example, "about 2 scoops or 1 ladle"). Add the Python-owned official
        # name here so the message remains self-contained without changing the
        # stored presentation phrase or any authoritative quantity.
        if not _contains_identity_token(quantity, item.display_name):
            quantity = f"{quantity} of {item.display_name.casefold()}"
        lines.append(f"- {quantity}")
    return "\n".join(lines)


def persist_rendered_meal_recommendation(
    rendered: MealRecommendationRendering,
    durable_state: object,
    *,
    plan_id: str | None = None,
    created_at: datetime | None = None,
) -> "PersistedMealPlan":
    """Persist an already rendered plan through the existing state API."""

    if not isinstance(rendered, MealRecommendationRendering):
        raise TypeError("rendered must be a MealRecommendationRendering")
    save = getattr(durable_state, "save_meal_plan", None)
    if not callable(save):
        raise TypeError("durable_state must provide save_meal_plan")
    return save(rendered.meal_plan, plan_id=plan_id, created_at=created_at)


def persist_meal_recommendation(
    recommendation: MealRecommendation,
    renderer: RecommendedPortionRenderer | RecommendedPortionSemanticRenderer,
    durable_state: object,
    *,
    plan_id: str | None = None,
    created_at: datetime | None = None,
) -> "PersistedMealPlan":
    """Explicitly render and persist one recommendation using ``DurableMealState``."""

    rendered = render_meal_recommendation(recommendation, renderer)
    return persist_rendered_meal_recommendation(
        rendered,
        durable_state,
        plan_id=plan_id,
        created_at=created_at,
    )


def _render_request_for_item(
    item: RecommendedMealItem,
) -> RecommendedPortionRenderRequest:
    return RecommendedPortionRenderRequest(
        official_food_display_name=item.record.name,
        physical_quantity=item.physical_quantity,
        station_name=item.occurrence.station_name,
        source_identifier=item.occurrence.source_identifier,
    )


def _resolved_food_for_item(item: RecommendedMealItem) -> ResolvedFood:
    """Create a plan-facing identity wrapper around the exact FD occurrence."""

    occurrence = item.occurrence
    return ResolvedFood(
        original_food_text=item.record.name,
        occurrence=occurrence,
        equivalent_occurrences=(occurrence,),
        source_identifier=occurrence.source_identifier,
        content_signature=occurrence.content_signature,
        nutrition_snapshot_id=occurrence.nutrition_snapshot_id,
        nutrition_record=item.record,
        resolution_method="exact_name",
    )


_WORD_TOKEN = re.compile(r"[a-z0-9]+")
_IDENTITY_STOP_WORDS = frozenset({"a", "an", "and", "of", "or", "the", "to", "with"})
_SEMANTIC_MEASUREMENT_TERMS = frozenset(
    {
        "cup",
        "cups",
        "fluid",
        "ounce",
        "ounces",
        "oz",
        "weight",
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
        "pound",
        "pounds",
    }
)
_FORBIDDEN_PRESENTATION_TERMS = frozenset(
    {
        "calorie",
        "calories",
        "carbohydrate",
        "carbohydrates",
        "fat",
        "fiber",
        "macro",
        "macros",
        "protein",
    }
)
_PRACTICAL_DESCRIPTOR_UTENSIL_WORDS = frozenset(
    {
        "ladle",
        "ladles",
        "ladleful",
        "ladlefuls",
        "scoop",
        "scoops",
        "spoon",
        "spoons",
        "spoonful",
        "spoonfuls",
    }
)
_PRACTICAL_DESCRIPTOR_WORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "around",
        "eight",
        "five",
        "four",
        "full",
        "generous",
        "half",
        "heaping",
        "ladle",
        "ladleful",
        "ladlefuls",
        "ladles",
        "large",
        "level",
        "light",
        "medium",
        "one",
        "or",
        "portion",
        "portions",
        "roughly",
        "scoop",
        "scoops",
        "serving",
        "seven",
        "six",
        "small",
        "spoon",
        "spoonful",
        "spoonfuls",
        "spoons",
        "three",
        "to",
        "two",
    }
)
_PRACTICAL_DESCRIPTOR_PREFIXES = frozenset({"about", "around", "roughly"})
_PRACTICAL_DESCRIPTOR_MAX_LENGTH = 96
_PRACTICAL_DESCRIPTOR_MAX_NUMBER = 8


def _descriptive_visual_for(request: RecommendedPortionRenderRequest) -> str | None:
    """Give familiar weight-based foods a rough, non-reversible serving cue."""

    quantity = request.physical_quantity
    food_name = " ".join(request.official_food_display_name.casefold().split())
    if not quantity.is_weight or quantity.unit.casefold().strip() not in {
        "ounce", "ounces", "ounce cooked weight", "ounces cooked weight",
    }:
        return None
    if not Decimal("2") <= quantity.amount <= Decimal("16"):
        return None
    plain_pasta = _is_plain_scoop_pasta(food_name)
    if _is_composite_food_component(food_name) and not plain_pasta:
        return None
    words = set(re.findall(r"[a-z]+", food_name))
    if not plain_pasta and words & {"soup", "stew", "chowder", "bisque", "mashed", "puree", "pureed",
                                  "salad", "noodles", "fries", "sauce", "gravy"}:
        return None
    if plain_pasta:
        unit, ounces = "scoop-sized", Decimal("4")
    elif words & {"egg", "eggs", "scramble", "scrambled"}:
        unit, ounces = "baseball-sized", Decimal("4")
    elif words & {"rice", "quinoa", "couscous"}:
        unit, ounces = "scoop-sized", Decimal("4")
    elif (
        words & {"potato", "potatoes", "vegetable", "vegetables", "broccoli",
                 "carrots", "cauliflower", "zucchini", "mushrooms", "squash",
                 "sprouts", "hashbrowns"}
        or {"green", "beans"} <= words
    ):
        unit, ounces = "fist-sized", Decimal("4")
    else:
        return None
    portions = int((quantity.amount / ounces).to_integral_value(rounding=ROUND_HALF_UP))
    noun = "portion" if portions == 1 else "portions"
    return f"about {portions} {unit} {noun}"


def _is_plain_scoop_pasta(food_name: str) -> bool:
    """Recognize plain pasta names while leaving mixed dishes in source units."""

    if food_name in {"mac and cheese", "macaroni and cheese"}:
        return True
    if _is_composite_food_component(food_name):
        return False
    words = set(re.findall(r"[a-z]+", food_name))
    pasta_words = {"pasta", "spaghetti", "noodles", "cavatappi", "penne", "macaroni",
                   "ramen", "fettuccine", "linguine", "rigatoni", "rotini", "farfalle"}
    plain_modifiers = {"corn", "dashi", "soba", "rice", "wheat", "whole", "grain",
                       "gluten", "free", "egg"}
    return bool(words & pasta_words) and words <= pasta_words | plain_modifiers


def _deterministic_render(
    request: RecommendedPortionRenderRequest,
) -> RecommendedPortionRendering | None:
    quantity = request.physical_quantity
    if quantity.is_unknown:
        return None
    calibration = DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS.find(
        request.source_identifier,
        quantity.official_serving,
    )
    if calibration is not None:
        calibrated = calibration.render(quantity.official_servings)
        return RecommendedPortionRendering(
            natural_quantity_text=calibrated.natural_quantity_text,
            natural_display_name=calibrated.display_food_name,
            confidence="high",
            presentation_practicality="practical",
            rendering_method="deterministic",
            reason="calibrated_serving_presentation",
            physical_quantity=quantity,
            canonical_quantity_text=calibrated.natural_quantity_text,
            semantic_descriptor_disposition="not_requested",
        )
    food_name = request.official_food_display_name.strip().casefold()
    amount_text = format_physical_amount(quantity)
    if quantity.is_discrete_count and quantity.count_unit in {"each", "count"}:
        quantity_text = _count_food_text(quantity.amount, food_name)
    elif quantity.is_weight and _is_composite_food_component(food_name):
        # A comma/conjunction food name is still one FD component. Making the
        # total explicit prevents a reader from applying the weight to every
        # named ingredient when an optional practical descriptor is unavailable.
        quantity_text = f"{amount_text} total of {food_name}"
    else:
        quantity_text = f"{amount_text} of {food_name}"
    return RecommendedPortionRendering(
        natural_quantity_text=quantity_text,
        confidence="high",
        presentation_practicality="practical",
        rendering_method="deterministic",
        reason="canonical_physical_quantity",
        physical_quantity=quantity,
        canonical_quantity_text=quantity_text,
        semantic_descriptor_disposition="not_requested",
    )


def _count_food_text(physical_count: Decimal, food_name: str) -> str:
    singular_food = _singular_food_phrase(food_name)
    number = _format_decimal(physical_count)
    noun = singular_food if physical_count == 1 else _plural_food_phrase(food_name)
    return f"{number} {noun}"


def _coerce_semantic_result(value: object) -> RecommendedPortionSemanticResult:
    """Accept the new descriptor contract and a narrow legacy test double."""

    if isinstance(value, RecommendedPortionSemanticResult):
        return value
    # Keeping this compatibility branch avoids making custom in-process
    # renderers fail merely because they still return the old presentation
    # object. Its old full text is treated only as a candidate descriptor; it
    # can never replace the deterministic canonical quantity or food identity.
    if isinstance(value, RecommendedPortionRendering):
        return RecommendedPortionSemanticResult(
            natural_descriptor=value.semantic_descriptor or value.natural_quantity_text,
            confidence=value.confidence,
            presentation_practicality=value.presentation_practicality,
            reason=value.reason,
        )
    raise RecommendedPortionRenderingOutputError(
        "recommendation semantic renderer returned invalid output"
    )


def _validate_final_rendering(
    rendering: RecommendedPortionRendering,
    request: RecommendedPortionRenderRequest,
) -> None:
    """Ensure the durable adapter receives Python-owned canonical wording."""

    if rendering.physical_quantity != request.physical_quantity:
        raise RecommendedPortionRenderingOutputError(
            "recommendation rendering changed canonical physical quantity"
        )
    canonical = _deterministic_render(request)
    if canonical is None or rendering.canonical_quantity_text != canonical.natural_quantity_text:
        raise RecommendedPortionRenderingOutputError(
            "recommendation rendering omitted canonical physical quantity text"
        )
    if rendering.natural_display_name != canonical.natural_display_name:
        raise RecommendedPortionRenderingOutputError(
            "recommendation rendering changed display presentation"
        )
    canonical_text = canonical.natural_quantity_text
    _validate_semantic_rendering(
        RecommendedPortionSemanticResult(
            natural_descriptor=rendering.semantic_descriptor,
            confidence=rendering.confidence,
            presentation_practicality=rendering.presentation_practicality,
            reason=rendering.reason,
        ),
        request,
    )
    if rendering.semantic_descriptor_disposition == "deterministic_visual":
        if (
            rendering.rendering_method != "deterministic"
            or rendering.semantic_descriptor is not None
            or rendering.natural_quantity_text != _descriptive_visual_for(request)
        ):
            raise RecommendedPortionRenderingOutputError(
                "recommendation rendering has an invalid visual description"
            )
    elif rendering.semantic_descriptor_disposition == "accepted":
        if (
            rendering.rendering_method != "luna"
            or rendering.semantic_descriptor is None
            or rendering.natural_quantity_text != rendering.semantic_descriptor
            or canonical.natural_display_name is not None
        ):
            raise RecommendedPortionRenderingOutputError(
                "recommendation rendering has an invalid practical descriptor"
            )
    elif (
        rendering.rendering_method != "deterministic"
        or rendering.semantic_descriptor is not None
        or rendering.natural_quantity_text != canonical_text
    ):
        raise RecommendedPortionRenderingOutputError(
            "recommendation rendering replaced canonical physical quantity text"
        )


def _validate_semantic_rendering(
    rendering: object,
    request: RecommendedPortionRenderRequest,
) -> None:
    semantic_result = _coerce_semantic_result(rendering)
    if semantic_result.natural_descriptor is not None:
        descriptor = semantic_result.natural_descriptor
        if descriptor not in request.approved_descriptive_options:
            raise RecommendedPortionRenderingOutputError(
                "semantic descriptor was not an approved presentation option"
            )
        if len(descriptor) > _PRACTICAL_DESCRIPTOR_MAX_LENGTH:
            raise RecommendedPortionRenderingOutputError(
                "semantic descriptor is too long"
            )
        if re.fullmatch(r"[a-zA-Z0-9\s\-–]+", descriptor) is None:
            raise RecommendedPortionRenderingOutputError(
                "semantic descriptor has invalid punctuation"
            )
        descriptor_tokens = set(_WORD_TOKEN.findall(descriptor.casefold()))
        if descriptor_tokens & _SEMANTIC_MEASUREMENT_TERMS:
            raise RecommendedPortionRenderingOutputError(
                "semantic descriptor must not replace canonical physical quantity"
            )
        if descriptor_tokens & _FORBIDDEN_PRESENTATION_TERMS:
            raise RecommendedPortionRenderingOutputError(
                "recommendation semantic renderer returned non-presentation wording"
            )
        if semantic_result.confidence == "low" or (
            semantic_result.presentation_practicality != "practical"
        ):
            raise RecommendedPortionRenderingOutputError(
                "semantic descriptor is not confident and practical"
            )
        _validate_practical_descriptor(descriptor, descriptor_tokens)


def _validate_practical_descriptor(descriptor: str, descriptor_tokens: set[str]) -> None:
    """Conservatively accept only a short, food-free serving-line phrase.

    This is intentionally a plausibility screen rather than a claimed
    scoop-to-weight conversion. The model is allowed to offer one or two
    recognition options, but it cannot output a full rewritten instruction,
    a food identity, or a physical/nutrition measurement.
    """

    tokens_in_order = _WORD_TOKEN.findall(descriptor.casefold())
    if not tokens_in_order or tokens_in_order[0] not in _PRACTICAL_DESCRIPTOR_PREFIXES:
        raise RecommendedPortionRenderingOutputError(
            "semantic descriptor must be an approximate serving-line phrase"
        )
    if descriptor_tokens - _PRACTICAL_DESCRIPTOR_WORDS - {
        token for token in descriptor_tokens if token.isdecimal()
    }:
        raise RecommendedPortionRenderingOutputError(
            "recommendation semantic renderer changed food identity"
        )
    numeric_values = [int(token) for token in descriptor_tokens if token.isdecimal()]
    if any(value <= 0 or value > _PRACTICAL_DESCRIPTOR_MAX_NUMBER for value in numeric_values):
        raise RecommendedPortionRenderingOutputError(
            "semantic descriptor has an implausible portion count"
        )
    if descriptor_tokens & {"serving", "portion", "portions"} and not (
        descriptor_tokens & {"spoon", "spoons", "spoonful", "spoonfuls"}
    ):
        raise RecommendedPortionRenderingOutputError(
            "semantic descriptor is not a practical utensil amount"
        )
    alternatives = re.split(r"\s+or\s+", descriptor.casefold())
    if len(alternatives) > 2:
        raise RecommendedPortionRenderingOutputError(
            "semantic descriptor has too many alternatives"
        )
    for alternative in alternatives:
        alternative_tokens = set(_WORD_TOKEN.findall(alternative))
        if not alternative_tokens & _PRACTICAL_DESCRIPTOR_UTENSIL_WORDS:
            raise RecommendedPortionRenderingOutputError(
                "semantic descriptor is not a practical utensil amount"
            )


def _semantic_fallback(
    deterministic: RecommendedPortionRendering,
    *,
    disposition: Literal["rejected", "unavailable"],
    reason: str,
) -> RecommendedPortionRendering:
    """Keep the exact Python wording when semantic presentation is unusable."""

    return replace(
        deterministic,
        presentation_practicality="awkward",
        reason=reason,
        semantic_descriptor_disposition=disposition,
    )


def _is_composite_food_component(food_name: str) -> bool:
    """Return whether a single menu-component name visibly lists ingredients."""

    return bool(
        "," in food_name
        or re.search(r"\b(?:and|with)\b|[&/]", food_name, flags=re.IGNORECASE)
    )


def _identity_overlap(left: str, right: str) -> bool:
    left_tokens = _identity_tokens(left)
    right_tokens = _identity_tokens(right)
    return bool(left_tokens & right_tokens)


def _contains_identity_token(text: str, official_name: str) -> bool:
    return _identity_overlap(text, official_name)


def _identity_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for token in _WORD_TOKEN.findall(value.casefold()):
        if token in _IDENTITY_STOP_WORDS or len(token) <= 1:
            continue
        tokens.add(token)
        if token.endswith("ies") and len(token) > 3:
            tokens.add(token[:-3] + "y")
        elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
            tokens.add(token[:-1])
    return tokens


def _singular_food_phrase(value: str) -> str:
    words = value.split()
    if not words:
        return value
    final = words[-1]
    if final in {"potatoes", "tomatoes"}:
        words[-1] = final[:-2]
    elif final.endswith("ies") and len(final) > 3:
        words[-1] = final[:-3] + "y"
    elif final.endswith("s") and not final.endswith(("ss", "us", "is")) and len(final) > 2:
        words[-1] = final[:-1]
    return " ".join(words)


def _plural_food_phrase(value: str) -> str:
    """Apply one conservative pluralization to a countable food name."""

    words = value.split()
    if not words:
        return value
    final = words[-1]
    if final.endswith("ies"):
        return value
    if final.endswith(("s", "ss", "us", "is")):
        return value
    if final.endswith("y") and len(final) > 1 and final[-2] not in "aeiou":
        words[-1] = final[:-1] + "ies"
    elif final in {"potato", "tomato"}:
        words[-1] = final + "es"
    elif final.endswith(("ch", "sh", "x", "z")):
        words[-1] = final + "es"
    else:
        words[-1] = final + "s"
    return " ".join(words)


def _format_decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{field_name} must be one line of text")
