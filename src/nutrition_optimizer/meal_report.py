"""Read-only reconciliation of a known meal plan and a natural-language report.

This module deliberately composes existing food-identity and portion boundaries
without importing the ledger, application commands, nutrition arithmetic, or
messaging.  Its result is a proposal for a later confirmation step only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, localcontext
import re
from typing import TYPE_CHECKING, Literal, Mapping, Protocol

from .food_resolution import (
    AmbiguousFood,
    FoodResolutionRequest,
    FoodResolutionResult,
    LocalFDFoodResolver,
    ResolvedFood,
    UnresolvedFood,
)
from .meal_identity import canonical_meal_name, meal_identity_matches
from .nutrition.models import NutritionRecord
from .presentation_binding import PresentationBinding
from .portion_interpretation import (
    AmbiguousPortion,
    InterpretedPortion,
    NaturalPortionInterpreter,
    PortionInterpretationRequest,
    UnresolvedPortion,
)

if TYPE_CHECKING:
    from .meal_report_structured import MealReportSemanticResult


__all__ = [
    "ClarificationItem",
    "MealPlan",
    "MealReportReconciler",
    "MealReportSemanticInterpreter",
    "PlannedMealItem",
    "ProposedEatenMealItem",
    "ProposedUnplannedMealItem",
    "ReconciledMealReport",
    "SkippedMealItem",
]


QuantitySource = Literal[
    "planned_quantity",
    "explicit_deterministic",
    "explicit_semantic_estimate",
]


@dataclass(frozen=True, slots=True)
class PlannedMealItem:
    """One exact current FD food and the authoritative planned quantity."""

    food: ResolvedFood
    recommended_official_servings: Decimal
    natural_quantity_text: str
    display_food_name: str | None = None
    presentation_binding: PresentationBinding | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.food, ResolvedFood):
            raise TypeError("food must be a ResolvedFood")
        _positive_decimal(self.recommended_official_servings, "recommended_official_servings")
        _text(self.natural_quantity_text, "natural_quantity_text")
        if self.presentation_binding is not None:
            self.presentation_binding.validate(self.food, self.recommended_official_servings, self.natural_quantity_text)
        if self.display_food_name is not None:
            _text(self.display_food_name, "display_food_name")

    @property
    def record(self) -> NutritionRecord:
        """Return the occurrence-linked immutable official nutrition record."""

        return self.food.nutrition_record

    @property
    def display_name(self) -> str:
        """Return the user-facing plan name without changing official identity."""

        return self.display_food_name or self.food.nutrition_record.name


@dataclass(frozen=True, slots=True)
class MealPlan:
    """An immutable recommended meal, suitable as context for a report."""

    service_date: date
    meal: str | int
    items: tuple[PlannedMealItem, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.service_date, date) or isinstance(self.service_date, datetime):
            raise TypeError("service_date must be a date")
        if isinstance(self.meal, bool) or not isinstance(self.meal, (str, int)):
            raise TypeError("meal must be text or an integer")
        if isinstance(self.meal, str):
            _text(self.meal, "meal")
        if not isinstance(self.items, tuple) or not self.items:
            raise ValueError("items must be a non-empty tuple")
        occurrence_ids: set[int] = set()
        for item in self.items:
            if not isinstance(item, PlannedMealItem):
                raise TypeError("items must contain PlannedMealItem values")
            occurrence = item.food.occurrence
            if occurrence.service_date != self.service_date:
                raise ValueError("planned food must use the plan service_date")
            if not _occurrence_matches_meal(occurrence, self.meal):
                raise ValueError("planned food must use the plan meal")
            if occurrence.occurrence_id in occurrence_ids:
                raise ValueError("items must not repeat an FD occurrence")
            occurrence_ids.add(occurrence.occurrence_id)

    def item_id(self, item: PlannedMealItem) -> str:
        """Return the compact, request-local opaque ID for a planned item."""

        try:
            return f"item_{self.items.index(item) + 1}"
        except ValueError:
            raise ValueError("item does not belong to this plan") from None

    def item_for_id(self, plan_item_id: str) -> PlannedMealItem | None:
        """Return exactly one supplied plan item, never a menu-wide match."""

        if not isinstance(plan_item_id, str):
            return None
        if not plan_item_id.startswith("item_"):
            return None
        try:
            index = int(plan_item_id.removeprefix("item_")) - 1
        except ValueError:
            return None
        if index < 0 or index >= len(self.items) or plan_item_id != f"item_{index + 1}":
            return None
        return self.items[index]

    def model_visible_items(self) -> list[dict[str, str | None]]:
        """Return compact, nutrition-free plan context exposed to the model.

        ``station_name`` is a current-menu fact for location-question routing;
        it is not nutrition, an official serving multiplier, or an instruction
        to invent a location when the local occurrence has none.
        """

        return [
            {
                "plan_item_id": self.item_id(item),
                "name": item.display_name,
                "display_quantity": item.natural_quantity_text,
                "station_name": item.food.occurrence.station_name,
            }
            for item in self.items
        ]


class MealReportSemanticInterpreter(Protocol):
    """Narrow semantic boundary over only the current immutable meal plan."""

    def interpret(self, plan: MealPlan, user_text: str) -> "MealReportSemanticResult":
        """Parse references and actions; never resolve food or quantities."""

    def interpret_with_context(
        self,
        plan: MealPlan,
        user_text: str,
        conversation_context: Mapping[str, object],
    ) -> "MealReportSemanticResult":
        """Optionally parse with compact durable conversation context."""


@dataclass(frozen=True, slots=True)
class ProposedEatenMealItem:
    """One resolved planned item proposed as eaten at an official quantity."""

    plan_item: PlannedMealItem
    official_servings: Decimal
    quantity_source: QuantitySource
    original_user_phrase: str
    confidence: Literal["high", "medium", "low"] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan_item, PlannedMealItem):
            raise TypeError("plan_item must be a PlannedMealItem")
        _positive_decimal(self.official_servings, "official_servings")
        if self.quantity_source not in {
            "planned_quantity",
            "explicit_deterministic",
            "explicit_semantic_estimate",
        }:
            raise ValueError("quantity_source is invalid")
        _text(self.original_user_phrase, "original_user_phrase")
        if self.confidence is not None and self.confidence not in {"high", "medium", "low"}:
            raise ValueError("confidence is invalid")
        if self.quantity_source == "planned_quantity" and self.confidence is not None:
            raise ValueError("planned quantities do not have interpretation confidence")

    @property
    def food(self) -> ResolvedFood:
        return self.plan_item.food

    @property
    def record(self) -> NutritionRecord:
        return self.plan_item.record


@dataclass(frozen=True, slots=True)
class SkippedMealItem:
    """One explicitly skipped plan item, distinct from an eaten zero quantity."""

    plan_item: PlannedMealItem
    original_user_phrase: str

    def __post_init__(self) -> None:
        if not isinstance(self.plan_item, PlannedMealItem):
            raise TypeError("plan_item must be a PlannedMealItem")
        _text(self.original_user_phrase, "original_user_phrase")


@dataclass(frozen=True, slots=True)
class ProposedUnplannedMealItem:
    """An additional food report, possibly unresolved in identity or quantity."""

    food_text: str
    quantity_text: str | None
    resolved_food: ResolvedFood | None
    official_servings: Decimal | None
    quantity_source: QuantitySource | None = None
    confidence: Literal["high", "medium", "low"] | None = None

    def __post_init__(self) -> None:
        _text(self.food_text, "food_text")
        if self.quantity_text is not None:
            _text(self.quantity_text, "quantity_text")
        if self.resolved_food is not None and not isinstance(self.resolved_food, ResolvedFood):
            raise TypeError("resolved_food must be a ResolvedFood or None")
        if self.official_servings is not None:
            _positive_decimal(self.official_servings, "official_servings")
        if self.official_servings is None:
            if self.quantity_source is not None or self.confidence is not None:
                raise ValueError("unresolved quantity cannot have source or confidence")
        elif self.quantity_source not in {"explicit_deterministic", "explicit_semantic_estimate"}:
            raise ValueError("resolved unplanned quantity needs an explicit source")
        if self.confidence is not None and self.confidence not in {"high", "medium", "low"}:
            raise ValueError("confidence is invalid")


@dataclass(frozen=True, slots=True)
class ClarificationItem:
    """One fail-closed issue a later confirmation interaction can present."""

    reason: str
    plan_item: PlannedMealItem | None = None
    food_text: str | None = None
    original_user_phrase: str | None = None

    def __post_init__(self) -> None:
        _text(self.reason, "reason")
        if self.plan_item is not None and not isinstance(self.plan_item, PlannedMealItem):
            raise TypeError("plan_item must be a PlannedMealItem or None")
        if self.food_text is not None:
            _text(self.food_text, "food_text")
        if self.original_user_phrase is not None:
            _text(self.original_user_phrase, "original_user_phrase")


@dataclass(frozen=True, slots=True)
class ReconciledMealReport:
    """Immutable, non-persisted proposal produced from one report."""

    plan: MealPlan
    eaten_items: tuple[ProposedEatenMealItem, ...]
    skipped_items: tuple[SkippedMealItem, ...]
    unspecified_items: tuple[PlannedMealItem, ...]
    unplanned_items: tuple[ProposedUnplannedMealItem, ...]
    clarification_items: tuple[ClarificationItem, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, MealPlan):
            raise TypeError("plan must be a MealPlan")
        for values, expected, name in (
            (self.eaten_items, ProposedEatenMealItem, "eaten_items"),
            (self.skipped_items, SkippedMealItem, "skipped_items"),
            (self.unspecified_items, PlannedMealItem, "unspecified_items"),
            (self.unplanned_items, ProposedUnplannedMealItem, "unplanned_items"),
            (self.clarification_items, ClarificationItem, "clarification_items"),
        ):
            if not isinstance(values, tuple) or not all(isinstance(value, expected) for value in values):
                raise TypeError(f"{name} has invalid values")
        statuses = [item.plan_item for item in self.eaten_items]
        statuses.extend(item.plan_item for item in self.skipped_items)
        statuses.extend(self.unspecified_items)
        if len(statuses) != len(self.plan.items) or set(statuses) != set(self.plan.items):
            raise ValueError("every plan item must have exactly one status")


class MealReportReconciler:
    """Compose plan parsing, local FD resolution, and portions without mutation."""

    def __init__(
        self,
        semantic_interpreter: MealReportSemanticInterpreter,
        food_resolver: LocalFDFoodResolver,
        portion_interpreter: NaturalPortionInterpreter,
    ) -> None:
        if not callable(getattr(semantic_interpreter, "interpret", None)):
            raise TypeError("semantic_interpreter must provide interpret")
        if not callable(getattr(food_resolver, "resolve", None)):
            raise TypeError("food_resolver must provide resolve")
        if not callable(getattr(portion_interpreter, "interpret", None)):
            raise TypeError("portion_interpreter must provide interpret")
        self._semantic_interpreter = semantic_interpreter
        self._food_resolver = food_resolver
        self._portion_interpreter = portion_interpreter

    def reconcile(self, plan: MealPlan, user_text: str) -> ReconciledMealReport:
        """Return a safe proposed outcome; any bad semantic output stays closed."""

        if not isinstance(plan, MealPlan):
            raise TypeError("plan must be a MealPlan")
        _text(user_text, "user_text")
        semantic = self.interpret_semantic(plan, user_text)
        if semantic is None:
            return self._closed(plan, "meal_report_semantic_failed", original_user_phrase=user_text)

        return self.reconcile_semantic(plan, semantic, user_text)

    def interpret_semantic(
        self,
        plan: MealPlan,
        user_text: str,
        *,
        conversation_context: Mapping[str, object] | None = None,
    ) -> "MealReportSemanticResult | None":
        """Invoke the semantic boundary with optional durable context.

        The optional method lets the production OpenClaw adapter see a compact
        report draft while preserving existing two-argument deterministic test
        interpreters and callers.  No context is interpreted by Python here.
        """

        if not isinstance(plan, MealPlan):
            raise TypeError("plan must be a MealPlan")
        _text(user_text, "user_text")
        if conversation_context is not None and not isinstance(conversation_context, Mapping):
            raise TypeError("conversation_context must be a mapping")
        try:
            contextual_interpret = getattr(self._semantic_interpreter, "interpret_with_context", None)
            semantic = (
                contextual_interpret(plan, user_text, conversation_context)
                if conversation_context is not None and callable(contextual_interpret)
                else self._semantic_interpreter.interpret(plan, user_text)
            )
        except Exception:
            return None

        from .meal_report_structured import MealReportSemanticResult

        if not isinstance(semantic, MealReportSemanticResult):
            return None
        return semantic

    def interpret_report_routing(
        self,
        user_text: str,
    ) -> "MealReportRoutingSemanticResult | None":
        """Return semantic report scope without granting persisted-plan authority.

        Older deterministic interpreters used by direct callers do not expose
        this optional pre-routing method and therefore retain unscoped behavior.
        A configured routing method that fails is different: it fails closed so
        an explicit different-slot report cannot be swallowed by an open draft.
        """

        _text(user_text, "user_text")
        routing_interpret = getattr(
            self._semantic_interpreter, "interpret_report_routing", None
        )
        from .meal_report_routing import MealReportRoutingSemanticResult

        if not callable(routing_interpret):
            return MealReportRoutingSemanticResult(
                intent="report_or_clarification",
                explicit_meal_slot=None,
            )
        try:
            semantic = routing_interpret(user_text)
        except Exception:
            return None
        if not isinstance(semantic, MealReportRoutingSemanticResult):
            return None
        return semantic

    def reconcile_semantic(
        self,
        plan: MealPlan,
        semantic: "MealReportSemanticResult",
        user_text: str,
        *,
        inherited_plan_item_quantities: Mapping[str, Decimal] | None = None,
        quantity_reference_item_id: str | None = None,
    ) -> ReconciledMealReport:
        """Reconcile an already validated semantic result without reinvoking AI.

        ``inherited_plan_item_quantities`` is restricted to an existing durable
        draft. ``quantity_reference_item_id`` is the sole outstanding item
        question selected by the caller from durable state, never model output.
        This also enables a semantic ``same_as_draft_item`` relation while
        Python, not the model, retrieves the exact previously resolved amount.
        """

        if not isinstance(plan, MealPlan):
            raise TypeError("plan must be a MealPlan")
        _text(user_text, "user_text")
        from .meal_report_structured import MealReportSemanticResult

        if not isinstance(semantic, MealReportSemanticResult):
            return self._closed(
                plan,
                "meal_report_semantic_response_invalid",
                original_user_phrase=user_text,
            )
        if inherited_plan_item_quantities is not None:
            if not isinstance(inherited_plan_item_quantities, Mapping):
                raise TypeError("inherited_plan_item_quantities must be a mapping")
            for item_id, quantity in inherited_plan_item_quantities.items():
                _text(item_id, "inherited plan item ID")
                _positive_decimal(quantity, "inherited plan item quantity")
        if quantity_reference_item_id is not None and plan.item_for_id(quantity_reference_item_id) is None:
            raise ValueError("quantity reference must identify a current plan item")
        return self._reconcile_semantic(
            plan,
            semantic,
            user_text,
            inherited_plan_item_quantities=inherited_plan_item_quantities or {},
            quantity_reference_item_id=quantity_reference_item_id,
        )

    def resolve_current_menu_food(
        self,
        plan: MealPlan,
        food_text: str,
    ) -> FoodResolutionResult | None:
        """Resolve one location-question food against the current scoped menu.

        The semantic layer may preserve a natural alias, but only the existing
        local resolver can choose a current authoritative occurrence.  This is
        intentionally identity-only; it never estimates quantity or nutrition.
        """

        if not isinstance(plan, MealPlan):
            raise TypeError("plan must be a MealPlan")
        _text(food_text, "food_text")
        try:
            return self._food_resolver.resolve(
                FoodResolutionRequest(food_text, plan.service_date, meal=plan.meal)
            )
        except Exception:
            return None

    def _reconcile_semantic(
        self,
        plan: MealPlan,
        semantic: "MealReportSemanticResult",
        user_text: str,
        *,
        inherited_plan_item_quantities: Mapping[str, Decimal],
        quantity_reference_item_id: str | None,
    ) -> ReconciledMealReport:
        by_id: dict[str, object] = {}
        invalid_reference = False
        for report in semantic.planned_items:
            item = plan.item_for_id(report.plan_item_id)
            if item is None:
                invalid_reference = True
                break
            prior = by_id.get(report.plan_item_id)
            if prior is not None and prior != report:
                return self._closed(plan, "contradictory_planned_report", original_user_phrase=user_text)
            by_id[report.plan_item_id] = report
        if invalid_reference:
            return self._closed(plan, "unknown_plan_item_reference", original_user_phrase=user_text)

        eaten: list[ProposedEatenMealItem] = []
        skipped: list[SkippedMealItem] = []
        unspecified: list[PlannedMealItem] = []
        clarifications: list[ClarificationItem] = []
        for item in plan.items:
            report = by_id.get(plan.item_id(item))
            if report is None:
                unspecified.append(item)
                continue
            # ``report`` originates from the strictly typed semantic result.
            action = report.action  # type: ignore[union-attr]
            if action == "skipped":
                if (
                    re.search(r"\bnone\b", user_text, re.IGNORECASE)
                    and any(_names_item(plan, other, user_text) for other in plan.items)
                    and not _names_item(plan, item, user_text)
                ) or (
                    len(plan.items) > 1
                    and re.search(r"\bnone of (?:it|that)\b", user_text, re.IGNORECASE)
                    and quantity_reference_item_id != plan.item_id(item)
                    and not _names_item(plan, item, user_text)
                ):
                    unspecified.append(item)
                    clarifications.append(ClarificationItem(
                        "ambiguous_reference", plan_item=item, original_user_phrase=user_text,
                    ))
                    continue
                skipped.append(SkippedMealItem(item, report.reference_text))  # type: ignore[union-attr]
                continue
            quantity_relation = report.quantity_relation  # type: ignore[union-attr]
            if quantity_relation == "as_recommended":
                if not _authorized_plan_attestation(
                    plan, item, user_text,
                    unambiguous_reference=quantity_reference_item_id == plan.item_id(item),
                ):
                    unspecified.append(item)
                    clarifications.append(
                        ClarificationItem(
                            reason="plan_attestation_required",
                            plan_item=item,
                            original_user_phrase=_literal_reported_amount(
                                user_text, report.reference_text,  # type: ignore[union-attr]
                            ),
                        )
                    )
                    continue
                eaten.append(
                    ProposedEatenMealItem(
                        item,
                        item.recommended_official_servings,
                        "planned_quantity",
                        report.reference_text,  # type: ignore[union-attr]
                    )
                )
                continue
            if quantity_relation == "fraction_of_recommended":
                quantity_text = report.quantity_text  # type: ignore[union-attr]
                fraction = _explicit_positive_fraction(quantity_text)
                if not _grounded_plan_fraction(
                    plan, item, quantity_text, user_text, quantity_reference_item_id,
                ):
                    fraction = None
                if fraction is not None:
                    with localcontext() as context:
                        context.prec = max(
                            36,
                            len(item.recommended_official_servings.as_tuple().digits)
                            + len(fraction.as_tuple().digits),
                        )
                        quantity = item.recommended_official_servings * fraction
                    eaten.append(
                        ProposedEatenMealItem(
                            item,
                            quantity,
                            "explicit_deterministic",
                            report.reference_text,  # type: ignore[union-attr]
                            "high",
                        )
                    )
                else:
                    unspecified.append(item)
                    clarifications.append(
                        ClarificationItem(
                            reason="planned_fraction_unresolved",
                            plan_item=item,
                            original_user_phrase=report.reference_text,  # type: ignore[union-attr]
                        )
                    )
                continue
            if quantity_relation == "same_as_draft_item":
                comparison_item_id = report.comparison_plan_item_id  # type: ignore[union-attr]
                comparison_item = (
                    plan.item_for_id(comparison_item_id)
                    if isinstance(comparison_item_id, str)
                    else None
                )
                inherited = (
                    inherited_plan_item_quantities.get(comparison_item_id)
                    if isinstance(comparison_item_id, str)
                    else None
                )
                if (
                    comparison_item is not None
                    and comparison_item != item
                    and inherited is not None
                    and _grounded_draft_reuse(
                        plan, item, comparison_item, user_text,
                        inherited_plan_item_quantities,
                        quantity_reference_item_id,
                    )
                ):
                    eaten.append(
                        ProposedEatenMealItem(
                            item,
                            inherited,
                            "explicit_deterministic",
                            report.reference_text,  # type: ignore[union-attr]
                            "high",
                        )
                    )
                else:
                    unspecified.append(item)
                    clarifications.append(
                        ClarificationItem(
                            reason="draft_quantity_reference_unresolved",
                            plan_item=item,
                            original_user_phrase=report.reference_text,  # type: ignore[union-attr]
                        )
                    )
                continue
            quantity_text = report.quantity_text  # type: ignore[union-attr]
            if quantity_relation != "modified" or quantity_text is None:
                # The structured contract normally prevents this.  Retain a
                # fail-closed guard in case a nonconforming interpreter
                # somehow constructs an otherwise typed result.
                unspecified.append(item)
                clarifications.append(
                    ClarificationItem(
                        reason="planned_quantity_relation_invalid",
                        plan_item=item,
                        original_user_phrase=report.reference_text,  # type: ignore[union-attr]
                    )
                )
                continue
            grounded_quantity = _grounded_modified_quantity_text(
                plan, item, quantity_text, user_text, quantity_reference_item_id,
            )
            if grounded_quantity is None:
                unspecified.append(item)
                clarifications.append(ClarificationItem(
                    "reported_quantity_not_grounded",
                    plan_item=item,
                    original_user_phrase=_literal_reported_quantity(user_text, quantity_text),
                ))
                continue
            portion = self._interpret_portion(item.food, grounded_quantity, planned_item=item)
            if (
                isinstance(portion, InterpretedPortion)
                and portion.interpretation_method != "semantic"
            ):
                eaten.append(
                    ProposedEatenMealItem(
                        item,
                        portion.estimated_official_servings,
                        "explicit_deterministic",
                        (
                            grounded_quantity
                            if grounded_quantity != quantity_text
                            else report.reference_text  # type: ignore[union-attr]
                        ),
                        portion.confidence,
                    )
                )
            else:
                unspecified.append(item)
                reason = (
                    "semantic_quantity_not_authoritative"
                    if isinstance(portion, InterpretedPortion)
                    else _portion_reason(portion)
                )
                clarifications.append(
                    ClarificationItem(
                        reason=reason,
                        plan_item=item,
                        original_user_phrase=report.reference_text,  # type: ignore[union-attr]
                    )
                )

        unplanned, unplanned_clarifications = self._reconcile_unplanned(plan, semantic, user_text)
        clarifications.extend(unplanned_clarifications)
        clarifications.extend(
            ClarificationItem(
                statement.reason,
                original_user_phrase=statement.reference_text,
            )
            for statement in semantic.unresolved_statements
        )
        return ReconciledMealReport(
            plan,
            tuple(eaten),
            tuple(skipped),
            tuple(unspecified),
            tuple(unplanned),
            tuple(clarifications),
        )

    def _reconcile_unplanned(
        self,
        plan: MealPlan,
        semantic: "MealReportSemanticResult",
        user_text: str,
    ) -> tuple[list[ProposedUnplannedMealItem], list[ClarificationItem]]:
        unplanned: list[ProposedUnplannedMealItem] = []
        clarifications: list[ClarificationItem] = []
        seen: set[tuple[str, str | None]] = set()
        for report in semantic.additional_foods:
            key = (report.food_text.casefold().strip(), report.quantity_text)
            if key in seen:
                continue
            seen.add(key)
            try:
                resolved = self._food_resolver.resolve(
                    FoodResolutionRequest(report.food_text, plan.service_date, meal=plan.meal)
                )
            except Exception:
                resolved = None
            if not isinstance(resolved, ResolvedFood):
                literal_food = _literal_unplanned_food_text(user_text, report.quantity_text)
                unplanned.append(
                    ProposedUnplannedMealItem(
                        literal_food, _literal_reported_quantity(user_text, report.quantity_text),
                        None, None,
                    )
                )
                reason = (
                    "unplanned_food_ambiguous"
                    if isinstance(resolved, AmbiguousFood)
                    else "unplanned_food_unresolved"
                )
                clarifications.append(ClarificationItem(reason, food_text=literal_food))
                continue
            if not _grounded_unplanned_identity(
                report.food_text, report.quantity_text, resolved, plan, user_text,
                self._food_resolver,
            ):
                literal_food = _literal_unplanned_food_text(user_text, report.quantity_text)
                unplanned.append(ProposedUnplannedMealItem(
                    literal_food, _literal_reported_quantity(user_text, report.quantity_text),
                    None, None,
                ))
                clarifications.append(ClarificationItem(
                    "unplanned_food_unresolved", food_text=literal_food,
                ))
                continue
            if report.quantity_text is None:
                unplanned.append(ProposedUnplannedMealItem(report.food_text, None, resolved, None))
                clarifications.append(
                    ClarificationItem("unplanned_quantity_required", food_text=report.food_text)
                )
                continue
            grounded_quantity = _grounded_unplanned_quantity_text(
                report.quantity_text, resolved, user_text,
                set().union(*(
                    _food_reference_tokens(item.display_name)
                    | _food_reference_tokens(item.record.name)
                    for item in plan.items
                )),
                set().union(*(_food_head_tokens(item) for item in plan.items)),
            )
            if grounded_quantity is None:
                literal = _literal_reported_quantity(user_text, report.quantity_text)
                unplanned.append(ProposedUnplannedMealItem(
                    report.food_text, literal, resolved, None,
                ))
                clarifications.append(ClarificationItem(
                    "reported_quantity_not_grounded", food_text=report.food_text,
                ))
                continue
            portion = self._interpret_portion(resolved, grounded_quantity)
            if (
                isinstance(portion, InterpretedPortion)
                and portion.interpretation_method != "semantic"
            ):
                unplanned.append(
                    ProposedUnplannedMealItem(
                        report.food_text,
                        grounded_quantity,
                        resolved,
                        portion.estimated_official_servings,
                        "explicit_deterministic",
                        portion.confidence,
                    )
                )
            else:
                unplanned.append(ProposedUnplannedMealItem(
                    report.food_text, grounded_quantity, resolved, None,
                ))
                reason = (
                    "semantic_quantity_not_authoritative"
                    if isinstance(portion, InterpretedPortion)
                    else _portion_reason(portion)
                )
                clarifications.append(ClarificationItem(reason, food_text=report.food_text))
        return unplanned, clarifications

    def _interpret_portion(self, food: ResolvedFood, quantity_text: str, *, planned_item: PlannedMealItem | None = None):
        try:
            return self._portion_interpreter.interpret(
                PortionInterpretationRequest(
                    food, quantity_text,
                    presentation_binding=planned_item.presentation_binding if planned_item else None,
                    planned_official_servings=planned_item.recommended_official_servings if planned_item else None,
                )
            )
        except Exception:
            return UnresolvedPortion(quantity_text, "portion_interpretation_failed")

    @staticmethod
    def _closed(
        plan: MealPlan,
        reason: str,
        *,
        original_user_phrase: str,
    ) -> ReconciledMealReport:
        return ReconciledMealReport(
            plan,
            (),
            (),
            plan.items,
            (),
            (ClarificationItem(reason, original_user_phrase=original_user_phrase),),
        )


def _portion_reason(result: AmbiguousPortion | UnresolvedPortion) -> str:
    if isinstance(result, AmbiguousPortion):
        return "explicit_quantity_ambiguous"
    return result.reason


def _literal_phrase_in(phrase: object, user_text: str, *, relative: bool = False) -> bool:
    if not isinstance(phrase, str) or not phrase.strip():
        return False
    literal = re.escape(" ".join(phrase.casefold().split()))
    normalized = " ".join(user_text.casefold().split())
    match = re.search(
        rf"(?<![\w.+/%−–—-]){literal}(?![\w%/]|\.\d)",
        normalized,
    )
    if match is None:
        return False
    # The fraction must describe the plan, not be the numeric part of a
    # physical measurement (e.g. "half a scoop" or "1 scoop"). A separate
    # item's explicit ounces elsewhere in a mixed report are unaffected.
    return not relative or not any(
        amount.start() < match.end() and match.start() < amount.end()
        for amount in _PHYSICAL_AMOUNT.finditer(normalized)
    )


_QUANTITY_WORDS = {
    "1": ("one",),
    "2": ("two",),
    "3": ("three",),
    "4": ("four",),
    "5": ("five",),
    "6": ("six",),
    "7": ("seven",),
    "8": ("eight",),
    "9": ("nine",),
    "10": ("ten",),
    "11": ("eleven",),
    "12": ("twelve",),
}


def _countable_unit_follows(tokens: list[str], index: int) -> bool:
    return any(
        _MEASUREMENT_AFTER_COUNT.fullmatch(" ".join(tokens[index + 1:index + 1 + width]))
        for width in (1, 2)
        if index + width < len(tokens)
    )


def _grounded_quantity_matches(
    phrase: str, user_text: str, *, allow_bare_article: bool = False,
) -> tuple[str, tuple[re.Match[str], ...]] | None:
    """Find the claimed words in the message; only number words may normalize."""

    normalized = " ".join(user_text.casefold().split())
    claimed = " ".join(phrase.casefold().split())
    if not claimed:
        return None
    boundary = r"(?<![\w.+/%−–—-]){}(?![\w%/]|\.\d)"
    exact = tuple(re.finditer(boundary.format(re.escape(claimed)), normalized))
    if re.fullmatch(r"[a-z0-9 ]+", claimed) is None:
        return (normalized, exact) if exact else None
    alternatives: list[str] = []
    claimed_tokens = claimed.split()
    for index, token in enumerate(claimed_tokens):
        number = next(
            (digit for digit, words in _QUANTITY_WORDS.items() if token == digit or token in words),
            None,
        )
        countable_article = (
            token in {"a", "an"}
            and _countable_unit_follows(claimed_tokens, index)
        )
        if number is None and not countable_article:
            alternatives.append(re.escape(token))
        else:
            number = number or "1"
            words = (number, *_QUANTITY_WORDS[number])
            if _countable_unit_follows(claimed_tokens, index) or (
                allow_bare_article and len(claimed_tokens) == 1 and number == "1"
            ):
                words += ("a", "an")
            alternatives.append("(?:" + "|".join(words) + ")")
    equivalent = tuple(re.finditer(boundary.format(r"\s+".join(alternatives)), normalized))
    matches = {match.span(): match for match in (*exact, *equivalent)}
    return None if not matches else (normalized, tuple(matches.values()))


_RELATION_FILLER = frozenset({
    "i", "we", "you", "ate", "had", "have", "eaten", "of", "the", "my", "our",
    "a", "an", "and", "only", "just", "actually", "for", "from", "those",
    "that", "it", "one", "item", "food",
})
_CORRECTION_MARKER = re.compile(
    r"\b(?:actually|rather|instead|correction|make that)\b|\bno(?=\s*[,;])"
)
_IDENTITY_NONFOOD = frozenset({
    "less", "more", "than", "almost", "about", "roughly", "approximately",
    "at", "least", "most", "up", "to", "under", "over", "around", "some",
    "serving", "servings", "official", "piece", "pieces", "quarter", "scoop",
    "scoops", "ounce", "ounces", "oz", "cup", "cups", "gram", "grams",
})


def _quantity_clause(text: str, match: re.Match[str]) -> tuple[str, str, str | None]:
    """Return the local clause, words outside the amount, and adjacent comma clause."""

    separators = list(re.finditer(r",|;|[.!?](?=\s|$)|\b(?:and|but)\b", text))
    previous = next((part for part in reversed(separators) if part.end() <= match.start()), None)
    start = previous.end() if previous is not None else 0
    end = min((part.start() for part in separators if part.start() >= match.end()), default=len(text))
    before = text[start:match.start()]
    after = text[match.end():end]
    prior_clause = None
    if previous is not None and previous.group() == ",":
        prior_start = max(
            (part.end() for part in separators if part.end() <= previous.start()), default=0,
        )
        prior_clause = text[prior_start:previous.start()]
    return text[start:end], before + " " + after, prior_clause


def _relation_words(outside: str) -> set[str]:
    """Any unexplained local word can change the amount or name its owner."""

    return set(re.findall(r"[a-z]+|\d+(?:\.\d+)?", outside.casefold())) - _RELATION_FILLER


def _later_local_correction(
    plan: MealPlan, item: PlannedMealItem, text: str, position: int,
) -> bool:
    """A subsequent correction targeting this relation cancels its earlier value."""

    for marker in _CORRECTION_MARKER.finditer(text, position):
        preceding = [
            part.strip() for part in re.split(
                r",|;|[.!?](?=\s|$)|\b(?:and|but)\b", text[:marker.start()],
            ) if part.strip()
        ]
        if not preceding:
            return True
        local = preceding[-1]
        named = [candidate for candidate in plan.items if _names_item(plan, candidate, local)]
        if named and item not in named:
            continue
        return True
    return False


def _food_head_tokens(item: PlannedMealItem) -> set[str]:
    """A lone food word must name the item, not just one of its ingredients."""

    heads: set[str] = set()
    for name in (item.display_name, item.record.name):
        words = re.findall(r"[a-z]+", name.casefold())
        while words and words[-1] in {"master"}:
            words.pop()
        if words:
            heads.update(_food_reference_tokens(words[-1]))
    return heads


def _planned_food_relation(
    plan: MealPlan,
    item: PlannedMealItem,
    clause: str,
    outside: str,
    prior_clause: str | None,
    quantity_reference_item_id: str | None,
    *,
    bare_count: bool = False,
) -> bool:
    if bare_count and _MEASUREMENT_AFTER_COUNT.search(outside):
        return False
    named = [candidate for candidate in plan.items if _names_item(plan, candidate, clause)]
    own = (
        _food_reference_tokens(item.display_name)
        | _food_reference_tokens(item.record.name)
        | set(re.findall(r"[a-z]+", (item.display_name + " " + item.record.name).casefold()))
    )
    remaining = _relation_words(outside) - own
    if remaining:
        return False
    if named:
        reference = _relation_words(outside)
        return (
            len(named) == 1
            and named[0] == item
            and (not bare_count or len(reference) != 1 or bool(reference & _food_head_tokens(item)))
        )
    if _relation_words(outside):
        return False
    if prior_clause is not None:
        preceding = [candidate for candidate in plan.items if _names_item(plan, candidate, prior_clause)]
        if preceding:
            reference = _relation_words(prior_clause)
            prior_words = reference - own
            return (
                len(preceding) == 1 and preceding[0] == item and not prior_words
                and (not bare_count or len(reference) != 1 or bool(reference & _food_head_tokens(item)))
            )
    return len(plan.items) == 1 or quantity_reference_item_id == plan.item_id(item)


_PLAN_REFERENT = re.compile(
    r"\b(?:recommendation|recommended|planned|plan|told me to)\b"
)
_RELATIVE_PRONOUN = re.compile(r"\b(?:of\s+)?(?:that|it)\b")


def _grounded_plan_fraction(
    plan: MealPlan,
    item: PlannedMealItem,
    phrase: str,
    user_text: str,
    quantity_reference_item_id: str | None,
) -> bool:
    """Prove a local fraction refers to this plan amount, not the physical food."""

    grounded = _grounded_quantity_matches(phrase, user_text)
    if grounded is None:
        return False
    text, matches = grounded
    if len(matches) != 1 or re.search(r"\b(?:negative|minus)\b", text):
        return False
    # A plan-relative fraction cannot erase another claim for this item merely
    # because the semantic result selected the fraction.
    selected = matches[0]
    own_tokens = _food_reference_tokens(item.display_name) | _food_reference_tokens(item.record.name)
    other_tokens = set().union(*(
        _food_reference_tokens(other.display_name) | _food_reference_tokens(other.record.name)
        for other in plan.items if other != item
    ))
    if _unaccounted_report_clause(
        text, selected.end(), own_tokens, other_tokens, before=selected.start(),
        food_head_tokens=_food_head_tokens(item),
        other_food_head_tokens=set().union(*(
            _food_head_tokens(other) for other in plan.items if other != item
        )),
    ):
        return False
    physical = tuple(_PHYSICAL_AMOUNT.finditer(text))
    for cue in _quantity_evidence_cues(text, physical):
        if selected.start() <= cue.start() and cue.end() <= selected.end():
            continue
        clause, _, _ = _quantity_clause(text, cue)
        named = [candidate for candidate in plan.items if _names_item(plan, candidate, clause)]
        if named and item not in named:
            continue
        before_cue = re.split(
            r",|;|[.!?](?=\s|$)|\b(?:and|but)\b", text[:cue.start()],
        )[-1]
        consumers = tuple(_QUANTITY_CONSUMER.finditer(before_cue))
        if consumers and consumers[-1].group("subject") not in {"i", "we"}:
            continue
        return False
    for consumer in _QUANTITY_CONSUMER.finditer(text, selected.end()):
        if consumer.group("subject") not in {"i", "we"}:
            continue
        clause, _, _ = _quantity_clause(text, consumer)
        named = [candidate for candidate in plan.items if _names_item(plan, candidate, clause)]
        if not named or item in named:
            return False
    for match in matches:
        if _later_local_correction(plan, item, text, match.end()):
            continue
        if any(
            amount.start() < match.end() and match.start() < amount.end()
            for amount in _PHYSICAL_AMOUNT.finditer(text)
        ):
            continue
        clause, _, prior_clause = _quantity_clause(text, match)
        named = [candidate for candidate in plan.items if _names_item(plan, candidate, clause)]
        if named and (len(named) != 1 or named[0] != item):
            continue
        if not named and len(plan.items) > 1 and quantity_reference_item_id != plan.item_id(item):
            continue
        cue = _PLAN_REFERENT.search(clause)
        if cue is not None:
            # Any food noun before the plan cue still describes a physical
            # fraction, including an unplanned food the model did not name.
            local_start = clause.find(match.group())
            if cue.start() < local_start:
                continue
            between = clause[local_start + len(match.group()):cue.start()]
            own = _food_reference_tokens(item.display_name) | _food_reference_tokens(item.record.name)
            if _relation_words(between) - {"what", "as", "like"}:
                continue
            if _relation_words(clause[cue.end():]) - own - {
                "amount", "portion", "serving", "servings", "what", "as", "like",
            }:
                continue
            return True
        # A lone percent or "half of that" may answer one deterministic plan
        # referent. A named food after the fraction instead names the physical
        # thing being divided, even in a one-item plan.
        if named or (prior_clause is not None and any(
            _names_item(plan, candidate, prior_clause) for candidate in plan.items
        )):
            continue
        after = clause[clause.find(match.group()) + len(match.group()):]
        if _food_reference_tokens(after) - {"it", "that"}:
            continue
        if (
            "%" in match.group()
            or "percent" in match.group()
            or _RELATIVE_PRONOUN.search(clause[clause.find(match.group()):])
            or (
                quantity_reference_item_id == plan.item_id(item)
                and _FRACTION_TEXT.fullmatch(phrase.casefold().strip()) is not None
            )
        ):
            return True
    return False


def _unplanned_local_contexts(
    user_text: str, quantity_text: str | None,
) -> tuple[tuple[str, str], ...]:
    if quantity_text is None:
        normalized = " ".join(user_text.casefold().split())
        return tuple((part, part) for part in re.split(r",|;|[.!?](?=\s|$)|\b(?:and|but)\b", normalized))
    grounded = _grounded_quantity_matches(
        quantity_text, user_text,
        allow_bare_article=_BARE_COUNT.fullmatch(quantity_text.casefold().strip()) is not None,
    )
    if grounded is None:
        return ()
    text, matches = grounded
    return tuple(
        (clause, outside)
        for match in matches
        for clause, outside, _ in (_quantity_clause(text, match),)
    )


def _literal_unplanned_food_text(user_text: str, quantity_text: str | None) -> str:
    """Retain user words, never a rejected model-selected menu identity."""

    contexts = _unplanned_local_contexts(user_text, quantity_text)
    if not contexts:
        contexts = _unplanned_local_contexts(user_text, None)
    for _, outside in contexts:
        words = [
            word for word in re.findall(r"[a-z]+", outside.casefold())
            if word not in _RELATION_FILLER and word not in _IDENTITY_NONFOOD
        ]
        if words:
            return " ".join(words)
    return user_text.strip()


def _grounded_unplanned_identity(
    food_text: str,
    quantity_text: str | None,
    food: ResolvedFood,
    plan: MealPlan,
    user_text: str,
    resolver: LocalFDFoodResolver,
) -> bool:
    """A model-selected menu name needs a unique literal local food reference."""

    name_tokens = _food_reference_tokens(food.record.name)
    list_candidates = getattr(resolver, "list_current_candidates", None)
    contexts = _unplanned_local_contexts(user_text, quantity_text)
    if not contexts:
        contexts = _unplanned_local_contexts(user_text, None)
    for clause, outside in contexts:
        reference = {
            word for word in _relation_words(outside) - _IDENTITY_NONFOOD
            if not word[0].isdigit()
        }
        if not reference or not reference <= name_tokens:
            continue
        if len(reference) < 2 and (
            food.resolution_method == "semantic"
            or not _literal_phrase_in(food_text, clause)
            or len(_food_reference_tokens(food.record.name)) != 1
        ):
            continue
        if callable(list_candidates):
            try:
                candidates = list_candidates(FoodResolutionRequest(
                    food_text, plan.service_date, meal=plan.meal,
                ))
            except Exception:
                continue
            matching = [
                candidate for candidate in candidates
                if reference <= _food_reference_tokens(candidate.official_display_name)
            ]
            if len(matching) != 1 or matching[0].source_identifier != food.source_identifier:
                continue
        elif food.resolution_method == "semantic":
            continue
        return True
    return False


def _grounded_unplanned_quantity(
    phrase: str, food: ResolvedFood, user_text: str,
) -> bool:
    """The entire local amount and food phrase must fit one menu identity."""

    name_tokens = _food_reference_tokens(food.record.name)
    return any(
        bool(_relation_words(outside))
        and _relation_words(outside) <= name_tokens
        for _, outside in _unplanned_local_contexts(user_text, phrase)
    )


def _grounded_unplanned_quantity_text(
    phrase: str, food: ResolvedFood, user_text: str,
    other_food_tokens: set[str] | None = None,
    other_food_head_tokens: set[str] | None = None,
) -> str | None:
    """Keep only a literal final amount owned by this unplanned food."""

    grounded = _grounded_unplanned_quantity(phrase, food, user_text)
    text = " ".join(user_text.casefold().split())
    amounts = tuple(_PHYSICAL_AMOUNT.finditer(text))
    selected = _grounded_quantity_matches(phrase, user_text)
    if selected is None:
        return None
    name_tokens = _food_reference_tokens(food.record.name)
    anchor = next((
        match for match in selected[1]
        if bool(_relation_words(_quantity_clause(text, match)[1]))
        and _relation_words(_quantity_clause(text, match)[1]) <= name_tokens
    ), selected[1][0])
    if _unaccounted_report_clause(
        text, anchor.end(), name_tokens, other_food_tokens or set(),
        before=anchor.start(),
        food_head_tokens=_food_reference_tokens(food.record.name.split()[-1]),
        other_food_head_tokens=other_food_head_tokens or set(),
    ):
        return None
    extra_named = _additional_named_quantity_cues(text, amounts, name_tokens)
    if _has_competing_followup_quantity_evidence(
        text, amounts, name_tokens, other_food_tokens=other_food_tokens,
    ):
        return None
    if extra_named and len(extra_named) + sum(
        bool(_food_reference_tokens(_quantity_clause(text, amount)[0]) & name_tokens)
        for amount in amounts
    ) > 1:
        return None
    owned: list[re.Match[str]] = []
    competing: list[re.Match[str]] = []
    for index, amount in enumerate(amounts):
        clause, outside, _ = _quantity_clause(text, amount)
        words = _relation_words(outside)
        if words and words <= name_tokens:
            owned.append(amount)
            competing.append(amount)
            continue
        # A quantity-only correction may inherit only its immediately prior
        # amount's owner. A new food name or unexplained words break the link.
        if (
            owned and index > 0 and amounts[index - 1] == owned[-1]
            and not (words - _CORRECTION_FILLER)
        ):
            owned.append(amount)
            competing.append(amount)
            continue
        # A connective can make an amount fail the strict local grammar while
        # still plainly naming this food. It must not disappear from the
        # per-food accounting and leave an earlier model choice authoritative.
        if (_food_reference_tokens(outside) & name_tokens) or (
            owned and words <= _FOLLOWUP_WORDS
        ):
            competing.append(amount)

    if len(competing) != len(owned):
        return None

    if len(owned) > 1:
        if any(
            not _local_quantity_correction(
                text, previous, current, amounts, name_tokens,
            )
            for previous, current in zip(owned, owned[1:])
        ):
            return None
        if not any(
            candidate.span() == amount.span()
            for candidate in selected[1] for amount in owned
        ):
            return None
        return owned[-1].group()

    if not grounded:
        return None
    if not owned:
        # Some official each-unit phrases (for example "half of one biscuit")
        # are deterministic for the portion interpreter but do not match the
        # generic physical-unit scanner. The full phrase and local food still
        # have to be literal, unique, and free of later correction.
        return phrase if len(selected[1]) == 1 and not _CORRECTION_MARKER.search(
            text[selected[1][0].end():]
        ) else None
    for candidate in selected[1]:
        if not any(candidate.span() == amount.span() for amount in owned):
            continue
        if _CORRECTION_MARKER.search(text[candidate.end():]):
            bare = _bare_local_correction(text, candidate, food_tokens=name_tokens)
            if bare is not None:
                return bare
            return None
        return phrase
    return None


def _grounded_modified_quantity(
    plan: MealPlan,
    item: PlannedMealItem,
    phrase: str,
    user_text: str,
    quantity_reference_item_id: str | None,
) -> bool:
    grounded = _grounded_quantity_matches(
        phrase, user_text,
        allow_bare_article=_BARE_COUNT.fullmatch(phrase.casefold().strip()) is not None,
    )
    if grounded is None:
        return False
    text, matches = grounded
    for match in matches:
        clause, outside, prior_clause = _quantity_clause(text, match)
        if _planned_food_relation(
            plan, item, clause, outside, prior_clause, quantity_reference_item_id,
            bare_count=_BARE_COUNT.fullmatch(phrase.casefold().strip()) is not None,
        ):
            return True
    return False


_CORRECTION_FILLER = frozenset({"no", "rather", "instead", "correction", "make"})
_FOLLOWUP_WORDS = frozenset({
    "later", "then", "afterward", "afterwards", "subsequently", "another",
    "also", "plus", "more", "i", "we", "had", "ate",
})
_LOCAL_QUANTITY_CUE = re.compile(
    r"\b(?:(?:half|quarter)\s+of\s+one|\d+(?:\.\d+)?|"
    r"one|two|three|four|five|six|seven|eight|"
    r"nine|ten|eleven|twelve|an?|half|quarter|"
    r"some|several|few|couple|a lot|little|bit|most|same amount)\b"
)
_QUANTITY_CONSUMER = re.compile(r"\b(?P<subject>[a-z]+)\s+(?:ate|had|consumed)\b")


def _unaccounted_report_clause(
    text: str, after: int, food_tokens: set[str], other_food_tokens: set[str],
    *, before: int | None = None,
    food_head_tokens: set[str] | None = None,
    other_food_head_tokens: set[str] | None = None,
) -> bool:
    """Reject report clauses outside one grounded relation unless accounted for.

    The clause need not contain a recognized amount or eating verb. Its meaning
    is deliberately left unresolved rather than inferred from a selected amount.
    """

    separators = tuple(re.finditer(
        r",|;|[.!?](?=\s|$)|\b(?:and|but)\b", text,
    ))
    starts = (0, *(separator.end() for separator in separators))
    ends = (*(separator.start() for separator in separators), len(text))
    correction_pending = False
    for start, end in zip(starts, ends):
        prior = before is not None and end <= before
        if start < after and not prior:
            continue
        clause = text[start:end].strip()
        if not clause:
            continue
        if not prior and _CORRECTION_MARKER.search(text[start:min(end + 1, len(text))]):
            # A stand-alone "no," corrects the following clause. The existing
            # local correction check decides whether that clause is valid.
            correction_pending = not bool(_quantity_evidence_cues(
                clause, tuple(_PHYSICAL_AMOUNT.finditer(clause)),
            ))
            continue
        if correction_pending:
            correction_pending = False
            continue
        named = _food_reference_tokens(clause)
        if (
            named & (other_food_head_tokens or set())
            and not named & (food_head_tokens or set())
        ):
            continue
        if named & other_food_tokens and not named & food_tokens:
            continue
        # A separately attributed person's statement cannot amend the user's
        # amount. A joint statement containing "I" or "we" remains unresolved.
        if not re.search(r"\b(?:i|we)\b", clause) and (
            re.match(r"^(?:he|she)\b", clause)
            or re.match(r"^(?:my|our|his|her|their)\s+friend\b", clause)
        ):
            continue
        if prior:
            # A food noun immediately before a comma amount is the supported
            # apposition, not another intake statement.
            if (
                end < len(text) and text[end] == ","
                and named & food_tokens
                and _relation_words(clause) <= food_tokens
                and not _quantity_evidence_cues(
                    clause, tuple(_PHYSICAL_AMOUNT.finditer(clause)),
                )
            ):
                continue
            if named & food_tokens or re.search(r"\b(?:i|we)\b", clause):
                return True
            continue
        return True
    return False


def _quantity_evidence_cues(
    text: str, physical: tuple[re.Match[str], ...],
) -> tuple[re.Match[str], ...]:
    """Find literal claim evidence without interpreting its amount."""

    cues = (
        *physical,
        *_LOCAL_QUANTITY_CUE.finditer(text),
        *_NON_FULL_AMOUNT.finditer(text),
        *_MEASUREMENT_AFTER_COUNT.finditer(text),
        *_PLAN_ATTESTATION.finditer(text),
    )
    return tuple(
        cue for cue in sorted(cues, key=lambda match: match.start())
        if (cue in physical or not any(
            amount.start() <= cue.start() and cue.end() <= amount.end()
            for amount in physical
        )) and (cue in physical or cue.group() not in {"a", "an", "not"})
    )


def _additional_named_quantity_cues(
    text: str, physical: tuple[re.Match[str], ...], food_tokens: set[str],
) -> tuple[re.Match[str], ...]:
    """Find food quantities outside the unit scanner, such as 'two potatoes'."""

    return tuple(
        count for count in _LOCAL_QUANTITY_CUE.finditer(text)
        if not any(
            amount.start() <= count.start() and count.end() <= amount.end()
            for amount in physical
        )
        and _food_reference_tokens(_quantity_clause(text, count)[0]) & food_tokens
    )


def _has_competing_followup_quantity_evidence(
    text: str, physical: tuple[re.Match[str], ...], food_tokens: set[str],
    *, allow_implicit_first: bool = False,
    other_food_tokens: set[str] | None = None,
) -> bool:
    """Keep later intake evidence from hiding behind an earlier food amount.

    This only blocks authority. It never assigns or interprets the later cue.
    An explicit different food in the follow-up clause breaks inheritance.
    """

    other_food_tokens = other_food_tokens or set()
    relevant_prior = False
    first_relevant_end: int | None = None
    for cue in _quantity_evidence_cues(text, physical):
        clause, _, _ = _quantity_clause(text, cue)
        clause_tokens = _food_reference_tokens(clause)
        if cue in physical:
            if clause_tokens & food_tokens or (
                allow_implicit_first and not (clause_tokens & other_food_tokens)
                and not relevant_prior
            ):
                relevant_prior = True
                if first_relevant_end is None:
                    first_relevant_end = cue.end()
            continue
        if not relevant_prior or (
            clause_tokens & other_food_tokens and not clause_tokens & food_tokens
        ):
            continue
        # The closest stated eater owns this cue. Another person's quantity
        # must not become a competing claim about the user's intake.
        before_cue = re.split(
            r",|;|[.!?](?=\s|$)|\b(?:and|but)\b", text[:cue.start()],
        )[-1]
        consumers = tuple(_QUANTITY_CONSUMER.finditer(before_cue))
        if consumers and consumers[-1].group("subject") not in {"i", "we"}:
            continue
        # A bare terminal digit can be a valid local correction; the existing
        # correction path must decide that relation, not this competing guard.
        if _CORRECTION_MARKER.search(clause):
            continue
        return True
    # A second first-person eating event without a distinct named food is
    # itself unresolved intake evidence. This catches wording outside the
    # known amount lexicon without assigning a quantity to it.
    if first_relevant_end is not None:
        for consumer in _QUANTITY_CONSUMER.finditer(text, first_relevant_end):
            if consumer.group("subject") not in {"i", "we"}:
                continue
            clause, _, _ = _quantity_clause(text, consumer)
            clause_tokens = _food_reference_tokens(clause)
            if (
                clause_tokens & other_food_tokens and not clause_tokens & food_tokens
            ) or _CORRECTION_MARKER.search(clause):
                continue
            return True
    return False


def _local_quantity_correction(
    text: str,
    previous: re.Match[str],
    current: re.Match[str],
    all_amounts: tuple[re.Match[str], ...],
    food_tokens: set[str],
) -> bool:
    """A correction must connect adjacent amounts of the same food."""

    if any(previous.end() <= amount.start() < current.start() for amount in all_amounts):
        return False
    between = text[previous.end():current.start()]
    for marker in _CORRECTION_MARKER.finditer(between):
        if _relation_words(between[:marker.start()]) - food_tokens:
            continue
        if _relation_words(between[marker.end():]) - _CORRECTION_FILLER:
            continue
        return True
    return False


def _bare_local_correction(
    text: str, previous: re.Match[str], *, food_tokens: set[str],
) -> str | None:
    """Inherit only the prior local unit for a terminal bare-number correction."""

    for marker in _CORRECTION_MARKER.finditer(text, previous.end()):
        if _relation_words(text[previous.end():marker.start()]) - food_tokens:
            continue
        number = re.fullmatch(
            r"\s*[,;]?\s*(\d+(?:\.\d+)?)\s*",
            text[marker.end():],
        )
        source = re.fullmatch(r"\S+\s+(.+)", previous.group())
        if number is not None and source is not None:
            return f"{number.group(1)} {source.group(1)}"
    return None


def _grounded_modified_quantity_text(
    plan: MealPlan,
    item: PlannedMealItem,
    phrase: str,
    user_text: str,
    quantity_reference_item_id: str | None,
) -> str | None:
    """Return the final literal amount when an unambiguous local correction wins."""

    grounded = _grounded_modified_quantity(
        plan, item, phrase, user_text, quantity_reference_item_id,
    )
    text = " ".join(user_text.casefold().split())
    physical = tuple(_PHYSICAL_AMOUNT.finditer(text))
    own_tokens = _food_reference_tokens(item.display_name) | _food_reference_tokens(item.record.name)
    extra_named = _additional_named_quantity_cues(text, physical, own_tokens)
    other_tokens = set().union(*(
        _food_reference_tokens(other.display_name) | _food_reference_tokens(other.record.name)
        for other in plan.items if other != item
    ))
    claimed = _grounded_quantity_matches(
        phrase, user_text,
        allow_bare_article=_BARE_COUNT.fullmatch(phrase.casefold().strip()) is not None,
    )
    anchor = None if claimed is None else next((
        match for match in claimed[1]
        if _planned_food_relation(
            plan, item, *_quantity_clause(text, match), quantity_reference_item_id,
            bare_count=_BARE_COUNT.fullmatch(phrase.casefold().strip()) is not None,
        )
    ), None)
    if anchor is None and claimed is not None:
        anchor = next((
            amount for match in claimed[1] for amount in physical
            if amount.end() < match.start()
            and _CORRECTION_MARKER.search(text[amount.end():match.start()])
            and _planned_food_relation(
                plan, item, *_quantity_clause(text, amount), quantity_reference_item_id,
            )
        ), None)
    if anchor is not None:
        corrected_source = next((
            amount for amount in physical
            if amount.end() < anchor.start()
            and _CORRECTION_MARKER.search(text[amount.end():anchor.start()])
            and _planned_food_relation(
                plan, item, *_quantity_clause(text, amount), quantity_reference_item_id,
            )
        ), None)
        if corrected_source is not None:
            anchor = corrected_source
    if anchor is None or _unaccounted_report_clause(
        text, anchor.end(), own_tokens, other_tokens, before=anchor.start(),
        food_head_tokens=_food_head_tokens(item),
        other_food_head_tokens=set().union(*(
            _food_head_tokens(other) for other in plan.items if other != item
        )),
    ):
        return None
    if _has_competing_followup_quantity_evidence(
        text, physical, own_tokens,
        allow_implicit_first=(
            len(plan.items) == 1 or quantity_reference_item_id == plan.item_id(item)
        ),
        other_food_tokens=other_tokens,
    ):
        return None
    if extra_named and len(extra_named) + sum(
        _names_item(plan, item, _quantity_clause(text, amount)[0])
        for amount in physical
    ) > 1:
        return None
    if len(physical) < 2:
        if grounded:
            for match in claimed[1]:
                if physical and match.start() > physical[0].end():
                    bare = _bare_local_correction(
                        text, physical[0], food_tokens=own_tokens,
                    )
                    if bare is not None and bare.split()[0] == match.group():
                        return bare
                if not _later_local_correction(plan, item, text, match.end()):
                    return phrase
                source = next((amount for amount in physical if amount.span() == match.span()), None)
                if source is not None:
                    bare = _bare_local_correction(text, source, food_tokens=own_tokens)
                    if bare is not None:
                        return bare
        return None

    owned: list[re.Match[str]] = []
    competing: list[re.Match[str]] = []
    for index, amount in enumerate(physical):
        clause, outside, prior_clause = _quantity_clause(text, amount)
        direct = _planned_food_relation(
            plan, item, clause, outside, prior_clause, quantity_reference_item_id,
        )
        if direct:
            owned.append(amount)
            competing.append(amount)
            continue
        # A quantity-only correction inherits exactly one immediately preceding
        # local owner. Any other food or unexplained words prevent inheritance.
        named = [candidate for candidate in plan.items if _names_item(plan, candidate, clause)]
        if (
            owned and index > 0 and physical[index - 1] == owned[-1]
            and not named
            and not (_relation_words(outside) - _CORRECTION_FILLER)
        ):
            owned.append(amount)
            competing.append(amount)
            continue
        # Strict ownership can reject a connective such as "later" or
        # "plus". A quantity that still names this food is competing evidence,
        # even if its relationship to the first amount is unsupported.
        if item in named or (
            owned and not named and _relation_words(outside) <= _FOLLOWUP_WORDS
        ):
            competing.append(amount)

    if len(competing) != len(owned):
        return None

    if not owned:
        return phrase if grounded else None
    if len(owned) == 1:
        if not grounded:
            return None
        if _later_local_correction(plan, item, text, owned[0].end()):
            return _bare_local_correction(text, owned[0], food_tokens=own_tokens)
        return phrase
    for previous, current in zip(owned, owned[1:]):
        if not _local_quantity_correction(
            text, previous, current, physical, own_tokens,
        ):
            return None
    claimed = _grounded_quantity_matches(phrase, user_text)
    if claimed is None or not any(
        selected.start() == amount.start() and selected.end() == amount.end()
        for selected in claimed[1] for amount in owned
    ):
        return None
    # The final phrase is still interpreted through the food's official unit
    # or frozen binding. A corrected scoop on an unbound food stays unresolved.
    return owned[-1].group()


_BARE_COUNT = re.compile(
    r"^(?:\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)$"
)
_MEASUREMENT_AFTER_COUNT = re.compile(
    r"\b(?:scoop(?:ful)?s?|ladle(?:ful)?s?|spoon(?:ful)?s?|"
    r"quarter\s+pieces?|pieces?|bowls?|cups?|ounces?|oz|grams?|"
    r"palms?(?:[ -]sized)?|handfuls?|servings?|counts?|items?|each|ea)\b"
)


def _grounded_draft_reuse(
    plan: MealPlan,
    item: PlannedMealItem,
    comparison_item: PlannedMealItem,
    user_text: str,
    inherited_quantities: Mapping[str, Decimal],
    quantity_reference_item_id: str | None,
) -> bool:
    if len(inherited_quantities) != 1 or inherited_quantities.get(plan.item_id(comparison_item)) is None:
        return False
    normalized = " ".join(user_text.casefold().split())
    reuse = re.compile(
        r"\b(?:same (?:amount|quantity|portion)(?: as (?:before|that))?|"
        r"same as (?:before|that)|that amount)\b"
    )
    for match in reuse.finditer(normalized):
        clause, outside, prior_clause = _quantity_clause(normalized, match)
        if _planned_food_relation(
            plan, item, clause, outside, prior_clause, quantity_reference_item_id,
        ) and not _later_local_correction(
            plan, item, normalized, match.end(),
        ) and not _unaccounted_report_clause(
            normalized, match.end(),
            _food_reference_tokens(item.display_name) | _food_reference_tokens(item.record.name),
            set().union(*(
                _food_reference_tokens(other.display_name) | _food_reference_tokens(other.record.name)
                for other in plan.items if other != item
            )),
            before=match.start(),
            food_head_tokens=_food_head_tokens(item),
            other_food_head_tokens=set().union(*(
                _food_head_tokens(other) for other in plan.items if other != item
            )),
        ):
            return True
    return False


def _literal_reported_quantity(user_text: str, claimed: str | None = None) -> str | None:
    matches = list(_PHYSICAL_AMOUNT.finditer(user_text))
    if not matches:
        return None
    if claimed is not None:
        words = re.findall(r"[a-z]+", claimed.casefold())
        if words:
            unit = words[-1]
            same_unit = [
                match for match in matches
                if unit in re.findall(r"[a-z]+", match.group().casefold())
            ]
            if len(same_unit) == 1:
                return same_unit[0].group()
    return matches[0].group() if len(matches) == 1 else None


# These are authority checks, not a food/intent parser. Unknown wording asks
# for clarification. In particular, display text is never consulted here.
_PHYSICAL_AMOUNT = re.compile(
    r"\b(?:\d+(?:\.\d+)?(?:\s*(?:%|percent))?|an?|one|two|three|four|five|"
    r"six|seven|eight|nine|ten|half)"
    r"\s+(?:of\s+)?(?:(?:an?|the)\s+)?"
    r"(?:(?:level|heaping|large|small|more|serving|quarter|tennis|golf)[ -]+)*"
    r"(?:scoop(?:ful)?s?|ladle(?:ful)?s?|spoons?|spoonfuls?|palms?(?:[ -]sized)?|handfuls?|"
    r"balls?(?:[ -]sized)?|pieces?|counts?|servings?|portions?|bowls?|cups?|"
    r"ounces?|oz|grams?|sandwich(?:es)?|meatballs?|eggs?|cookies?)\b",
    re.IGNORECASE,
)
_NON_FULL_AMOUNT = re.compile(
    r"\b(?:half|quarter|third|none|not|didn't|didn’t|less|more|percent|some|part|"
    r"bit|little|an?|few|several|couple|dozen|twice|double|most|almost|nearly|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|"
    r"scoop(?:ful)?s?|ladle(?:ful)?s?|spoon(?:ful)?s?|palms?|handfuls?|pieces?)\b|%|\d",
    re.IGNORECASE,
)


def _literal_reported_amount(user_text: str, reference_text: str) -> str:
    if _literal_phrase_in(reference_text, user_text) and _PHYSICAL_AMOUNT.search(reference_text):
        user_text = reference_text
    matches = list(_PHYSICAL_AMOUNT.finditer(user_text))
    return matches[0].group() if len(matches) == 1 else user_text


def _food_reference_tokens(value: str) -> set[str]:
    tokens = set(re.findall(r"[a-z]+", value.casefold())) - {
        "a", "an", "the", "of", "and", "with", "based", "plant", "all", "whole",
    }
    for token in tuple(tokens):
        if token.endswith("oes"):
            tokens.add(token[:-2])
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            tokens.add(token[:-1])
    return tokens


def _names_item(plan: MealPlan, item: PlannedMealItem, text: str) -> bool:
    """Require a distinctive literal food token for item-scoped attestations."""
    tokens = _food_reference_tokens
    own = tokens(item.display_name) | tokens(item.record.name)
    others = set().union(*(
        tokens(other.display_name) | tokens(other.record.name)
        for other in plan.items if other != item
    ))
    return bool((own - others) & tokens(text))


_PLAN_ATTESTATION = re.compile(
    r"\b(?:everything you recommended|what you recommended|"
    r"amounts? you (?:recommended|told me to)|"
    r"recommended amounts? of|the (?:whole |full )?recommended (?:amount|portion)|"
    r"(?:as|like) (?:you )?recommended|as planned|all)\b"
)
_WHOLE_MEAL_ATTESTATION = re.compile(
    r"\b(?:everything(?: you recommended)?|the whole meal(?: as (?:you )?recommended)?|"
    r"all of it (?:but|except)|what you recommended)\b"
)
_SHARED_PREPOSED_ATTESTATION = re.compile(
    r"\b(?:recommended amounts? of|all of the recommended)\s+"
    r"(?P<foods>[^.;!?]+?)(?=\s+but\b|[.;!?]|$)"
)
_PLAN_ATTESTATION_WORDS = frozenset({
    "as", "like", "recommended", "planned", "what", "amount", "amounts",
    "portion", "told", "me", "to", "whole", "full", "all", "everything",
    "meal", "eat", "well",
})
_MEAL_SCOPE_WORDS = frozenset({"breakfast", "lunch", "dinner", "brunch"})


def _authorized_plan_attestation(
    plan: MealPlan, item: PlannedMealItem, text: str, *, unambiguous_reference: bool,
) -> bool:
    """Require a whole-meal statement or an attestation local to this item."""

    normalized = " ".join(text.casefold().split())
    own_tokens = _food_reference_tokens(item.display_name) | _food_reference_tokens(item.record.name)
    other_tokens = set().union(*(
        _food_reference_tokens(other.display_name) | _food_reference_tokens(other.record.name)
        for other in plan.items if other != item
    ))
    own_heads = _food_head_tokens(item)
    other_heads = set().union(*(
        _food_head_tokens(other) for other in plan.items if other != item
    ))
    whole = _WHOLE_MEAL_ATTESTATION.search(normalized)
    whole_clause = None if whole is None else _quantity_clause(normalized, whole)[0]
    if whole is not None and (
        whole.group().startswith("all of it ")
        or re.match(r"\s+(?:but|except)\b", normalized[whole.end():]) is not None
        or not any(_names_item(plan, candidate, whole_clause) for candidate in plan.items)
    ):
        if (
            _PHYSICAL_AMOUNT.search(normalized)
            or _NON_FULL_AMOUNT.search(normalized)
            or _later_local_correction(plan, item, normalized, whole.end())
            or _unaccounted_report_clause(
                normalized, whole.end(), own_tokens, other_tokens, before=whole.start(),
                food_head_tokens=own_heads, other_food_head_tokens=other_heads,
            )
        ):
            return False
        exclusion = re.split(r"\b(?:but|except)\b", normalized, maxsplit=1)
        return len(exclusion) == 1 or not _names_item(plan, item, exclusion[1])
    for shared in _SHARED_PREPOSED_ATTESTATION.finditer(normalized):
        parts = [part.strip() for part in re.split(r"\band\b", shared.group("foods"))]
        if len(parts) < 2 or any(
            _PHYSICAL_AMOUNT.search(part) or _NON_FULL_AMOUNT.search(part)
            for part in parts
        ):
            continue
        owners = []
        for part in parts:
            named = [candidate for candidate in plan.items if _names_item(plan, candidate, part)]
            if len(named) != 1:
                break
            own = _food_reference_tokens(named[0].display_name) | _food_reference_tokens(named[0].record.name)
            if _relation_words(part) - own:
                break
            owners.append(named[0])
        if (
            len(owners) == len(parts) and len(set(owners)) == len(owners)
            and item in owners
            and not _later_local_correction(plan, item, normalized, shared.end())
            and not _unaccounted_report_clause(
                normalized, shared.end(), own_tokens, other_tokens, before=shared.start(),
                food_head_tokens=own_heads, other_food_head_tokens=other_heads,
            )
        ):
            return True
    for match in _PLAN_ATTESTATION.finditer(normalized):
        if _later_local_correction(plan, item, normalized, match.end()) or (
            _unaccounted_report_clause(
                normalized, match.end(), own_tokens, other_tokens, before=match.start(),
                food_head_tokens=own_heads, other_food_head_tokens=other_heads,
            )
        ):
            continue
        clause, _, _ = _quantity_clause(normalized, match)
        if _PHYSICAL_AMOUNT.search(clause) or _NON_FULL_AMOUNT.search(clause):
            continue
        named = [candidate for candidate in plan.items if _names_item(plan, candidate, clause)]
        own = own_tokens
        scope_words = _relation_words(clause) & _MEAL_SCOPE_WORDS
        if scope_words:
            provider_scope = canonical_meal_name(plan.meal)
            if provider_scope is None:
                continue
            allowed_scope = {provider_scope}
            if provider_scope == "lunch":
                # Product lunch and brunch may share FD period 2. Routing has
                # already selected the persisted product slot before this call.
                allowed_scope.add("brunch")
            if not scope_words <= allowed_scope:
                continue
        if _relation_words(clause) - own - _PLAN_ATTESTATION_WORDS - scope_words:
            continue
        if named:
            if len(named) == 1 and named[0] == item:
                return True
        elif len(plan.items) == 1 or unambiguous_reference:
            return True
    return False


_FRACTION_SUFFIX = r"(?:\s+of\s+(?:it|that|the recommended amount|what you recommended))?"
_PERCENT_TEXT = re.compile(
    rf"^(?P<percent>\d+(?:\.\d+)?)\s*(?:%|percent){_FRACTION_SUFFIX}$"
)
_FRACTION_TEXT = re.compile(
    rf"^(?P<numerator>\d+)\s*/\s*(?P<denominator>\d+){_FRACTION_SUFFIX}$"
)
_DECIMAL_FRACTION_TEXT = re.compile(
    rf"^(?P<decimal>0?\.\d+|1(?:\.0+)?){_FRACTION_SUFFIX}$"
)
_FRACTION_WORDS: dict[str, Decimal] = {
    "half": Decimal("0.5"),
    "a half": Decimal("0.5"),
    "one half": Decimal("0.5"),
    "half of it": Decimal("0.5"),
    "a half of it": Decimal("0.5"),
    "half of that": Decimal("0.5"),
    "half of the recommended amount": Decimal("0.5"),
    "half of what you recommended": Decimal("0.5"),
    "quarter": Decimal("0.25"),
    "a quarter": Decimal("0.25"),
    "one quarter": Decimal("0.25"),
    "quarter of it": Decimal("0.25"),
    "quarter of that": Decimal("0.25"),
    "a quarter of that": Decimal("0.25"),
    "quarter of what you recommended": Decimal("0.25"),
    "a quarter of it": Decimal("0.25"),
    "third": Decimal("0.333333333333333333333333333333333333"),
    "a third": Decimal("0.333333333333333333333333333333333333"),
    "one third": Decimal("0.333333333333333333333333333333333333"),
}


def _explicit_positive_fraction(value: object) -> Decimal | None:
    """Parse only an explicit user fraction for one planned quantity relation.

    This is deliberately not a physical scoop/ladle/container conversion.  It
    merely applies a phrase such as ``1/4`` or ``half`` to the exact quantity
    already stored on this one persisted plan item.
    """

    if not isinstance(value, str):
        return None
    normalized = " ".join(value.casefold().strip().split())
    if normalized in _FRACTION_WORDS:
        return _FRACTION_WORDS[normalized]
    percent_match = _PERCENT_TEXT.fullmatch(normalized)
    if percent_match is not None:
        percent = Decimal(percent_match.group("percent"))
        if not Decimal("0") < percent <= Decimal("100"):
            return None
        with localcontext() as context:
            context.prec = 36
            return percent / Decimal("100")
    decimal_match = _DECIMAL_FRACTION_TEXT.fullmatch(normalized)
    if decimal_match is not None:
        try:
            decimal_fraction = Decimal(decimal_match.group("decimal"))
        except (InvalidOperation, ValueError):
            return None
        return (
            decimal_fraction
            if decimal_fraction.is_finite() and Decimal("0") < decimal_fraction <= Decimal("1")
            else None
        )
    match = _FRACTION_TEXT.fullmatch(normalized)
    if match is None:
        return None
    try:
        numerator = Decimal(match.group("numerator"))
        denominator = Decimal(match.group("denominator"))
        if numerator <= 0 or denominator <= 0 or numerator > denominator:
            return None
        with localcontext() as context:
            context.prec = 36
            fraction = numerator / denominator
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return fraction if fraction.is_finite() and fraction > 0 else None


def _positive_decimal(value: Decimal, field_name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{field_name} must be finite and greater than zero")


def _text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")


def _occurrence_matches_meal(occurrence: object, meal: str | int) -> bool:
    occurrence_name = getattr(occurrence, "meal_period_name", None)
    occurrence_id = getattr(occurrence, "meal_period_id", None)
    if meal_identity_matches(meal, occurrence_id) or meal_identity_matches(meal, occurrence_name):
        return True
    if isinstance(meal, int):
        return occurrence_id == meal or str(occurrence_id) == str(meal)
    return isinstance(occurrence_name, str) and occurrence_name.casefold() == meal.casefold()
