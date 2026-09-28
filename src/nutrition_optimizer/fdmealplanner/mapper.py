"""Strict conversion of FDMealPlanner recipe components into domain records."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import re
from typing import Any
import xml.etree.ElementTree as ET

from nutrition_optimizer.nutrition.models import (
    NutritionProvenance,
    NutritionRecord,
    NutrientProfile,
    Serving,
    SourceIdentifier,
)

from .models import FDMealOccurrence, FDMealsPayload

SUPPORTED_RECIPE_COMPONENT_TYPE = 181
EXPECTED_UOMS = {
    "calories": "kcal",
    "protein": "g",
    "carbohydrates": "g",
    "fat": "g",
    "sodium": "mg",
}

OPTIONAL_NUTRIENT_UOMS = {
    "dietaryFiber": "g",
}


@dataclass(frozen=True, slots=True)
class RecordDiagnostic:
    """Machine-readable reason a public component was not mapped."""

    reason: str
    message: str
    component_id: int | None = None
    component_type_id: int | None = None
    display_name: str | None = None


@dataclass(frozen=True, slots=True)
class FDMappingResult:
    """Either one valid domain record or one explicit rejection diagnostic."""

    record: NutritionRecord | None
    diagnostic: RecordDiagnostic | None
    occurrence: FDMealOccurrence | None = None
    content_signature: str | None = None

    @property
    def accepted(self) -> bool:
        return self.record is not None


@dataclass(frozen=True, slots=True)
class FDMappingBatch:
    """Mapped monthly occurrences, retaining valid records and diagnostics."""

    results: tuple[FDMappingResult, ...]

    @property
    def records(self) -> tuple[NutritionRecord, ...]:
        return tuple(result.record for result in self.results if result.record is not None)

    @property
    def diagnostics(self) -> tuple[RecordDiagnostic, ...]:
        return tuple(
            result.diagnostic for result in self.results if result.diagnostic is not None
        )


def _diagnostic(
    reason: str,
    message: str,
    *,
    component_id: int | None = None,
    component_type_id: int | None = None,
    display_name: str | None = None,
) -> RecordDiagnostic:
    return RecordDiagnostic(
        reason=reason,
        message=message,
        component_id=component_id,
        component_type_id=component_type_id,
        display_name=display_name,
    )


def _parse_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        return None
    if not parsed.is_finite() or parsed < 0:
        return None
    return parsed


def _parse_positive_integer(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        return None
    if not parsed.is_finite() or parsed <= 0 or parsed != parsed.to_integral_value():
        return None
    return int(parsed)


def _display_name(component: Mapping[str, Any]) -> str | None:
    # englishAlternateName is the public UI label.  componentName is retained
    # as the narrow fallback used by the frontend when the label is absent.
    for field in ("englishAlternateName", "componentName"):
        value = component.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def classify_component(component: Mapping[str, Any]) -> RecordDiagnostic | None:
    """Return a concise rejection reason, or ``None`` when mapping is safe."""

    if not isinstance(component, Mapping):
        return _diagnostic("invalid_payload", "component is not an object")

    component_type_id = _parse_positive_integer(component.get("componentTypeId"))
    component_id = _parse_positive_integer(component.get("componentId"))
    display_name = _display_name(component)

    if component_type_id == 180:
        return _diagnostic(
            "unsupported_component_type",
            "product-like componentTypeId is not a per-serving recipe record",
            component_id=component_id,
            component_type_id=component_type_id,
            display_name=display_name,
        )
    if component_type_id is None or component_id is None or display_name is None:
        return _diagnostic(
            "missing_identity",
            "component type, component ID, and display name are required",
            component_id=component_id,
            component_type_id=component_type_id,
            display_name=display_name,
        )
    if component_type_id != SUPPORTED_RECIPE_COMPONENT_TYPE:
        return _diagnostic(
            "unsupported_component_type",
            "component type is not a supported recipe type",
            component_id=component_id,
            component_type_id=component_type_id,
            display_name=display_name,
        )

    serving_quantity = _parse_decimal(component.get("recipePortionSize"))
    serving_raw = component.get("recipePortionSize")
    if serving_raw is None or (isinstance(serving_raw, str) and not serving_raw.strip()):
        return _diagnostic(
            "missing_serving",
            "recipe portion size is absent",
            component_id=component_id,
            component_type_id=component_type_id,
            display_name=display_name,
        )
    if serving_quantity is None or serving_quantity <= 0:
        return _diagnostic(
            "invalid_serving",
            "recipe portion size must be finite and positive",
            component_id=component_id,
            component_type_id=component_type_id,
            display_name=display_name,
        )
    serving_unit = component.get("recipePortionSizeUnit")
    if not isinstance(serving_unit, str) or not serving_unit.strip():
        return _diagnostic(
            "missing_serving",
            "recipe portion unit is absent",
            component_id=component_id,
            component_type_id=component_type_id,
            display_name=display_name,
        )

    for nutrient, expected_uom in EXPECTED_UOMS.items():
        value = _parse_decimal(component.get(nutrient))
        if value is None:
            return _diagnostic(
                "invalid_nutrient",
                f"{nutrient} must be a finite nonnegative number",
                component_id=component_id,
                component_type_id=component_type_id,
                display_name=display_name,
            )
        uom = component.get(f"{nutrient}UOM")
        if not isinstance(uom, str) or uom != expected_uom:
            return _diagnostic(
                "unexpected_uom",
                f"{nutrient} must use {expected_uom}",
                component_id=component_id,
                component_type_id=component_type_id,
                display_name=display_name,
            )
    for nutrient, expected_uom in OPTIONAL_NUTRIENT_UOMS.items():
        raw_value = component.get(nutrient)
        if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
            continue
        if _parse_decimal(raw_value) is None:
            return _diagnostic(
                "invalid_nutrient",
                f"{nutrient} must be a finite nonnegative number when supplied",
                component_id=component_id,
                component_type_id=component_type_id,
                display_name=display_name,
            )
        uom = component.get(f"{nutrient}UOM")
        if not isinstance(uom, str) or uom != expected_uom:
            return _diagnostic(
                "unexpected_uom",
                f"{nutrient} must use {expected_uom}",
                component_id=component_id,
                component_type_id=component_type_id,
                display_name=display_name,
            )
    return None


def stable_source_identifier(
    *,
    tenant_id: int,
    component_type_id: int,
    component_id: int,
) -> SourceIdentifier:
    """Build the stable provider identity, excluding occurrence context."""

    tenant = _parse_positive_integer(tenant_id)
    component_type = _parse_positive_integer(component_type_id)
    component = _parse_positive_integer(component_id)
    if tenant is None or component_type is None or component is None:
        raise ValueError("tenant and component identity values must be positive integers")
    return SourceIdentifier(kind="component", value=f"{tenant}:{component_type}:{component}")


def _source_text_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ()


def is_student_visible_component(component: Mapping[str, Any]) -> bool:
    """Return whether the public MealPlanner UI would display a component.

    The guest calendar uses ``isShowOnMealPlanner`` for food-bar records when
    that field is present, and ``isShowOnMenu`` for ordinary records.  Missing
    flags are deliberately treated as not visible.
    """

    if not isinstance(component, Mapping):
        return False
    is_food_bar = component.get("isFoodBar")
    if _truthy_flag(is_food_bar) and "isShowOnMealPlanner" in component:
        return _truthy_flag(component.get("isShowOnMealPlanner"))
    return _truthy_flag(component.get("isShowOnMenu"))


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return isinstance(value, str) and value.strip().casefold() in {"1", "true", "yes", "y"}


def _serving_text(quantity: Decimal, unit: str) -> str:
    normalized = format(quantity.normalize(), "f")
    return f"{normalized} {unit.strip()}"


def map_component(
    component: Mapping[str, Any],
    *,
    tenant_id: int,
    retrieved_at: datetime | None = None,
    source_reference: str | None = None,
    occurrence: FDMealOccurrence | None = None,
) -> FDMappingResult:
    """Strictly map one FD recipe component into ``NutritionRecord``."""

    diagnostic = classify_component(component)
    if diagnostic is not None:
        return FDMappingResult(record=None, diagnostic=diagnostic, occurrence=occurrence)

    # The classifier guarantees these values, but parse again to keep this
    # function locally deterministic if fields are changed between calls.
    component_id = _parse_positive_integer(component.get("componentId"))
    component_type_id = _parse_positive_integer(component.get("componentTypeId"))
    name = _display_name(component)
    serving_quantity = _parse_decimal(component.get("recipePortionSize"))
    serving_unit = component.get("recipePortionSizeUnit")
    if (
        component_id is None
        or component_type_id is None
        or name is None
        or serving_quantity is None
        or not isinstance(serving_unit, str)
    ):
        # This is unreachable for ordinary mappings, but prevents a future
        # mutation between validation and mapping from fabricating a record.
        return FDMappingResult(
            record=None,
            diagnostic=_diagnostic("invalid_payload", "component changed during validation"),
            occurrence=occurrence,
        )

    nutrients: dict[str, Decimal] = {}
    for nutrient in EXPECTED_UOMS:
        parsed = _parse_decimal(component.get(nutrient))
        if parsed is None:
            return FDMappingResult(
                record=None,
                diagnostic=_diagnostic("invalid_nutrient", f"{nutrient} could not be parsed"),
                occurrence=occurrence,
            )
        nutrients[nutrient] = parsed
    dietary_fiber = _parse_decimal(component.get("dietaryFiber"))

    if retrieved_at is None:
        retrieved_at = datetime.now(timezone.utc)
    identifier = stable_source_identifier(
        tenant_id=tenant_id,
        component_type_id=component_type_id,
        component_id=component_id,
    )
    record = NutritionRecord(
        name=name,
        serving=Serving(
            quantity=serving_quantity,
            unit=serving_unit.strip(),
            text=_serving_text(serving_quantity, serving_unit),
        ),
        nutrients=NutrientProfile(
            calories_kcal=nutrients["calories"],
            protein_g=nutrients["protein"],
            carbohydrates_g=nutrients["carbohydrates"],
            fat_g=nutrients["fat"],
            sodium_mg=nutrients["sodium"],
            dietary_fiber_g=dietary_fiber,
        ),
        provenance=NutritionProvenance(
            provider="FDMealPlanner",
            retrieved_at=retrieved_at,
            record_type="recipe",
            source_reference=source_reference,
            identifiers=(identifier,),
        ),
        ingredients=(
            (component["ingredientStatement"],)
            if isinstance(component.get("ingredientStatement"), str)
            and component["ingredientStatement"].strip()
            else ()
        ),
        allergens=_source_text_tuple(component.get("allergenName")),
    )
    return FDMappingResult(
        record=record,
        diagnostic=None,
        occurrence=occurrence,
        content_signature=content_signature(component),
    )


def _canonical_content_value(value: Any, *, numeric: bool = False) -> Any:
    if numeric and value is not None and not isinstance(value, bool):
        try:
            parsed = Decimal(str(value).strip())
        except (InvalidOperation, ValueError, TypeError):
            parsed = None
        if parsed is not None and parsed.is_finite():
            return str(parsed.normalize())
    if isinstance(value, Decimal):
        return str(value.normalize())
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_content_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_content_value(item) for item in value]
    return value


def content_signature(component: Mapping[str, Any]) -> str:
    """Return a deterministic digest of stable nutrition/source content.

    Menu dates, stations, meal periods, and ``MenuDetailId`` are intentionally
    absent so occurrence movement does not look like a catalog revision.
    """

    fields = (
        "componentId",
        "componentTypeId",
        "englishAlternateName",
        "componentName",
        "descriptionEnglish",
        "recipePortionSize",
        "recipePortionSizeUnit",
        "productMeasuringSize",
        "productMeasuringSizeUnit",
        "calories",
        "caloriesUOM",
        "protein",
        "proteinUOM",
        "carbohydrates",
        "carbohydratesUOM",
        "fat",
        "fatUOM",
        "sodium",
        "sodiumUOM",
        "cholesterol",
        "cholesterolUOM",
        "dietaryFiber",
        "dietaryFiberUOM",
        "totalSugars",
        "totalSugarsUOM",
        "saturatedFat",
        "saturatedFatUOM",
        "transFattyAcid",
        "transFattyAcidUOM",
        "calcium",
        "calciumUOM",
        "iron",
        "ironUOM",
        "vitaminA",
        "vitaminAUOM",
        "vitaminC",
        "vitaminCUOM",
        "ingredientStatement",
        "allergenName",
        "recipeProductDietaryName",
    )
    numeric_fields = {
        "componentId",
        "componentTypeId",
        "recipePortionSize",
        "productMeasuringSize",
        "calories",
        "protein",
        "carbohydrates",
        "fat",
        "sodium",
        "cholesterol",
        "dietaryFiber",
        "totalSugars",
        "saturatedFat",
        "transFattyAcid",
        "calcium",
        "iron",
        "vitaminA",
        "vitaminC",
    }
    content = {
        field: _canonical_content_value(component.get(field), numeric=field in numeric_fields)
        for field in fields
    }
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


_DETAIL_RE = re.compile(
    r"<Details\b[^>]*\bMenuDetailId=[\"']([^\"']+)[\"'][^>]*\bComponentId=[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)


def _menu_detail_queues(day: Mapping[str, Any]) -> dict[str, deque[str]]:
    raw_xml = day.get("xmlMenuRecipes")
    if not isinstance(raw_xml, str) or not raw_xml.strip():
        return {}
    queues: dict[str, deque[str]] = defaultdict(deque)
    for menu_detail_id, component_id in _DETAIL_RE.findall(raw_xml):
        queues[component_id].append(menu_detail_id)
    # A malformed or unexpected XML string must not prevent nutrition parsing;
    # the occurrence ID is optional context only.  ElementTree is used only as
    # a cheap well-formedness check for strings that look like XML.
    if not queues:
        try:
            ET.fromstring(raw_xml)
        except ET.ParseError:
            return {}
    return queues


def _integer_or_text(value: Any) -> int | str | None:
    parsed = _parse_positive_integer(value)
    return parsed if parsed is not None else (value if isinstance(value, str) and value.strip() else None)


def iter_meal_occurrences(
    payload: FDMealsPayload | Mapping[str, Any],
    *,
    default_meal_period_id: int | None = None,
    default_meal_period_name: str | None = None,
) -> tuple[FDMealOccurrence, ...]:
    """Flatten daily/station recipe arrays while retaining occurrence context."""

    days = payload.days if isinstance(payload, FDMealsPayload) else FDMealsPayload.from_payload(payload).days
    occurrences: list[FDMealOccurrence] = []
    global_station_by_row: dict[str, tuple[int | str | None, str | None]] = {}
    for day in days:
        concepts = day.get("conceptData")
        if not isinstance(concepts, list):
            continue
        for concept in concepts:
            if not isinstance(concept, Mapping) or concept.get("rowId") is None:
                continue
            global_station_by_row[str(concept["rowId"])] = (
                _integer_or_text(concept.get("conceptId")),
                concept.get("conceptName")
                if isinstance(concept.get("conceptName"), str)
                else None,
            )
    for day in days:
        station_by_row = dict(global_station_by_row)
        concepts = day.get("conceptData")
        if isinstance(concepts, list):
            for concept in concepts:
                if not isinstance(concept, Mapping):
                    continue
                row_id = concept.get("rowId")
                if row_id is None:
                    continue
                station_by_row[str(row_id)] = (
                    _integer_or_text(concept.get("conceptId")),
                    concept.get("conceptName") if isinstance(concept.get("conceptName"), str) else None,
                )
        detail_queues = _menu_detail_queues(day)
        recipes = day.get("allMenuRecipes")
        if not isinstance(recipes, list):
            continue
        menu_date = day.get("strMenuForDate") or day.get("menuForDate")
        menu_date_text = menu_date.strip() if isinstance(menu_date, str) and menu_date.strip() else None
        meal_period_id = _integer_or_text(day.get("mealPeriodId"))
        if not isinstance(meal_period_id, int):
            meal_period_id = default_meal_period_id
        meal_period_name_value = day.get("mealPeriodName") or day.get("mealName")
        if isinstance(meal_period_name_value, str) and meal_period_name_value.strip():
            meal_period_name = meal_period_name_value.strip()
        elif isinstance(default_meal_period_name, str) and default_meal_period_name.strip():
            meal_period_name = default_meal_period_name.strip()
        else:
            meal_period_name = None
        menu_id = _integer_or_text(day.get("menuId"))
        occurrence_ordinals: dict[tuple[str | None, str, str, str, str], int] = defaultdict(int)
        for component in recipes:
            if not isinstance(component, Mapping):
                continue
            station = station_by_row.get(str(component.get("rowId")))
            component_id = component.get("componentId")
            queue = detail_queues.get(str(component_id))
            direct_menu_detail_id = component.get("MenuDetailId")
            if isinstance(direct_menu_detail_id, (str, int)) and str(direct_menu_detail_id).strip():
                menu_detail_id = str(direct_menu_detail_id)
            else:
                menu_detail_id = queue.popleft() if queue else None
            ordinal_key = (
                menu_date_text,
                str(meal_period_id),
                str(station[0]) if station else "",
                str(station[1]) if station else "",
                f"{component.get('componentTypeId')}:{component.get('componentId')}",
            )
            occurrence_ordinal = occurrence_ordinals[ordinal_key]
            occurrence_ordinals[ordinal_key] += 1
            occurrences.append(
                FDMealOccurrence(
                    component=component,
                    menu_date=menu_date_text,
                    meal_period_id=meal_period_id,
                    station_concept_id=station[0] if station else None,
                    station_name=station[1] if station else None,
                    menu_detail_id=menu_detail_id,
                    menu_id=menu_id,
                    meal_period_name=meal_period_name,
                    occurrence_ordinal=occurrence_ordinal,
                )
            )
    return tuple(occurrences)


def map_meals_payload(
    payload: FDMealsPayload | Mapping[str, Any],
    *,
    tenant_id: int,
    retrieved_at: datetime | None = None,
    source_reference: str | None = None,
    default_meal_period_id: int | None = None,
    default_meal_period_name: str | None = None,
) -> FDMappingBatch:
    """Map all recipe occurrences in one bounded monthly response."""

    results = tuple(
        map_component(
            occurrence.component,
            tenant_id=tenant_id,
            retrieved_at=retrieved_at,
            source_reference=source_reference,
            occurrence=occurrence,
        )
        for occurrence in iter_meal_occurrences(
            payload,
            default_meal_period_id=default_meal_period_id,
            default_meal_period_name=default_meal_period_name,
        )
    )
    return FDMappingBatch(results=results)
