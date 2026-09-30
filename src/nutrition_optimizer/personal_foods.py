"""Luke's narrow recommendation preferences and one approved personal estimate."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import re

from .fdmealplanner.catalog import FDMenuOccurrence
from .nutrition.models import (
    NutrientProfile, NutritionProvenance, NutritionRecord, Serving, SourceIdentifier,
)


DERIVED_BISCUIT_PROVIDER = "personal_derived_estimate"
_BISCUIT_ESTIMATE_VERSION = "luke-biscuit-v1"
_BISCUIT_PARENT_NAME = "biscuits and gravy"


def is_biscuit_parent(occurrence: FDMenuOccurrence) -> bool:
    """Only the approved FD composite can offer the personal biscuit estimate."""

    return (
        occurrence.nutrition_record.provenance.provider == "FDMealPlanner"
        and " ".join(occurrence.nutrition_record.name.casefold().split()) == _BISCUIT_PARENT_NAME
    )


def is_personally_excluded(record: NutritionRecord) -> bool:
    """Exclude genuine plant-based egg/sausage substitutes by food identity."""

    name = " ".join(re.findall(r"[a-z0-9]+", record.name.casefold()))
    qualifier = r"(?:vegan|plant based|meatless|vegetarian|veggie|beyond(?: meat)?|impossible)"
    nearby = r"(?:\s+(?:scrambled|liquid|breakfast|style)){0,2}\s+"
    return bool(
        re.search(rf"\b{qualifier}{nearby}eggs?\b|\bjust egg\b", name)
        or re.search(
            rf"\b{qualifier}{nearby}(?:sausages?|breakfast patt(?:y|ies))\b",
            name,
        )
    )


def derived_biscuit_definition(
    parent: FDMenuOccurrence,
) -> tuple[NutritionRecord, SourceIdentifier, str]:
    """Return the immutable per-biscuit estimate and parent-linked identity."""

    if not is_biscuit_parent(parent):
        raise ValueError("personal biscuit estimate requires Biscuits and Gravy")
    source = SourceIdentifier(
        "personal_biscuit",
        f"{parent.source_identifier.kind}:{parent.source_identifier.value}",
    )
    signature = sha256(
        f"{_BISCUIT_ESTIMATE_VERSION}:{parent.content_signature}".encode("utf-8")
    ).hexdigest()
    record = NutritionRecord(
        name="Biscuit",
        serving=Serving(Decimal("1"), "Each", "1 Each"),
        nutrients=NutrientProfile(
            calories_kcal=Decimal("200"), protein_g=Decimal("5"),
            carbohydrates_g=Decimal("24"), fat_g=Decimal("10"),
            dietary_fiber_g=Decimal("1"),
        ),
        provenance=NutritionProvenance(
            provider=DERIVED_BISCUIT_PROVIDER,
            retrieved_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
            record_type="approximate_personal_estimate",
            source_reference=(
                "Approved per-biscuit estimate; parent FDMealPlanner "
                f"{_BISCUIT_PARENT_NAME} {parent.source_identifier.kind}:"
                f"{parent.source_identifier.value} {parent.content_signature}"
            ),
            identifiers=(source,),
        ),
    )
    return record, source, signature
