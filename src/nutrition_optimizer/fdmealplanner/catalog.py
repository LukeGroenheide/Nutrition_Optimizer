"""SQLite catalog for immutable official FD nutrition and menu snapshots."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Literal

from nutrition_optimizer.nutrition.models import (
    NutritionProvenance,
    NutritionRecord,
    NutrientProfile,
    Serving,
    SourceIdentifier,
)

from ..meal_identity import (
    canonical_meal_id,
    canonical_meal_name,
    legacy_meal_slot,
    meal_context_key,
)
from .mapper import (
    FDMappingBatch,
    FDMappingResult,
    classify_component,
    content_signature as fd_content_signature,
    is_student_visible_component,
)
from .models import FDMealOccurrence

DEFAULT_CATALOG_PATH = Path("data/state/nutrition.sqlite3")
CATALOG_SCHEMA_VERSION = 14
DEFAULT_PROVIDER = "FDMealPlanner"


class NutritionCatalogError(RuntimeError):
    """Base error for local official-nutrition catalog operations."""


class NutritionCatalogValidationError(NutritionCatalogError):
    """An observation or source identity is not safe to persist."""


class NutritionCatalogSerializationError(NutritionCatalogError):
    """A persisted snapshot is malformed and cannot be reconstructed safely."""


@dataclass(frozen=True, slots=True)
class CatalogObservation:
    """One official record observation supplied to the catalog."""

    record: NutritionRecord
    content_signature: str
    observed_at: datetime
    source_identifier: SourceIdentifier | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.record, NutritionRecord):
            raise TypeError("catalog observation record must be a NutritionRecord")
        if not isinstance(self.content_signature, str) or not self.content_signature.strip():
            raise ValueError("catalog observation content_signature must be non-empty text")
        _normalize_timestamp(self.observed_at)
        if self.source_identifier is not None and not isinstance(
            self.source_identifier, SourceIdentifier
        ):
            raise TypeError("catalog observation source_identifier must be a SourceIdentifier")


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    """One immutable serialized nutrition snapshot and observation metadata."""

    snapshot_id: int
    provider: str
    source_identifier: SourceIdentifier
    content_signature: str
    record: NutritionRecord
    first_observed_at: datetime
    last_observed_at: datetime


CatalogOutcome = Literal["inserted", "unchanged", "new_version"]


@dataclass(frozen=True, slots=True)
class CatalogWriteResult:
    """Result of one idempotent catalog observation."""

    outcome: CatalogOutcome
    snapshot: CatalogSnapshot


@dataclass(frozen=True, slots=True)
class DietaryFiberBackfillResult:
    """Exact-signature result of an immutable dietary-fiber backfill."""

    source_observations: int
    matched_snapshots: int
    inserted: int
    unchanged: int
    unmatched: tuple[tuple[SourceIdentifier, str], ...]


@dataclass(frozen=True, slots=True)
class FDRefreshResult:
    """Compact result of applying mapped FD records to the catalog."""

    inserted: int
    unchanged: int
    new_versions: int
    rejected: int
    writes: tuple[CatalogWriteResult, ...]


@dataclass(frozen=True, slots=True)
class FDMenuOccurrence:
    """One locally cached published FD menu occurrence and its exact record."""

    occurrence_id: int
    occurrence_key: str
    service_date: date
    meal_period_id: int | str
    meal_period_name: str
    station_concept_id: int | str | None
    station_name: str | None
    source_identifier: SourceIdentifier
    content_signature: str
    nutrition_snapshot_id: int
    nutrition_record: NutritionRecord
    menu_detail_id: str | None
    menu_id: int | str | None
    first_observed_at: datetime
    last_observed_at: datetime

    @property
    def meal(self) -> str:
        """Return the display name used for the meal-period filter."""

        return self.meal_period_name

    @property
    def station(self) -> str | None:
        """Return the station display name used for the station filter."""

        return self.station_name

    @property
    def record(self) -> NutritionRecord:
        """Compatibility alias for callers that use ``record`` terminology."""

        return self.nutrition_record

    @property
    def nutrition(self) -> NutritionRecord:
        """Short alias for application code that calls the record nutrition."""

        return self.nutrition_record


@dataclass(frozen=True, slots=True)
class FDMenuRefreshResult:
    """Atomic nutrition-plus-menu result for one completed FD refresh."""

    catalog: FDRefreshResult
    refresh_id: int
    occurrences_observed: int
    current_logical_occurrences: int
    occurrence_additions: int
    occurrence_changes: int
    occurrence_removals: int

    @property
    def current_occurrence_count(self) -> int:
        return self.current_logical_occurrences


@dataclass(frozen=True, slots=True)
class _OccurrenceObservation:
    """Validated occurrence fields ready for one normalized database row."""

    occurrence_key: str
    service_date: str
    meal_period_id: str
    meal_period_name: str
    station_concept_id: str | None
    station_name: str | None
    source_identifier: SourceIdentifier
    content_signature: str
    nutrition_snapshot_id: int
    menu_detail_id: str | None
    menu_id: str | None


def _normalize_timestamp(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("observation timestamps must be datetime values")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("observation timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return _normalize_timestamp(value).isoformat(timespec="microseconds")


def _parse_timestamp(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise NutritionCatalogSerializationError(f"snapshot {field_name} is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise NutritionCatalogSerializationError(f"snapshot {field_name} is invalid") from exc
    try:
        return _normalize_timestamp(parsed)
    except (TypeError, ValueError) as exc:
        raise NutritionCatalogSerializationError(f"snapshot {field_name} is not timezone-aware") from exc


def _decimal_to_json(value: Decimal | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, Decimal) or not value.is_finite():
        raise NutritionCatalogValidationError("nutrition values must be finite Decimals or None")
    return str(value)


def _decimal_from_json(value: Any, *, field_name: str) -> Decimal | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise NutritionCatalogSerializationError(f"snapshot {field_name} must be a Decimal string or null")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise NutritionCatalogSerializationError(f"snapshot {field_name} is not a Decimal string") from exc
    if not parsed.is_finite():
        raise NutritionCatalogSerializationError(f"snapshot {field_name} is not finite")
    return parsed


def _required_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NutritionCatalogSerializationError(f"snapshot {field_name} must be non-empty text")
    return value


def _optional_text(value: Any, *, field_name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise NutritionCatalogSerializationError(f"snapshot {field_name} must be text or null")
    return value


def _text_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise NutritionCatalogSerializationError(f"snapshot {field_name} must be a JSON list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise NutritionCatalogSerializationError(f"snapshot {field_name} contains invalid text")
    return tuple(value)


def nutrition_record_to_mapping(
    record: NutritionRecord,
    *,
    include_dietary_fiber: bool = True,
) -> dict[str, Any]:
    """Return the deterministic JSON-compatible representation of a record."""

    if not isinstance(record, NutritionRecord):
        raise TypeError("record must be a NutritionRecord")
    nutrients: dict[str, str | None] = {
        "calories_kcal": _decimal_to_json(record.nutrients.calories_kcal),
        "protein_g": _decimal_to_json(record.nutrients.protein_g),
        "carbohydrates_g": _decimal_to_json(record.nutrients.carbohydrates_g),
        "fat_g": _decimal_to_json(record.nutrients.fat_g),
        "sodium_mg": _decimal_to_json(record.nutrients.sodium_mg),
    }
    if include_dietary_fiber:
        nutrients["dietary_fiber_g"] = _decimal_to_json(record.nutrients.dietary_fiber_g)
    return {
        "name": record.name,
        "serving": {
            "quantity": _decimal_to_json(record.serving.quantity),
            "unit": record.serving.unit,
            "text": record.serving.text,
        },
        "nutrients": nutrients,
        "provenance": {
            "provider": record.provenance.provider,
            "retrieved_at": record.provenance.retrieved_at.isoformat(),
            "record_type": record.provenance.record_type,
            "source_reference": record.provenance.source_reference,
            "identifiers": [
                {"kind": identifier.kind, "value": identifier.value}
                for identifier in record.provenance.identifiers
            ],
        },
        "ingredients": list(record.ingredients),
        "allergens": list(record.allergens),
    }


def serialize_nutrition_record(
    record: NutritionRecord,
    *,
    include_dietary_fiber: bool = True,
) -> str:
    """Serialize a record as canonical JSON with Decimal values as strings."""

    try:
        return json.dumps(
            nutrition_record_to_mapping(record, include_dietary_fiber=include_dietary_fiber),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise NutritionCatalogSerializationError("nutrition record cannot be serialized") from exc


def nutrition_record_from_mapping(payload: Mapping[str, Any]) -> NutritionRecord:
    """Reconstruct a domain record, rejecting malformed persisted data."""

    if not isinstance(payload, Mapping):
        raise NutritionCatalogSerializationError("snapshot JSON must be an object")
    serving = payload.get("serving")
    nutrients = payload.get("nutrients")
    provenance = payload.get("provenance")
    if not isinstance(serving, Mapping):
        raise NutritionCatalogSerializationError("snapshot serving is invalid")
    if not isinstance(nutrients, Mapping):
        raise NutritionCatalogSerializationError("snapshot nutrients are invalid")
    if not isinstance(provenance, Mapping):
        raise NutritionCatalogSerializationError("snapshot provenance is invalid")

    identifiers_payload = provenance.get("identifiers")
    if not isinstance(identifiers_payload, list):
        raise NutritionCatalogSerializationError("snapshot provenance identifiers are invalid")
    identifiers: list[SourceIdentifier] = []
    for index, identifier in enumerate(identifiers_payload):
        if not isinstance(identifier, Mapping):
            raise NutritionCatalogSerializationError(f"snapshot identifier[{index}] is invalid")
        try:
            identifiers.append(
                SourceIdentifier(
                    kind=_required_text(identifier.get("kind"), field_name="identifier kind"),
                    value=_required_text(identifier.get("value"), field_name="identifier value"),
                )
            )
        except (TypeError, ValueError) as exc:
            raise NutritionCatalogSerializationError(f"snapshot identifier[{index}] is invalid") from exc

    retrieved_at = _parse_timestamp(provenance.get("retrieved_at"), field_name="retrieved_at")
    try:
        return NutritionRecord(
            name=_required_text(payload.get("name"), field_name="name"),
            serving=Serving(
                quantity=_decimal_from_json(serving.get("quantity"), field_name="serving quantity"),
                unit=_optional_text(serving.get("unit"), field_name="serving unit"),
                text=_optional_text(serving.get("text"), field_name="serving text"),
            ),
            nutrients=NutrientProfile(
                calories_kcal=_decimal_from_json(
                    nutrients.get("calories_kcal"), field_name="calories_kcal"
                ),
                protein_g=_decimal_from_json(nutrients.get("protein_g"), field_name="protein_g"),
                carbohydrates_g=_decimal_from_json(
                    nutrients.get("carbohydrates_g"), field_name="carbohydrates_g"
                ),
                fat_g=_decimal_from_json(nutrients.get("fat_g"), field_name="fat_g"),
                sodium_mg=_decimal_from_json(nutrients.get("sodium_mg"), field_name="sodium_mg"),
                dietary_fiber_g=_decimal_from_json(
                    nutrients.get("dietary_fiber_g"), field_name="dietary_fiber_g"
                ),
            ),
            provenance=NutritionProvenance(
                provider=_required_text(provenance.get("provider"), field_name="provider"),
                retrieved_at=retrieved_at,
                record_type=_optional_text(provenance.get("record_type"), field_name="record_type"),
                source_reference=_optional_text(
                    provenance.get("source_reference"), field_name="source_reference"
                ),
                identifiers=tuple(identifiers),
            ),
            ingredients=_text_tuple(payload.get("ingredients"), field_name="ingredients"),
            allergens=_text_tuple(payload.get("allergens"), field_name="allergens"),
        )
    except NutritionCatalogSerializationError:
        raise
    except (TypeError, ValueError) as exc:
        raise NutritionCatalogSerializationError("snapshot fields do not form a valid NutritionRecord") from exc


def deserialize_nutrition_record(serialized: str) -> NutritionRecord:
    """Deserialize canonical record JSON without guessing at corrupt fields."""

    if not isinstance(serialized, str):
        raise NutritionCatalogSerializationError("serialized snapshot must be text")
    try:
        payload = json.loads(serialized)
    except (TypeError, ValueError) as exc:
        raise NutritionCatalogSerializationError("snapshot JSON is invalid") from exc
    return nutrition_record_from_mapping(payload)


def _source_identifier_for(
    record: NutritionRecord,
    explicit: SourceIdentifier | None,
) -> SourceIdentifier:
    if explicit is not None:
        if explicit not in record.provenance.identifiers:
            raise NutritionCatalogValidationError(
                "explicit source_identifier must be present in record provenance"
            )
        return explicit
    component_identifiers = tuple(
        identifier
        for identifier in record.provenance.identifiers
        if identifier.kind == "component"
    )
    if len(component_identifiers) == 1:
        return component_identifiers[0]
    if len(record.provenance.identifiers) == 1:
        return record.provenance.identifiers[0]
    raise NutritionCatalogValidationError(
        "record must have one stable source identifier or an explicit source_identifier"
    )


_NUTRITION_SCHEMA_STATEMENTS = (
    """
CREATE TABLE IF NOT EXISTS nutrition_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
    source_kind TEXT NOT NULL CHECK (length(trim(source_kind)) > 0),
    source_value TEXT NOT NULL CHECK (length(trim(source_value)) > 0),
    content_signature TEXT NOT NULL CHECK (length(trim(content_signature)) > 0),
    snapshot_json TEXT NOT NULL CHECK (length(snapshot_json) > 0),
    first_observed_at TEXT NOT NULL,
    last_observed_at TEXT NOT NULL,
    UNIQUE(provider, source_kind, source_value, content_signature)
)
""",
    """
CREATE INDEX IF NOT EXISTS nutrition_snapshots_current_idx
ON nutrition_snapshots(provider, source_kind, source_value, last_observed_at DESC, snapshot_id DESC)
""",
)

_OCCURRENCE_SCHEMA_STATEMENTS = (
    """
CREATE TABLE IF NOT EXISTS fd_refresh_runs (
    refresh_id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
    requested_start_date TEXT NOT NULL CHECK (length(requested_start_date) = 10),
    requested_end_date TEXT NOT NULL CHECK (length(requested_end_date) = 10),
    observed_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('in_progress', 'complete')),
    CHECK (requested_start_date <= requested_end_date)
)
""",

    """
CREATE TABLE IF NOT EXISTS fd_menu_occurrences (
    occurrence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
    occurrence_key TEXT NOT NULL CHECK (length(trim(occurrence_key)) > 0),
    service_date TEXT NOT NULL CHECK (length(service_date) = 10),
    meal_period_id TEXT NOT NULL CHECK (length(trim(meal_period_id)) > 0),
    meal_period_name TEXT NOT NULL CHECK (length(trim(meal_period_name)) > 0),
    station_concept_id TEXT,
    station_name TEXT,
    source_kind TEXT NOT NULL CHECK (length(trim(source_kind)) > 0),
    source_value TEXT NOT NULL CHECK (length(trim(source_value)) > 0),
    content_signature TEXT NOT NULL CHECK (length(trim(content_signature)) > 0),
    nutrition_snapshot_id INTEGER NOT NULL,
    menu_detail_id TEXT,
    menu_id TEXT,
    first_observed_at TEXT NOT NULL,
    last_observed_at TEXT NOT NULL,
    UNIQUE(provider, occurrence_key, nutrition_snapshot_id),
    FOREIGN KEY (nutrition_snapshot_id) REFERENCES nutrition_snapshots(snapshot_id),
    FOREIGN KEY (provider, source_kind, source_value, content_signature)
        REFERENCES nutrition_snapshots(provider, source_kind, source_value, content_signature)
)
""",
    """
CREATE TABLE IF NOT EXISTS fd_refresh_occurrences (
    refresh_id INTEGER NOT NULL,
    occurrence_id INTEGER NOT NULL,
    PRIMARY KEY (refresh_id, occurrence_id),
    FOREIGN KEY (refresh_id) REFERENCES fd_refresh_runs(refresh_id),
    FOREIGN KEY (occurrence_id) REFERENCES fd_menu_occurrences(occurrence_id)
)
""",
    """
CREATE INDEX IF NOT EXISTS fd_refresh_runs_coverage_idx
ON fd_refresh_runs(provider, status, requested_start_date, requested_end_date, refresh_id DESC)
""",
    """
CREATE INDEX IF NOT EXISTS fd_menu_occurrences_date_idx
ON fd_menu_occurrences(provider, service_date, occurrence_key, occurrence_id)
""",
    """
CREATE INDEX IF NOT EXISTS fd_refresh_occurrences_occurrence_idx
ON fd_refresh_occurrences(occurrence_id, refresh_id)
""",
)


_APPLICATION_STATE_SCHEMA_STATEMENTS = (
    """
CREATE TABLE IF NOT EXISTS meal_plans (
    plan_id TEXT PRIMARY KEY CHECK (length(trim(plan_id)) > 0),
    service_date TEXT NOT NULL CHECK (length(service_date) = 10),
    meal TEXT NOT NULL CHECK (length(trim(meal)) > 0),
    meal_kind TEXT NOT NULL CHECK (meal_kind IN ('text', 'integer')),
    meal_context TEXT NOT NULL CHECK (length(trim(meal_context)) > 0),
    meal_slot TEXT NOT NULL CHECK (meal_slot IN ('breakfast', 'lunch', 'dinner', 'brunch')),
    created_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'applied', 'superseded')) DEFAULT 'active'
)


""",
    """
CREATE UNIQUE INDEX IF NOT EXISTS meal_plans_one_active_slot_idx
ON meal_plans(service_date, meal_slot)
WHERE status = 'active'
""",
    """
CREATE TABLE IF NOT EXISTS meal_plan_items (
    plan_id TEXT NOT NULL,
    plan_item_id TEXT NOT NULL CHECK (length(trim(plan_item_id)) > 0),
    item_position INTEGER NOT NULL CHECK (item_position >= 0),
    occurrence_id INTEGER NOT NULL,
    nutrition_snapshot_id INTEGER NOT NULL,
    source_kind TEXT NOT NULL CHECK (length(trim(source_kind)) > 0),
    source_value TEXT NOT NULL CHECK (length(trim(source_value)) > 0),
    content_signature TEXT NOT NULL CHECK (length(trim(content_signature)) > 0),
    recommended_official_servings TEXT NOT NULL CHECK (length(trim(recommended_official_servings)) > 0),
    natural_quantity_text TEXT NOT NULL CHECK (length(trim(natural_quantity_text)) > 0),
    display_food_name TEXT,
    PRIMARY KEY (plan_id, plan_item_id),
    UNIQUE (plan_id, item_position),
    FOREIGN KEY (plan_id) REFERENCES meal_plans(plan_id),
    FOREIGN KEY (occurrence_id) REFERENCES fd_menu_occurrences(occurrence_id),
    FOREIGN KEY (nutrition_snapshot_id) REFERENCES nutrition_snapshots(snapshot_id)
)
""",
    """
CREATE TABLE IF NOT EXISTS meal_report_applications (
    source_event_id TEXT PRIMARY KEY CHECK (length(trim(source_event_id)) > 0),
    plan_id TEXT NOT NULL,
    applied_at TEXT NOT NULL,
    FOREIGN KEY (plan_id) REFERENCES meal_plans(plan_id)
)
""",
    """
CREATE TABLE IF NOT EXISTS accepted_intake_entries (
    intake_id TEXT PRIMARY KEY CHECK (length(trim(intake_id)) > 0),
    source_event_id TEXT NOT NULL,
    service_date TEXT NOT NULL CHECK (length(service_date) = 10),
    recorded_at TEXT NOT NULL,
    meal TEXT,
    plan_id TEXT,
    plan_item_id TEXT,
    occurrence_id INTEGER NOT NULL,
    nutrition_snapshot_id INTEGER NOT NULL,
    source_kind TEXT NOT NULL CHECK (length(trim(source_kind)) > 0),
    source_value TEXT NOT NULL CHECK (length(trim(source_value)) > 0),
    content_signature TEXT NOT NULL CHECK (length(trim(content_signature)) > 0),
    official_servings TEXT NOT NULL CHECK (length(trim(official_servings)) > 0),
    quantity_source TEXT NOT NULL CHECK (quantity_source IN (
        'planned_quantity', 'explicit_deterministic',
        'explicit_semantic_estimate', 'manual_other'
    )),
    original_reference_text TEXT,
    original_quantity_text TEXT,
    item_position INTEGER NOT NULL CHECK (item_position >= 0),
    FOREIGN KEY (source_event_id) REFERENCES meal_report_applications(source_event_id),
    FOREIGN KEY (plan_id) REFERENCES meal_plans(plan_id),
    FOREIGN KEY (occurrence_id) REFERENCES fd_menu_occurrences(occurrence_id),
    FOREIGN KEY (nutrition_snapshot_id) REFERENCES nutrition_snapshots(snapshot_id)
)
""",
    """
CREATE INDEX IF NOT EXISTS accepted_intake_entries_date_idx
ON accepted_intake_entries(service_date, recorded_at, intake_id)
""",
    """
CREATE INDEX IF NOT EXISTS meal_plan_items_occurrence_idx
ON meal_plan_items(occurrence_id, nutrition_snapshot_id)
""",
)


# Independent operational intake. Absence means no reminder was claimed.
_SHAKE_INTAKE_SCHEMA_STATEMENTS = (
    """
CREATE TABLE IF NOT EXISTS shake_intake (
    local_date TEXT NOT NULL CHECK (length(local_date) = 10),
    shake_slot TEXT NOT NULL CHECK (shake_slot IN ('breakfast_shake', 'dinner_shake')),
    chat_guid TEXT NOT NULL CHECK (length(trim(chat_guid)) > 0),
    reminder_claimed_at TEXT NOT NULL,
    reminder_delivered_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending', 'confirmed', 'skipped')),
    resolved_at TEXT,
    resolution_source_event_id TEXT UNIQUE,
    product_name TEXT,
    calories_kcal TEXT,
    protein_g TEXT,
    carbohydrates_g TEXT,
    fat_g TEXT,
    PRIMARY KEY (local_date, shake_slot),
    CHECK (
        (status = 'pending' AND resolved_at IS NULL AND resolution_source_event_id IS NULL
         AND product_name IS NULL AND calories_kcal IS NULL AND protein_g IS NULL
         AND carbohydrates_g IS NULL AND fat_g IS NULL)
        OR
        (status = 'skipped' AND resolved_at IS NOT NULL AND resolution_source_event_id IS NOT NULL
         AND product_name IS NULL AND calories_kcal IS NULL AND protein_g IS NULL
         AND carbohydrates_g IS NULL AND fat_g IS NULL)
        OR
        (status = 'confirmed' AND resolved_at IS NOT NULL AND resolution_source_event_id IS NOT NULL
         AND product_name IS NOT NULL AND calories_kcal IS NOT NULL AND protein_g IS NOT NULL
         AND carbohydrates_g IS NOT NULL AND fat_g IS NOT NULL)
    )
)
""",
)


_DIETARY_FIBER_EXTENSION_SCHEMA_STATEMENTS = (
    """
CREATE TABLE IF NOT EXISTS nutrition_snapshot_dietary_fiber (
    nutrition_snapshot_id INTEGER PRIMARY KEY,
    dietary_fiber_g TEXT NOT NULL CHECK (length(trim(dietary_fiber_g)) > 0),
    FOREIGN KEY (nutrition_snapshot_id) REFERENCES nutrition_snapshots(snapshot_id)
)
""",
)


# A scheduled recommendation is deliberately separate from inbound
# ``meal_report_applications``.  The record owns one exact persisted plan for
# one Detroit service-date/FD-meal context and tracks only outbound delivery.
# A missing row means the scheduler has never prepared that opportunity.
_SCHEDULED_RECOMMENDATION_SCHEMA_STATEMENTS = (
    """
CREATE TABLE IF NOT EXISTS scheduled_recommendation_dispatches (
    service_date TEXT NOT NULL CHECK (length(service_date) = 10),
    meal_context TEXT NOT NULL CHECK (length(trim(meal_context)) > 0),
    meal_slot TEXT NOT NULL CHECK (meal_slot IN ('breakfast', 'lunch', 'dinner', 'brunch')),
    plan_id TEXT NOT NULL UNIQUE,
    delivery_token TEXT NOT NULL UNIQUE CHECK (length(trim(delivery_token)) > 0),
    status TEXT NOT NULL CHECK (status IN (
        'pending_delivery', 'sending', 'delivered', 'expired'
    )),
    created_at TEXT NOT NULL,
    delivery_started_at TEXT,
    delivered_at TEXT,
    expired_at TEXT,
    PRIMARY KEY (service_date, meal_slot),
    FOREIGN KEY (plan_id) REFERENCES meal_plans(plan_id)
)
""",
    """
CREATE INDEX IF NOT EXISTS scheduled_recommendation_dispatches_status_idx
ON scheduled_recommendation_dispatches(status, service_date, meal_slot)
""",
)


# Conversational report state is intentionally separate from both accepted
# intake and scheduler delivery.  A draft contains only deterministic,
# structured facts required to continue a clarification after a process
# restart.  It never stores model reasoning or grants the model authority over
# food identity, official quantities, nutrition, or plan lifecycle.
_CONVERSATIONAL_MEAL_REPORT_SCHEMA_STATEMENTS = (
    """
CREATE TABLE IF NOT EXISTS meal_report_drafts (
    draft_id TEXT PRIMARY KEY CHECK (length(trim(draft_id)) > 0),
    chat_guid TEXT NOT NULL CHECK (length(trim(chat_guid)) > 0),
    plan_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN (
        'draft', 'awaiting_clarification', 'completed', 'cancelled'
    )),
    revision INTEGER NOT NULL CHECK (revision >= 0),
    last_prompt TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    cancelled_at TEXT,
    FOREIGN KEY (plan_id) REFERENCES meal_plans(plan_id)
)
""",
    """
CREATE UNIQUE INDEX IF NOT EXISTS meal_report_drafts_one_open_chat_idx
ON meal_report_drafts(chat_guid, plan_id)
WHERE status IN ('draft', 'awaiting_clarification')
""",
    """
CREATE INDEX IF NOT EXISTS meal_report_drafts_plan_status_idx
ON meal_report_drafts(plan_id, status, updated_at)
""",
    """
CREATE TABLE IF NOT EXISTS meal_report_draft_planned_items (
    draft_id TEXT NOT NULL,
    plan_item_id TEXT NOT NULL CHECK (length(trim(plan_item_id)) > 0),
    action TEXT NOT NULL CHECK (action IN ('eaten', 'skipped')),
    official_servings TEXT,
    quantity_source TEXT CHECK (quantity_source IN (
        'planned_quantity', 'explicit_deterministic',
        'explicit_semantic_estimate'
    )),
    original_reference_text TEXT NOT NULL CHECK (length(trim(original_reference_text)) > 0),
    original_quantity_text TEXT,
    quantity_status TEXT NOT NULL CHECK (quantity_status IN (
        'resolved', 'unresolved', 'not_applicable'
    )),
    unresolved_reason TEXT,
    last_source_event_id TEXT,
    fact_revision INTEGER NOT NULL DEFAULT 0 CHECK (fact_revision >= 0),
    PRIMARY KEY (draft_id, plan_item_id),
    FOREIGN KEY (draft_id) REFERENCES meal_report_drafts(draft_id),
    CHECK (
        (action = 'skipped' AND quantity_status = 'not_applicable'
         AND official_servings IS NULL AND quantity_source IS NULL
         AND unresolved_reason IS NULL)
        OR
        (action = 'eaten' AND quantity_status = 'resolved'
         AND official_servings IS NOT NULL AND quantity_source IS NOT NULL
         AND unresolved_reason IS NULL)
        OR
        (action = 'eaten' AND quantity_status = 'unresolved'
         AND official_servings IS NULL AND quantity_source IS NULL
         AND unresolved_reason IS NOT NULL)
    )
)
""",
    """
CREATE TABLE IF NOT EXISTS meal_report_draft_unplanned_items (
    draft_id TEXT NOT NULL,
    item_position INTEGER NOT NULL CHECK (item_position >= 0),
    fact_id TEXT NOT NULL CHECK (length(trim(fact_id)) > 0),
    action TEXT NOT NULL CHECK (action = 'eaten'),
    food_text TEXT NOT NULL CHECK (length(trim(food_text)) > 0),
    quantity_text TEXT,
    identity_status TEXT NOT NULL CHECK (identity_status IN ('resolved', 'unresolved')),
    identity_unresolved_reason TEXT,
    occurrence_id INTEGER,
    nutrition_snapshot_id INTEGER,
    source_kind TEXT,
    source_value TEXT,
    content_signature TEXT,
    quantity_status TEXT NOT NULL CHECK (quantity_status IN ('resolved', 'unresolved')),
    official_servings TEXT,
    quantity_source TEXT CHECK (quantity_source IN (
        'explicit_deterministic', 'explicit_semantic_estimate'
    )),
    quantity_unresolved_reason TEXT,
    confidence TEXT CHECK (confidence IN ('high', 'medium', 'low')),
    last_source_event_id TEXT,
    fact_revision INTEGER NOT NULL DEFAULT 0 CHECK (fact_revision >= 0),
    PRIMARY KEY (draft_id, item_position),
    UNIQUE (draft_id, fact_id),
    FOREIGN KEY (draft_id) REFERENCES meal_report_drafts(draft_id),
    FOREIGN KEY (occurrence_id) REFERENCES fd_menu_occurrences(occurrence_id),
    FOREIGN KEY (nutrition_snapshot_id) REFERENCES nutrition_snapshots(snapshot_id),
    CHECK (
        (identity_status = 'resolved' AND identity_unresolved_reason IS NULL
         AND occurrence_id IS NOT NULL AND nutrition_snapshot_id IS NOT NULL
         AND source_kind IS NOT NULL AND source_value IS NOT NULL
         AND content_signature IS NOT NULL)
        OR
        (identity_status = 'unresolved' AND identity_unresolved_reason IS NOT NULL
         AND occurrence_id IS NULL AND nutrition_snapshot_id IS NULL
         AND source_kind IS NULL AND source_value IS NULL
         AND content_signature IS NULL)
    ),
    CHECK (
        (quantity_status = 'resolved' AND official_servings IS NOT NULL
         AND quantity_source IS NOT NULL AND quantity_unresolved_reason IS NULL)
        OR
        (quantity_status = 'unresolved' AND official_servings IS NULL
         AND quantity_source IS NULL AND quantity_unresolved_reason IS NOT NULL
         AND confidence IS NULL)
    )
)
""",
    """
CREATE TABLE IF NOT EXISTS meal_report_draft_clarifications (
    draft_id TEXT NOT NULL,
    clarification_position INTEGER NOT NULL CHECK (clarification_position >= 0),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    plan_item_id TEXT,
    food_text TEXT,
    quantity_text TEXT,
    PRIMARY KEY (draft_id, clarification_position),
    FOREIGN KEY (draft_id) REFERENCES meal_report_drafts(draft_id)
)
""",
    """
CREATE TABLE IF NOT EXISTS meal_report_draft_planned_item_history (
    draft_id TEXT NOT NULL,
    plan_item_id TEXT NOT NULL,
    fact_revision INTEGER NOT NULL CHECK (fact_revision >= 0),
    action TEXT NOT NULL CHECK (action IN ('eaten', 'skipped')),
    official_servings TEXT,
    quantity_source TEXT,
    original_reference_text TEXT NOT NULL,
    original_quantity_text TEXT,
    quantity_status TEXT NOT NULL,
    unresolved_reason TEXT,
    last_source_event_id TEXT,
    superseded_by_source_event_id TEXT NOT NULL,
    superseded_at TEXT NOT NULL,
    PRIMARY KEY (draft_id, plan_item_id, fact_revision),
    FOREIGN KEY (draft_id) REFERENCES meal_report_drafts(draft_id)
)
""",
    """
CREATE TABLE IF NOT EXISTS meal_report_draft_unplanned_item_history (
    draft_id TEXT NOT NULL,
    fact_id TEXT NOT NULL,
    fact_revision INTEGER NOT NULL CHECK (fact_revision >= 0),
    item_position INTEGER NOT NULL,
    action TEXT NOT NULL,
    food_text TEXT NOT NULL,
    quantity_text TEXT,
    identity_status TEXT NOT NULL,
    identity_unresolved_reason TEXT,
    occurrence_id INTEGER,
    nutrition_snapshot_id INTEGER,
    source_kind TEXT,
    source_value TEXT,
    content_signature TEXT,
    quantity_status TEXT NOT NULL,
    official_servings TEXT,
    quantity_source TEXT,
    quantity_unresolved_reason TEXT,
    confidence TEXT,
    last_source_event_id TEXT,
    superseded_by_source_event_id TEXT NOT NULL,
    superseded_at TEXT NOT NULL,
    PRIMARY KEY (draft_id, fact_id, fact_revision),
    FOREIGN KEY (draft_id) REFERENCES meal_report_drafts(draft_id)
)
""",
    """
CREATE TABLE IF NOT EXISTS meal_report_message_events (
    source_event_id TEXT PRIMARY KEY CHECK (length(trim(source_event_id)) > 0),
    chat_guid TEXT NOT NULL CHECK (length(trim(chat_guid)) > 0),
    outcome_type TEXT NOT NULL CHECK (outcome_type IN (
        'routed_interaction', 'draft_update', 'draft_cancelled',
        'final_application', 'ambiguous_meal_context',
        'unavailable_meal_slot', 'no_reportable_context',
        'routing_failure', 'pending_meal_request'
    )),
    plan_id TEXT,
    draft_id TEXT,
    application_source_event_id TEXT,
    intent TEXT NOT NULL CHECK (intent IN (
        'meal_report', 'clarification_answer', 'location_question',
        'replacement_request', 'meal_request', 'unsupported_or_ambiguous'
    )),
    reply_text TEXT NOT NULL CHECK (length(trim(reply_text)) > 0),
    processed_at TEXT NOT NULL,
    FOREIGN KEY (plan_id) REFERENCES meal_plans(plan_id),
    FOREIGN KEY (draft_id) REFERENCES meal_report_drafts(draft_id),
    FOREIGN KEY (application_source_event_id)
        REFERENCES meal_report_applications(source_event_id),
    CHECK (
        (outcome_type IN (
            'ambiguous_meal_context', 'unavailable_meal_slot',
            'no_reportable_context', 'routing_failure', 'pending_meal_request'
         ) AND plan_id IS NULL AND draft_id IS NULL
           AND application_source_event_id IS NULL)
        OR
        (outcome_type IN ('routed_interaction', 'draft_update', 'draft_cancelled')
         AND plan_id IS NOT NULL AND application_source_event_id IS NULL)
        OR
        (outcome_type = 'final_application' AND plan_id IS NOT NULL
         AND application_source_event_id = source_event_id)
    )
)
""",
    """
CREATE INDEX IF NOT EXISTS meal_report_message_events_draft_idx
ON meal_report_message_events(draft_id, processed_at)
""",
)


# A replacement is a new authoritative recommendation, not an intake event or
# a second scheduler dispatch.  It records the exact prior/replacement plan
# lineage and an independent outbound state so a failed send can retry the
# already-persisted replacement without re-optimizing.  The original scheduled
# dispatch, when present, remains historically intact and terminal.
_MEAL_RECOMMENDATION_REPLACEMENT_SCHEMA_STATEMENTS = (
    """
CREATE TABLE IF NOT EXISTS meal_recommendation_replacements (
    source_event_id TEXT PRIMARY KEY CHECK (length(trim(source_event_id)) > 0),
    chat_guid TEXT NOT NULL CHECK (length(trim(chat_guid)) > 0),
    prior_plan_id TEXT NOT NULL,
    replacement_plan_id TEXT NOT NULL UNIQUE,
    scheduled_origin_plan_id TEXT,
    request_kind TEXT NOT NULL CHECK (request_kind IN ('rejection', 'meal_request')) DEFAULT 'rejection',
    whole_meal INTEGER NOT NULL CHECK (whole_meal IN (0, 1)),
    delivery_token TEXT NOT NULL UNIQUE CHECK (length(trim(delivery_token)) > 0),
    status TEXT NOT NULL CHECK (status IN ('pending_delivery', 'sending', 'delivered')),
    created_at TEXT NOT NULL,
    delivery_started_at TEXT,
    delivered_at TEXT,
    FOREIGN KEY (prior_plan_id) REFERENCES meal_plans(plan_id),
    FOREIGN KEY (replacement_plan_id) REFERENCES meal_plans(plan_id),
    FOREIGN KEY (scheduled_origin_plan_id) REFERENCES meal_plans(plan_id)
)
""",
    """
CREATE INDEX IF NOT EXISTS meal_recommendation_replacements_prior_idx
ON meal_recommendation_replacements(prior_plan_id, status)
""",
    """
CREATE INDEX IF NOT EXISTS meal_recommendation_replacements_origin_idx
ON meal_recommendation_replacements(scheduled_origin_plan_id, status)
""",
    """
CREATE TABLE IF NOT EXISTS meal_recommendation_replacement_rejections (
    source_event_id TEXT NOT NULL,
    plan_item_id TEXT NOT NULL CHECK (length(trim(plan_item_id)) > 0),
    PRIMARY KEY (source_event_id, plan_item_id),
    FOREIGN KEY (source_event_id)
        REFERENCES meal_recommendation_replacements(source_event_id)
)
""",
    """
CREATE TABLE IF NOT EXISTS meal_recommendation_replacement_requested_foods (
    source_event_id TEXT NOT NULL,
    food_position INTEGER NOT NULL CHECK (food_position >= 0),
    food_text TEXT NOT NULL CHECK (length(trim(food_text)) > 0),
    source_kind TEXT NOT NULL CHECK (length(trim(source_kind)) > 0),
    source_value TEXT NOT NULL CHECK (length(trim(source_value)) > 0),
    content_signature TEXT NOT NULL CHECK (length(trim(content_signature)) > 0),
    PRIMARY KEY (source_event_id, food_position),
    UNIQUE (source_event_id, source_kind, source_value, content_signature),
    FOREIGN KEY (source_event_id)
        REFERENCES meal_recommendation_replacements(source_event_id)
)
""",
)


# A positive request before its service opportunity has no meal plan yet.  It
# is intentionally scoped to one chat, Detroit service date, and canonical
# meal context; the scheduler later consumes it in the same transaction that
# creates its scheduled plan.  The source GUID and reply text make an inbound
# webhook/poller retry deterministic without recording intake.
_PENDING_MEAL_REQUEST_SCHEMA_STATEMENTS = (
    """
CREATE TABLE IF NOT EXISTS pending_meal_requests (
    source_event_id TEXT PRIMARY KEY CHECK (length(trim(source_event_id)) > 0),
    chat_guid TEXT NOT NULL CHECK (length(trim(chat_guid)) > 0),
    service_date TEXT NOT NULL CHECK (length(service_date) = 10),
    meal TEXT NOT NULL CHECK (length(trim(meal)) > 0),
    meal_kind TEXT NOT NULL CHECK (meal_kind IN ('text', 'integer')),
    meal_context TEXT NOT NULL CHECK (length(trim(meal_context)) > 0),
    meal_slot TEXT NOT NULL CHECK (meal_slot IN ('breakfast', 'lunch', 'dinner', 'brunch')),
    whole_meal INTEGER NOT NULL CHECK (whole_meal IN (0, 1)),
    status TEXT NOT NULL CHECK (status IN ('pending', 'consumed', 'superseded')),
    reply_text TEXT NOT NULL CHECK (length(trim(reply_text)) > 0),
    created_at TEXT NOT NULL,
    consumed_at TEXT,
    consumed_plan_id TEXT,
    FOREIGN KEY (consumed_plan_id) REFERENCES meal_plans(plan_id)
)
""",
    """
CREATE UNIQUE INDEX IF NOT EXISTS pending_meal_requests_one_open_context_idx
ON pending_meal_requests(chat_guid, service_date, meal_slot)
WHERE status = 'pending'
""",
    """
CREATE INDEX IF NOT EXISTS pending_meal_requests_scheduler_idx
ON pending_meal_requests(chat_guid, service_date, meal_slot, status)
""",
    """
CREATE TABLE IF NOT EXISTS pending_meal_request_foods (
    source_event_id TEXT NOT NULL,
    food_position INTEGER NOT NULL CHECK (food_position >= 0),
    food_text TEXT NOT NULL CHECK (length(trim(food_text)) > 0),
    source_kind TEXT NOT NULL CHECK (length(trim(source_kind)) > 0),
    source_value TEXT NOT NULL CHECK (length(trim(source_value)) > 0),
    content_signature TEXT NOT NULL CHECK (length(trim(content_signature)) > 0),
    PRIMARY KEY (source_event_id, food_position),
    UNIQUE (source_event_id, source_kind, source_value, content_signature),
    FOREIGN KEY (source_event_id) REFERENCES pending_meal_requests(source_event_id)
)
""",
)


# An immediate user-requested meal has no prior plan lineage, so it carries its
# own stable outbound hand-off record rather than overloading a rejection
# replacement.  It shares the same exact plan/items and one-time delivery
# semantics as the established replacement dispatcher.
_IMMEDIATE_MEAL_REQUEST_SCHEMA_STATEMENTS = (
    """
CREATE TABLE IF NOT EXISTS immediate_meal_request_dispatches (
    source_event_id TEXT PRIMARY KEY CHECK (length(trim(source_event_id)) > 0),
    chat_guid TEXT NOT NULL CHECK (length(trim(chat_guid)) > 0),
    plan_id TEXT NOT NULL UNIQUE,
    whole_meal INTEGER NOT NULL CHECK (whole_meal IN (0, 1)),
    delivery_token TEXT NOT NULL UNIQUE CHECK (length(trim(delivery_token)) > 0),
    status TEXT NOT NULL CHECK (status IN ('pending_delivery', 'sending', 'delivered')),
    created_at TEXT NOT NULL,
    delivery_started_at TEXT,
    delivered_at TEXT,
    FOREIGN KEY (plan_id) REFERENCES meal_plans(plan_id)
)
""",
    """
CREATE INDEX IF NOT EXISTS immediate_meal_request_dispatches_plan_idx
ON immediate_meal_request_dispatches(plan_id, status)
""",
    """
CREATE TABLE IF NOT EXISTS immediate_meal_request_requested_foods (
    source_event_id TEXT NOT NULL,
    food_position INTEGER NOT NULL CHECK (food_position >= 0),
    food_text TEXT NOT NULL CHECK (length(trim(food_text)) > 0),
    source_kind TEXT NOT NULL CHECK (length(trim(source_kind)) > 0),
    source_value TEXT NOT NULL CHECK (length(trim(source_value)) > 0),
    content_signature TEXT NOT NULL CHECK (length(trim(content_signature)) > 0),
    PRIMARY KEY (source_event_id, food_position),
    UNIQUE (source_event_id, source_kind, source_value, content_signature),
    FOREIGN KEY (source_event_id)
        REFERENCES immediate_meal_request_dispatches(source_event_id)
)
""",
)


def _meal_report_message_events_need_meal_request_migration(
    connection: sqlite3.Connection,
) -> bool:
    """Return whether the v8 CHECK constraint lacks ``meal_request``."""

    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'meal_report_message_events'"
    ).fetchone()
    if row is None or row["sql"] is None:
        raise NutritionCatalogError("meal report message event state is missing")
    return "'meal_request'" not in str(row["sql"]).casefold()


def _migrate_meal_request_state_v9(connection: sqlite3.Connection) -> None:
    """Add explicit positive-request state while preserving all v8 history."""

    columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(meal_recommendation_replacements)"
        ).fetchall()
    }
    if not columns:
        raise NutritionCatalogError("meal recommendation replacement state is missing")
    if "request_kind" not in columns:
        connection.execute(
            "ALTER TABLE meal_recommendation_replacements "
            "ADD COLUMN request_kind TEXT NOT NULL DEFAULT 'rejection' "
            "CHECK (request_kind IN ('rejection', 'meal_request'))"
        )

    if _meal_report_message_events_need_meal_request_migration(connection):
        rows = tuple(
            connection.execute(
                """
                SELECT source_event_id, chat_guid, plan_id, draft_id, intent,
                       reply_text, processed_at
                FROM meal_report_message_events
                ORDER BY processed_at, source_event_id
                """
            ).fetchall()
        )
        connection.execute("DROP INDEX IF EXISTS meal_report_message_events_draft_idx")
        connection.execute(
            "ALTER TABLE meal_report_message_events RENAME TO meal_report_message_events_v8"
        )
        connection.execute(
            """
            CREATE TABLE meal_report_message_events (
                source_event_id TEXT PRIMARY KEY CHECK (length(trim(source_event_id)) > 0),
                chat_guid TEXT NOT NULL CHECK (length(trim(chat_guid)) > 0),
                plan_id TEXT NOT NULL,
                draft_id TEXT,
                intent TEXT NOT NULL CHECK (intent IN (
                    'meal_report', 'clarification_answer', 'location_question',
                    'replacement_request', 'meal_request', 'unsupported_or_ambiguous'
                )),
                reply_text TEXT NOT NULL CHECK (length(trim(reply_text)) > 0),
                processed_at TEXT NOT NULL,
                FOREIGN KEY (plan_id) REFERENCES meal_plans(plan_id),
                FOREIGN KEY (draft_id) REFERENCES meal_report_drafts(draft_id)
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO meal_report_message_events
            (source_event_id, chat_guid, plan_id, draft_id, intent, reply_text, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(tuple(row) for row in rows),
        )
        connection.execute("DROP TABLE meal_report_message_events_v8")
        connection.execute(
            """
            CREATE INDEX meal_report_message_events_draft_idx
            ON meal_report_message_events(draft_id, processed_at)
            """
        )

    for statement in _MEAL_RECOMMENDATION_REPLACEMENT_SCHEMA_STATEMENTS:
        connection.execute(statement)
    for statement in _PENDING_MEAL_REQUEST_SCHEMA_STATEMENTS:
        connection.execute(statement)
    for statement in _IMMEDIATE_MEAL_REQUEST_SCHEMA_STATEMENTS:
        connection.execute(statement)


def _meal_slot_migration_required(connection: sqlite3.Connection) -> bool:
    """Return whether the persisted application graph predates schema v10."""

    plan_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(meal_plans)").fetchall()
    }
    return bool(plan_columns) and "meal_slot" not in plan_columns


def _legacy_slot_from_row(row: sqlite3.Row) -> str:
    """Classify one v9 date/provider pair without menu or AI inference."""

    try:
        service_date = date.fromisoformat(str(row["service_date"]))
        meal = (
            int(str(row["meal"]))
            if str(row["meal_kind"]) == "integer"
            else str(row["meal"])
        )
        if str(row["meal_context"]) != meal_context_key(meal):
            raise ValueError("meal context mismatch")
        return legacy_meal_slot(service_date, meal)
    except (TypeError, ValueError):
        raise NutritionCatalogError(
            "persisted v9 meal context cannot be assigned a deterministic meal slot"
        ) from None


def _migrate_meal_slots_v10(connection: sqlite3.Connection) -> None:
    """Add explicit product slots while preserving provider identity and history.

    Foreign keys are disabled by the caller only for this transactional table
    rebuild.  Existing child rows continue to reference the final ``meal_plans``
    table name, and ``foreign_key_check`` validates the complete graph before
    ``user_version`` advances.
    """

    plan_rows = tuple(
        connection.execute(
            """
            SELECT plan_id, service_date, meal, meal_kind, meal_context, created_at, status
            FROM meal_plans ORDER BY created_at, plan_id
            """
        ).fetchall()
    )
    slots_by_plan = {str(row["plan_id"]): _legacy_slot_from_row(row) for row in plan_rows}

    dispatch_rows = tuple(
        connection.execute(
            """
            SELECT service_date, meal_context, plan_id, delivery_token, status,
                   created_at, delivery_started_at, delivered_at, expired_at
            FROM scheduled_recommendation_dispatches
            ORDER BY service_date, meal_context
            """
        ).fetchall()
    )
    pending_rows = tuple(
        connection.execute(
            """
            SELECT source_event_id, chat_guid, service_date, meal, meal_kind,
                   meal_context, whole_meal, status, reply_text, created_at,
                   consumed_at, consumed_plan_id
            FROM pending_meal_requests ORDER BY created_at, source_event_id
            """
        ).fetchall()
    )
    pending_slots = {
        str(row["source_event_id"]): _legacy_slot_from_row(row) for row in pending_rows
    }

    connection.execute(
        """
        CREATE TABLE meal_plans_v10 (
            plan_id TEXT PRIMARY KEY CHECK (length(trim(plan_id)) > 0),
            service_date TEXT NOT NULL CHECK (length(service_date) = 10),
            meal TEXT NOT NULL CHECK (length(trim(meal)) > 0),
            meal_kind TEXT NOT NULL CHECK (meal_kind IN ('text', 'integer')),
            meal_context TEXT NOT NULL CHECK (length(trim(meal_context)) > 0),
            meal_slot TEXT NOT NULL CHECK (
                meal_slot IN ('breakfast', 'lunch', 'dinner', 'brunch')
            ),
            created_at TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('active', 'applied', 'superseded'))
                DEFAULT 'active'
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO meal_plans_v10
        (plan_id, service_date, meal, meal_kind, meal_context, meal_slot, created_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        tuple(
            (
                row["plan_id"], row["service_date"], row["meal"], row["meal_kind"],
                row["meal_context"], slots_by_plan[str(row["plan_id"])],
                row["created_at"], row["status"],
            )
            for row in plan_rows
        ),
    )
    connection.execute("DROP TABLE meal_plans")
    connection.execute("ALTER TABLE meal_plans_v10 RENAME TO meal_plans")
    connection.execute(
        """
        CREATE UNIQUE INDEX meal_plans_one_active_slot_idx
        ON meal_plans(service_date, meal_slot) WHERE status = 'active'
        """
    )

    connection.execute(
        """
        CREATE TABLE scheduled_recommendation_dispatches_v10 (
            service_date TEXT NOT NULL CHECK (length(service_date) = 10),
            meal_context TEXT NOT NULL CHECK (length(trim(meal_context)) > 0),
            meal_slot TEXT NOT NULL CHECK (
                meal_slot IN ('breakfast', 'lunch', 'dinner', 'brunch')
            ),
            plan_id TEXT NOT NULL UNIQUE,
            delivery_token TEXT NOT NULL UNIQUE CHECK (length(trim(delivery_token)) > 0),
            status TEXT NOT NULL CHECK (status IN (
                'pending_delivery', 'sending', 'delivered', 'expired'
            )),
            created_at TEXT NOT NULL,
            delivery_started_at TEXT,
            delivered_at TEXT,
            expired_at TEXT,
            PRIMARY KEY (service_date, meal_slot),
            FOREIGN KEY (plan_id) REFERENCES meal_plans(plan_id)
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO scheduled_recommendation_dispatches_v10
        (service_date, meal_context, meal_slot, plan_id, delivery_token, status,
         created_at, delivery_started_at, delivered_at, expired_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        tuple(
            (
                row["service_date"], row["meal_context"],
                slots_by_plan[str(row["plan_id"])], row["plan_id"],
                row["delivery_token"], row["status"], row["created_at"],
                row["delivery_started_at"], row["delivered_at"], row["expired_at"],
            )
            for row in dispatch_rows
        ),
    )
    connection.execute("DROP TABLE scheduled_recommendation_dispatches")
    connection.execute(
        "ALTER TABLE scheduled_recommendation_dispatches_v10 "
        "RENAME TO scheduled_recommendation_dispatches"
    )
    connection.execute(
        """
        CREATE INDEX scheduled_recommendation_dispatches_status_idx
        ON scheduled_recommendation_dispatches(status, service_date, meal_slot)
        """
    )

    connection.execute(
        """
        CREATE TABLE pending_meal_requests_v10 (
            source_event_id TEXT PRIMARY KEY CHECK (length(trim(source_event_id)) > 0),
            chat_guid TEXT NOT NULL CHECK (length(trim(chat_guid)) > 0),
            service_date TEXT NOT NULL CHECK (length(service_date) = 10),
            meal TEXT NOT NULL CHECK (length(trim(meal)) > 0),
            meal_kind TEXT NOT NULL CHECK (meal_kind IN ('text', 'integer')),
            meal_context TEXT NOT NULL CHECK (length(trim(meal_context)) > 0),
            meal_slot TEXT NOT NULL CHECK (
                meal_slot IN ('breakfast', 'lunch', 'dinner', 'brunch')
            ),
            whole_meal INTEGER NOT NULL CHECK (whole_meal IN (0, 1)),
            status TEXT NOT NULL CHECK (status IN ('pending', 'consumed', 'superseded')),
            reply_text TEXT NOT NULL CHECK (length(trim(reply_text)) > 0),
            created_at TEXT NOT NULL,
            consumed_at TEXT,
            consumed_plan_id TEXT,
            FOREIGN KEY (consumed_plan_id) REFERENCES meal_plans(plan_id)
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO pending_meal_requests_v10
        (source_event_id, chat_guid, service_date, meal, meal_kind, meal_context,
         meal_slot, whole_meal, status, reply_text, created_at, consumed_at, consumed_plan_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        tuple(
            (
                row["source_event_id"], row["chat_guid"], row["service_date"],
                row["meal"], row["meal_kind"], row["meal_context"],
                pending_slots[str(row["source_event_id"])], row["whole_meal"],
                row["status"], row["reply_text"], row["created_at"],
                row["consumed_at"], row["consumed_plan_id"],
            )
            for row in pending_rows
        ),
    )
    connection.execute("DROP TABLE pending_meal_requests")
    connection.execute("ALTER TABLE pending_meal_requests_v10 RENAME TO pending_meal_requests")
    connection.execute(
        """
        CREATE UNIQUE INDEX pending_meal_requests_one_open_context_idx
        ON pending_meal_requests(chat_guid, service_date, meal_slot)
        WHERE status = 'pending'
        """
    )
    connection.execute(
        """
        CREATE INDEX pending_meal_requests_scheduler_idx
        ON pending_meal_requests(chat_guid, service_date, meal_slot, status)
        """
    )

    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise NutritionCatalogError("v10 meal-slot migration produced invalid foreign keys")


def _draft_fact_migration_required(connection: sqlite3.Connection) -> bool:
    """Return whether conversational report state predates granular v11 facts."""

    columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(meal_report_draft_planned_items)"
        ).fetchall()
    }
    return bool(columns) and "quantity_status" not in columns


def _migrate_report_draft_facts_v11(connection: sqlite3.Connection) -> None:
    """Rebuild v10 draft children as independently resolved effective facts."""

    connection.execute("DROP INDEX IF EXISTS meal_report_drafts_one_open_chat_idx")
    connection.execute(
        "ALTER TABLE meal_report_draft_planned_items "
        "RENAME TO meal_report_draft_planned_items_v10"
    )
    connection.execute(
        "ALTER TABLE meal_report_draft_unplanned_items "
        "RENAME TO meal_report_draft_unplanned_items_v10"
    )
    connection.execute(
        "ALTER TABLE meal_report_draft_clarifications "
        "RENAME TO meal_report_draft_clarifications_v10"
    )
    for statement in _CONVERSATIONAL_MEAL_REPORT_SCHEMA_STATEMENTS:
        connection.execute(statement)

    provenance = """
        SELECT event.source_event_id
        FROM meal_report_message_events AS event
        WHERE event.draft_id = legacy.draft_id
        ORDER BY event.processed_at DESC, event.source_event_id DESC
        LIMIT 1
    """
    connection.execute(
        f"""
        INSERT INTO meal_report_draft_planned_items
        (draft_id, plan_item_id, action, official_servings, quantity_source,
         original_reference_text, original_quantity_text, quantity_status,
         unresolved_reason, last_source_event_id, fact_revision)
        SELECT legacy.draft_id, legacy.plan_item_id, legacy.action,
               legacy.official_servings, legacy.quantity_source,
               legacy.original_reference_text, legacy.original_quantity_text,
               CASE WHEN legacy.action = 'skipped' THEN 'not_applicable' ELSE 'resolved' END,
               NULL, ({provenance}), 0
        FROM meal_report_draft_planned_items_v10 AS legacy
        """
    )
    connection.execute(
        f"""
        INSERT INTO meal_report_draft_unplanned_items
        (draft_id, item_position, fact_id, action, food_text, quantity_text,
         identity_status, identity_unresolved_reason, occurrence_id,
         nutrition_snapshot_id, source_kind, source_value, content_signature,
         quantity_status, official_servings, quantity_source,
         quantity_unresolved_reason, confidence, last_source_event_id, fact_revision)
        SELECT legacy.draft_id, legacy.item_position,
               'legacy:' || CAST(legacy.item_position AS TEXT), 'eaten',
               legacy.food_text, legacy.quantity_text, 'resolved', NULL,
               legacy.occurrence_id, legacy.nutrition_snapshot_id,
               legacy.source_kind, legacy.source_value, legacy.content_signature,
               'resolved', legacy.official_servings, legacy.quantity_source,
               NULL, legacy.confidence, ({provenance}), 0
        FROM meal_report_draft_unplanned_items_v10 AS legacy
        """
    )
    connection.execute(
        """
        INSERT INTO meal_report_draft_clarifications
        (draft_id, clarification_position, reason, plan_item_id, food_text, quantity_text)
        SELECT draft_id, clarification_position, reason, plan_item_id, food_text, NULL
        FROM meal_report_draft_clarifications_v10
        """
    )
    connection.execute("DROP TABLE meal_report_draft_planned_items_v10")
    connection.execute("DROP TABLE meal_report_draft_unplanned_items_v10")
    connection.execute("DROP TABLE meal_report_draft_clarifications_v10")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise NutritionCatalogError("v11 report-draft migration produced invalid foreign keys")


def _message_outcome_migration_required(connection: sqlite3.Connection) -> bool:
    """Return whether inbound message events predate durable pre-route outcomes."""

    columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(meal_report_message_events)"
        ).fetchall()
    }
    return bool(columns) and "outcome_type" not in columns


def _migrate_message_outcomes_v12(connection: sqlite3.Connection) -> None:
    """Allow plan-free outcomes while preserving every routed v11 event."""

    rows = tuple(
        connection.execute(
            """
            SELECT event.source_event_id, event.chat_guid, event.plan_id,
                   event.draft_id, event.intent, event.reply_text,
                   event.processed_at,
                   CASE
                     WHEN application.source_event_id IS NOT NULL
                       THEN 'final_application'
                     WHEN event.draft_id IS NOT NULL THEN 'draft_update'
                     ELSE 'routed_interaction'
                   END AS outcome_type,
                   application.source_event_id AS application_source_event_id
            FROM meal_report_message_events AS event
            LEFT JOIN meal_report_applications AS application
              ON application.source_event_id = event.source_event_id
            ORDER BY event.processed_at, event.source_event_id
            """
        ).fetchall()
    )
    connection.execute("DROP INDEX IF EXISTS meal_report_message_events_draft_idx")
    connection.execute(
        "ALTER TABLE meal_report_message_events RENAME TO meal_report_message_events_v11"
    )
    for statement in _CONVERSATIONAL_MEAL_REPORT_SCHEMA_STATEMENTS:
        connection.execute(statement)
    connection.executemany(
        """
        INSERT INTO meal_report_message_events
        (source_event_id, chat_guid, outcome_type, plan_id, draft_id,
         application_source_event_id, intent, reply_text, processed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        tuple(
            (
                row["source_event_id"],
                row["chat_guid"],
                row["outcome_type"],
                row["plan_id"],
                row["draft_id"],
                row["application_source_event_id"],
                row["intent"],
                row["reply_text"],
                row["processed_at"],
            )
            for row in rows
        ),
    )
    connection.execute(
        """
        INSERT INTO meal_report_message_events
        (source_event_id, chat_guid, outcome_type, plan_id, draft_id,
         application_source_event_id, intent, reply_text, processed_at)
        SELECT request.source_event_id, request.chat_guid,
               'pending_meal_request', NULL, NULL, NULL, 'meal_request',
               request.reply_text, request.created_at
        FROM pending_meal_requests AS request
        WHERE NOT EXISTS (
            SELECT 1 FROM meal_report_message_events AS event
            WHERE event.source_event_id = request.source_event_id
        )
        """
    )
    connection.execute("DROP TABLE meal_report_message_events_v11")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise NutritionCatalogError("v12 message-outcome migration produced invalid foreign keys")


def _application_lifecycle_needs_migration(connection: sqlite3.Connection) -> bool:
    """Return whether a v3/v4 application-state table needs the v5 rebuild."""

    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(meal_plans)").fetchall()
    }
    if not columns:
        raise NutritionCatalogError("meal plan application state is missing")
    table_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'meal_plans'"
    ).fetchone()
    schema_sql = "" if table_row is None or table_row["sql"] is None else str(table_row["sql"])
    normalized_schema = schema_sql.casefold()
    return (
        "meal_context" not in columns
        or "applied" not in normalized_schema
        or "superseded" not in normalized_schema
    )


def _migrate_application_lifecycle_v5(connection: sqlite3.Connection) -> None:
    """Rebuild v3/v4 application tables with durable plan lifecycle state.

    SQLite cannot widen the legacy ``meal_plans.status`` CHECK constraint in
    place.  The caller disables foreign-key enforcement and opens one explicit
    transaction before this function runs, so the complete child-first rebuild
    either commits together or leaves the prior schema and history untouched.
    """

    plan_rows = tuple(
        connection.execute(
            """
            SELECT plan_id, service_date, meal, meal_kind, created_at
            FROM meal_plans
            ORDER BY created_at, plan_id
            """
        ).fetchall()
    )
    item_rows = tuple(
        connection.execute(
            """
            SELECT plan_id, plan_item_id, item_position, occurrence_id,
                   nutrition_snapshot_id, source_kind, source_value,
                   content_signature, recommended_official_servings,
                   natural_quantity_text, display_food_name
            FROM meal_plan_items
            ORDER BY plan_id, item_position
            """
        ).fetchall()
    )
    application_rows = tuple(
        connection.execute(
            """
            SELECT source_event_id, plan_id, applied_at
            FROM meal_report_applications
            ORDER BY source_event_id
            """
        ).fetchall()
    )
    intake_rows = tuple(
        connection.execute(
            """
            SELECT intake_id, source_event_id, service_date, recorded_at, meal,
                   plan_id, plan_item_id, occurrence_id, nutrition_snapshot_id,
                   source_kind, source_value, content_signature,
                   official_servings, quantity_source, original_reference_text,
                   original_quantity_text, item_position
            FROM accepted_intake_entries
            ORDER BY source_event_id, item_position, intake_id
            """
        ).fetchall()
    )

    contexts_by_plan: dict[str, str] = {}
    slots_by_plan: dict[str, str] = {}
    uncompleted_by_context: dict[tuple[str, str], list[sqlite3.Row]] = {}
    applied_plan_ids = {str(row["plan_id"]) for row in application_rows}
    for row in plan_rows:
        plan_id = str(row["plan_id"])
        try:
            meal = (
                int(str(row["meal"]))
                if row["meal_kind"] == "integer"
                else str(row["meal"])
            )
            context = meal_context_key(meal)
            slot = legacy_meal_slot(date.fromisoformat(str(row["service_date"])), meal)
        except (TypeError, ValueError):
            raise NutritionCatalogError("persisted meal plan has an invalid meal context") from None
        contexts_by_plan[plan_id] = context
        slots_by_plan[plan_id] = slot
        if plan_id not in applied_plan_ids:
            uncompleted_by_context.setdefault((str(row["service_date"]), context), []).append(row)

    active_plan_ids = {
        str(max(rows, key=lambda row: (str(row["created_at"]), str(row["plan_id"])))["plan_id"])
        for rows in uncompleted_by_context.values()
    }

    # Recreate the FK graph child-first.  SQLite has no ALTER TABLE operation
    # for replacing a CHECK constraint, and preserving every row is safer than
    # a destructive status-only rewrite.
    for table_name in (
        "accepted_intake_entries",
        "meal_report_applications",
        "meal_plan_items",
        "meal_plans",
    ):
        connection.execute(f"DROP TABLE {table_name}")
    for statement in _APPLICATION_STATE_SCHEMA_STATEMENTS:
        connection.execute(statement)

    for row in plan_rows:
        plan_id = str(row["plan_id"])
        status = (
            "applied"
            if plan_id in applied_plan_ids
            else "active"
            if plan_id in active_plan_ids
            else "superseded"
        )
        connection.execute(
            """
            INSERT INTO meal_plans
            (plan_id, service_date, meal, meal_kind, meal_context, meal_slot, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan_id,
                row["service_date"],
                row["meal"],
                row["meal_kind"],
                contexts_by_plan[plan_id],
                slots_by_plan[plan_id],
                row["created_at"],
                status,
            ),
        )
    for row in item_rows:
        connection.execute(
            """
            INSERT INTO meal_plan_items
            (plan_id, plan_item_id, item_position, occurrence_id,
             nutrition_snapshot_id, source_kind, source_value,
             content_signature, recommended_official_servings,
             natural_quantity_text, display_food_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(row),
        )
    for row in application_rows:
        connection.execute(
            """
            INSERT INTO meal_report_applications (source_event_id, plan_id, applied_at)
            VALUES (?, ?, ?)
            """,
            tuple(row),
        )
    for row in intake_rows:
        connection.execute(
            """
            INSERT INTO accepted_intake_entries
            (intake_id, source_event_id, service_date, recorded_at, meal, plan_id,
             plan_item_id, occurrence_id, nutrition_snapshot_id, source_kind,
             source_value, content_signature, official_servings, quantity_source,
             original_reference_text, original_quantity_text, item_position)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(row),
        )

    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise NutritionCatalogError("meal plan lifecycle migration violates foreign keys")


def _occurrence_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (str, int)):
        text = str(value).strip()
        return text or None
    return None


def _occurrence_date_text(occurrence: FDMealOccurrence) -> str:
    value = occurrence.menu_date
    if not isinstance(value, str) or not value.strip():
        raise NutritionCatalogValidationError("FD occurrence is missing a service date")
    try:
        return date.fromisoformat(value.strip()[:10]).isoformat()
    except ValueError as exc:
        raise NutritionCatalogValidationError("FD occurrence has an invalid service date") from exc


def _service_date_parameter(value: date) -> str:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise TypeError("service_date must be a date")
    return value.isoformat()


def _occurrence_key_payload(
    occurrence: FDMealOccurrence,
    *,
    service_date: str,
    source_identifier: SourceIdentifier,
) -> dict[str, str | None]:
    meal_period_id = _occurrence_text(occurrence.meal_period_id)
    meal_token = meal_period_id or _occurrence_text(occurrence.meal_period_name)
    station_id = _occurrence_text(occurrence.station_concept_id)
    station_token = station_id or _occurrence_text(occurrence.station_name)
    menu_detail_id = _occurrence_text(occurrence.menu_detail_id)
    if menu_detail_id is not None:
        occurrence_token = f"menu_detail:{menu_detail_id}"
    else:
        menu_id = _occurrence_text(occurrence.menu_id) or ""
        ordinal = "0" if occurrence.occurrence_ordinal is None else str(occurrence.occurrence_ordinal)
        occurrence_token = f"menu:{menu_id}:ordinal:{ordinal}"
    return {
        "service_date": service_date,
        "meal": meal_token,
        "station": station_token,
        "source_kind": source_identifier.kind,
        "source_value": source_identifier.value,
        "occurrence": occurrence_token,
    }


def occurrence_key_for_fd_result(result: FDMappingResult) -> str:
    """Return a stable logical key that excludes the nutrition version."""

    if not isinstance(result, FDMappingResult) or result.record is None:
        raise NutritionCatalogValidationError("an accepted FD mapping result is required")
    occurrence = result.occurrence
    if occurrence is None:
        raise NutritionCatalogValidationError("accepted FD result is missing occurrence context")
    source_identifier = _source_identifier_for(result.record, None)
    payload = _occurrence_key_payload(
        occurrence,
        service_date=_occurrence_date_text(occurrence),
        source_identifier=source_identifier,
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stored_identifier(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return value
    return parsed if str(parsed) == value or value.lstrip("+") == str(parsed) else value


def _migrate_presentation_bindings_v13(connection: sqlite3.Connection) -> None:
    """Add empty historical bindings; never infer authority from legacy text."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS meal_plan_item_presentation_bindings (
            plan_id TEXT NOT NULL,
            plan_item_id TEXT NOT NULL,
            binding_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (plan_id, plan_item_id),
            FOREIGN KEY (plan_id, plan_item_id)
                REFERENCES meal_plan_items(plan_id, plan_item_id)
        )
    """)
    connection.execute("""
        CREATE TRIGGER IF NOT EXISTS immutable_meal_presentation_binding
        BEFORE UPDATE ON meal_plan_item_presentation_bindings
        BEGIN SELECT RAISE(ABORT, 'presentation bindings are immutable'); END
    """)
    connection.execute("""
        CREATE TRIGGER IF NOT EXISTS retain_meal_presentation_binding
        BEFORE DELETE ON meal_plan_item_presentation_bindings
        BEGIN SELECT RAISE(ABORT, 'presentation bindings are immutable'); END
    """)
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise sqlite3.IntegrityError("presentation migration foreign-key violation")


class OfficialNutritionCatalog:
    """A single-user SQLite catalog of immutable official nutrition snapshots."""

    def __init__(
        self,
        path: str | Path = DEFAULT_CATALOG_PATH,
        *,
        read_only: bool = False,
    ) -> None:
        self.path = Path(path).expanduser()
        database = str(self.path)
        if not isinstance(read_only, bool):
            raise TypeError("read_only must be a bool")
        self._read_only = read_only
        if read_only and database == ":memory:":
            raise ValueError("a read-only catalog requires a file path")
        if database != ":memory:" and not read_only:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise NutritionCatalogError("unable to create nutrition catalog directory") from exc
        try:
            if read_only:
                database = f"{self.path.resolve().as_uri()}?mode=ro"
                self._connection = sqlite3.connect(database, uri=True)
            else:
                self._connection = sqlite3.connect(database)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            if read_only:
                self._validate_read_only_schema()
            else:
                self._initialize_schema()
        except NutritionCatalogError:
            self.close()
            raise
        except sqlite3.Error as exc:
            self.close()
            raise NutritionCatalogError("unable to initialize nutrition catalog") from exc

    def _validate_read_only_schema(self) -> None:
        """Reject a stale schema instead of mutating it for a diagnostic read."""

        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version != CATALOG_SCHEMA_VERSION:
            raise NutritionCatalogError(
                f"nutrition catalog schema version {version} is not readable without migration "
                f"to {CATALOG_SCHEMA_VERSION}"
            )

    def _initialize_schema(self) -> None:
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version > CATALOG_SCHEMA_VERSION:
            raise NutritionCatalogError(
                f"nutrition catalog schema version {version} is newer than supported version "
                f"{CATALOG_SCHEMA_VERSION}"
            )
        lifecycle_migration_required = (
            3 <= version <= 4
            and _application_lifecycle_needs_migration(self._connection)
        )
        meal_slot_migration_required = (
            3 <= version <= 9 and _meal_slot_migration_required(self._connection)
        )
        draft_fact_migration_required = (
            7 <= version <= 10 and _draft_fact_migration_required(self._connection)
        )
        message_outcome_migration_required = (
            7 <= version <= 11 and _message_outcome_migration_required(self._connection)
        )
        table_rebuild_required = (
            lifecycle_migration_required
            or meal_slot_migration_required
            or draft_fact_migration_required
            or message_outcome_migration_required
        )
        foreign_keys_were_enabled = False
        if table_rebuild_required:
            foreign_keys_were_enabled = bool(
                self._connection.execute("PRAGMA foreign_keys").fetchone()[0]
            )
            # FK enforcement must be toggled outside a transaction so the
            # v5 table rebuild can replace the parent and all children as one
            # transaction.  ``foreign_key_check`` below still validates every
            # copied relationship before commit.
            self._connection.execute("PRAGMA foreign_keys = OFF")
        try:
            if version < CATALOG_SCHEMA_VERSION:
                # SQLite's Python context manager does not begin a transaction
                # for DDL. Start one explicitly so ALTER/table rebuild work and
                # user_version update either commit together or all roll back.
                self._connection.execute("BEGIN")
            with self._connection:
                if version == 0:
                    for statement in _NUTRITION_SCHEMA_STATEMENTS:
                        self._connection.execute(statement)
                if version <= 1:
                    for statement in _OCCURRENCE_SCHEMA_STATEMENTS:
                        self._connection.execute(statement)
                if version <= 2:
                    for statement in _APPLICATION_STATE_SCHEMA_STATEMENTS:
                        self._connection.execute(statement)
                if version <= 3:
                    for statement in _DIETARY_FIBER_EXTENSION_SCHEMA_STATEMENTS:
                        self._connection.execute(statement)
                if lifecycle_migration_required:
                    _migrate_application_lifecycle_v5(self._connection)
                if version <= 5:
                    for statement in _SCHEDULED_RECOMMENDATION_SCHEMA_STATEMENTS:
                        self._connection.execute(statement)
                if version <= 6:
                    for statement in _CONVERSATIONAL_MEAL_REPORT_SCHEMA_STATEMENTS:
                        self._connection.execute(statement)
                if version <= 7:
                    for statement in _MEAL_RECOMMENDATION_REPLACEMENT_SCHEMA_STATEMENTS:
                        self._connection.execute(statement)
                if version <= 8:
                    _migrate_meal_request_state_v9(self._connection)
                if _meal_slot_migration_required(self._connection):
                    _migrate_meal_slots_v10(self._connection)
                if _draft_fact_migration_required(self._connection):
                    _migrate_report_draft_facts_v11(self._connection)
                if _message_outcome_migration_required(self._connection):
                    _migrate_message_outcomes_v12(self._connection)
                if version < 13:
                    _migrate_presentation_bindings_v13(self._connection)
                if version < 14:
                    for statement in _SHAKE_INTAKE_SCHEMA_STATEMENTS:
                        self._connection.execute(statement)
                self._connection.execute(f"PRAGMA user_version = {CATALOG_SCHEMA_VERSION}")
        except sqlite3.Error as exc:
            raise NutritionCatalogError("unable to migrate nutrition catalog schema") from exc
        finally:
            if table_rebuild_required:
                self._connection.execute(
                    f"PRAGMA foreign_keys = {1 if foreign_keys_were_enabled else 0}"
                )

    def close(self) -> None:
        if getattr(self, "_connection", None) is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "OfficialNutritionCatalog":
        if self._connection is None:
            raise NutritionCatalogError("nutrition catalog is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise NutritionCatalogError("nutrition catalog is closed")
        return self._connection

    def _snapshot_from_row(self, row: sqlite3.Row) -> CatalogSnapshot:
        try:
            source_identifier = SourceIdentifier(
                kind=row["source_kind"],
                value=row["source_value"],
            )
            content_signature = _required_text(
                row["content_signature"], field_name="content_signature"
            )
            provider = _required_text(row["provider"], field_name="provider")
            snapshot_id = int(row["snapshot_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise NutritionCatalogSerializationError("catalog identity metadata is corrupt") from exc
        record = self._hydrate_dietary_fiber(
            deserialize_nutrition_record(row["snapshot_json"]), snapshot_id
        )
        if record.provenance.provider != provider or source_identifier not in record.provenance.identifiers:
            raise NutritionCatalogSerializationError(
                "catalog identity does not match the serialized NutritionRecord"
            )
        return CatalogSnapshot(
            snapshot_id=snapshot_id,
            provider=provider,
            source_identifier=source_identifier,
            content_signature=content_signature,
            record=record,
            first_observed_at=_parse_timestamp(
                row["first_observed_at"], field_name="first_observed_at"
            ),
            last_observed_at=_parse_timestamp(
                row["last_observed_at"], field_name="last_observed_at"
            ),
        )

    def _hydrate_dietary_fiber(
        self,
        record: NutritionRecord,
        snapshot_id: int,
    ) -> NutritionRecord:
        row = self._require_connection().execute(
            "SELECT dietary_fiber_g FROM nutrition_snapshot_dietary_fiber WHERE nutrition_snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
        if row is None:
            return record
        fiber = _decimal_from_json(row["dietary_fiber_g"], field_name="dietary_fiber_g")
        if fiber is None:
            raise NutritionCatalogSerializationError("dietary fiber extension is invalid")
        return replace(record, nutrients=replace(record.nutrients, dietary_fiber_g=fiber))

    def _upsert_dietary_fiber_extension(
        self,
        snapshot_id: int,
        dietary_fiber_g: Decimal | None,
    ) -> Literal["inserted", "unchanged"]:
        if dietary_fiber_g is None:
            return "unchanged"
        value = _decimal_to_json(dietary_fiber_g)
        assert value is not None
        connection = self._require_connection()
        existing = connection.execute(
            "SELECT dietary_fiber_g FROM nutrition_snapshot_dietary_fiber WHERE nutrition_snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
        if existing is not None:
            if existing["dietary_fiber_g"] != value:
                raise NutritionCatalogValidationError(
                    "dietary fiber extension conflicts with the immutable snapshot"
                )
            return "unchanged"
        connection.execute(
            "INSERT INTO nutrition_snapshot_dietary_fiber (nutrition_snapshot_id, dietary_fiber_g) VALUES (?, ?)",
            (snapshot_id, value),
        )
        return "inserted"

    def _upsert_observation(self, observation: CatalogObservation) -> CatalogWriteResult:
        connection = self._require_connection()
        source_identifier = _source_identifier_for(
            observation.record, observation.source_identifier
        )
        provider = observation.record.provenance.provider
        signature = observation.content_signature.strip()
        observed_at = _timestamp_text(observation.observed_at)
        snapshot_json = serialize_nutrition_record(
            observation.record, include_dietary_fiber=False
        )
        identity_params = (provider, source_identifier.kind, source_identifier.value, signature)
        existing = connection.execute(
            """
            SELECT snapshot_id, first_observed_at, last_observed_at
            FROM nutrition_snapshots
            WHERE provider = ? AND source_kind = ? AND source_value = ?
              AND content_signature = ?
            """,
            identity_params,
        ).fetchone()
        if existing is not None:
            first_observed_at = min(existing["first_observed_at"], observed_at)
            last_observed_at = max(existing["last_observed_at"], observed_at)
            connection.execute(
                """
                UPDATE nutrition_snapshots
                SET first_observed_at = ?, last_observed_at = ?
                WHERE snapshot_id = ?
                """,
                (first_observed_at, last_observed_at, existing["snapshot_id"]),
            )
            self._upsert_dietary_fiber_extension(
                int(existing["snapshot_id"]), observation.record.nutrients.dietary_fiber_g
            )
            row = connection.execute(
                "SELECT * FROM nutrition_snapshots WHERE snapshot_id = ?",
                (existing["snapshot_id"],),
            ).fetchone()
            assert row is not None
            return CatalogWriteResult("unchanged", self._snapshot_from_row(row))

        identity_exists = connection.execute(
            """
            SELECT 1
            FROM nutrition_snapshots
            WHERE provider = ? AND source_kind = ? AND source_value = ?
            LIMIT 1
            """,
            (provider, source_identifier.kind, source_identifier.value),
        ).fetchone()
        try:
            cursor = connection.execute(
                """
                INSERT INTO nutrition_snapshots (
                    provider, source_kind, source_value, content_signature,
                    snapshot_json, first_observed_at, last_observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provider,
                    source_identifier.kind,
                    source_identifier.value,
                    signature,
                    snapshot_json,
                    observed_at,
                    observed_at,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise NutritionCatalogError("unable to store nutrition snapshot") from exc
        snapshot_id = int(cursor.lastrowid)
        self._upsert_dietary_fiber_extension(
            snapshot_id, observation.record.nutrients.dietary_fiber_g
        )
        row = connection.execute(
            "SELECT * FROM nutrition_snapshots WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
        assert row is not None
        return CatalogWriteResult(
            "new_version" if identity_exists is not None else "inserted",
            self._snapshot_from_row(row),
        )

    def store(
        self,
        record: NutritionRecord,
        *,
        content_signature: str,
        observed_at: datetime,
        source_identifier: SourceIdentifier | None = None,
    ) -> CatalogWriteResult:
        """Store one observation idempotently, preserving changed versions."""

        observation = CatalogObservation(
            record=record,
            content_signature=content_signature,
            observed_at=observed_at,
            source_identifier=source_identifier,
        )
        return self.store_many((observation,))[0]

    def store_many(self, observations: Iterable[CatalogObservation]) -> tuple[CatalogWriteResult, ...]:
        """Store a batch in one transaction; any failure rolls back the batch."""

        connection = self._require_connection()
        try:
            with connection:
                return self._store_observations(observations)
        except sqlite3.Error as exc:
            raise NutritionCatalogError("nutrition catalog transaction failed") from exc

    def _store_observations(
        self,
        observations: Iterable[CatalogObservation],
    ) -> tuple[CatalogWriteResult, ...]:
        """Store observations inside the caller's transaction boundary."""

        results: list[CatalogWriteResult] = []
        for observation in observations:
            if not isinstance(observation, CatalogObservation):
                raise TypeError("store_many accepts CatalogObservation values")
            results.append(self._upsert_observation(observation))
        return tuple(results)

    def backfill_dietary_fiber(
        self,
        observations: Iterable[CatalogObservation],
    ) -> DietaryFiberBackfillResult:
        """Attach fiber only to exact existing identity/signature snapshots.

        This never creates a snapshot, rewrites snapshot JSON, or changes an
        existing extension. A conflicting source assertion rolls back the
        complete batch rather than guessing.
        """

        unique: dict[tuple[str, str, str, str], CatalogObservation] = {}
        for observation in observations:
            if not isinstance(observation, CatalogObservation):
                raise TypeError("observations must contain CatalogObservation values")
            identifier = _source_identifier_for(observation.record, observation.source_identifier)
            key = (
                observation.record.provenance.provider,
                identifier.kind,
                identifier.value,
                observation.content_signature.strip(),
            )
            existing = unique.get(key)
            if existing is not None and (
                existing.record.nutrients.dietary_fiber_g
                != observation.record.nutrients.dietary_fiber_g
            ):
                raise NutritionCatalogValidationError(
                    "one fiber backfill contains conflicting source values"
                )
            unique.setdefault(key, observation)

        matched = inserted = unchanged = 0
        unmatched: list[tuple[SourceIdentifier, str]] = []
        connection = self._require_connection()
        try:
            with connection:
                for provider, kind, value, signature in sorted(unique):
                    row = connection.execute(
                        """
                        SELECT snapshot_id FROM nutrition_snapshots
                        WHERE provider = ? AND source_kind = ? AND source_value = ?
                          AND content_signature = ?
                        """,
                        (provider, kind, value, signature),
                    ).fetchone()
                    if row is None:
                        unmatched.append((SourceIdentifier(kind, value), signature))
                        continue
                    matched += 1
                    outcome = self._upsert_dietary_fiber_extension(
                        int(row["snapshot_id"]),
                        unique[(provider, kind, value, signature)].record.nutrients.dietary_fiber_g,
                    )
                    if outcome == "inserted":
                        inserted += 1
                    else:
                        unchanged += 1
        except sqlite3.Error as exc:
            raise NutritionCatalogError("dietary fiber backfill failed atomically") from exc
        return DietaryFiberBackfillResult(
            source_observations=len(unique),
            matched_snapshots=matched,
            inserted=inserted,
            unchanged=unchanged,
            unmatched=tuple(unmatched),
        )

    def _identity(self, source_identifier: SourceIdentifier, provider: str) -> tuple[str, str, str]:
        if not isinstance(source_identifier, SourceIdentifier):
            raise TypeError("source_identifier must be a SourceIdentifier")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider must be non-empty text")
        return provider, source_identifier.kind, source_identifier.value

    def get_history(
        self,
        source_identifier: SourceIdentifier,
        *,
        provider: str = DEFAULT_PROVIDER,
    ) -> tuple[CatalogSnapshot, ...]:
        """Return all versions newest-observed first, deterministically."""

        connection = self._require_connection()
        identity = self._identity(source_identifier, provider)
        rows = connection.execute(
            """
            SELECT * FROM nutrition_snapshots
            WHERE provider = ? AND source_kind = ? AND source_value = ?
            ORDER BY last_observed_at DESC, snapshot_id DESC
            """,
            identity,
        ).fetchall()
        return tuple(self._snapshot_from_row(row) for row in rows)

    def get_current_snapshot(
        self,
        source_identifier: SourceIdentifier,
        *,
        provider: str = DEFAULT_PROVIDER,
    ) -> CatalogSnapshot | None:
        """Return the version with the newest observation timestamp."""

        history = self.get_history(source_identifier, provider=provider)
        return history[0] if history else None

    def get_current_record(
        self,
        source_identifier: SourceIdentifier,
        *,
        provider: str = DEFAULT_PROVIDER,
    ) -> NutritionRecord | None:
        snapshot = self.get_current_snapshot(source_identifier, provider=provider)
        return snapshot.record if snapshot is not None else None

    def has_snapshot(
        self,
        source_identifier: SourceIdentifier,
        *,
        provider: str = DEFAULT_PROVIDER,
        content_signature: str,
    ) -> bool:
        connection = self._require_connection()
        identity = self._identity(source_identifier, provider)
        if not isinstance(content_signature, str) or not content_signature.strip():
            raise ValueError("content_signature must be non-empty text")
        row = connection.execute(
            """
            SELECT 1 FROM nutrition_snapshots
            WHERE provider = ? AND source_kind = ? AND source_value = ?
              AND content_signature = ?
            LIMIT 1
            """,
            (*identity, content_signature.strip()),
        ).fetchone()
        return row is not None

    def get_snapshot(
        self,
        source_identifier: SourceIdentifier,
        *,
        content_signature: str,
        provider: str = DEFAULT_PROVIDER,
    ) -> CatalogSnapshot | None:
        """Return one exact immutable snapshot by identity and signature."""

        connection = self._require_connection()
        identity = self._identity(source_identifier, provider)
        if not isinstance(content_signature, str) or not content_signature.strip():
            raise ValueError("content_signature must be non-empty text")
        row = connection.execute(
            """
            SELECT * FROM nutrition_snapshots
            WHERE provider = ? AND source_kind = ? AND source_value = ?
              AND content_signature = ?
            """,
            (*identity, content_signature.strip()),
        ).fetchone()
        return self._snapshot_from_row(row) if row is not None else None

    def get_snapshot_by_id(self, snapshot_id: int) -> CatalogSnapshot | None:
        """Return an immutable snapshot by its durable SQLite identity."""

        if isinstance(snapshot_id, bool) or not isinstance(snapshot_id, int) or snapshot_id <= 0:
            raise ValueError("snapshot_id must be a positive integer")
        row = self._require_connection().execute(
            "SELECT * FROM nutrition_snapshots WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
        return self._snapshot_from_row(row) if row is not None else None

    def get_occurrence_by_id(self, occurrence_id: int) -> FDMenuOccurrence | None:
        """Load one historical occurrence without consulting current menu state."""

        if isinstance(occurrence_id, bool) or not isinstance(occurrence_id, int) or occurrence_id <= 0:
            raise ValueError("occurrence_id must be a positive integer")
        row = self._require_connection().execute(
            """
            SELECT
                o.occurrence_id, o.occurrence_key, o.service_date,
                o.meal_period_id, o.meal_period_name, o.station_concept_id,
                o.station_name, o.source_kind, o.source_value,
                o.nutrition_snapshot_id, o.menu_detail_id, o.menu_id,
                o.first_observed_at, o.last_observed_at,
                s.provider AS snapshot_provider,
                s.content_signature AS snapshot_content_signature,
                s.snapshot_json
            FROM fd_menu_occurrences AS o
            JOIN nutrition_snapshots AS s
              ON s.snapshot_id = o.nutrition_snapshot_id
            WHERE o.occurrence_id = ?
            """,
            (occurrence_id,),
        ).fetchone()
        return self._cached_occurrence_from_row(row) if row is not None else None

    def _current_occurrence_state(
        self,
        start_date: date,
        end_date: date,
        *,
        provider: str,
    ) -> dict[str, str]:
        """Return logical occurrence keys in the latest completed coverage."""

        start_text = _service_date_parameter(start_date)
        end_text = _service_date_parameter(end_date)
        connection = self._require_connection()
        rows = connection.execute(
            """
            SELECT o.occurrence_key, o.content_signature
            FROM fd_menu_occurrences AS o
            JOIN fd_refresh_occurrences AS membership
              ON membership.occurrence_id = o.occurrence_id
            JOIN fd_refresh_runs AS run
              ON run.refresh_id = membership.refresh_id
            WHERE o.provider = ?
              AND o.service_date BETWEEN ? AND ?
              AND run.status = 'complete'
              AND run.refresh_id = (
                  SELECT MAX(latest.refresh_id)
                  FROM fd_refresh_runs AS latest
                  WHERE latest.provider = o.provider
                    AND latest.status = 'complete'
                    AND latest.requested_start_date <= o.service_date
                    AND latest.requested_end_date >= o.service_date
              )
            ORDER BY o.occurrence_key, o.content_signature, o.occurrence_id
            """,
            (provider, start_text, end_text),
        ).fetchall()
        state: dict[str, str] = {}
        for row in rows:
            key = str(row["occurrence_key"])
            signature = str(row["content_signature"])
            previous = state.setdefault(key, signature)
            if previous != signature:
                raise NutritionCatalogError(
                    "current FD refresh contains conflicting nutrition versions for one occurrence"
                )
        return state

    def _occurrence_observation(
        self,
        result: FDMappingResult,
        *,
        snapshot: CatalogSnapshot,
    ) -> _OccurrenceObservation | None:
        if result.record is None or result.occurrence is None:
            return None
        occurrence = result.occurrence
        if classify_component(occurrence.component) is not None:
            return None
        if not is_student_visible_component(occurrence.component):
            return None
        source_identifier = _source_identifier_for(result.record, None)
        signature = result.content_signature
        if not isinstance(signature, str) or not signature.strip():
            signature = fd_content_signature(occurrence.component)
        signature = signature.strip()
        if (
            snapshot.provider != result.record.provenance.provider
            or snapshot.source_identifier != source_identifier
            or snapshot.content_signature != signature
        ):
            raise NutritionCatalogValidationError(
                "FD occurrence does not match the exact nutrition snapshot being linked"
            )
        service_date = _occurrence_date_text(occurrence)
        meal_period_id = _occurrence_text(occurrence.meal_period_id)
        if meal_period_id is None:
            raise NutritionCatalogValidationError("FD occurrence is missing a meal period ID")
        meal_period_name = _occurrence_text(occurrence.meal_period_name)
        if meal_period_name is None:
            meal_period_name = canonical_meal_name(meal_period_id) or meal_period_id
        return _OccurrenceObservation(
            occurrence_key=occurrence_key_for_fd_result(result),
            service_date=service_date,
            meal_period_id=meal_period_id,
            meal_period_name=meal_period_name,
            station_concept_id=_occurrence_text(occurrence.station_concept_id),
            station_name=_occurrence_text(occurrence.station_name),
            source_identifier=source_identifier,
            content_signature=signature,
            nutrition_snapshot_id=snapshot.snapshot_id,
            menu_detail_id=_occurrence_text(occurrence.menu_detail_id),
            menu_id=_occurrence_text(occurrence.menu_id),
        )

    def _upsert_occurrence(
        self,
        observation: _OccurrenceObservation,
        *,
        observed_at: str,
    ) -> int:
        connection = self._require_connection()
        identity = (
            DEFAULT_PROVIDER,
            observation.occurrence_key,
            observation.nutrition_snapshot_id,
        )
        existing = connection.execute(
            """
            SELECT occurrence_id, first_observed_at, last_observed_at
            FROM fd_menu_occurrences
            WHERE provider = ? AND occurrence_key = ? AND nutrition_snapshot_id = ?
            """,
            identity,
        ).fetchone()
        if existing is not None:
            first_observed_at = min(existing["first_observed_at"], observed_at)
            last_observed_at = max(existing["last_observed_at"], observed_at)
            connection.execute(
                """
                UPDATE fd_menu_occurrences
                SET meal_period_id = ?, meal_period_name = ?, station_concept_id = ?,
                    station_name = ?, source_kind = ?, source_value = ?,
                    content_signature = ?, menu_detail_id = ?, menu_id = ?,
                    first_observed_at = ?, last_observed_at = ?
                WHERE occurrence_id = ?
                """,
                (
                    observation.meal_period_id,
                    observation.meal_period_name,
                    observation.station_concept_id,
                    observation.station_name,
                    observation.source_identifier.kind,
                    observation.source_identifier.value,
                    observation.content_signature,
                    observation.menu_detail_id,
                    observation.menu_id,
                    first_observed_at,
                    last_observed_at,
                    existing["occurrence_id"],
                ),
            )
            return int(existing["occurrence_id"])

        cursor = connection.execute(
            """
            INSERT INTO fd_menu_occurrences (
                provider, occurrence_key, service_date, meal_period_id,
                meal_period_name, station_concept_id, station_name,
                source_kind, source_value, content_signature,
                nutrition_snapshot_id, menu_detail_id, menu_id,
                first_observed_at, last_observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                DEFAULT_PROVIDER,
                observation.occurrence_key,
                observation.service_date,
                observation.meal_period_id,
                observation.meal_period_name,
                observation.station_concept_id,
                observation.station_name,
                observation.source_identifier.kind,
                observation.source_identifier.value,
                observation.content_signature,
                observation.nutrition_snapshot_id,
                observation.menu_detail_id,
                observation.menu_id,
                observed_at,
                observed_at,
            ),
        )
        if cursor.lastrowid is None:  # pragma: no cover - sqlite always supplies this key
            raise NutritionCatalogError("unable to identify stored FD menu occurrence")
        return int(cursor.lastrowid)

    def synchronize_fd_refresh(
        self,
        mapped_results: FDMappingBatch | Iterable[FDMappingResult],
        *,
        requested_start: date,
        requested_end: date,
        observed_at: datetime,
    ) -> FDMenuRefreshResult:
        """Atomically store nutrition snapshots and the completed menu state."""

        if not isinstance(requested_start, date) or isinstance(requested_start, datetime):
            raise TypeError("requested_start must be a date")
        if not isinstance(requested_end, date) or isinstance(requested_end, datetime):
            raise TypeError("requested_end must be a date")
        if requested_start > requested_end:
            raise ValueError("requested_start must not be after requested_end")
        timestamp = _normalize_timestamp(observed_at)
        results = tuple(
            mapped_results.results if isinstance(mapped_results, FDMappingBatch) else mapped_results
        )
        for result in results:
            if not isinstance(result, FDMappingResult):
                raise TypeError("mapped_results must contain FDMappingResult values")

        accepted: list[FDMappingResult] = []
        for result in results:
            if result.record is None or result.occurrence is None:
                continue
            if classify_component(result.occurrence.component) is not None:
                continue
            if not is_student_visible_component(result.occurrence.component):
                continue
            signature = result.content_signature
            if not isinstance(signature, str) or not signature.strip():
                signature = fd_content_signature(result.occurrence.component)
            if not signature.strip():
                continue
            occurrence_date = date.fromisoformat(_occurrence_date_text(result.occurrence))
            if not requested_start <= occurrence_date <= requested_end:
                continue
            accepted.append(result)

        unique_results: dict[tuple[str, str, str, str], FDMappingResult] = {}
        observations: list[CatalogObservation] = []
        for result in accepted:
            source_identifier = _source_identifier_for(result.record, None)  # type: ignore[arg-type]
            signature = result.content_signature
            if not isinstance(signature, str) or not signature.strip():
                assert result.occurrence is not None
                signature = fd_content_signature(result.occurrence.component)
            key = (
                result.record.provenance.provider,  # type: ignore[union-attr]
                source_identifier.kind,
                source_identifier.value,
                signature.strip(),
            )
            unique_results.setdefault(key, result)
        for key in sorted(unique_results):
            result = unique_results[key]
            assert result.record is not None
            signature = key[3]
            observations.append(
                CatalogObservation(
                    record=result.record,
                    content_signature=signature,
                    observed_at=timestamp,
                )
            )

        previous_state = self._current_occurrence_state(
            requested_start,
            requested_end,
            provider=DEFAULT_PROVIDER,
        )
        occurrence_inputs: dict[str, FDMappingResult] = {}
        occurrence_signatures: dict[str, str] = {}
        for result in accepted:
            assert result.record is not None
            occurrence = result.occurrence
            assert occurrence is not None
            signature = result.content_signature
            if not isinstance(signature, str) or not signature.strip():
                signature = fd_content_signature(occurrence.component)
            occurrence_key = occurrence_key_for_fd_result(result)
            existing_signature = occurrence_signatures.get(occurrence_key)
            if existing_signature is not None and existing_signature != signature.strip():
                raise NutritionCatalogValidationError(
                    "one FD refresh contains conflicting nutrition versions for one occurrence"
                )
            occurrence_inputs.setdefault(occurrence_key, result)
            occurrence_signatures[occurrence_key] = signature.strip()

        connection = self._require_connection()
        occurrence_rows: dict[tuple[str, int], int] = {}
        refresh_id: int
        try:
            with connection:
                writes = self._store_observations(observations)
                snapshot_by_key = {
                    (
                        write.snapshot.provider,
                        write.snapshot.source_identifier.kind,
                        write.snapshot.source_identifier.value,
                        write.snapshot.content_signature,
                    ): write.snapshot
                    for write in writes
                }
                cursor = connection.execute(
                    """
                    INSERT INTO fd_refresh_runs (
                        provider, requested_start_date, requested_end_date,
                        observed_at, status
                    ) VALUES (?, ?, ?, ?, 'in_progress')
                    """,
                    (
                        DEFAULT_PROVIDER,
                        requested_start.isoformat(),
                        requested_end.isoformat(),
                        _timestamp_text(timestamp),
                    ),
                )
                if cursor.lastrowid is None:  # pragma: no cover - sqlite always supplies this key
                    raise NutritionCatalogError("unable to identify FD refresh run")
                refresh_id = int(cursor.lastrowid)
                for occurrence_key in sorted(occurrence_inputs):
                    result = occurrence_inputs[occurrence_key]
                    assert result.record is not None
                    source_identifier = _source_identifier_for(result.record, None)
                    signature = occurrence_signatures[occurrence_key]
                    snapshot_key = (
                        result.record.provenance.provider,
                        source_identifier.kind,
                        source_identifier.value,
                        signature,
                    )
                    snapshot = snapshot_by_key.get(snapshot_key)
                    if snapshot is None:
                        raise NutritionCatalogError(
                            "FD occurrence has no persisted exact nutrition snapshot"
                        )
                    occurrence_observation = self._occurrence_observation(
                        result,
                        snapshot=snapshot,
                    )
                    if occurrence_observation is None:
                        continue
                    occurrence_id = self._upsert_occurrence(
                        occurrence_observation,
                        observed_at=_timestamp_text(timestamp),
                    )
                    occurrence_rows[(occurrence_key, snapshot.snapshot_id)] = occurrence_id
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO fd_refresh_occurrences (refresh_id, occurrence_id)
                        VALUES (?, ?)
                        """,
                        (refresh_id, occurrence_id),
                    )
                connection.execute(
                    """
                    UPDATE fd_refresh_runs
                    SET status = 'complete', completed_at = ?
                    WHERE refresh_id = ?
                    """,
                    (_timestamp_text(timestamp), refresh_id),
                )
        except sqlite3.Error as exc:
            raise NutritionCatalogError("FD refresh transaction failed") from exc

        new_state = {
            key: occurrence_signatures[key]
            for key in occurrence_inputs
            if any(row_key[0] == key for row_key in occurrence_rows)
        }
        additions = set(new_state) - set(previous_state)
        removals = set(previous_state) - set(new_state)
        changes = {
            key
            for key in set(previous_state) & set(new_state)
            if previous_state[key] != new_state[key]
        }
        nutrition_result = FDRefreshResult(
            inserted=sum(write.outcome == "inserted" for write in writes),
            unchanged=sum(write.outcome == "unchanged" for write in writes),
            new_versions=sum(write.outcome == "new_version" for write in writes),
            rejected=sum(result.record is None for result in results),
            writes=writes,
        )
        return FDMenuRefreshResult(
            catalog=nutrition_result,
            refresh_id=refresh_id,
            occurrences_observed=len(accepted),
            current_logical_occurrences=len(new_state),
            occurrence_additions=len(additions),
            occurrence_changes=len(changes),
            occurrence_removals=len(removals),
        )

    def _latest_completed_refresh_id(self, service_date: str, provider: str) -> int | None:
        connection = self._require_connection()
        row = connection.execute(
            """
            SELECT refresh_id
            FROM fd_refresh_runs
            WHERE provider = ? AND status = 'complete'
              AND requested_start_date <= ? AND requested_end_date >= ?
            ORDER BY refresh_id DESC
            LIMIT 1
            """,
            (provider, service_date, service_date),
        ).fetchone()
        return int(row["refresh_id"]) if row is not None else None

    def _cached_occurrence_from_row(self, row: sqlite3.Row) -> FDMenuOccurrence:
        try:
            source_identifier = SourceIdentifier(
                kind=str(row["source_kind"]),
                value=str(row["source_value"]),
            )
            service_date = date.fromisoformat(str(row["service_date"]))
            meal_period_id = _stored_identifier(str(row["meal_period_id"]))
            if meal_period_id is None:
                raise ValueError("meal period ID is missing")
            meal_period_name = _required_text(
                row["meal_period_name"], field_name="meal_period_name"
            )
            content_signature = _required_text(
                row["snapshot_content_signature"], field_name="content_signature"
            )
            snapshot_id = int(row["nutrition_snapshot_id"])
            record = self._hydrate_dietary_fiber(
                deserialize_nutrition_record(row["snapshot_json"]), snapshot_id
            )
            if (
                record.provenance.provider != row["snapshot_provider"]
                or source_identifier not in record.provenance.identifiers
            ):
                raise NutritionCatalogSerializationError(
                    "occurrence snapshot identity does not match its NutritionRecord"
                )
            occurrence_id = int(row["occurrence_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise NutritionCatalogSerializationError("cached FD occurrence metadata is corrupt") from exc
        return FDMenuOccurrence(
            occurrence_id=occurrence_id,
            occurrence_key=_required_text(row["occurrence_key"], field_name="occurrence_key"),
            service_date=service_date,
            meal_period_id=meal_period_id,
            meal_period_name=meal_period_name,
            station_concept_id=_stored_identifier(row["station_concept_id"]),
            station_name=row["station_name"],
            source_identifier=source_identifier,
            content_signature=content_signature,
            nutrition_snapshot_id=snapshot_id,
            nutrition_record=record,
            menu_detail_id=row["menu_detail_id"],
            menu_id=_stored_identifier(row["menu_id"]),
            first_observed_at=_parse_timestamp(
                row["first_observed_at"], field_name="first_observed_at"
            ),
            last_observed_at=_parse_timestamp(
                row["last_observed_at"], field_name="last_observed_at"
            ),
        )

    def list_current_meal_occurrences(
        self,
        service_date: date,
        meal: str | int | None = None,
        station: str | int | None = None,
        *,
        provider: str = DEFAULT_PROVIDER,
    ) -> tuple[FDMenuOccurrence, ...]:
        """Read the current published menu entirely from the local SQLite cache."""

        service_date_text = _service_date_parameter(service_date)
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider must be non-empty text")
        refresh_id = self._latest_completed_refresh_id(service_date_text, provider)
        if refresh_id is None:
            return ()
        filters = ["membership.refresh_id = ?", "o.provider = ?", "o.service_date = ?"]
        params: list[Any] = [refresh_id, provider, service_date_text]
        if meal is not None:
            meal_text = str(meal).strip()
            if not meal_text:
                raise ValueError("meal must be non-empty when provided")
            meal_id = canonical_meal_id(meal)
            meal_name = canonical_meal_name(meal)
            if meal_id is not None and meal_name is not None:
                filters.append(
                    "(o.meal_period_id = ? OR lower(o.meal_period_name) = lower(?))"
                )
                params.extend((str(meal_id), meal_name))
            else:
                filters.append(
                    "(o.meal_period_id = ? OR lower(o.meal_period_name) = lower(?))"
                )
                params.extend((meal_text, meal_text))
        if station is not None:
            station_text = str(station).strip()
            if not station_text:
                raise ValueError("station must be non-empty when provided")
            filters.append(
                "(o.station_concept_id = ? OR lower(COALESCE(o.station_name, '')) = lower(?))"
            )
            params.extend((station_text, station_text))
        connection = self._require_connection()
        rows = connection.execute(
            f"""
            SELECT
                o.occurrence_id, o.occurrence_key, o.service_date,
                o.meal_period_id, o.meal_period_name, o.station_concept_id,
                o.station_name, o.source_kind, o.source_value,
                o.nutrition_snapshot_id, o.menu_detail_id, o.menu_id,
                o.first_observed_at, o.last_observed_at,
                s.provider AS snapshot_provider,
                s.content_signature AS snapshot_content_signature,
                s.snapshot_json
            FROM fd_menu_occurrences AS o
            JOIN fd_refresh_occurrences AS membership
              ON membership.occurrence_id = o.occurrence_id
            JOIN nutrition_snapshots AS s
              ON s.snapshot_id = o.nutrition_snapshot_id
            WHERE {' AND '.join(filters)}
            ORDER BY
                lower(COALESCE(o.meal_period_name, '')),
                lower(COALESCE(o.station_name, '')),
                COALESCE(o.station_concept_id, ''),
                o.source_value,
                COALESCE(o.menu_detail_id, ''),
                o.occurrence_key,
                o.occurrence_id
            """,
            tuple(params),
        ).fetchall()
        return tuple(self._cached_occurrence_from_row(row) for row in rows)

    def list_current_occurrences(
        self,
        service_date: date,
        meal: str | int | None = None,
        station: str | int | None = None,
        *,
        provider: str = DEFAULT_PROVIDER,
    ) -> tuple[FDMenuOccurrence, ...]:
        """Alias for the application-facing current-menu query."""

        return self.list_current_meal_occurrences(
            service_date,
            meal=meal,
            station=station,
            provider=provider,
        )

    @property
    def snapshot_count(self) -> int:
        connection = self._require_connection()
        return int(connection.execute("SELECT COUNT(*) FROM nutrition_snapshots").fetchone()[0])

    @property
    def dietary_fiber_extension_count(self) -> int:
        """Return immutable snapshots with a known first-class fiber value."""

        return int(
            self._require_connection().execute(
                "SELECT COUNT(*) FROM nutrition_snapshot_dietary_fiber"
            ).fetchone()[0]
        )


def refresh_from_fd_results(
    catalog: OfficialNutritionCatalog,
    mapped_results: FDMappingBatch | Iterable[FDMappingResult],
    *,
    observed_at: datetime | None = None,
) -> FDRefreshResult:
    """Persist valid mapped FD results without deciding what to fetch."""

    if not isinstance(catalog, OfficialNutritionCatalog):
        raise TypeError("catalog must be an OfficialNutritionCatalog")
    timestamp = observed_at or datetime.now(timezone.utc)
    _normalize_timestamp(timestamp)
    results = mapped_results.results if isinstance(mapped_results, FDMappingBatch) else mapped_results
    observations: list[CatalogObservation] = []
    rejected = 0
    for mapped in results:
        if not isinstance(mapped, FDMappingResult):
            raise TypeError("mapped_results must contain FDMappingResult values")
        if mapped.record is None:
            rejected += 1
            continue
        signature = mapped.content_signature
        if not signature and mapped.occurrence is not None:
            signature = fd_content_signature(mapped.occurrence.component)
        if not signature:
            rejected += 1
            continue
        observations.append(
            CatalogObservation(
                record=mapped.record,
                content_signature=signature,
                observed_at=timestamp,
            )
        )

    writes = catalog.store_many(observations)
    return FDRefreshResult(
        inserted=sum(write.outcome == "inserted" for write in writes),
        unchanged=sum(write.outcome == "unchanged" for write in writes),
        new_versions=sum(write.outcome == "new_version" for write in writes),
        rejected=rejected,
        writes=writes,
    )


def synchronize_fd_refresh(
    catalog: OfficialNutritionCatalog,
    mapped_results: FDMappingBatch | Iterable[FDMappingResult],
    *,
    requested_start: date,
    requested_end: date,
    observed_at: datetime,
) -> FDMenuRefreshResult:
    """Public wrapper for the atomic nutrition-plus-menu synchronization."""

    if not isinstance(catalog, OfficialNutritionCatalog):
        raise TypeError("catalog must be an OfficialNutritionCatalog")
    return catalog.synchronize_fd_refresh(
        mapped_results,
        requested_start=requested_start,
        requested_end=requested_end,
        observed_at=observed_at,
    )
