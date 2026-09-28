"""Frozen presentation authority. Display text is never parsed for conversion.

The database is trusted application storage, not an adversarial trust boundary.
The envelope digest detects accidental corruption; it is not a signature.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal
from enum import Enum
import hashlib
import json

from .food_resolution import ResolvedFood
from .nutrition.models import Serving, SourceIdentifier
from .serving_presentation import ServingPresentationCalibration, _positive_decimal, _require_text


class PresentationKind(str, Enum):
    EXACT_AUTHORITATIVE = "exact_authoritative"
    REVERSIBLE_CALIBRATED = "reversible_calibrated"
    DESCRIPTIVE_ONLY = "descriptive_only"


@dataclass(frozen=True, slots=True)
class PresentationBinding:
    """One selected rendering and its original, self-contained authority.

    Official servings repeat the plan multiplier solely as an equality guard.
    Scope records the original occurrence, including absent station metadata.
    Calibration ratios/aliases are frozen values, never a current-registry key.
    """
    format_version: int
    kind: PresentationKind
    display_text: str
    source_identifier: SourceIdentifier
    official_serving: Serving
    official_servings: Decimal
    occurrence_id: int
    nutrition_snapshot_id: int
    content_signature: str
    station_concept_id: int | str | None
    station_name: str | None
    calibration: ServingPresentationCalibration | None = None

    def __post_init__(self) -> None:
        if type(self.format_version) is not int or self.format_version != 1:
            raise ValueError("unsupported presentation binding format")
        if not isinstance(self.kind, PresentationKind):
            raise ValueError("invalid presentation classification")
        _require_text(self.display_text, "display_text")
        _positive_decimal(self.official_servings, "official_servings")
        if not isinstance(self.source_identifier, SourceIdentifier) or not isinstance(self.official_serving, Serving):
            raise ValueError("invalid presentation identity/serving guard")
        for value in (self.occurrence_id, self.nutrition_snapshot_id):
            if type(value) is not int or value <= 0:
                raise ValueError("invalid presentation occurrence/snapshot guard")
        _require_text(self.content_signature, "content_signature")
        if self.kind == PresentationKind.REVERSIBLE_CALIBRATED:
            if not isinstance(self.calibration, ServingPresentationCalibration):
                raise ValueError("reversible presentation requires calibration")
            if not self.calibration.matches(self.source_identifier, self.official_serving):
                raise ValueError("calibration does not match presentation guards")
        elif self.calibration is not None:
            raise ValueError("non-calibrated presentation cannot carry reverse authority")

    def validate(self, food: ResolvedFood, amount: Decimal, display_text: str) -> None:
        if (self.source_identifier != food.source_identifier
            or self.official_serving != food.nutrition_record.serving
            or self.official_servings != amount
            or self.display_text != display_text
            or self.occurrence_id != food.occurrence.occurrence_id
            or self.nutrition_snapshot_id != food.nutrition_snapshot_id
            or self.content_signature != food.content_signature
            or self.station_concept_id != food.occurrence.station_concept_id
            or self.station_name != food.occurrence.station_name):
            raise ValueError("presentation binding does not match frozen plan item")

    def official_servings_for_phrase(self, food: ResolvedFood, amount: Decimal, phrase: str) -> Decimal | None:
        self.validate(food, amount, self.display_text)
        if self.calibration is None:
            return None
        return self.calibration.official_servings_for_phrase(phrase)

    def to_json(self) -> str:
        payload = _encode(self)
        text = _canonical(payload)
        return _canonical({"payload": payload, "sha256": hashlib.sha256(text.encode()).hexdigest()})

    @classmethod
    def from_json(cls, value: str) -> PresentationBinding:
        try:
            envelope = json.loads(value)
            if set(envelope) != {"payload", "sha256"}:
                raise ValueError("invalid presentation envelope")
            data = envelope["payload"]
            if hashlib.sha256(_canonical(data).encode()).hexdigest() != envelope["sha256"]:
                raise ValueError("presentation binding checksum mismatch")
            data = dict(data)
            data['kind'] = PresentationKind(data['kind'])
            data['source_identifier'] = SourceIdentifier(**data['source_identifier'])
            serving = dict(data['official_serving'])
            if serving['quantity'] is not None:
                serving['quantity'] = _decimal_string(serving['quantity'])
            data['official_serving'] = Serving(**serving)
            data['official_servings'] = _decimal_string(data['official_servings'])
            if data['calibration'] is not None:
                calibration = dict(data['calibration'])
                calibration['source_identifier'] = SourceIdentifier(**calibration['source_identifier'])
                for name in (
                    'expected_serving_quantity',
                    'presentation_units_per_official_serving',
                    'whole_items_per_official_serving',
                    'whole_equivalent_minimum',
                    'display_rounding_increment',
                ):
                    if name not in calibration:
                        continue
                    if calibration[name] is not None:
                        calibration[name] = _decimal_string(calibration[name])
                for name in ('presentation_aliases', 'whole_item_aliases'):
                    calibration[name] = tuple(calibration[name])
                data['calibration'] = ServingPresentationCalibration(**calibration)
            return cls(**data)
        except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
            raise ValueError("invalid persisted presentation binding") from exc


def freeze_presentation(food: ResolvedFood, amount: Decimal, display_text: str,
                        kind: PresentationKind, calibration: ServingPresentationCalibration | None = None) -> PresentationBinding:
    """Python-only constructor called after deterministic rendering validation."""
    return PresentationBinding(1, kind, display_text, food.source_identifier,
        food.nutrition_record.serving, amount, food.occurrence.occurrence_id,
        food.nutrition_snapshot_id, food.content_signature,
        food.occurrence.station_concept_id, food.occurrence.station_name, calibration)


def _decimal_string(value: object) -> Decimal:
    if not isinstance(value, str):
        raise ValueError("authoritative decimal must be a string")
    return Decimal(value)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _encode(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    if hasattr(value, '__dataclass_fields__'):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    return value
