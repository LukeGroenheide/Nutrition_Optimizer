"""Data structures for the Hope menu ingestion layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FetchResult:
    """The decoded response and the exact bytes returned by Hope."""

    url: str
    fetched_at: datetime
    raw_bytes: bytes
    payload: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ServiceHour:
    """A source-provided meal/service period for Phelps."""

    meal_name: str
    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class MenuEntry:
    """A food line or an obvious notice extracted from station HTML."""

    name: str
    dietary_markers: tuple[str, ...]
    allergens: tuple[str, ...]
    is_notice: bool
    raw_text: str


@dataclass(frozen=True, slots=True)
class MenuBlock:
    """One station menu block and its source validity interval."""

    station_name: str
    start: datetime
    end: datetime
    entries: tuple[MenuEntry, ...]


@dataclass(frozen=True, slots=True)
class PhelpsMenu:
    """Normalized Phelps menu data derived from one source payload."""

    location_name: str
    service_hours: tuple[ServiceHour, ...]
    menu_blocks: tuple[MenuBlock, ...]


@dataclass(frozen=True, slots=True)
class IngestResult:
    """The raw snapshot location and normalized menu from one ingestion."""

    snapshot_path: Path
    fetched_at: datetime
    menu: PhelpsMenu
