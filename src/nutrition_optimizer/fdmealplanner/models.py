"""Small transport and occurrence types for the FDMealPlanner adapter.

The service response is intentionally kept as mappings.  The public API adds
fields periodically, and the mapper only accepts fields whose meaning and
units have been explicitly validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class FDApplicationConfiguration:
    """Public routing values extracted from the frontend bootstrap response."""

    tenant_id: int
    security_prefix: str
    menu_url: str
    meal_period_url: str
    location_search_url: str

    @property
    def api_origin(self) -> str:
        """Return the HTTPS origin used by the tenant's data APIs."""

        from urllib.parse import urlsplit

        parsed = urlsplit(self.menu_url)
        return f"{parsed.scheme}://{parsed.netloc}"


@dataclass(frozen=True, slots=True)
class FDMealsPayload:
    """A validated monthly meal response and its daily result rows."""

    payload: Mapping[str, Any]
    days: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FDMealsPayload":
        if not isinstance(payload, Mapping):
            raise TypeError("FD meals payload must be a mapping")
        result = payload.get("result")
        if not isinstance(result, list):
            raise ValueError("FD meals payload result must be a list")
        days: list[Mapping[str, Any]] = []
        for index, day in enumerate(result):
            if not isinstance(day, Mapping):
                raise ValueError(f"FD meals result[{index}] must be an object")
            days.append(day)
        return cls(payload=payload, days=tuple(days))


@dataclass(frozen=True, slots=True)
class FDMealOccurrence:
    """One menu occurrence with context kept separate from catalog identity."""

    component: Mapping[str, Any]
    menu_date: str | None
    meal_period_id: int | None
    station_concept_id: int | str | None = None
    station_name: str | None = None
    menu_detail_id: str | None = None
    menu_id: int | str | None = None
    meal_period_name: str | None = None
    occurrence_ordinal: int | None = None

    @property
    def meal(self) -> str | None:
        """Return the human-readable meal period when the source supplied it."""

        return self.meal_period_name

    @property
    def station(self) -> str | None:
        """Return the station display name when the source supplied it."""

        return self.station_name
