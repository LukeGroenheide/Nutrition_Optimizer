"""Hope menu fetching, snapshotting, and normalization."""

from .fetch import MENU_URL, MenuFetchError, fetch_menu
from .models import (
    FetchResult,
    IngestResult,
    MenuBlock,
    MenuEntry,
    PhelpsMenu,
    ServiceHour,
)
from .normalize import MenuNormalizationError, PhelpsNotFoundError, normalize_phelps
from .resolver import (
    AmbiguousFDMatch,
    DiningBucketOccurrence,
    FDMatchResult,
    FDStationScope,
    FD_STATION_CONCEPT_IDS,
    MEAL_PERIOD_IDS,
    NoFDMatch,
    NoMatchReason,
    ResolvedFDMatch,
    resolve_occurrence,
    resolve_station_scope,
)

__all__ = [
    "FetchResult",
    "IngestResult",
    "MENU_URL",
    "MenuBlock",
    "MenuEntry",
    "MenuFetchError",
    "MenuNormalizationError",
    "PhelpsMenu",
    "PhelpsNotFoundError",
    "ServiceHour",
    "fetch_menu",
    "normalize_phelps",
    "AmbiguousFDMatch",
    "DiningBucketOccurrence",
    "FDMatchResult",
    "FDStationScope",
    "FD_STATION_CONCEPT_IDS",
    "MEAL_PERIOD_IDS",
    "NoFDMatch",
    "NoMatchReason",
    "ResolvedFDMatch",
    "resolve_occurrence",
    "resolve_station_scope",
]
