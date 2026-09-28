"""Pure normalization of Hope's Phelps menu payload."""

from __future__ import annotations

from datetime import datetime
from html.parser import HTMLParser
import re
from typing import Any, Sequence

from .models import MenuBlock, MenuEntry, PhelpsMenu, ServiceHour


class MenuNormalizationError(ValueError):
    """Raised when a menu payload cannot be normalized safely."""


class PhelpsNotFoundError(MenuNormalizationError):
    """Raised when the source payload contains no Phelps location."""


_MARKER_RE = re.compile(r"^(?P<markers>(?:\[[^\]]+\]\s*)+)(?P<rest>.*)$")
_MARKER_RUN_RE = re.compile(r"(?:\[[^\]]+\]\s*)+")
_ALLERGEN_RE = re.compile(r"\b(?:CONTAINS|MAY\s+CONTAIN)\s*:\s*", re.IGNORECASE)


class _MenuHTMLParser(HTMLParser):
    """Collect text from source list items while ignoring presentation markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._items: list[list[str]] = []
        self.items: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() == "li":
            self._items.append([])
        elif tag.lower() == "br" and self._items:
            self._items[-1].append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "li" and self._items:
            raw_text = _clean_text("".join(self._items.pop()))
            if raw_text:
                self.items.append(raw_text)

    def handle_data(self, data: str) -> None:
        if self._items:
            self._items[-1].append(data)


def normalize_phelps(payload: Sequence[dict[str, Any]]) -> PhelpsMenu:
    """Convert a decoded Hope payload into normalized Phelps menu data."""

    location = next(
        (
            candidate
            for candidate in payload
            if isinstance(candidate.get("name"), str)
            and candidate["name"].strip().casefold() == "phelps"
        ),
        None,
    )
    if location is None:
        raise PhelpsNotFoundError("Payload does not contain a Phelps location")

    hours_value = location.get("hours", [])
    menus_value = location.get("menus", [])
    if not isinstance(hours_value, list):
        raise MenuNormalizationError("Phelps hours must be a list")
    if not isinstance(menus_value, list):
        raise MenuNormalizationError("Phelps menus must be a list")

    service_hours = tuple(_normalize_service_hour(item, index) for index, item in enumerate(hours_value))
    menu_blocks = tuple(_normalize_menu_block(item, index) for index, item in enumerate(menus_value))

    return PhelpsMenu(
        location_name=location["name"].strip(),
        service_hours=service_hours,
        menu_blocks=menu_blocks,
    )


def _normalize_service_hour(value: Any, index: int) -> ServiceHour:
    item = _require_mapping(value, f"Phelps hour at index {index}")
    context = f"Phelps hour at index {index}"
    start = _parse_interval_value(item, "start", context)
    end = _parse_interval_value(item, "end", context)
    _validate_interval(start, end, context)
    return ServiceHour(
        meal_name=_require_text(item, "name", context),
        start=start,
        end=end,
    )


def _normalize_menu_block(value: Any, index: int) -> MenuBlock:
    item = _require_mapping(value, f"Phelps menu at index {index}")
    context = f"Phelps menu at index {index}"
    html = item.get("foodAtStation", "")
    if not isinstance(html, str):
        raise MenuNormalizationError(
            f"Phelps menu at index {index} has non-text foodAtStation content"
        )
    start = _parse_interval_value(item, "start", context)
    end = _parse_interval_value(item, "end", context)
    _validate_interval(start, end, context)

    return MenuBlock(
        station_name=_require_text(item, "stationName", context),
        start=start,
        end=end,
        entries=tuple(_normalize_entry(raw_text) for raw_text in _extract_list_items(html)),
    )


def _extract_list_items(html: str) -> list[str]:
    parser = _MenuHTMLParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:
        raise MenuNormalizationError("Could not parse menu HTML") from exc
    items: list[str] = []
    for raw_text in parser.items:
        items.extend(_split_marker_prefixed_items(raw_text))
    return items


def _split_marker_prefixed_items(raw_text: str) -> list[str]:
    """Split source list items that contain multiple marker-prefixed foods."""

    marker_runs = list(_MARKER_RUN_RE.finditer(raw_text))
    if len(marker_runs) <= 1 or marker_runs[0].start() != 0:
        return [raw_text]

    starts = [match.start() for match in marker_runs]
    shared_allergen = _ALLERGEN_RE.search(raw_text)
    shared_suffix = raw_text[shared_allergen.start() :].strip() if shared_allergen else ""

    items = []
    for start, end in zip(starts, [*starts[1:], len(raw_text)]):
        item = raw_text[start:end].strip()
        if shared_suffix and not _ALLERGEN_RE.search(item):
            item = f"{item} {shared_suffix}"
        items.append(item)
    return items


def _normalize_entry(raw_text: str) -> MenuEntry:
    marker_match = _MARKER_RE.match(raw_text)
    if marker_match:
        dietary_markers = tuple(
            marker.upper() for marker in re.findall(r"\[([^\]]+)\]", marker_match.group("markers"))
        )
        text = marker_match.group("rest").strip()
    else:
        dietary_markers = ()
        text = raw_text

    allergen_match = _ALLERGEN_RE.search(text)
    standalone_allergen = bool(
        allergen_match and not text[: allergen_match.start()].strip()
    )
    allergens: tuple[str, ...]
    if allergen_match:
        name = text[: allergen_match.start()].strip("-, ")
        allergen_text = text[allergen_match.end() :]
        allergens = tuple(
            part.strip(" .,-").upper()
            for part in allergen_text.split(",")
            if part.strip(" .,-")
        )
    else:
        name = text.strip("-, ")
        allergens = ()

    if not name:
        name = raw_text

    is_notice = bool(
        re.match(r"^ALL\b", name, re.IGNORECASE)
        or re.search(r"\bMAY\s+CONTAIN\b", raw_text, re.IGNORECASE)
        or standalone_allergen
    )
    return MenuEntry(
        name=name,
        dietary_markers=dietary_markers,
        allergens=allergens,
        is_notice=is_notice,
        raw_text=raw_text,
    )


def _clean_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MenuNormalizationError(f"{context} must be an object")
    return value


def _require_text(value: dict[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise MenuNormalizationError(f"{context} has no valid {key}")
    return result.strip()


def _parse_interval_value(value: dict[str, Any], key: str, context: str) -> datetime:
    raw_value = value.get(key)
    if not isinstance(raw_value, str):
        raise MenuNormalizationError(f"{context} has no valid {key}")
    try:
        parsed = datetime.fromisoformat(raw_value)
    except ValueError as exc:
        raise MenuNormalizationError(f"{context} has invalid {key}: {raw_value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MenuNormalizationError(f"{context} {key} must include a timezone")
    return parsed


def _validate_interval(start: datetime, end: datetime, context: str) -> None:
    if start > end:
        raise MenuNormalizationError(f"{context} start cannot be after end")
