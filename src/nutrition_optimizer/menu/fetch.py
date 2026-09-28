"""HTTP fetching for Hope's public dining menu feed."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from curl_cffi import requests

from .models import FetchResult

MENU_URL = "https://diningbucket.hope.edu/menus-and-hours.json"
REQUEST_HEADERS = {
    "Origin": "https://hope.edu",
    "Referer": "https://hope.edu/",
}


class MenuFetchError(RuntimeError):
    """Raised when the menu endpoint cannot be fetched or has an invalid shape."""


def fetch_menu(url: str = MENU_URL, timeout: float = 30.0) -> FetchResult:
    """Fetch and minimally validate Hope's menu payload.

    The response bytes are retained so callers can save the exact JSON body
    before attempting any normalization.
    """

    try:
        response = requests.get(
            url,
            impersonate="chrome",
            headers=REQUEST_HEADERS,
            timeout=timeout,
        )
    except Exception as exc:  # curl_cffi exposes several request exception types.
        raise MenuFetchError(f"Could not fetch menu from {url}: {exc}") from exc

    status_code = getattr(response, "status_code", None)
    if not isinstance(status_code, int) or not 200 <= status_code < 300:
        raise MenuFetchError(f"Menu request returned HTTP status {status_code!r}")

    raw_bytes = bytes(getattr(response, "content", b""))
    try:
        payload = json.loads(raw_bytes)
    except (TypeError, ValueError) as exc:
        raise MenuFetchError("Menu response was not valid JSON") from exc

    if not isinstance(payload, list):
        raise MenuFetchError("Menu response root must be a JSON list")

    for index, location in enumerate(payload):
        if not isinstance(location, dict):
            raise MenuFetchError(f"Menu location at index {index} must be an object")
        if not isinstance(location.get("name"), str):
            raise MenuFetchError(f"Menu location at index {index} has no valid name")

    fetched_at = datetime.now(timezone.utc)
    return FetchResult(
        url=url,
        fetched_at=fetched_at,
        raw_bytes=raw_bytes,
        payload=payload,
    )
