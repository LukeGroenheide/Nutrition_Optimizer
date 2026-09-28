"""Application-local time semantics for Nutrition Optimizer service dates.

The Dell's operating-system timezone is deliberately not part of this
boundary.  Nutrition service dates belong to the Phelps/Hope College context,
whose authoritative timezone is America/Detroit; audit and retrieval
timestamps remain UTC at their existing boundaries.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo


__all__ = [
    "DEFAULT_NUTRITION_APPLICATION_CLOCK",
    "NUTRITION_APPLICATION_TIMEZONE",
    "NUTRITION_APPLICATION_TIMEZONE_NAME",
    "NutritionApplicationClock",
    "nutrition_service_date",
]


NUTRITION_APPLICATION_TIMEZONE_NAME = "America/Detroit"
NUTRITION_APPLICATION_TIMEZONE = ZoneInfo(NUTRITION_APPLICATION_TIMEZONE_NAME)

_NowProvider = Callable[[], datetime]


class NutritionApplicationClock:
    """Convert an absolute, aware instant into Nutrition application time."""

    def __init__(self, now_provider: _NowProvider | None = None) -> None:
        if now_provider is not None and not callable(now_provider):
            raise TypeError("now_provider must be callable")
        self._now_provider = now_provider or _utc_now

    def now(self) -> datetime:
        """Return the current aware datetime in America/Detroit."""

        instant = self._now_provider()
        if not isinstance(instant, datetime):
            raise TypeError("now_provider must return a datetime")
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("now_provider must return a timezone-aware datetime")
        return instant.astimezone(NUTRITION_APPLICATION_TIMEZONE)

    def service_date(self) -> date:
        """Return today's Nutrition service date in America/Detroit."""

        return self.now().date()


def _utc_now() -> datetime:
    """Read the system clock as an absolute UTC instant, never a local date."""

    return datetime.now(timezone.utc)


DEFAULT_NUTRITION_APPLICATION_CLOCK = NutritionApplicationClock()


def nutrition_service_date() -> date:
    """Return the current Phelps service date without consulting host-local time."""

    return DEFAULT_NUTRITION_APPLICATION_CLOCK.service_date()
