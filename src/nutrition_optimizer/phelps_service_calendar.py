"""Explicit Phelps availability rules in Nutrition application time.

The weekly rules below are a small, replaceable local configuration boundary
for the current Hope College/Phelps deployment.  They intentionally model
published facility hours separately from meal-period transitions: the
continuous weekday windows say that Phelps is open, but do *not* fabricate
breakfast, lunch, or dinner hand-off times.

All public methods receive an absolute, timezone-aware instant and normalize
it through :mod:`nutrition_optimizer.application_clock`.  Consequently these
rules are independent of the Dell's operating-system timezone and continue to
follow America/Detroit DST transitions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from types import MappingProxyType

from .application_clock import NUTRITION_APPLICATION_TIMEZONE
from .meal_identity import canonical_meal_id


__all__ = [
    "DEFAULT_PHELPS_SERVICE_CALENDAR",
    "PHELPS_WEEKLY_SERVICE_WINDOWS",
    "PhelpsServiceAvailability",
    "PhelpsServiceCalendar",
    "PhelpsServiceWindow",
    "PhelpsServiceWindowDefinition",
]


@dataclass(frozen=True, slots=True)
class PhelpsServiceWindowDefinition:
    """One local weekly opening interval.

    ``meal_id`` is an optional canonical FD meal-period ID.  ``None`` means a
    facility-level opening interval only; it deliberately grants no invented
    meal-period boundary.  Date-specific overrides use the same structure and
    replace the weekly windows for that date.
    """

    label: str
    opens_at: time
    closes_at: time
    meal_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("service-window label must be non-empty text")
        if not isinstance(self.opens_at, time) or not isinstance(self.closes_at, time):
            raise TypeError("service-window times must be time values")
        if self.opens_at.tzinfo is not None or self.closes_at.tzinfo is not None:
            raise ValueError("service-window times must be local naive times")
        if self.opens_at >= self.closes_at:
            raise ValueError("service-window closing time must follow opening time")
        if self.meal_id is not None and canonical_meal_id(self.meal_id) != self.meal_id:
            raise ValueError("meal_id must be a known canonical FD meal ID")


@dataclass(frozen=True, slots=True)
class PhelpsServiceWindow:
    """A configured interval concretized for one Detroit service date."""

    service_date: date
    label: str
    starts_at: datetime
    ends_at: datetime
    meal_id: int | None

    def __post_init__(self) -> None:
        _require_service_date(self.service_date)
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("service-window label must be non-empty text")
        _require_aware_datetime(self.starts_at, "starts_at")
        _require_aware_datetime(self.ends_at, "ends_at")
        if self.starts_at.tzinfo != NUTRITION_APPLICATION_TIMEZONE:
            raise ValueError("starts_at must use the Nutrition application timezone")
        if self.ends_at.tzinfo != NUTRITION_APPLICATION_TIMEZONE:
            raise ValueError("ends_at must use the Nutrition application timezone")
        if self.starts_at >= self.ends_at:
            raise ValueError("service-window end must follow start")
        if self.meal_id is not None and canonical_meal_id(self.meal_id) != self.meal_id:
            raise ValueError("meal_id must be a known canonical FD meal ID")

    def contains(self, local_instant: datetime) -> bool:
        """Return whether the interval contains ``local_instant``.

        The closing instant is exclusive, so a published 8:00 p.m. close is
        unavailable at exactly 8:00 p.m.
        """

        local = _normalize_local_instant(local_instant)
        return self.starts_at <= local < self.ends_at


@dataclass(frozen=True, slots=True)
class PhelpsServiceAvailability:
    """Read-only deterministic diagnostic for one Detroit-local instant."""

    local_now: datetime
    service_date: date
    windows: tuple[PhelpsServiceWindow, ...]

    def __post_init__(self) -> None:
        _require_aware_datetime(self.local_now, "local_now")
        if self.local_now.tzinfo != NUTRITION_APPLICATION_TIMEZONE:
            raise ValueError("local_now must use the Nutrition application timezone")
        _require_service_date(self.service_date)
        if self.service_date != self.local_now.date():
            raise ValueError("service_date must match local_now")
        if not isinstance(self.windows, tuple) or not all(
            isinstance(window, PhelpsServiceWindow) for window in self.windows
        ):
            raise TypeError("windows must be PhelpsServiceWindow values")
        if any(window.service_date != self.service_date for window in self.windows):
            raise ValueError("windows must belong to service_date")

    @property
    def open_windows(self) -> tuple[PhelpsServiceWindow, ...]:
        """Return every published interval containing ``local_now``."""

        return tuple(window for window in self.windows if window.contains(self.local_now))

    @property
    def is_open(self) -> bool:
        """Return whether Phelps is open for any configured opportunity."""

        return bool(self.open_windows)


# Source: Hope College Dining Services' published "Standard Semester Dining
# Hours" for Phelps.  The published pages say actual hours may change, so this
# local value is intentionally the one place an operator updates the current
# semester schedule.  Saturday/Sunday use explicit meal IDs only where the
# current FD cache corroborates the FD period representation (brunch -> fd:2).
PHELPS_WEEKLY_SERVICE_WINDOWS: Mapping[int, tuple[PhelpsServiceWindowDefinition, ...]] = (
    MappingProxyType(
        {
            # Monday through Thursday: continuous facility dining only.
            0: (PhelpsServiceWindowDefinition("continuous dining", time(7, 30), time(20)),),
            1: (PhelpsServiceWindowDefinition("continuous dining", time(7, 30), time(20)),),
            2: (PhelpsServiceWindowDefinition("continuous dining", time(7, 30), time(20)),),
            3: (PhelpsServiceWindowDefinition("continuous dining", time(7, 30), time(20)),),
            # Friday: continuous facility dining only.
            4: (PhelpsServiceWindowDefinition("continuous dining", time(7, 30), time(19)),),
            5: (
                PhelpsServiceWindowDefinition("breakfast", time(7, 30), time(9, 30), 1),
                PhelpsServiceWindowDefinition("lunch", time(10, 30), time(13, 30), 2),
                PhelpsServiceWindowDefinition("dinner", time(17), time(19), 3),
            ),
            6: (
                # Current cached Sunday menu occurrences use the FD `lunch`
                # period (fd:2) for Hope's published brunch interval.
                PhelpsServiceWindowDefinition("brunch", time(10, 30), time(13, 30), 2),
                PhelpsServiceWindowDefinition("dinner", time(17), time(19), 3),
            ),
        }
    )
)


class PhelpsServiceCalendar:
    """Resolve replaceable Phelps availability without host-local time.

    ``date_overrides`` is deliberately small: when supplied, an entry replaces
    the normal weekly windows for that exact date.  This supports a future
    closure or special-hours calendar without changing the public API; the
    production default intentionally contains no guessed holiday exceptions.
    """

    def __init__(
        self,
        weekly_windows: Mapping[int, Sequence[PhelpsServiceWindowDefinition]] = PHELPS_WEEKLY_SERVICE_WINDOWS,
        *,
        date_overrides: Mapping[date, Sequence[PhelpsServiceWindowDefinition]] | None = None,
    ) -> None:
        self._weekly_windows = _normalize_weekly_windows(weekly_windows)
        self._date_overrides = _normalize_date_overrides(date_overrides)

    def local_datetime_at(self, instant: datetime) -> datetime:
        """Normalize one absolute instant to an aware America/Detroit value."""

        return _normalize_local_instant(instant)

    def windows_for(self, service_date: date) -> tuple[PhelpsServiceWindow, ...]:
        """Return all configured windows for one explicit service date."""

        _require_service_date(service_date)
        definitions = self._date_overrides.get(
            service_date,
            self._weekly_windows[service_date.weekday()],
        )
        return tuple(
            PhelpsServiceWindow(
                service_date,
                definition.label,
                datetime.combine(
                    service_date,
                    definition.opens_at,
                    tzinfo=NUTRITION_APPLICATION_TIMEZONE,
                ),
                datetime.combine(
                    service_date,
                    definition.closes_at,
                    tzinfo=NUTRITION_APPLICATION_TIMEZONE,
                ),
                definition.meal_id,
            )
            for definition in definitions
        )

    def availability_at(self, instant: datetime) -> PhelpsServiceAvailability:
        """Return a read-only availability snapshot for an absolute instant."""

        local_now = self.local_datetime_at(instant)
        return PhelpsServiceAvailability(
            local_now,
            local_now.date(),
            self.windows_for(local_now.date()),
        )

    def is_open_at(self, instant: datetime) -> bool:
        """Return whether Phelps is open at ``instant``."""

        return self.availability_at(instant).is_open

    def is_meal_context_eligible_at(
        self,
        service_date: date,
        meal: str | int,
        instant: datetime,
    ) -> bool:
        """Return whether a known FD meal context is eligible at ``instant``.

        A generic continuous-dining window is facility availability only.  It
        permits an otherwise-known context while Phelps is open because there
        is no published meal transition to disprove it; it does not assign a
        meal label or invent a transition.  Explicit weekend windows are
        narrower and require the matching canonical FD meal ID.
        """

        _require_service_date(service_date)
        meal_id = canonical_meal_id(meal)
        if meal_id is None:
            return False
        local_now = self.local_datetime_at(instant)
        if local_now.date() != service_date:
            return False
        windows = self.windows_for(service_date)
        specific = tuple(window for window in windows if window.meal_id == meal_id)
        if specific:
            return any(window.contains(local_now) for window in specific)
        generic = tuple(window for window in windows if window.meal_id is None)
        return any(window.contains(local_now) for window in generic)

    def has_plan_become_stale(
        self,
        service_date: date,
        meal: str | int,
        instant: datetime,
    ) -> bool:
        """Return whether a plan's Phelps opportunity has definitely ended.

        A prior Detroit service date is always stale.  On the current date a
        plan retires only after its known meal-specific opportunity ends, or,
        where no meal transition is published, after the final facility window
        ends.  This conservative fallback prevents weekday breakfast/lunch/
        dinner plans from being retired at invented transition times. Durable
        meal state layers the narrow post-delivery reportability grace for
        scheduler-owned plans above this facility/recommendation decision.
        """

        _require_service_date(service_date)
        local_now = self.local_datetime_at(instant)
        current_date = local_now.date()
        if service_date < current_date:
            return True
        if service_date > current_date:
            return False

        windows = self.windows_for(service_date)
        if not windows:
            # A configured date-specific closure has no remaining service
            # opportunity for a current-date active plan.
            return True

        meal_id = canonical_meal_id(meal)
        specific = (
            tuple(window for window in windows if window.meal_id == meal_id)
            if meal_id is not None
            else ()
        )
        applicable = specific or tuple(
            window for window in windows if window.meal_id is None
        )
        # If this context has no explicit or generic mapping, do not claim it
        # expired during a weekend gap; it is definitively stale only after
        # every published facility opportunity for the date has ended.
        if not applicable:
            applicable = windows
        return local_now >= max(window.ends_at for window in applicable)


def _normalize_weekly_windows(
    weekly_windows: Mapping[int, Sequence[PhelpsServiceWindowDefinition]],
) -> Mapping[int, tuple[PhelpsServiceWindowDefinition, ...]]:
    if not isinstance(weekly_windows, Mapping):
        raise TypeError("weekly_windows must be a mapping")
    normalized: dict[int, tuple[PhelpsServiceWindowDefinition, ...]] = {}
    for weekday in range(7):
        definitions = weekly_windows.get(weekday)
        if definitions is None:
            raise ValueError("weekly_windows must define every weekday")
        if isinstance(definitions, (str, bytes)) or not isinstance(definitions, Sequence):
            raise TypeError("weekly service windows must be sequences")
        values = tuple(definitions)
        if not all(isinstance(value, PhelpsServiceWindowDefinition) for value in values):
            raise TypeError("weekly service windows must use PhelpsServiceWindowDefinition")
        normalized[weekday] = values
    if set(weekly_windows) != set(range(7)):
        raise ValueError("weekly_windows keys must be weekdays 0 through 6")
    return MappingProxyType(normalized)


def _normalize_date_overrides(
    date_overrides: Mapping[date, Sequence[PhelpsServiceWindowDefinition]] | None,
) -> Mapping[date, tuple[PhelpsServiceWindowDefinition, ...]]:
    if date_overrides is None:
        return MappingProxyType({})
    if not isinstance(date_overrides, Mapping):
        raise TypeError("date_overrides must be a mapping")
    normalized: dict[date, tuple[PhelpsServiceWindowDefinition, ...]] = {}
    for service_date, definitions in date_overrides.items():
        _require_service_date(service_date)
        if isinstance(definitions, (str, bytes)) or not isinstance(definitions, Sequence):
            raise TypeError("date override windows must be sequences")
        values = tuple(definitions)
        if not all(isinstance(value, PhelpsServiceWindowDefinition) for value in values):
            raise TypeError("date override windows must use PhelpsServiceWindowDefinition")
        normalized[service_date] = values
    return MappingProxyType(normalized)


def _require_service_date(value: object) -> None:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise TypeError("service_date must be a date")


def _require_aware_datetime(value: object, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _normalize_local_instant(instant: datetime) -> datetime:
    _require_aware_datetime(instant, "instant")
    return instant.astimezone(NUTRITION_APPLICATION_TIMEZONE)


DEFAULT_PHELPS_SERVICE_CALENDAR = PhelpsServiceCalendar()
