"""Production composition and one-shot CLI for automatic recommendations.

Systemd invokes ``run-scheduler-once`` periodically; it does not encode meal
times or retain scheduling state.  Detroit-local timing, Phelps eligibility,
and durable dispatch idempotency all stay in
``nutrition_optimizer.recommendation_scheduler``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys

from ..application import MealRecommendationOrchestrator, load_production_daily_targets
from ..application_clock import (
    DEFAULT_NUTRITION_APPLICATION_CLOCK,
    NutritionApplicationClock,
)
from ..durable_state import DurableMealState
from ..fdmealplanner import DEFAULT_CATALOG_PATH, OfficialNutritionCatalog
from ..openclaw_recommendation_rendering import OpenClawRecommendedPortionRenderer
from ..phelps_service_calendar import DEFAULT_PHELPS_SERVICE_CALENDAR
from ..recommendation_rendering import RecommendedPortionRenderer
from ..recommendation_scheduler import (
    RecommendationScheduler,
    RecommendationSchedulerRunResult,
    RecommendationSchedulerStatus,
)
from .application import ApplicationMessagingConfig, CHAT_GUID_ENV_VAR
from .bluebubbles import BlueBubblesAPIError, BlueBubblesClient


__all__ = [
    "main",
    "run_production_scheduler_once",
    "scheduler_status",
]


def run_production_scheduler_once(
    *,
    environ: Mapping[str, str] | None = None,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    clock: NutritionApplicationClock | None = None,
) -> RecommendationSchedulerRunResult:
    """Compose and execute one mutable production scheduler evaluation.

    Runtime configuration is read only when a dispatch actually needs target
    arithmetic; the target loader itself remains lazy inside the scheduler.
    This function is intentionally the only production path that creates an
    outbound BlueBubbles client for automatic recommendations.
    """

    config = ApplicationMessagingConfig.from_env(environ)
    chat_guid = config.expected_chat_guid
    if chat_guid is None:
        raise ValueError(f"{CHAT_GUID_ENV_VAR} must be set for outbound delivery")
    client = BlueBubblesClient.from_env(environ)
    application_clock = clock or DEFAULT_NUTRITION_APPLICATION_CLOCK
    with OfficialNutritionCatalog(catalog_path) as catalog:
        # This is the same presentation composition used by the existing
        # manual delivery command.  The OpenClaw renderer never controls food
        # identity, quantities, nutrition, lifecycle, or dispatch state.
        renderer = RecommendedPortionRenderer(OpenClawRecommendedPortionRenderer.from_env(environ))
        scheduler = RecommendationScheduler(
            DurableMealState(catalog),
            clock=application_clock,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
            recommendation_preparer=MealRecommendationOrchestrator(
                catalog,
                renderer,
                service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
                clock=application_clock,
            ),
            target_loader=lambda: load_production_daily_targets(environ),
            outbound_sender=client,
            chat_guid=chat_guid,
            # A non-2xx response is a known failed hand-off and can retry the
            # same immutable plan.  A network interruption is deliberately
            # left as ``sending`` because the remote outcome is unknowable;
            # automatic retry would risk a second iMessage.
            is_known_delivery_failure=lambda exc: isinstance(exc, BlueBubblesAPIError),
        )
        return scheduler.run_once()


def scheduler_status(
    *,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    clock: NutritionApplicationClock | None = None,
) -> RecommendationSchedulerStatus:
    """Read scheduler status without a migration, lifecycle write, or sender."""

    application_clock = clock or DEFAULT_NUTRITION_APPLICATION_CLOCK
    # ``read_only=True`` refuses an old schema rather than silently performing
    # the v8 migration during an operator's diagnostic command.
    with OfficialNutritionCatalog(catalog_path, read_only=True) as catalog:
        return RecommendationScheduler(
            DurableMealState(catalog),
            clock=application_clock,
            service_calendar=DEFAULT_PHELPS_SERVICE_CALENDAR,
        ).status()


def main(argv: list[str] | None = None) -> int:
    """Run one automatic dispatch pass or inspect it without side effects."""

    parser = argparse.ArgumentParser(
        description="Run or inspect Nutrition Optimizer automatic recommendation scheduling"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    status_parser = subparsers.add_parser(
        "scheduler-status",
        help="inspect Detroit-local opportunities and durable dispatch state without writes",
    )
    status_parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)
    run_parser = subparsers.add_parser(
        "run-scheduler-once",
        help="evaluate due opportunities once; timer-safe across retries and restarts",
    )
    run_parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)
    args = parser.parse_args(argv)
    try:
        if args.command == "scheduler-status":
            _print_status(scheduler_status(catalog_path=args.catalog_path))
            return 0
        result = run_production_scheduler_once(catalog_path=args.catalog_path)
        _print_run_result(result)
        return 1 if result.has_operational_failure else 0
    except (RuntimeError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 2  # pragma: no cover - argparse.error always raises SystemExit.


def _print_status(status: RecommendationSchedulerStatus) -> None:
    print(f"Detroit now: {status.local_now.isoformat()}", flush=True)
    for item in status.items:
        dispatch = item.dispatch_status or "not-created"
        active = item.active_plan_id or "none"
        print(
            f"fd:{item.opportunity.meal_id} {item.opportunity.label}: "
            f"{item.timing}; Phelps eligible={'yes' if item.phelps_eligible else 'no'}; "
            f"reportable={'yes' if item.reportable else 'no'}; "
            f"dispatch={dispatch}; active_plan={active}",
            flush=True,
        )


def _print_run_result(result: RecommendationSchedulerRunResult) -> None:
    print(f"Detroit now: {result.local_now.isoformat()}", flush=True)
    print(
        f"Stale active plans retired: {len(result.stale_retirement.retired_plan_ids)}",
        flush=True,
    )
    for decision in result.decisions:
        plan = f"; plan_id={decision.plan_id}" if decision.plan_id is not None else ""
        dispatch = (
            f"; dispatch={decision.dispatch_status}"
            if decision.dispatch_status is not None
            else ""
        )
        reason = (
            f"; reason={decision.diagnostic_reason}"
            if decision.diagnostic_reason is not None
            else ""
        )
        print(
            f"fd:{decision.opportunity.meal_id} {decision.opportunity.label}: "
            f"{decision.outcome}{reason}{plan}{dispatch}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
