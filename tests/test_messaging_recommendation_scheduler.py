"""Offline tests for the production scheduler composition and diagnostics."""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from nutrition_optimizer.application_clock import NutritionApplicationClock
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.messaging.application import CHAT_GUID_ENV_VAR
from nutrition_optimizer.messaging.bluebubbles import PASSWORD_ENV_VAR
from nutrition_optimizer.messaging.recommendation_scheduler import (
    run_production_scheduler_once,
    scheduler_status,
)


class ProductionRecommendationSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "state" / "nutrition.sqlite3"
        with OfficialNutritionCatalog(self.path):
            pass

    @staticmethod
    def clock_at(instant: datetime) -> NutritionApplicationClock:
        return NutritionApplicationClock(lambda: instant)

    def test_scheduler_status_is_read_only_and_never_constructs_a_sender(self) -> None:
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        clock = self.clock_at(datetime(2026, 8, 31, 12, tzinfo=timezone.utc))

        with patch(
            "nutrition_optimizer.messaging.recommendation_scheduler.BlueBubblesClient.from_env"
        ) as client_from_env:
            status = scheduler_status(catalog_path=self.path, clock=clock)

        after = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertEqual(status.local_now.date(), date(2026, 8, 31))
        self.assertFalse(client_from_env.called)

    def test_non_due_production_pass_defers_target_loading_and_delivery(self) -> None:
        environment = {
            CHAT_GUID_ENV_VAR: "chat-guid",
            PASSWORD_ENV_VAR: "runtime-secret",
        }
        clock = self.clock_at(datetime(2026, 8, 31, 11, 25, tzinfo=timezone.utc))
        # 11:25 UTC is 07:25 EDT: before the weekday breakfast activation.
        with (
            patch(
                "nutrition_optimizer.messaging.recommendation_scheduler.BlueBubblesClient.from_env"
            ) as client_from_env,
            patch(
                "nutrition_optimizer.messaging.recommendation_scheduler.load_production_daily_targets"
            ) as target_loader,
        ):
            result = run_production_scheduler_once(
                environ=environment,
                catalog_path=self.path,
                clock=clock,
            )

        self.assertFalse(target_loader.called)
        self.assertTrue(client_from_env.called)
        breakfast = next(item for item in result.decisions if item.opportunity.meal_id == 1)
        self.assertEqual(breakfast.outcome, "not_due")

    def test_scheduler_timer_uses_aligned_wall_clock_cadence(self) -> None:
        timer_path = (
            Path(__file__).resolve().parents[1]
            / "systemd"
            / "nutrition-recommendation-scheduler.timer"
        )
        timer_text = timer_path.read_text(encoding="utf-8")

        self.assertIn("OnCalendar=*-*-* *:00/5:00", timer_text)
        self.assertIn("Persistent=true", timer_text)
        self.assertIn("AccuracySec=1s", timer_text)
        self.assertNotIn("OnUnitActiveSec=", timer_text)
        self.assertNotIn("OnBootSec=", timer_text)

        systemd_analyze = shutil.which("systemd-analyze")
        if systemd_analyze is None:  # pragma: no cover - systemd is present on deployment hosts
            self.skipTest("systemd-analyze is unavailable")
        service_path = timer_path.with_suffix(".service")
        verification = subprocess.run(
            [systemd_analyze, "verify", str(service_path), str(timer_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(verification.returncode, 0, verification.stderr)


if __name__ == "__main__":
    unittest.main()
