from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nutrition_optimizer.menu.fetch import MenuFetchError, fetch_menu
from nutrition_optimizer.menu.ingest import ingest_phelps, write_raw_snapshot
from nutrition_optimizer.menu.normalize import (
    MenuNormalizationError,
    PhelpsNotFoundError,
    normalize_phelps,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "menu_payload.json"


def load_fixture() -> list[dict[str, object]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class NormalizePhelpsTests(unittest.TestCase):
    def test_normalizes_phelps_and_preserves_source_order(self) -> None:
        menu = normalize_phelps(load_fixture())

        self.assertEqual(menu.location_name, "phelps")
        self.assertEqual([hour.meal_name for hour in menu.service_hours], ["Lunch"])
        self.assertEqual([block.station_name for block in menu.menu_blocks], ["Homestyle 2", "ZONE 2"])

        first_entries = menu.menu_blocks[0].entries
        self.assertEqual(
            [entry.name for entry in first_entries],
            [
                "Roasted Zucchini & Squash",
                "Chicken Gravy",
                "ALL SANDWICH ITEMS",
                "Green Beans",
                "Corn",
                "Sauce A",
                "Sauce B",
                "CONTAINS: SESAME",
            ],
        )
        self.assertEqual(first_entries[0].dietary_markers, ("V",))
        self.assertEqual(first_entries[1].dietary_markers, ("CG",))
        self.assertEqual(first_entries[1].allergens, ("WHEAT", "MILK"))
        self.assertTrue(first_entries[2].is_notice)
        self.assertEqual(first_entries[2].allergens, ("WHEAT", "MILK"))
        self.assertEqual(first_entries[3].dietary_markers, ("V",))
        self.assertEqual(first_entries[4].dietary_markers, ("V",))
        self.assertEqual(first_entries[5].allergens, ("EGG",))
        self.assertEqual(first_entries[6].allergens, ("EGG",))
        self.assertTrue(first_entries[7].is_notice)
        self.assertEqual(first_entries[7].allergens, ("SESAME",))

        self.assertEqual(
            [entry.name for entry in menu.menu_blocks[1].entries],
            ["Roasted Zucchini & Squash", "Roasted Zucchini & Squash"],
        )
        self.assertTrue(all(block.start.tzinfo is not None for block in menu.menu_blocks))

    def test_missing_phelps_is_explicit(self) -> None:
        with self.assertRaises(PhelpsNotFoundError):
            normalize_phelps([{"name": "Other Hall", "hours": [], "menus": []}])

    def test_reversed_interval_is_rejected(self) -> None:
        payload = load_fixture()
        payload[1]["hours"][0]["end"] = "2026-08-16T11:00:00-04:00"

        with self.assertRaises(MenuNormalizationError):
            normalize_phelps(payload)


class FetchTests(unittest.TestCase):
    def test_fetch_validates_response_and_retains_exact_bytes(self) -> None:
        raw_bytes = b'[{"name":"phelps","hours":[],"menus":[]}]'

        class FakeResponse:
            status_code = 200
            content = raw_bytes

        with patch("nutrition_optimizer.menu.fetch.requests.get", return_value=FakeResponse()) as get:
            result = fetch_menu(url="https://example.test/menu.json", timeout=12)

        self.assertEqual(result.raw_bytes, raw_bytes)
        self.assertEqual(result.payload[0]["name"], "phelps")
        get.assert_called_once_with(
            "https://example.test/menu.json",
            impersonate="chrome",
            headers={"Origin": "https://hope.edu", "Referer": "https://hope.edu/"},
            timeout=12,
        )

    def test_fetch_rejects_non_success_status(self) -> None:
        class FakeResponse:
            status_code = 403
            content = b"{}"

        with patch("nutrition_optimizer.menu.fetch.requests.get", return_value=FakeResponse()):
            with self.assertRaises(MenuFetchError):
                fetch_menu()

    def test_fetch_rejects_non_list_json(self) -> None:
        class FakeResponse:
            status_code = 200
            content = b'{}'

        with patch("nutrition_optimizer.menu.fetch.requests.get", return_value=FakeResponse()):
            with self.assertRaises(MenuFetchError):
                fetch_menu()


class SnapshotTests(unittest.TestCase):
    def test_snapshot_preserves_exact_bytes(self) -> None:
        fetched_at = datetime(2026, 8, 16, 20, 0, 1, 123456, tzinfo=timezone.utc)
        raw_bytes = b'{"raw": true}\n'

        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_raw_snapshot(Path(temp_dir), raw_bytes, fetched_at)

            self.assertEqual(path.name, "hope-menus-20260816T200001.123456Z.json")
            self.assertEqual(path.read_bytes(), raw_bytes)

    @patch("nutrition_optimizer.menu.ingest.fetch_menu")
    def test_ingest_snapshots_before_normalizing(self, fetch_mock) -> None:
        from nutrition_optimizer.menu.models import FetchResult

        raw_bytes = FIXTURE_PATH.read_bytes()
        fetch_mock.return_value = FetchResult(
            url="https://example.test/menu.json",
            fetched_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            raw_bytes=raw_bytes,
            payload=load_fixture(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ingest_phelps(Path(temp_dir))

            self.assertTrue(result.snapshot_path.exists())
            self.assertEqual(result.snapshot_path.read_bytes(), raw_bytes)
            self.assertEqual(result.menu.location_name, "phelps")


if __name__ == "__main__":
    unittest.main()
