"""Orchestration and command-line entry point for Phelps menu ingestion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .fetch import MENU_URL, fetch_menu
from .models import IngestResult
from .normalize import normalize_phelps


def ingest_phelps(
    output_dir: Path = Path("data/raw"),
    *,
    url: str = MENU_URL,
    timeout: float = 30.0,
) -> IngestResult:
    """Fetch Hope's menu, snapshot it, and normalize the Phelps portion."""

    fetched = fetch_menu(url=url, timeout=timeout)
    snapshot_path = write_raw_snapshot(output_dir, fetched.raw_bytes, fetched.fetched_at)
    menu = normalize_phelps(fetched.payload)
    return IngestResult(
        snapshot_path=snapshot_path,
        fetched_at=fetched.fetched_at,
        menu=menu,
    )


def write_raw_snapshot(output_dir: Path, raw_bytes: bytes, fetched_at: datetime) -> Path:
    """Write the exact response body to a timestamped JSON snapshot."""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = fetched_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    snapshot_path = output_dir / f"hope-menus-{timestamp}.json"
    snapshot_path.write_bytes(raw_bytes)
    return snapshot_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and normalize the Phelps menu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="directory for the unmodified raw JSON snapshot",
    )
    args = parser.parse_args(argv)

    result = ingest_phelps(output_dir=args.output_dir)
    entry_count = sum(len(block.entries) for block in result.menu.menu_blocks)
    print(f"snapshot: {result.snapshot_path}")
    print(f"service hours: {len(result.menu.service_hours)}")
    print(f"menu blocks: {len(result.menu.menu_blocks)}")
    print(f"entries: {entry_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
