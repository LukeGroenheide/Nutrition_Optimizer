"""Audit physical quantities in real local-menu recommendations.

This is a read-only demonstration with deliberately synthetic targets. It
never persists a plan or intake and never invokes OpenClaw. Run from the
repository root:

    .venv/bin/python benchmarks/physical_recommendation_audit.py
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
import argparse

from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.meal_optimizer import LocalMealOptimizer
from nutrition_optimizer.nutrition import DailyLedger, DailyTargets
from nutrition_optimizer.recommendation_rendering import (
    RecommendedPortionRenderRequest,
    RecommendedPortionRenderer,
)


DEFAULT_DATABASE = "data/state/nutrition.sqlite3"
DEFAULT_SERVICE_DATE = date(2026, 8, 29)
MEALS = ("breakfast", "lunch", "dinner")


def main() -> int:
    arguments = _arguments()
    catalog = OfficialNutritionCatalog(arguments.database)
    try:
        optimizer = LocalMealOptimizer(catalog)
        renderer = RecommendedPortionRenderer()
        targets = DailyTargets(
            Decimal("2000"),
            Decimal("150"),
            Decimal("250"),
            Decimal("80"),
        )
        print(f"service_date: {arguments.service_date.isoformat()}")
        print("targets: synthetic test-only values; no production targets selected")
        for meal in MEALS:
            recommendation = optimizer.optimize_meal(
                arguments.service_date,
                meal,
                targets,
                DailyLedger(),
            )
            print(
                f"{meal}: outcome={recommendation.diagnostics.outcome} "
                f"items={len(recommendation.items)}"
            )
            for item in recommendation.items:
                quantity = item.physical_quantity
                rendering = renderer.render(
                    RecommendedPortionRenderRequest(
                        official_food_display_name=item.record.name,
                        physical_quantity=quantity,
                        station_name=item.occurrence.station_name,
                    )
                )
                if rendering.physical_quantity != quantity:
                    raise AssertionError("rendering changed the optimizer physical quantity")
                print(
                    f"  {item.record.name} | official={_serving_text(item.record.serving)} | "
                    f"physical={quantity.amount} {quantity.unit} | "
                    f"official_multiplier={item.official_servings} | "
                    f"instruction={rendering.natural_quantity_text}"
                )
        return 0
    finally:
        catalog.close()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument(
        "--date",
        dest="service_date",
        type=date.fromisoformat,
        default=DEFAULT_SERVICE_DATE,
    )
    return parser.parse_args()


def _serving_text(serving) -> str:
    if serving.text is not None:
        return serving.text
    if serving.quantity is not None and serving.unit is not None:
        return f"{serving.quantity} {serving.unit}"
    return serving.unit or "<unspecified>"


if __name__ == "__main__":
    raise SystemExit(main())
