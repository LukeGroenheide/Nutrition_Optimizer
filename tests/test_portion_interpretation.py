"""Offline tests for separate natural-portion interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import nutrition_optimizer.portion_interpretation as portion_interpretation
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import FoodResolutionRequest, LocalFDFoodResolver, ResolvedFood
from nutrition_optimizer.nutrition import DailyLedger
from nutrition_optimizer.portion_interpretation import (
    AmbiguousPortion,
    InterpretedPortion,
    NaturalPortionInterpreter,
    PortionInterpretationRequest,
    PortionSemanticDecision,
    PortionSemanticGatewayUnavailableError,
    PortionSemanticOutputError,
    PortionSemanticRuntimeUnavailableError,
    PortionSemanticTimeoutError,
    PortionSemanticTransportError,
    UnresolvedPortion,
    parse_official_serving_multiplier,
)
from tests.test_food_resolution import DAY1, T1, component, mapped, menu_day


def make_resolved_food(
    *,
    name: str = "Test Food",
    serving_quantity: str = "1",
    serving_unit: str = "Serving",
) -> ResolvedFood:
    """Build an immutable exact occurrence record without a network call."""

    with TemporaryDirectory() as directory:
        catalog = OfficialNutritionCatalog(Path(directory) / "state" / "nutrition.sqlite3")
        try:
            recipe = component(1, name)
            recipe["recipePortionSize"] = serving_quantity
            recipe["recipePortionSizeUnit"] = serving_unit
            catalog.synchronize_fd_refresh(
                mapped(menu_day(DAY1, 3, "Dinner", ((recipe, "Station"),))),
                requested_start=DAY1,
                requested_end=DAY1,
                observed_at=T1,
            )
            result = LocalFDFoodResolver(catalog).resolve(
                FoodResolutionRequest(name, DAY1, meal="Dinner", station="Station")
            )
            assert isinstance(result, ResolvedFood)
            return result
        finally:
            catalog.close()


@dataclass
class FakePortionMatcher:
    result: object

    def __post_init__(self) -> None:
        self.calls: list[PortionInterpretationRequest] = []

    def decide(self, request: PortionInterpretationRequest) -> object:
        self.calls.append(request)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class NaturalPortionInterpreterTests(unittest.TestCase):
    def request(
        self,
        quantity: str,
        *,
        name: str = "Test Food",
        serving_quantity: str = "1",
        serving_unit: str = "Serving",
    ) -> PortionInterpretationRequest:
        return PortionInterpretationRequest(
            resolved_food=make_resolved_food(
                name=name,
                serving_quantity=serving_quantity,
                serving_unit=serving_unit,
            ),
            original_quantity_text=quantity,
        )

    def test_explicit_numeric_servings_are_exact_and_decimal(self) -> None:
        result = NaturalPortionInterpreter().interpret(self.request("2 servings"))

        self.assertEqual(
            result,
            InterpretedPortion(
                original_quantity_text="2 servings",
                estimated_official_servings=Decimal("2"),
                confidence="high",
                interpretation_method="explicit_official_servings",
                reason="explicit_official_servings",
            ),
        )

    def test_explicit_decimal_servings_are_exact_and_decimal(self) -> None:
        result = NaturalPortionInterpreter().interpret(self.request("1.5 servings"))

        self.assertIsInstance(result, InterpretedPortion)
        self.assertEqual(result.estimated_official_servings, Decimal("1.5"))
        self.assertNotIsInstance(result.estimated_official_servings, float)
        self.assertEqual(result.interpretation_method, "explicit_official_servings")

    def test_half_serving_is_exact(self) -> None:
        result = NaturalPortionInterpreter().interpret(self.request("half a serving"))

        self.assertIsInstance(result, InterpretedPortion)
        self.assertEqual(result.estimated_official_servings, Decimal("0.5"))
        self.assertEqual(result.confidence, "high")

    def test_safe_count_mapping_requires_one_each_serving(self) -> None:
        each_result = NaturalPortionInterpreter().interpret(
            self.request("two pieces", name="Hamburger Patty", serving_unit="Each")
        )
        half_result = NaturalPortionInterpreter().interpret(
            self.request("half of one", name="Hamburger Patty", serving_unit="Each")
        )
        weighted_result = NaturalPortionInterpreter().interpret(
            self.request("two pieces", name="Roast Chicken", serving_quantity="4", serving_unit="Ounce")
        )

        self.assertIsInstance(each_result, InterpretedPortion)
        self.assertEqual(each_result.estimated_official_servings, Decimal("2"))
        self.assertEqual(each_result.interpretation_method, "count_based_each")
        self.assertIsInstance(half_result, InterpretedPortion)
        self.assertEqual(half_result.estimated_official_servings, Decimal("0.5"))
        self.assertIsInstance(weighted_result, UnresolvedPortion)
        self.assertEqual(weighted_result.reason, "portion_semantic_matcher_unavailable")

    def test_explicit_ounce_matches_only_the_foods_exact_official_ounce_serving(self) -> None:
        result = NaturalPortionInterpreter().interpret(
            self.request(
                "6 oz of the pork",
                name="Herbed Pork Loin",
                serving_quantity="6",
                serving_unit="Ounce",
            )
        )
        scoop = NaturalPortionInterpreter().interpret(
            self.request(
                "one scoop",
                name="Herbed Pork Loin",
                serving_quantity="6",
                serving_unit="Ounce",
            )
        )

        self.assertIsInstance(result, InterpretedPortion)
        self.assertEqual(result.estimated_official_servings, Decimal("1"))
        self.assertEqual(result.reason, "explicit_matching_official_ounce_unit")
        self.assertIsInstance(scoop, UnresolvedPortion)
        self.assertEqual(scoop.reason, "portion_semantic_matcher_unavailable")

    def test_explicit_invalid_quantities_fail_closed_without_semantic_fallback(self) -> None:
        for quantity in ("0 servings", "13 servings"):
            with self.subTest(quantity=quantity):
                matcher = FakePortionMatcher(
                    PortionSemanticDecision("estimate", Decimal("1"), "high", "unused")
                )
                result = NaturalPortionInterpreter(semantic_matcher=matcher).interpret(
                    self.request(quantity)
                )

                self.assertIsInstance(result, UnresolvedPortion)
                self.assertEqual(result.reason, "invalid_explicit_official_servings")
                self.assertEqual(matcher.calls, [])

    def test_semantic_estimate_keeps_resolved_food_immutable(self) -> None:
        request = self.request("a bowl", name="Cereal", serving_unit="Cup")
        original_record = request.resolved_food.nutrition_record
        original_identity = request.resolved_food.source_identifier
        matcher = FakePortionMatcher(
            PortionSemanticDecision("estimate", Decimal("1.5"), "medium", "bowl_volume")
        )

        result = NaturalPortionInterpreter(semantic_matcher=matcher).interpret(request)

        self.assertIsInstance(result, InterpretedPortion)
        self.assertEqual(result.estimated_official_servings, Decimal("1.5"))
        self.assertEqual(result.confidence, "medium")
        self.assertEqual(result.interpretation_method, "semantic")
        self.assertEqual(request.resolved_food.nutrition_record, original_record)
        self.assertEqual(request.resolved_food.source_identifier, original_identity)
        self.assertEqual(matcher.calls, [request])

    def test_semantic_ambiguous_and_no_estimate_stay_non_numeric(self) -> None:
        request = self.request("some", name="Mashed Potatoes", serving_unit="Cup")
        ambiguous = NaturalPortionInterpreter(
            semantic_matcher=FakePortionMatcher(
                PortionSemanticDecision("ambiguous", None, None, "physical_amount_unclear")
            )
        ).interpret(request)
        unresolved = NaturalPortionInterpreter(
            semantic_matcher=FakePortionMatcher(
                PortionSemanticDecision("no_estimate", None, None, "insufficient_context")
            )
        ).interpret(request)

        self.assertIsInstance(ambiguous, AmbiguousPortion)
        self.assertEqual(ambiguous.reason, "portion_semantic_ambiguous")
        self.assertIsInstance(unresolved, UnresolvedPortion)
        self.assertEqual(unresolved.reason, "portion_semantic_no_estimate")

    def test_typed_semantic_failures_remain_closed_and_distinct(self) -> None:
        request = self.request("a scoop", name="Rice", serving_unit="Cup")
        cases = (
            (PortionSemanticTimeoutError("private timeout"), "portion_semantic_timeout"),
            (
                PortionSemanticGatewayUnavailableError("private gateway"),
                "portion_semantic_gateway_unavailable",
            ),
            (PortionSemanticTransportError("private transport"), "portion_semantic_transport_failed"),
            (PortionSemanticOutputError("private model"), "portion_semantic_output_invalid"),
            (
                PortionSemanticRuntimeUnavailableError("private runtime"),
                "portion_semantic_runtime_unavailable",
            ),
        )
        for error, reason in cases:
            with self.subTest(reason=reason):
                matcher = FakePortionMatcher(error)
                result = NaturalPortionInterpreter(semantic_matcher=matcher).interpret(request)
                self.assertIsInstance(result, UnresolvedPortion)
                self.assertEqual(result.reason, reason)
                self.assertEqual(len(matcher.calls), 1)

    def test_decimal_parser_rejects_nonfinite_negative_and_absurd_values(self) -> None:
        for value in ("0", "-1", "NaN", "Infinity", "1e3", "13", "", "1.2.3"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_official_serving_multiplier(value)

    def test_module_has_no_catalog_network_ledger_intake_or_messaging_dependency(self) -> None:
        source = Path(portion_interpretation.__file__).read_text(encoding="utf-8")

        self.assertNotIn("OfficialNutritionCatalog", source)
        self.assertNotIn("FDMealPlannerClient", source)
        self.assertNotIn("DiningBucket", source)
        self.assertNotIn("DailyLedger", source)
        self.assertNotIn("RecordIntakeCommand", source)
        self.assertNotIn("BlueBubbles", source)
        self.assertNotIn("OpenClaw", source)
        self.assertEqual(DailyLedger().entries, ())


if __name__ == "__main__":
    unittest.main()
