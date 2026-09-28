"""Offline tests for current-menu FD food identity resolution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import nutrition_optimizer.food_resolution as food_resolution
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog, map_meals_payload
from nutrition_optimizer.food_resolution import (
    AmbiguousFood,
    FoodResolutionRequest,
    FoodSemanticDecision,
    FoodSemanticGatewayUnavailableError,
    FoodSemanticOutputError,
    FoodSemanticRuntimeUnavailableError,
    FoodSemanticTimeoutError,
    FoodSemanticTransportError,
    LocalFDFoodResolver,
    ResolvedFood,
    UnresolvedFood,
)
from nutrition_optimizer.nutrition import DailyLedger


T1 = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
DAY1 = date(2026, 8, 25)
DAY2 = date(2026, 8, 26)


def component(
    component_id: int,
    name: str,
    *,
    protein: str = "10",
    visible: bool = True,
    component_type_id: int = 181,
    ingredient_statement: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "componentId": component_id,
        "componentTypeId": component_type_id,
        "englishAlternateName": name,
        "componentName": f"{name}-master",
        "recipePortionSize": "1",
        "recipePortionSizeUnit": "Serving",
        "calories": "100",
        "caloriesUOM": "kcal",
        "protein": protein,
        "proteinUOM": "g",
        "carbohydrates": "0",
        "carbohydratesUOM": "g",
        "fat": "2",
        "fatUOM": "g",
        "sodium": "50",
        "sodiumUOM": "mg",
        "isShowOnMenu": "1" if visible else "0",
        "isFoodBar": "0",
    }
    if ingredient_statement is not None:
        value["ingredientStatement"] = ingredient_statement
    return value


def menu_day(
    service_date: date,
    meal_id: int,
    meal_name: str,
    recipes: tuple[tuple[dict[str, object], str], ...],
) -> dict[str, object]:
    concepts: list[dict[str, object]] = []
    all_recipes: list[dict[str, object]] = []
    for index, (recipe, station_name) in enumerate(recipes):
        row_id = f"row-{meal_id}-{index}"
        concepts.append(
            {"rowId": row_id, "conceptId": 40 + index, "conceptName": station_name}
        )
        all_recipes.append({**recipe, "rowId": row_id})
    return {
        "menuForDate": service_date.isoformat(),
        "strMenuForDate": service_date.isoformat(),
        "menuId": meal_id,
        "mealPeriodId": meal_id,
        "mealPeriodName": meal_name,
        "conceptData": concepts,
        "allMenuRecipes": all_recipes,
    }


def mapped(*days: dict[str, object]):
    return map_meals_payload(
        {"result": list(days)},
        tenant_id=7,
        retrieved_at=T1,
    ).results


@dataclass
class FakeSemanticMatcher:
    result: object

    def __post_init__(self) -> None:
        self.calls: list[tuple[FoodResolutionRequest, tuple[object, ...]]] = []

    def decide(self, request: FoodResolutionRequest, candidates: tuple[object, ...]) -> object:
        self.calls.append((request, candidates))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class LocalFDFoodResolverTests(unittest.TestCase):
    def path(self, directory: TemporaryDirectory[str]) -> Path:
        return Path(directory.name) / "state" / "nutrition.sqlite3"

    def open_catalog(self) -> tuple[TemporaryDirectory[str], OfficialNutritionCatalog]:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return directory, OfficialNutritionCatalog(self.path(directory))

    def sync(
        self,
        catalog: OfficialNutritionCatalog,
        results,
        *,
        start: date = DAY1,
        end: date = DAY2,
        observed_at: datetime = T1,
    ) -> None:
        catalog.synchronize_fd_refresh(
            results,
            requested_start=start,
            requested_end=end,
            observed_at=observed_at,
        )

    def test_exact_local_name_resolves_without_semantic_matcher(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(menu_day(DAY1, 3, "Dinner", ((component(1, "Chicken"), "Grill"),))),
                end=DAY1,
            )
            matcher = FakeSemanticMatcher(FoodSemanticDecision("no_match", None, "unused"))
            result = LocalFDFoodResolver(catalog, semantic_matcher=matcher).resolve(
                FoodResolutionRequest("Chicken", DAY1, meal="Dinner")
            )

        self.assertIsInstance(result, ResolvedFood)
        self.assertEqual(result.resolution_method, "exact_name")
        self.assertEqual(result.source_identifier.value, "7:181:1")
        self.assertEqual(matcher.calls, [])

    def test_normalized_local_name_resolves_without_ai(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(
                    menu_day(
                        DAY1,
                        3,
                        "Dinner",
                        ((component(1, "Mac and Cheese*"), "Trattoria"),),
                    )
                ),
                end=DAY1,
            )
            matcher = FakeSemanticMatcher(FoodSemanticDecision("no_match", None, "unused"))
            result = LocalFDFoodResolver(catalog, semantic_matcher=matcher).resolve(
                FoodResolutionRequest("  mac & cheese  ", DAY1)
            )

        self.assertIsInstance(result, ResolvedFood)
        self.assertEqual(result.resolution_method, "normalized_name")
        self.assertEqual(matcher.calls, [])

    def test_verified_zone_prefix_resolves_without_ai(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(
                    menu_day(
                        DAY1,
                        3,
                        "Dinner",
                        ((component(1, "Zone Tandoori Chicken"), "ZONE"),),
                    )
                ),
                end=DAY1,
            )
            result = LocalFDFoodResolver(catalog).resolve(
                FoodResolutionRequest("tandoori chicken", DAY1, meal="dinner", station="zone")
            )

        self.assertIsInstance(result, ResolvedFood)
        self.assertEqual(result.resolution_method, "station_context_name")
        self.assertEqual(result.nutrition_record.name, "Zone Tandoori Chicken")

    def test_candidates_come_only_from_current_requested_date(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(
                    menu_day(DAY1, 3, "Dinner", ((component(1, "Chicken"), "Grill"),)),
                    menu_day(DAY2, 3, "Dinner", ((component(2, "Rice"), "Grill"),)),
                ),
            )
            resolver = LocalFDFoodResolver(catalog)
            candidates = resolver.list_current_candidates(FoodResolutionRequest("anything", DAY1))

        self.assertEqual([candidate.source_identifier.value for candidate in candidates], ["7:181:1"])

    def test_wrong_date_food_is_not_resolved(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(menu_day(DAY2, 3, "Dinner", ((component(2, "Rice"), "Grill"),))),
                start=DAY2,
                end=DAY2,
            )
            result = LocalFDFoodResolver(catalog).resolve(FoodResolutionRequest("Rice", DAY1))

        self.assertIsInstance(result, UnresolvedFood)
        self.assertEqual(result.reason, "no_current_local_candidates")

    def test_meal_constraint_is_enforced_before_matching(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(
                    menu_day(DAY1, 1, "Breakfast", ((component(1, "Eggs"), "Grill"),)),
                    menu_day(DAY1, 3, "Dinner", ((component(2, "Eggs"), "Grill"),)),
                ),
                end=DAY1,
            )
            result = LocalFDFoodResolver(catalog).resolve(
                FoodResolutionRequest("eggs", DAY1, meal="dinner")
            )

        self.assertIsInstance(result, ResolvedFood)
        self.assertEqual(result.source_identifier.value, "7:181:2")
        self.assertEqual(result.occurrence.meal, "Dinner")

    def test_station_constraint_is_enforced_before_matching(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(
                    menu_day(
                        DAY1,
                        3,
                        "Dinner",
                        ((component(1, "Rice"), "Grill"), (component(2, "Rice"), "Wok")),
                    )
                ),
                end=DAY1,
            )
            result = LocalFDFoodResolver(catalog).resolve(
                FoodResolutionRequest("rice", DAY1, station="Wok")
            )

        self.assertIsInstance(result, ResolvedFood)
        self.assertEqual(result.source_identifier.value, "7:181:2")
        self.assertEqual(result.occurrence.station, "Wok")

    def test_duplicate_same_identity_and_snapshot_is_not_nutrition_ambiguous(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(
                    menu_day(DAY1, 1, "Breakfast", ((component(1, "Chicken Soup"), "Deli"),)),
                    menu_day(DAY1, 3, "Dinner", ((component(1, "Chicken Soup"), "Global Bowls"),)),
                ),
                end=DAY1,
            )
            result = LocalFDFoodResolver(catalog).resolve(FoodResolutionRequest("chicken soup", DAY1))

        self.assertIsInstance(result, ResolvedFood)
        self.assertEqual(result.source_identifier.value, "7:181:1")
        self.assertEqual(len(result.equivalent_occurrences), 2)
        self.assertEqual(result.nutrition_record, result.occurrence.nutrition_record)

    def test_different_stable_identities_remain_ambiguous(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(
                    menu_day(
                        DAY1,
                        3,
                        "Dinner",
                        ((component(1, "Chicken"), "Grill"), (component(2, "Chicken"), "Wok")),
                    )
                ),
                end=DAY1,
            )
            matcher = FakeSemanticMatcher(FoodSemanticDecision("match", "7:181:1", "unused"))
            result = LocalFDFoodResolver(catalog, semantic_matcher=matcher).resolve(
                FoodResolutionRequest("chicken", DAY1)
            )

        self.assertIsInstance(result, AmbiguousFood)
        self.assertEqual({candidate.source_identifier.value for candidate in result.candidates}, {"7:181:1", "7:181:2"})
        self.assertEqual(matcher.calls, [])

    def test_semantic_fallback_receives_only_scoped_current_candidates(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(
                    menu_day(DAY1, 1, "Breakfast", ((component(1, "Eggs"), "Grill"),)),
                    menu_day(DAY1, 3, "Dinner", ((component(2, "Chicken Parmesan"), "Trattoria"),)),
                    menu_day(DAY2, 3, "Dinner", ((component(3, "Chicken Parmesan"), "Trattoria"),)),
                ),
            )
            matcher = FakeSemanticMatcher(
                FoodSemanticDecision("match", "7:181:2", "ordinary wording")
            )
            result = LocalFDFoodResolver(catalog, semantic_matcher=matcher).resolve(
                FoodResolutionRequest("parm chicken", DAY1, meal="Dinner", station="Trattoria")
            )

        self.assertIsInstance(result, ResolvedFood)
        self.assertEqual(result.resolution_method, "semantic")
        self.assertEqual(len(matcher.calls), 1)
        _, candidates = matcher.calls[0]
        self.assertEqual([candidate.source_identifier.value for candidate in candidates], ["7:181:2"])
        self.assertEqual(candidates[0].occurrence.service_date, DAY1)
        self.assertEqual(candidates[0].occurrence.meal, "Dinner")
        self.assertEqual(candidates[0].occurrence.station, "Trattoria")

    def test_semantic_match_returns_exact_current_occurrence_record(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            first = mapped(
                menu_day(DAY1, 3, "Dinner", ((component(1, "Chicken", protein="10"), "Grill"),))
            )
            second = mapped(
                menu_day(DAY2, 3, "Dinner", ((component(1, "Chicken", protein="11"), "Grill"),))
            )
            self.sync(catalog, first, end=DAY1, observed_at=T1)
            self.sync(catalog, second, start=DAY2, end=DAY2, observed_at=T2)
            matcher = FakeSemanticMatcher(FoodSemanticDecision("match", "7:181:1", "short name"))
            result = LocalFDFoodResolver(catalog, semantic_matcher=matcher).resolve(
                FoodResolutionRequest("the chicken", DAY1)
            )
            later = catalog.list_current_meal_occurrences(DAY2)[0]

        self.assertIsInstance(result, ResolvedFood)
        self.assertEqual(result.nutrition_record.nutrients.protein_g, Decimal("10"))
        self.assertNotEqual(result.content_signature, later.content_signature)
        self.assertNotEqual(result.nutrition_snapshot_id, later.nutrition_snapshot_id)
        self.assertEqual(later.nutrition_record.nutrients.protein_g, Decimal("11"))

    def test_same_identity_with_two_current_snapshots_remains_ambiguous(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(
                    menu_day(DAY1, 1, "Breakfast", ((component(1, "Chicken", protein="10"), "Grill"),)),
                    menu_day(DAY1, 3, "Dinner", ((component(1, "Chicken", protein="11"), "Grill"),)),
                ),
                end=DAY1,
            )
            matcher = FakeSemanticMatcher(FoodSemanticDecision("match", "7:181:1", "one food"))
            direct = LocalFDFoodResolver(catalog, semantic_matcher=matcher).resolve(
                FoodResolutionRequest("Chicken", DAY1)
            )
            semantic = LocalFDFoodResolver(catalog, semantic_matcher=matcher).resolve(
                FoodResolutionRequest("the chicken", DAY1)
            )

        self.assertIsInstance(direct, AmbiguousFood)
        self.assertEqual(len(direct.candidates), 2)
        self.assertEqual(matcher.calls[0][0].food_text, "the chicken")
        self.assertIsInstance(semantic, AmbiguousFood)
        self.assertEqual(semantic.reason, "semantic_identity_has_multiple_current_snapshots")

    def test_semantic_unknown_identity_fails_closed(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(menu_day(DAY1, 3, "Dinner", ((component(1, "Chicken"), "Grill"),))),
                end=DAY1,
            )
            matcher = FakeSemanticMatcher(FoodSemanticDecision("match", "7:181:999", "bad"))
            result = LocalFDFoodResolver(catalog, semantic_matcher=matcher).resolve(
                FoodResolutionRequest("the protein", DAY1)
            )

        self.assertIsInstance(result, UnresolvedFood)
        self.assertEqual(result.reason, "semantic_candidate_not_admissible")
        self.assertEqual(len(matcher.calls), 1)

    def test_malformed_semantic_output_fails_closed(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(menu_day(DAY1, 3, "Dinner", ((component(1, "Chicken"), "Grill"),))),
                end=DAY1,
            )
            matcher = FakeSemanticMatcher({"decision": "match", "component_identity": "7:181:1"})
            result = LocalFDFoodResolver(catalog, semantic_matcher=matcher).resolve(
                FoodResolutionRequest("the protein", DAY1)
            )

        self.assertIsInstance(result, UnresolvedFood)
        self.assertEqual(result.reason, "semantic_response_invalid")

    def test_semantic_timeout_fails_closed(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(menu_day(DAY1, 3, "Dinner", ((component(1, "Chicken"), "Grill"),))),
                end=DAY1,
            )
            result = LocalFDFoodResolver(
                catalog,
                semantic_matcher=FakeSemanticMatcher(
                    FoodSemanticTimeoutError("test timeout")
                ),
            ).resolve(FoodResolutionRequest("the protein", DAY1))

        self.assertIsInstance(result, UnresolvedFood)
        self.assertEqual(result.reason, "semantic_timeout")

    def test_typed_semantic_runtime_failures_remain_closed_and_distinct(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(menu_day(DAY1, 3, "Dinner", ((component(1, "Chicken"), "Grill"),))),
                end=DAY1,
            )
            cases = (
                (
                    FoodSemanticGatewayUnavailableError("private gateway detail"),
                    "semantic_gateway_unavailable",
                ),
                (FoodSemanticTransportError("private transport detail"), "semantic_transport_failed"),
                (FoodSemanticOutputError("private model detail"), "semantic_output_invalid"),
                (
                    FoodSemanticRuntimeUnavailableError("private runtime detail"),
                    "semantic_runtime_unavailable",
                ),
            )
            for error, reason in cases:
                with self.subTest(reason=reason):
                    matcher = FakeSemanticMatcher(error)
                    result = LocalFDFoodResolver(catalog, semantic_matcher=matcher).resolve(
                        FoodResolutionRequest("the protein", DAY1)
                    )
                    self.assertIsInstance(result, UnresolvedFood)
                    self.assertEqual(result.reason, reason)
                    self.assertEqual(len(matcher.calls), 1)

    def test_semantic_ambiguous_remains_ambiguous(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(
                    menu_day(
                        DAY1,
                        3,
                        "Dinner",
                        ((component(1, "Chicken Parmesan"), "Trattoria"), (component(2, "Chicken Piccata"), "Trattoria")),
                    )
                ),
                end=DAY1,
            )
            matcher = FakeSemanticMatcher(
                FoodSemanticDecision("ambiguous", None, "two chicken dishes")
            )
            result = LocalFDFoodResolver(catalog, semantic_matcher=matcher).resolve(
                FoodResolutionRequest("the chicken", DAY1)
            )

        self.assertIsInstance(result, AmbiguousFood)
        self.assertEqual(result.reason, "semantic_ambiguous")
        self.assertEqual(len(matcher.calls), 1)

    def test_semantic_no_match_remains_unresolved(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(menu_day(DAY1, 3, "Dinner", ((component(1, "Chicken"), "Grill"),))),
                end=DAY1,
            )
            matcher = FakeSemanticMatcher(
                FoodSemanticDecision("no_match", None, "not on menu")
            )
            result = LocalFDFoodResolver(catalog, semantic_matcher=matcher).resolve(
                FoodResolutionRequest("sushi", DAY1)
            )

        self.assertIsInstance(result, UnresolvedFood)
        self.assertEqual(result.reason, "semantic_no_match")
        self.assertEqual(len(matcher.calls), 1)

    def test_removed_food_is_not_in_current_candidate_universe(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            initial = mapped(
                menu_day(
                    DAY1,
                    3,
                    "Dinner",
                    ((component(1, "Chicken"), "Grill"), (component(2, "Rice"), "Grill")),
                )
            )
            later = mapped(menu_day(DAY1, 3, "Dinner", ((component(1, "Chicken"), "Grill"),)))
            self.sync(catalog, initial, end=DAY1, observed_at=T1)
            self.sync(catalog, later, end=DAY1, observed_at=T2)
            resolver = LocalFDFoodResolver(catalog)
            candidates = resolver.list_current_candidates(FoodResolutionRequest("rice", DAY1))
            result = resolver.resolve(FoodResolutionRequest("rice", DAY1))

        self.assertEqual([candidate.official_display_name for candidate in candidates], ["Chicken"])
        self.assertIsInstance(result, UnresolvedFood)

    def test_hidden_and_product_components_cannot_be_candidates(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(
                    menu_day(
                        DAY1,
                        3,
                        "Dinner",
                        (
                            (component(1, "Chicken"), "Grill"),
                            (component(2, "Hidden", visible=False), "Grill"),
                            (component(3, "Product", component_type_id=180), "Grill"),
                        ),
                    )
                ),
                end=DAY1,
            )
            candidates = LocalFDFoodResolver(catalog).list_current_candidates(
                FoodResolutionRequest("anything", DAY1)
            )

        self.assertEqual([candidate.official_display_name for candidate in candidates], ["Chicken"])

    def test_source_objects_are_not_mutated(self) -> None:
        source = component(1, "Chicken")
        before = deepcopy(source)
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(menu_day(DAY1, 3, "Dinner", ((source, "Grill"),))),
                end=DAY1,
            )
            LocalFDFoodResolver(catalog).resolve(FoodResolutionRequest("Chicken", DAY1))

        self.assertEqual(source, before)

    def test_food_resolution_has_no_network_diningbucket_ledger_or_messaging_dependency(self) -> None:
        source = inspect.getsource(food_resolution)

        self.assertNotIn("FDMealPlannerClient", source)
        self.assertNotIn("DiningBucket", source)
        self.assertNotIn("OpenClaw", source)
        self.assertNotIn("OpenAI", source)
        self.assertNotIn("RecordIntakeCommand", source)
        self.assertNotIn("BlueBubbles", source)
        self.assertNotIn("DailyLedger", source)
        self.assertEqual(DailyLedger().entries, ())

    def test_resolved_food_has_no_serving_count_and_can_pair_with_future_portion(self) -> None:
        _, catalog = self.open_catalog()
        with catalog:
            self.sync(
                catalog,
                mapped(menu_day(DAY1, 3, "Dinner", ((component(1, "Chicken"), "Grill"),))),
                end=DAY1,
            )
            result = LocalFDFoodResolver(catalog).resolve(FoodResolutionRequest("chicken", DAY1))

        self.assertIsInstance(result, ResolvedFood)
        self.assertFalse(hasattr(result, "servings"))
        self.assertFalse(hasattr(FoodResolutionRequest, "servings"))
        # A later portion interpreter can attach its own result without changing
        # the immutable food identity or replacing its official serving record.
        future_pair = (result, {"original_quantity_text": "a bowl", "estimated_official_servings": "1.6"})
        self.assertEqual(future_pair[0].nutrition_record.serving.text, "1 Serving")


if __name__ == "__main__":
    unittest.main()
