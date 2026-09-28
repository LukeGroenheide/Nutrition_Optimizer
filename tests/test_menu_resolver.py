"""Offline tests for deterministic DiningBucket to FD occurrence matching."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
import unittest

from nutrition_optimizer.fdmealplanner import (
    FDMealOccurrence,
    map_component,
)
from nutrition_optimizer.menu import (
    AmbiguousFDMatch,
    DiningBucketOccurrence,
    NoFDMatch,
    ResolvedFDMatch,
    resolve_occurrence,
    resolve_station_scope,
)


RETRIEVED_AT = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
FRIDAY = date(2026, 8, 21)


def recipe_component(**overrides: object) -> dict[str, object]:
    component: dict[str, object] = {
        "componentId": 69534,
        "componentTypeId": 181,
        "englishAlternateName": "Herb Roasted Chicken",
        "componentName": "Herb Roasted Chicken-Master",
        "recipePortionSize": "4",
        "recipePortionSizeUnit": "Ounce Cooked Weight",
        "calories": "175",
        "caloriesUOM": "kcal",
        "protein": "27",
        "proteinUOM": "g",
        "carbohydrates": "0",
        "carbohydratesUOM": "g",
        "fat": "4",
        "fatUOM": "g",
        "sodium": "303",
        "sodiumUOM": "mg",
        "isShowOnMenu": "1",
        "isFoodBar": "0",
    }
    component.update(overrides)
    return component


def fd_result(
    name: str,
    *,
    component_id: int = 69534,
    service_date: date = FRIDAY,
    meal_period_id: int | str = 2,
    station_concept_id: int | str = 48,
    station_name: str = "AMERICAN GRILLE",
    **overrides: object,
):
    component = recipe_component(
        englishAlternateName=name,
        componentName=name,
        componentId=component_id,
        **overrides,
    )
    occurrence = FDMealOccurrence(
        component=component,
        menu_date=service_date.isoformat(),
        meal_period_id=meal_period_id,
        station_concept_id=station_concept_id,
        station_name=station_name,
    )
    return map_component(
        component,
        tenant_id=7,
        retrieved_at=RETRIEVED_AT,
        occurrence=occurrence,
    )


def source(
    name: str,
    *,
    service_date: date = FRIDAY,
    meal: str = "lunch",
    station_name: str = "Grille 1",
) -> DiningBucketOccurrence:
    return DiningBucketOccurrence(
        service_date=service_date,
        meal=meal,
        station_name=station_name,
        name=name,
    )


class StationRuleTests(unittest.TestCase):
    def test_static_station_aliases_resolve_to_verified_concepts(self) -> None:
        self.assertEqual(
            resolve_station_scope("Global 1", service_date=FRIDAY, meal="lunch").concept_id,
            203,
        )
        self.assertEqual(
            resolve_station_scope("Trattoria 2", service_date=FRIDAY, meal="lunch").concept_id,
            114,
        )
        self.assertEqual(
            resolve_station_scope("ZONE 2", service_date=FRIDAY, meal="lunch").concept_id,
            121,
        )
        self.assertEqual(
            resolve_station_scope("Plant Forward", service_date=FRIDAY, meal="lunch").concept_id,
            156,
        )

    def test_comfort_corner_weekday_lunch_maps_to_american_grille(self) -> None:
        scope = resolve_station_scope("Comfort Corner", service_date=FRIDAY, meal="lunch")
        self.assertIsNotNone(scope)
        self.assertEqual(scope.key, "american grille")
        self.assertEqual(scope.concept_id, 48)

    def test_comfort_corner_weekend_lunch_maps_to_homestyle(self) -> None:
        scope = resolve_station_scope("Comfort Corner", service_date=date(2026, 8, 22), meal="lunch")
        self.assertIsNotNone(scope)
        self.assertEqual(scope.key, "homestyle")
        self.assertEqual(scope.concept_id, 81)

    def test_comfort_corner_dinner_maps_to_homestyle(self) -> None:
        scope = resolve_station_scope("Comfort Corner", service_date=FRIDAY, meal="dinner")
        self.assertIsNotNone(scope)
        self.assertEqual(scope.key, "homestyle")

    def test_comfort_corner_breakfast_is_unresolved(self) -> None:
        result = resolve_occurrence(
            source("Eggs", meal="breakfast", station_name="Comfort Corner"),
            (),
        )
        self.assertIsInstance(result, NoFDMatch)
        self.assertEqual(result.reason, "unknown_station_context")

    def test_known_station_without_candidates_is_explicitly_empty(self) -> None:
        result = resolve_occurrence(source("Eggs", station_name="American Grille"), ())
        self.assertIsInstance(result, NoFDMatch)
        self.assertEqual(result.reason, "known_context_but_fd_empty")
        self.assertEqual(result.station_scope.concept_id, 48)

    def test_unknown_station_does_not_broaden_search(self) -> None:
        result = resolve_occurrence(source("Eggs", station_name="Za'atar"), ())
        self.assertIsInstance(result, NoFDMatch)
        self.assertEqual(result.reason, "unknown_station_context")


class DeterministicMatchingTests(unittest.TestCase):
    def test_exact_same_context_resolves_to_stable_component_identity(self) -> None:
        candidate = fd_result("General Tso's Chicken", component_id=72525)
        result = resolve_occurrence(source("General Tso's Chicken"), (candidate,))
        self.assertIsInstance(result, ResolvedFDMatch)
        self.assertEqual(result.rule, "exact_display_name")
        self.assertEqual(result.source_identifier.kind, "component")
        self.assertEqual(result.source_identifier.value, "7:181:72525")

    def test_case_and_whitespace_normalization_resolves(self) -> None:
        candidate = fd_result("General Tso's Chicken")
        result = resolve_occurrence(source("  GENERAL tso's   chicken  "), (candidate,))
        self.assertIsInstance(result, ResolvedFDMatch)
        self.assertEqual(result.rule, "normalized_display_name")

    def test_punctuation_normalization_resolves(self) -> None:
        candidate = fd_result("Chicken-Noodle Soup")
        result = resolve_occurrence(source("Chicken Noodle Soup"), (candidate,))
        self.assertIsInstance(result, ResolvedFDMatch)
        self.assertEqual(result.rule, "normalized_display_name")

    def test_trailing_menu_marker_is_ignored(self) -> None:
        candidate = fd_result("General Tso's Chicken")
        result = resolve_occurrence(source("General Tso's Chicken**"), (candidate,))
        self.assertIsInstance(result, ResolvedFDMatch)
        self.assertEqual(result.rule, "normalized_display_name")

    def test_ampersand_and_are_conservative_lexical_normalization(self) -> None:
        candidate = fd_result("Chickpea and Tomato Salad with Basil")
        result = resolve_occurrence(source("Chickpea & Tomato Salad with Basil"), (candidate,))
        self.assertIsInstance(result, ResolvedFDMatch)
        self.assertEqual(result.rule, "conservative_lexical_normalization")

    def test_zone_context_prefix_is_allowed_only_in_zone(self) -> None:
        candidate = fd_result(
            "Zone Sweet and Sour Chicken",
            component_id=70371,
            station_concept_id=121,
            station_name="ZONE",
        )
        result = resolve_occurrence(
            source("Sweet & Sour Chicken", station_name="the Zone"),
            (candidate,),
        )
        self.assertIsInstance(result, ResolvedFDMatch)
        self.assertEqual(result.source_identifier.value, "7:181:70371")

    def test_same_name_on_another_date_does_not_resolve(self) -> None:
        candidate = fd_result("General Tso's Chicken", service_date=date(2026, 8, 22))
        result = resolve_occurrence(source("General Tso's Chicken"), (candidate,))
        self.assertIsInstance(result, NoFDMatch)
        self.assertNotEqual(result.reason, "no_name_match")

    def test_same_name_in_another_meal_does_not_resolve(self) -> None:
        candidate = fd_result("General Tso's Chicken", meal_period_id=3)
        result = resolve_occurrence(source("General Tso's Chicken"), (candidate,))
        self.assertIsInstance(result, NoFDMatch)

    def test_same_name_at_another_station_does_not_resolve(self) -> None:
        candidate = fd_result("General Tso's Chicken", station_concept_id=81, station_name="HOMESTYLE")
        result = resolve_occurrence(source("General Tso's Chicken"), (candidate,))
        self.assertIsInstance(result, NoFDMatch)

    def test_occurrence_context_does_not_change_stable_source_identity(self) -> None:
        friday_candidate = fd_result("Green Beans", component_id=54215)
        saturday_candidate = fd_result(
            "Green Beans",
            component_id=54215,
            service_date=date(2026, 8, 22),
            station_concept_id=81,
            station_name="HOMESTYLE",
        )
        friday = resolve_occurrence(source("Green Beans"), (friday_candidate,))
        saturday = resolve_occurrence(
            source(
                "Green Beans",
                service_date=date(2026, 8, 22),
                station_name="Homestyle",
            ),
            (saturday_candidate,),
        )
        self.assertIsInstance(friday, ResolvedFDMatch)
        self.assertIsInstance(saturday, ResolvedFDMatch)
        self.assertEqual(friday.source_identifier, saturday.source_identifier)

    def test_hidden_product_and_invalid_records_are_excluded(self) -> None:
        hidden = fd_result("General Tso's Chicken", isShowOnMenu="0")
        product = fd_result(
            "General Tso's Chicken",
            component_id=18001,
            componentTypeId=180,
            recipePortionSize="0",
            recipePortionSizeUnit="",
        )
        incomplete = fd_result(
            "General Tso's Chicken",
            component_id=18002,
            recipePortionSize="0",
        )
        result = resolve_occurrence(source("General Tso's Chicken"), (hidden, product, incomplete))
        self.assertIsInstance(result, NoFDMatch)
        self.assertEqual(result.reason, "known_context_but_fd_empty")

    def test_food_bar_uses_meal_planner_visibility_flag(self) -> None:
        candidate = fd_result(
            "Food Bar Item",
            isFoodBar="1",
            isShowOnMenu="0",
            isShowOnMealPlanner="1",
        )
        result = resolve_occurrence(source("Food Bar Item"), (candidate,))
        self.assertIsInstance(result, ResolvedFDMatch)

    def test_two_component_ids_at_one_rule_are_ambiguous(self) -> None:
        first = fd_result("Rice", component_id=10001)
        second = fd_result("Rice", component_id=10002)
        result = resolve_occurrence(source("Rice"), (first, second))
        self.assertIsInstance(result, AmbiguousFDMatch)
        self.assertEqual(
            {identifier.value for identifier in result.source_identifiers},
            {"7:181:10001", "7:181:10002"},
        )

    def test_eligible_candidates_with_no_matching_name_are_no_name_match(self) -> None:
        candidate = fd_result("Different Food")
        result = resolve_occurrence(source("Rice"), (candidate,))
        self.assertIsInstance(result, NoFDMatch)
        self.assertEqual(result.reason, "no_name_match")
        self.assertEqual(result.eligible_candidate_count, 1)

    def test_unsafe_roasted_and_grilled_names_do_not_match(self) -> None:
        candidate = fd_result("Grilled Chicken")
        result = resolve_occurrence(source("Roasted Chicken"), (candidate,))
        self.assertIsInstance(result, NoFDMatch)
        self.assertEqual(result.reason, "no_name_match")

    def test_arbitrary_context_prefix_is_not_stripped(self) -> None:
        candidate = fd_result(
            "Taqueria Black Bean Corn Salsa",
            station_concept_id=203,
            station_name="Global Bowls",
        )
        result = resolve_occurrence(
            source("Black Bean Corn Salsa", station_name="Global 1"),
            (candidate,),
        )
        self.assertIsInstance(result, NoFDMatch)

    def test_composite_dish_is_not_decomposed(self) -> None:
        candidate = fd_result("Biscuits and Gravy")
        result = resolve_occurrence(source("Sausage Gravy"), (candidate,))
        self.assertIsInstance(result, NoFDMatch)

    def test_source_and_candidate_objects_are_not_mutated(self) -> None:
        original_source = source("General Tso's Chicken")
        candidate = fd_result("General Tso's Chicken")
        original_component = deepcopy(candidate.occurrence.component)
        result = resolve_occurrence(original_source, (candidate,))
        self.assertIsInstance(result, ResolvedFDMatch)
        self.assertEqual(original_source.name, "General Tso's Chicken")
        self.assertEqual(candidate.occurrence.component, original_component)

    def test_resolver_has_no_ai_or_private_service_dependency(self) -> None:
        import nutrition_optimizer.menu.resolver as resolver_module

        self.assertNotIn("openai", resolver_module.__dict__)
        self.assertNotIn("openclaw", resolver_module.__dict__)


if __name__ == "__main__":
    unittest.main()
