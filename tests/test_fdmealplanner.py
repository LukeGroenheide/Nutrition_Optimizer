"""Offline tests for the public FDMealPlanner source adapter."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import json
import unittest

from nutrition_optimizer.fdmealplanner import (
    FDMealPlannerClient,
    FDMealPlannerConfigurationError,
    classify_component,
    content_signature,
    map_component,
    map_meals_payload,
    parse_initial_application_data,
    stable_source_identifier,
)


RETRIEVED_AT = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


def recipe_component(**overrides: object) -> dict[str, object]:
    component: dict[str, object] = {
        "componentId": 69534,
        "componentTypeId": 181,
        "englishAlternateName": "Herb Roasted Chicken",
        "componentName": "Herb Roasted Chicken-Master",
        "descriptionEnglish": "A source description",
        "recipePortionSize": 4.0,
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
        "ingredientStatement": "Chicken, oil, salt, herbs",
        "allergenName": "WHEAT, MILK",
        "recipeProductDietaryName": "",
    }
    component.update(overrides)
    return component


def meals_payload(*components: dict[str, object], menu_date: str = "2026-08-20") -> dict[str, object]:
    return {
        "result": [
            {
                "menuForDate": menu_date,
                "strMenuForDate": menu_date,
                "menuId": 0,
                "mealPeriodId": 2,
                "conceptData": [
                    {"rowId": "station-row", "conceptId": 333, "conceptName": "Grill"}
                ],
                "allMenuRecipes": [
                    {**component, "rowId": "station-row"} for component in components
                ],
            }
        ]
    }


class FakeResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> object:
        return self._payload


class FakeHTTP:
    def __init__(self, initial: dict[str, object], meals: dict[str, object]) -> None:
        self.initial = initial
        self.meals = meals
        self.get_calls: list[dict[str, object]] = []
        self.post_calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.get_calls.append({"url": url, **kwargs})
        if "initial-application-data" in url:
            return FakeResponse(self.initial)
        if "search-location" in url:
            return FakeResponse({"data": {"result": [{"locationId": 10112}]}})
        if url.endswith("/mealPeriods"):
            return FakeResponse({"data": [{"mealPeriodId": 2, "mealPeriodName": "Lunch"}]})
        return FakeResponse(self.meals)

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.post_calls.append({"url": url, **kwargs})
        return FakeResponse({"data": {"accessToken": "anonymous-token"}})


def initial_payload() -> dict[str, object]:
    return {
        "success": True,
        "data": {
            "configuration": [
                {"key": "APP_SEC_PREFIX", "value": "public-rsa-prefix"},
                {
                    "key": "API_CONFIGURATION",
                    "value": json.dumps(
                        [
                            {
                                "tenantId": 7,
                                "tenantName": "CDS",
                                "URL": {
                                    "MENU_PLANNER_DATA_URL": "https://tenant-cds.example/api/v1/data-locator-webapi/{tenantId}/meals",
                                    "MEAL_PERIOD_API_URL": "https://tenant-cds.example/api/v1/data-locator-webapi/{tenantId}/mealPeriods",
                                },
                            }
                        ]
                    ),
                },
            ]
        },
    }


class FDConfigurationTests(unittest.TestCase):
    def test_public_configuration_parses_tenant_routing(self) -> None:
        configuration = parse_initial_application_data(initial_payload())
        self.assertEqual(configuration.tenant_id, 7)
        self.assertEqual(
            configuration.menu_url,
            "https://tenant-cds.example/api/v1/data-locator-webapi/7/meals",
        )
        self.assertEqual(configuration.api_origin, "https://tenant-cds.example")

    def test_missing_tenant_mapping_is_explicit(self) -> None:
        with self.assertRaises(FDMealPlannerConfigurationError):
            parse_initial_application_data(initial_payload(), tenant_id=8)


class FDClientTests(unittest.TestCase):
    def make_client(self, meals: dict[str, object] | None = None) -> tuple[FDMealPlannerClient, FakeHTTP]:
        http = FakeHTTP(initial_payload(), meals or meals_payload(recipe_component()))
        client = FDMealPlannerClient(
            http=http,
            encryptor=lambda prefix, message: "encrypted-public-bootstrap",
            clock=lambda: 1_725_000_000_123,
        )
        return client, http

    def test_bootstrap_and_month_request_are_public_and_bounded(self) -> None:
        client, http = self.make_client()
        payload = client.fetch_meals(
            tenant_id=7,
            account_id=10033,
            location_id=10112,
            meal_period_id=2,
            year=2026,
            month=8,
            start_date=date(2026, 8, 17),
            end_date=date(2026, 8, 31),
        )
        self.assertEqual(len(payload.days), 1)
        self.assertEqual(len(http.post_calls), 1)
        token_body = http.post_calls[0]["json"]
        self.assertEqual(token_body["requestId"], "1725000000123")
        self.assertEqual(token_body["messageJson"], "encrypted-public-bootstrap")
        self.assertNotIn("anonymous-token", repr(token_body))
        api_call = http.get_calls[-1]
        self.assertEqual(api_call["url"], "https://tenant-cds.example/api/v1/data-locator-webapi/7/meals")
        self.assertEqual(
            api_call["params"],
            {
                "menuId": 0,
                "accountId": 10033,
                "locationId": 10112,
                "mealPeriodId": 2,
                "tenantId": 7,
                "monthId": "08",
                "startDate": "2026/08/17",
                "endDate": "2026/08/31",
                "timeOffset": 0,
            },
        )
        self.assertEqual(api_call["headers"]["Authorization"], "Bearer anonymous-token")

    def test_meal_period_route_uses_same_public_token(self) -> None:
        client, http = self.make_client()
        periods = client.fetch_meal_periods()
        self.assertEqual(periods[0]["mealPeriodName"], "Lunch")
        self.assertTrue(http.get_calls[-1]["url"].endswith("/7/mealPeriods"))

    def test_location_directory_accepts_nested_public_result(self) -> None:
        client, http = self.make_client()
        locations = client.search_locations(search_text="Hope")
        self.assertEqual(locations[0]["locationId"], 10112)
        self.assertEqual(http.get_calls[-1]["params"], {"searchText": "Hope"})


class FDMapperTests(unittest.TestCase):
    def test_herb_roasted_chicken_maps_exactly_and_preserves_zero(self) -> None:
        result = map_component(recipe_component(), tenant_id=7, retrieved_at=RETRIEVED_AT)
        self.assertTrue(result.accepted)
        assert result.record is not None
        self.assertEqual(result.record.name, "Herb Roasted Chicken")
        self.assertEqual(result.record.serving.quantity, Decimal("4.0"))
        self.assertEqual(result.record.serving.text, "4 Ounce Cooked Weight")
        self.assertEqual(result.record.nutrients.calories_kcal, Decimal("175"))
        self.assertEqual(result.record.nutrients.protein_g, Decimal("27"))
        self.assertEqual(result.record.nutrients.carbohydrates_g, Decimal("0"))
        self.assertEqual(result.record.nutrients.fat_g, Decimal("4"))
        self.assertEqual(result.record.nutrients.sodium_mg, Decimal("303"))
        self.assertEqual(result.record.ingredients, ("Chicken, oil, salt, herbs",))
        self.assertEqual(result.record.allergens, ("WHEAT", "MILK"))
        self.assertEqual(
            result.record.provenance.identifiers[0],
            stable_source_identifier(tenant_id=7, component_type_id=181, component_id=69534),
        )

    def test_product_type_is_rejected_before_batch_scale_nutrition_can_map(self) -> None:
        result = map_component(
            recipe_component(
                componentId=31740,
                componentTypeId=180,
                englishAlternateName="",
                componentName="Kimchi",
                recipePortionSize=0,
                recipePortionSizeUnit="",
                calories="1201",
                sodium="32429",
            ),
            tenant_id=7,
        )
        self.assertIsNone(result.record)
        self.assertEqual(result.diagnostic.reason, "unsupported_component_type")

    def test_incomplete_and_malformed_records_fail_closed(self) -> None:
        incomplete = map_component(
            recipe_component(recipePortionSize=0, recipePortionSizeUnit=""), tenant_id=7
        )
        self.assertEqual(incomplete.diagnostic.reason, "invalid_serving")
        malformed_nutrient = map_component(recipe_component(protein="not-a-number"), tenant_id=7)
        self.assertEqual(malformed_nutrient.diagnostic.reason, "invalid_nutrient")
        unexpected_uom = map_component(recipe_component(sodiumUOM="g"), tenant_id=7)
        self.assertEqual(unexpected_uom.diagnostic.reason, "unexpected_uom")

    def test_stable_identity_and_content_signature_ignore_occurrence_context(self) -> None:
        first = map_meals_payload(meals_payload(recipe_component()), tenant_id=7)
        second = map_meals_payload(
            meals_payload(recipe_component(), menu_date="2026-08-27"), tenant_id=7
        )
        self.assertEqual(first.records[0].provenance.identifiers, second.records[0].provenance.identifiers)
        self.assertEqual(content_signature(recipe_component()), content_signature(recipe_component(rowId="other")))
        self.assertNotEqual(content_signature(recipe_component()), content_signature(recipe_component(protein="28")))

    def test_repeated_occurrences_retain_context_without_changing_record_identity(self) -> None:
        payload = meals_payload(recipe_component(), recipe_component(), menu_date="2026-08-20")
        batch = map_meals_payload(payload, tenant_id=7)
        self.assertEqual(len(batch.records), 2)
        self.assertEqual(batch.results[0].occurrence.station_name, "Grill")
        self.assertEqual(batch.records[0].provenance.identifiers, batch.records[1].provenance.identifiers)

    def test_null_station_container_maps_complete_recipe_without_inventing_station(self) -> None:
        payload = meals_payload(recipe_component(), menu_date="2026-09-10")
        day = payload["result"][0]
        assert isinstance(day, dict)
        # Sanitized exact live form: conceptData is JSON null even though the
        # recipe array is a complete, structurally usable list.
        day["conceptData"] = None

        batch = map_meals_payload(payload, tenant_id=7, retrieved_at=RETRIEVED_AT)

        self.assertEqual(len(batch.records), 1)
        occurrence = batch.results[0].occurrence
        self.assertIsNotNone(occurrence)
        assert occurrence is not None
        self.assertIsNone(occurrence.station_concept_id)
        self.assertIsNone(occurrence.station_name)
        self.assertEqual(occurrence.menu_date, "2026-09-10")
        self.assertEqual(occurrence.meal_period_id, 2)


if __name__ == "__main__":
    unittest.main()
