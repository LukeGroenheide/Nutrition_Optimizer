from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import inspect
import unittest

from nutrition_optimizer.fdmealplanner.catalog import FDMenuOccurrence
from nutrition_optimizer.meal_optimizer import (
    LocalMealOptimizer,
    MealOptimizationPolicy,
    RecommendedMealItem,
    optimize_meal,
)
from nutrition_optimizer.meal_request import RequestedMealFood
from nutrition_optimizer.physical_quantity import (
    PhysicalQuantityError,
    official_servings_for_physical_count,
    physical_quantity_for,
)
from nutrition_optimizer.nutrition import (
    DailyLedger,
    DailyMinimums,
    DailyTargets,
    IntakeEntry,
    NutritionProvenance,
    NutritionRecord,
    NutrientProfile,
    Serving,
    SourceIdentifier,
    add_nutrients,
    calculate_daily_balance,
)


DAY = date(2026, 8, 25)


def targets() -> DailyTargets:
    return DailyTargets(Decimal("1000"), Decimal("100"), Decimal("100"), Decimal("50"))


def record(
    name: str,
    calories: str = "100",
    protein: str = "20",
    carbohydrates: str = "10",
    fat: str = "5",
    *,
    fiber: str | None = None,
    serving_quantity: Decimal | None = Decimal("1"),
    serving_unit: str = "Cup",
) -> NutritionRecord:
    return NutritionRecord(
        name=name,
        # The optimizer now excludes unknown provider units.  Use a known
        # continuous unit for this generic scoring fixture so these tests keep
        # exercising the existing objective and grid behavior.
        serving=Serving(
            quantity=serving_quantity,
            unit=serving_unit,
            text=f"{serving_quantity} {serving_unit}" if serving_quantity is not None else None,
        ),
        nutrients=NutrientProfile(
            calories_kcal=Decimal(calories) if calories is not None else None,
            protein_g=Decimal(protein) if protein is not None else None,
            carbohydrates_g=Decimal(carbohydrates) if carbohydrates is not None else None,
            fat_g=Decimal(fat) if fat is not None else None,
            sodium_mg=Decimal("100"),
            dietary_fiber_g=Decimal(fiber) if fiber is not None else None,
        ),
        provenance=NutritionProvenance("test", datetime(2026, 8, 1, tzinfo=timezone.utc)),
    )


def occurrence(
    identifier: int,
    food: NutritionRecord,
    *,
    service_date: date = DAY,
    meal: str = "lunch",
    source_value: str | None = None,
    snapshot_id: int | None = None,
) -> FDMenuOccurrence:
    return FDMenuOccurrence(
        occurrence_id=identifier,
        occurrence_key=f"occurrence-{identifier}",
        service_date=service_date,
        meal_period_id=meal,
        meal_period_name=meal,
        station_concept_id="station",
        station_name="Station",
        source_identifier=SourceIdentifier("recipe", source_value or str(identifier)),
        content_signature=f"signature-{identifier}",
        nutrition_snapshot_id=snapshot_id or identifier,
        nutrition_record=food,
        menu_detail_id=str(identifier),
        menu_id="menu",
        first_observed_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        last_observed_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )


def ledger(*entries: tuple[NutritionRecord, str]) -> DailyLedger:
    return DailyLedger(tuple(IntakeEntry(food, Decimal(servings)) for food, servings in entries))


def fixed_policy(**overrides: object) -> MealOptimizationPolicy:
    values: dict[str, object] = {
        "minimum_servings": Decimal("0.25"),
        "maximum_servings_per_food": Decimal("2"),
        "serving_increment": Decimal("0.25"),
        "max_distinct_foods": 2,
        "beam_width": 100,
    }
    values.update(overrides)
    return MealOptimizationPolicy(**values)  # type: ignore[arg-type]


class FakeCatalog:
    def __init__(self, occurrences: tuple[FDMenuOccurrence, ...]) -> None:
        self.occurrences = occurrences
        self.calls: list[tuple[date, str | int | None]] = []

    def list_current_meal_occurrences(
        self, service_date: date, meal: str | int | None = None
    ) -> tuple[FDMenuOccurrence, ...]:
        self.calls.append((service_date, meal))
        return self.occurrences


class MealOptimizerTests(unittest.TestCase):
    def test_required_foods_leave_room_for_complements_with_a_narrow_beam(self) -> None:
        complement = occurrence(1, record("Chicken", "100", "20", "0", "0"))
        pizza = occurrence(2, record("Pizza", "100", "0", "10", "5"))
        required = RequestedMealFood("pizza", "recipe", "2", "signature-2")
        result = optimize_meal(
            DAY, "lunch", targets(), DailyLedger(), (complement, pizza),
            required_foods=(required,),
            policy=fixed_policy(
                minimum_servings=Decimal("1"), maximum_servings_per_food=Decimal("1"),
                serving_increment=Decimal("1"), beam_width=1,
            ),
        )
        self.assertEqual({item.record.name for item in result.items}, {"Chicken", "Pizza"})

    def test_multiple_required_foods_cannot_be_pruned_by_beam_score(self) -> None:
        first = occurrence(1, record("Pizza", "10", "1", "1", "1"))
        second = occurrence(2, record("Chicken", "100", "20", "10", "5"))
        required = tuple(
            RequestedMealFood(item.nutrition_record.name, "recipe", str(item.occurrence_id), item.content_signature)
            for item in (first, second)
        )
        result = optimize_meal(
            DAY, "lunch", targets(), DailyLedger(), (first, second),
            required_foods=required,
            policy=fixed_policy(beam_width=1),
        )
        self.assertTrue(result.is_recommendation)
        self.assertEqual({item.record.name for item in result.items}, {"Chicken", "Pizza"})

    def test_required_food_is_included_even_when_confirmed_intake_exceeds_targets(self) -> None:
        pizza = occurrence(1, record("Pizza"))
        consumed = ledger((record("Confirmed intake", "2000", "200", "200", "100"), "1"))
        result = optimize_meal(
            DAY, "lunch", targets(), consumed, (pizza,),
            required_foods=(RequestedMealFood("pizza", "recipe", "1", "signature-1"),),
            policy=fixed_policy(),
        )
        self.assertTrue(result.is_recommendation)
        self.assertGreater(result.objective_score, result.diagnostics.baseline_objective_score)
        self.assertEqual(
            result.projected_daily_total,
            add_nutrients(consumed.total_consumed_nutrients, result.projected_meal_nutrition),
        )
        self.assertEqual(result.projected_balance, calculate_daily_balance(targets(), result.projected_daily_total))

    def test_required_foods_exceeding_item_bounds_fail_instead_of_being_dropped(self) -> None:
        menu = tuple(occurrence(n, record(f"Food {n}")) for n in range(1, 4))
        result = optimize_meal(
            DAY, "lunch", targets(), DailyLedger(), menu, policy=fixed_policy(max_distinct_foods=2),
            required_foods=tuple(RequestedMealFood(
                item.nutrition_record.name, "recipe", str(item.occurrence_id), item.content_signature,
            ) for item in menu),
        )
        self.assertFalse(result.is_recommendation)
        self.assertEqual(result.diagnostics.outcome, "required_food_infeasible")

    def test_default_policy_uses_benchmark_beam_width(self) -> None:
        self.assertEqual(MealOptimizationPolicy().beam_width, 1000)

    def test_explicit_beam_width_is_honored_and_wider_search_recovers_fiber(self) -> None:
        targets_with_fiber = DailyTargets(
            Decimal("100"), Decimal("100"), Decimal("100"), Decimal("50"),
            DailyMinimums(Decimal("3")),
        )
        complement = occurrence(
            1,
            record(
                "Fiber complement",
                calories="30",
                protein="30",
                carbohydrates="0",
                fat="0",
                fiber="3",
            ),
            meal="dinner",
        )
        near_miss = tuple(
            occurrence(
                identifier,
                record(
                    f"Fiber near miss {identifier}",
                    calories="70",
                    protein="70",
                    carbohydrates="0",
                    fat="0",
                    fiber="2.75",
                ),
                meal="dinner",
            )
            for identifier in range(2, 252)
        )
        menu = (complement, *near_miss)

        def policy(beam_width: int) -> MealOptimizationPolicy:
            return MealOptimizationPolicy(
                minimum_servings=Decimal("1"),
                maximum_servings_per_food=Decimal("1"),
                serving_increment=Decimal("1"),
                max_distinct_foods=2,
                beam_width=beam_width,
                dinner_stage_fraction=Decimal("1"),
            )

        narrow = optimize_meal(
            DAY, "dinner", targets_with_fiber, DailyLedger(), menu,
            policy=policy(250),
        )
        wide = optimize_meal(
            DAY, "dinner", targets_with_fiber, DailyLedger(), menu,
            policy=policy(1000),
        )

        self.assertEqual(MealOptimizationPolicy(beam_width=17).beam_width, 17)
        self.assertEqual(len(narrow.items), 1)
        self.assertEqual(narrow.items[0].record.name, "Fiber near miss 2")
        self.assertEqual(
            narrow.projected_balance.minimums.dietary_fiber_deficit_g,
            Decimal("0.25"),
        )
        self.assertEqual(
            [item.occurrence.occurrence_id for item in wide.items],
            [1, 2],
        )
        self.assertEqual(wide.projected_balance.minimums.dietary_fiber_deficit_g, Decimal("0"))
        self.assertLess(wide.objective_score, narrow.objective_score)

    def test_equal_scores_use_stable_occurrence_id_tie_breaking(self) -> None:
        food = record("Tied food")
        first = occurrence(1, food, meal="dinner")
        second = occurrence(2, food, meal="dinner")
        policy = MealOptimizationPolicy(
            minimum_servings=Decimal("1"),
            maximum_servings_per_food=Decimal("1"),
            serving_increment=Decimal("1"),
            max_distinct_foods=1,
            beam_width=1,
            dinner_stage_fraction=Decimal("1"),
        )

        result = optimize_meal(
            DAY, "dinner", targets(), DailyLedger(), (second, first), policy=policy
        )

        self.assertEqual(result.items[0].occurrence.occurrence_id, 1)

    def test_fd_meal_ids_and_names_share_stage_fractions(self) -> None:
        policy = fixed_policy(
            breakfast_stage_fraction=Decimal("0.35"),
            lunch_stage_fraction=Decimal("0.60"),
            dinner_stage_fraction=Decimal("1"),
            other_meal_stage_fraction=Decimal("0.20"),
        )

        for expected, values in (
            (Decimal("0.35"), (1, "1", "breakfast", " Breakfast ")),
            (Decimal("0.60"), (2, "2", "lunch")),
            (Decimal("1"), (3, "3", "dinner")),
        ):
            for meal in values:
                with self.subTest(meal=meal):
                    self.assertEqual(policy.stage_fraction_for(meal), expected)

        self.assertEqual(policy.stage_fraction_for("snack"), Decimal("0.20"))
        self.assertEqual(policy.stage_fraction_for(4), Decimal("0.20"))

    def test_three_each_grid_uses_whole_physical_choices_and_decimal_ratios(self) -> None:
        serving = Serving(Decimal("3"), "Each", "3 Each")
        policy = fixed_policy()

        multipliers = policy.serving_multipliers_for(serving)

        self.assertEqual(
            multipliers,
            (
                Decimal("0.333333333333333333333333333333333333"),
                Decimal("0.666666666666666666666666666666666667"),
                Decimal("1"),
                Decimal("1.33333333333333333333333333333333333"),
                Decimal("1.66666666666666666666666666666666667"),
                Decimal("2"),
            ),
        )
        self.assertEqual(
            [physical_quantity_for(serving, value).amount for value in multipliers],
            [Decimal(index) for index in range(1, 7)],
        )

    def test_one_each_slice_and_sticks_never_generate_fractional_physical_items(self) -> None:
        policy = fixed_policy()
        definitions = (
            Serving(Decimal("1"), "Each", "1 Each"),
            Serving(Decimal("1"), "Slice", "1 Slice"),
            Serving(Decimal("2"), "Sticks", "2 Sticks"),
        )

        for serving in definitions:
            with self.subTest(unit=serving.unit):
                multipliers = policy.serving_multipliers_for(serving)
                self.assertTrue(multipliers)
                self.assertTrue(
                    all(
                        physical_quantity_for(serving, value).amount
                        == physical_quantity_for(serving, value).amount.to_integral_value()
                        for value in multipliers
                    )
                )

        self.assertEqual(
            policy.serving_multipliers_for(definitions[0]),
            (Decimal("1"), Decimal("2")),
        )
        self.assertEqual(
            [physical_quantity_for(definitions[2], value).amount for value in policy.serving_multipliers_for(definitions[2])],
            [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4")],
        )

    def test_unknown_provider_unit_is_conservatively_excluded(self) -> None:
        unknown = occurrence(
            99,
            record("Unknown unit", serving_quantity=Decimal("1"), serving_unit="Serving"),
        )

        result = optimize_meal(DAY, "lunch", targets(), DailyLedger(), (unknown,))

        self.assertFalse(result.is_recommendation)
        self.assertEqual(result.diagnostics.eligible_candidates, 0)
        self.assertEqual(result.diagnostics.excluded_invalid_serving, 1)
        self.assertEqual(MealOptimizationPolicy().serving_multipliers_for(unknown.nutrition_record.serving), ())

    def test_derived_count_multiplier_drives_existing_nutrition_arithmetic(self) -> None:
        food = record(
            "Chicken Tenders",
            calories="300",
            protein="30",
            carbohydrates="15",
            fat="9",
            serving_quantity=Decimal("3"),
            serving_unit="Each",
        )
        ratio = official_servings_for_physical_count(food.serving, Decimal("2"))
        item = RecommendedMealItem(occurrence(100, food), food, ratio)

        self.assertEqual(item.physical_quantity.amount, Decimal("2"))
        self.assertEqual(item.nutrition_contribution.calories_kcal, Decimal("300") * ratio)
        self.assertEqual(item.nutrition_contribution.protein_g, Decimal("30") * ratio)
        self.assertEqual(item.nutrition_contribution.carbohydrates_g, Decimal("15") * ratio)
        self.assertEqual(item.nutrition_contribution.fat_g, Decimal("9") * ratio)

        with self.assertRaises(PhysicalQuantityError):
            RecommendedMealItem(occurrence(101, food), food, Decimal("0.75"))

    def test_count_bounds_are_translated_without_exceeding_physical_maximum(self) -> None:
        serving = Serving(Decimal("3"), "Each", "3 Each")
        policy = fixed_policy(
            minimum_servings=Decimal("0.25"),
            maximum_servings_per_food=Decimal("0.50"),
        )

        self.assertEqual(
            policy.serving_multipliers_for(serving),
            (Decimal("0.333333333333333333333333333333333333"),),
        )

    def test_uses_only_catalog_current_meal_query_and_filters_wrong_context(self) -> None:
        valid = occurrence(1, record("Valid"))
        wrong_date = occurrence(2, record("Wrong date"), service_date=date(2026, 8, 26))
        wrong_meal = occurrence(3, record("Wrong meal"), meal="dinner")
        catalog = FakeCatalog((valid, wrong_date, wrong_meal))

        result = LocalMealOptimizer(catalog).optimize_meal(DAY, "lunch", targets(), DailyLedger())

        self.assertEqual(catalog.calls, [(DAY, "lunch")])
        self.assertEqual(result.diagnostics.candidate_occurrences, 3)
        self.assertEqual(result.diagnostics.excluded_wrong_context, 2)
        self.assertTrue(all(item.occurrence == valid for item in result.items))

    def test_missing_required_nutrition_and_invalid_serving_are_excluded(self) -> None:
        missing = occurrence(1, record("Missing", protein=None))
        invalid_serving = occurrence(2, record("No numeric serving", serving_quantity=None))

        result = optimize_meal(DAY, "lunch", targets(), DailyLedger(), (missing, invalid_serving))

        self.assertFalse(result.is_recommendation)
        self.assertEqual(result.diagnostics.outcome, "no_eligible_candidates")
        self.assertEqual(result.diagnostics.excluded_missing_required_nutrition, 1)
        self.assertEqual(result.diagnostics.excluded_invalid_serving, 1)

    def test_decimal_servings_granularity_and_maximum_are_preserved(self) -> None:
        food = occurrence(1, record("Protein", "50", "10", "0", "0"), meal="dinner")
        policy = fixed_policy(maximum_servings_per_food=Decimal("0.75"), max_distinct_foods=1)

        result = optimize_meal(DAY, "dinner", targets(), DailyLedger(), (food,), policy=policy)

        self.assertEqual(result.items[0].official_servings, Decimal("0.75"))
        self.assertLessEqual(result.items[0].official_servings, policy.maximum_servings_per_food)
        self.assertEqual(result.items[0].official_servings % policy.serving_increment, Decimal("0"))

    def test_projected_nutrition_total_and_balance_use_existing_arithmetic(self) -> None:
        eaten = record("Breakfast", "200", "10", "20", "10")
        candidate = occurrence(1, record("Lunch", "100", "20", "10", "5"), meal="dinner")
        current = ledger((eaten, "1"))
        policy = fixed_policy(maximum_servings_per_food=Decimal("1"), max_distinct_foods=1)

        result = optimize_meal(DAY, "dinner", targets(), current, (candidate,), policy=policy)
        expected_meal = result.items[0].nutrition_contribution
        expected_total = add_nutrients(current.total_consumed_nutrients, expected_meal)

        self.assertEqual(result.projected_meal_nutrition, expected_meal)
        self.assertEqual(result.projected_daily_total, expected_total)
        self.assertEqual(result.projected_balance, calculate_daily_balance(targets(), expected_total))

    def test_actual_intake_changes_lunch_quantity_after_skipped_breakfast_food(self) -> None:
        protein_food = occurrence(1, record("Protein", "100", "20", "0", "0"))
        breakfast_adequate = record("Actual breakfast", "300", "80", "100", "50")
        breakfast_under_eaten = record("Actual breakfast", "300", "0", "100", "50")
        policy = fixed_policy(max_distinct_foods=1)

        adequate = optimize_meal(
            DAY, "lunch", targets(), ledger((breakfast_adequate, "1")), (protein_food,), policy=policy
        )
        under_eaten = optimize_meal(
            DAY, "lunch", targets(), ledger((breakfast_under_eaten, "1")), (protein_food,), policy=policy
        )

        self.assertLess(adequate.items[0].official_servings, under_eaten.items[0].official_servings)
        self.assertEqual(adequate.items[0].official_servings, Decimal("0.50"))
        self.assertEqual(under_eaten.items[0].official_servings, Decimal("2.00"))

    def test_dinner_is_more_aggressive_than_breakfast_under_same_deficit(self) -> None:
        food = occurrence(1, record("Balanced", "200", "20", "20", "10"), meal="breakfast")
        breakfast = optimize_meal(DAY, "breakfast", targets(), DailyLedger(), (food,))
        dinner_food = occurrence(1, food.nutrition_record, meal="dinner")
        dinner = optimize_meal(DAY, "dinner", targets(), DailyLedger(), (dinner_food,))

        self.assertLess(breakfast.items[0].official_servings, dinner.items[0].official_servings)

    def test_overshoot_conflict_prefers_protein_dense_food_when_calories_are_nearly_met(self) -> None:
        existing = record("Already eaten", "950", "0", "100", "50")
        dense = occurrence(1, record("Protein dense", "20", "40", "0", "0"), meal="dinner")
        calorie_heavy = occurrence(2, record("Protein heavy calories", "500", "40", "0", "0"), meal="dinner")
        policy = fixed_policy(minimum_servings=Decimal("1"), maximum_servings_per_food=Decimal("1"), serving_increment=Decimal("1"), max_distinct_foods=1)

        result = optimize_meal(DAY, "dinner", targets(), ledger((existing, "1")), (dense, calorie_heavy), policy=policy)

        self.assertEqual(result.items[0].record.name, "Protein dense")

    def test_duplicate_identical_occurrences_are_collapsed(self) -> None:
        food = record("Same food")
        first = occurrence(1, food, source_value="same", snapshot_id=9)
        duplicate = occurrence(2, food, source_value="same", snapshot_id=9)

        result = optimize_meal(DAY, "lunch", targets(), DailyLedger(), (first, duplicate))

        self.assertEqual(result.diagnostics.duplicate_occurrences_collapsed, 1)
        self.assertLessEqual(len(result.items), 1)
        self.assertEqual(result.items[0].occurrence.occurrence_id, 1)

    def test_max_distinct_count_empty_menu_and_already_satisfied_are_explicit(self) -> None:
        foods = tuple(occurrence(index, record(f"Food {index}")) for index in range(1, 5))
        limited = optimize_meal(
            DAY, "lunch", targets(), DailyLedger(), foods,
            policy=fixed_policy(max_distinct_foods=1),
        )
        empty = optimize_meal(DAY, "lunch", targets(), DailyLedger(), ())
        met = optimize_meal(
            DAY, "lunch", targets(),
            ledger((record("Met", "1000", "100", "100", "50"), "1")), foods,
        )

        self.assertLessEqual(len(limited.items), 1)
        self.assertEqual(empty.diagnostics.outcome, "empty_menu")
        self.assertEqual(met.diagnostics.outcome, "targets_satisfied_or_exceeded")

    def test_materially_worsening_options_return_no_recommendation(self) -> None:
        nearly_met = record("Nearly met", "999", "99", "99", "49")
        excessive = occurrence(1, record("Excessive", "100", "20", "20", "10"), meal="dinner")

        result = optimize_meal(
            DAY, "dinner", targets(), ledger((nearly_met, "1")), (excessive,)
        )

        self.assertFalse(result.is_recommendation)
        self.assertEqual(result.diagnostics.outcome, "no_improving_recommendation")

    def test_identical_input_is_deterministic_and_has_no_ai_network_or_db_dependencies(self) -> None:
        foods = (occurrence(1, record("A")), occurrence(2, record("B", "125", "10", "15", "2")))
        first = optimize_meal(DAY, "lunch", targets(), DailyLedger(), foods)
        second = optimize_meal(DAY, "lunch", targets(), DailyLedger(), foods)
        source = inspect.getsource(__import__("nutrition_optimizer.meal_optimizer", fromlist=["*"]))

        self.assertEqual(first, second)
        for forbidden in ("OpenAI", "DiningBucket", "FDMealPlannerClient", "sqlite3", "requests"):
            self.assertNotIn(forbidden, source)

    def test_unknown_existing_balance_returns_no_recommendation_without_zero_filling(self) -> None:
        unknown = record("Unknown protein", "100", None, "10", "5")
        candidate = occurrence(1, record("Candidate"))

        result = optimize_meal(DAY, "lunch", targets(), ledger((unknown, "1")), (candidate,))

        self.assertFalse(result.is_recommendation)
        self.assertEqual(result.diagnostics.outcome, "unknown_current_balance")


if __name__ == "__main__":
    unittest.main()
