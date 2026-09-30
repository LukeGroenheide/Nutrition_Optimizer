"""Phase 2A authority checks using isolated catalogs and fixed semantic output."""
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nutrition_optimizer.durable_state import DurableMealState
from nutrition_optimizer.presentation_binding import freeze_presentation, PresentationKind
from nutrition_optimizer.serving_presentation import DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog
from nutrition_optimizer.food_resolution import FoodResolutionRequest, LocalFDFoodResolver
from nutrition_optimizer.meal_report import MealPlan, PlannedMealItem, MealReportReconciler
from nutrition_optimizer.meal_report_conversation import MealReportConversationOrchestrator
from nutrition_optimizer.portion_interpretation import NaturalPortionInterpreter
from tests.test_meal_report_conversation import (
    DAY, OBSERVED_AT, CHAT_GUID, CurrentPlanResolver, ScriptedInterpreter,
    food, semantic, eaten, skipped,
)
from tests.test_food_resolution import mapped, menu_day


class MealQuantityAuthorityTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'state.sqlite3'
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(lambda: self.catalog.close())
        self.catalog.synchronize_fd_refresh(
            mapped(menu_day(DAY, 2, 'Lunch', (
                (food(75146, 'Corn Spaghetti', serving_quantity='4', serving_unit='Ounce'), 'Pasta'),
                (food(71776, 'Meatball Sub Sandwich', serving_quantity='1', serving_unit='Each'), 'Deli'),
                (food(62889, 'Baked Sweet Potatoes', serving_quantity='1', serving_unit='Each'), 'Homestyle'),
            ))), requested_start=DAY, requested_end=DAY, observed_at=OBSERVED_AT,
        )
        resolver = LocalFDFoodResolver(self.catalog)
        items = []
        for name, amount, display in (
            ('Corn Spaghetti', '1.75', 'about 2 scoops'),
            ('Meatball Sub Sandwich', '1', '1 sandwich'),
            ('Baked Sweet Potatoes', '2', '8 sweet potato quarter pieces'),
        ):
            resolved = resolver.resolve(FoodResolutionRequest(name, DAY, meal=2))
            calibration = DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS.find(resolved.source_identifier, resolved.nutrition_record.serving)
            binding = (freeze_presentation(resolved, Decimal(amount), display,
                PresentationKind.REVERSIBLE_CALIBRATED, calibration) if calibration else None)
            items.append(PlannedMealItem(resolved, Decimal(amount), display, presentation_binding=binding))
        self.plan = MealPlan(DAY, 2, tuple(items))
        self.state = DurableMealState(self.catalog)

    def reconcile(self, text, reports, *, plan=None, reference=None):
        reconciler = MealReportReconciler(
            ScriptedInterpreter({}), LocalFDFoodResolver(self.catalog), NaturalPortionInterpreter(),
        )
        return reconciler.reconcile_semantic(
            plan or self.plan, semantic(planned=reports), text,
            quantity_reference_item_id=reference,
        )

    def assert_no_intake(self):
        self.assertEqual(self.state.get_daily_intake(DAY), ())
        self.assertEqual(self.catalog._connection.execute(
            'SELECT count(*) FROM meal_report_applications'
        ).fetchone()[0], 0)

    def test_explicit_plan_attestations_use_frozen_quantity(self):
        for text in ('I ate what you recommended.',
                     'I ate the amount you told me to for spaghetti.',
                     'I ate the whole amount you recommended for spaghetti.',
                     'I ate all of the spaghetti you told me to eat.'):
            with self.subTest(text=text):
                result = self.reconcile(text, [eaten('item_1', text)])
                self.assertEqual(result.eaten_items[0].official_servings, Decimal('1.75'))
                self.assertEqual(result.eaten_items[0].quantity_source, 'planned_quantity')

    def test_forged_attestation_cannot_override_literal_visual_amount(self):
        for amount in ('a scoop', '1 scoop', 'two ladles', 'one serving spoon',
                       'one palm-sized amount', 'a handful', 'one tennis ball',
                       'one golf-ball-sized amount', 'three pieces', '2 meatballs'):
            with self.subTest(amount=amount):
                result = self.reconcile('I ate ' + amount, [eaten('item_1', 'what you recommended')])
                self.assertEqual(result.eaten_items, ())
                self.assertEqual(result.clarification_items[0].reason, 'plan_attestation_required')
                self.assert_no_intake()

    def test_reference_to_display_does_not_authorize_a_physical_amount(self):
        for text in ('I ate 1 scoop, like you recommended', 'I ate a fistful as recommended',
                     'I ate scoops of what you recommended', 'I ate at most what you recommended'):
            with self.subTest(text=text):
                result = self.reconcile(text, [eaten('item_1', 'recommended')])
                self.assertEqual(result.eaten_items, ())

    def test_bare_food_mention_does_not_attest_to_an_amount(self):
        for text in ('I ate spaghetti', 'I ate recommended spaghetti'):
            with self.subTest(text=text):
                result = self.reconcile(text, [eaten('item_1', 'spaghetti')])
                self.assertEqual(result.eaten_items, ())

    def test_partial_amount_cannot_be_misclassified_as_full_attestation(self):
        for text in ('I ate half of what you recommended', 'I ate 0.5 of what you recommended',
                     'I ate some of what you recommended', 'I did not eat what you recommended'):
            with self.subTest(text=text):
                self.assertEqual(self.reconcile(text, [eaten('item_1', text)]).eaten_items, ())

    def test_relative_fractions_use_exact_selected_plan(self):
        single = MealPlan(DAY, 2, (self.plan.items[0],))
        for phrase, factor in (
            ('half of that', '.5'), ('half of it', '.5'),
            ('half of what you recommended', '.5'), ('quarter of that', '.25'),
            ('quarter of it', '.25'), ('1/4 of that', '.25'), ('0.5 of that', '.5'),
            ('50%', '.5'), ('50 percent', '.5'), ('25%', '.25'), ('25 percent', '.25'),
        ):
            with self.subTest(phrase=phrase):
                result = self.reconcile('I ate ' + phrase, [eaten(
                    'item_1', phrase, relation='fraction_of_recommended', quantity_text=phrase,
                )], plan=single)
                self.assertEqual(result.eaten_items[0].official_servings, Decimal('1.75') * Decimal(factor))

    def test_invalid_relative_quantities_fail_closed(self):
        for phrase in ('-50%', '150%', '50%%', 'NaN%', '1e2%', '0%', '-1/4', '2/1', '50 per cent-ish'):
            with self.subTest(phrase=phrase):
                result = self.reconcile(phrase, [eaten(
                    'item_1', phrase, relation='fraction_of_recommended', quantity_text=phrase,
                )])
                self.assertEqual(result.eaten_items, ())
                self.assert_no_intake()

    def test_half_preserves_all_digits_of_the_frozen_quantity(self):
        item = replace(self.plan.items[0], recommended_official_servings=Decimal(
            '1.000000000000000000000000000000000001',
        ))
        result = self.reconcile('half of that', [eaten(
            'item_1', 'half of that', relation='fraction_of_recommended', quantity_text='half of that',
        )], plan=MealPlan(DAY, 2, (item,)))
        self.assertEqual(result.eaten_items[0].official_servings, Decimal(
            '0.5000000000000000000000000000000000005',
        ))

    def test_model_cannot_strip_invalid_fraction_sign_or_digits(self):
        single = MealPlan(DAY, 2, (self.plan.items[0],))
        for text in ('-50%', '−50%', '150%', 'negative 50%', 'minus 50%'):
            with self.subTest(text=text):
                result = self.reconcile(text, [eaten(
                    'item_1', text, relation='fraction_of_recommended', quantity_text='50%',
                )], plan=single)
                self.assertEqual(result.eaten_items, ())

    def test_ambiguous_relative_reference_is_not_selected_by_model(self):
        for text in ('half of that', '50%'):
            with self.subTest(text=text):
                result = self.reconcile(text, [eaten(
                    'item_1', text, relation='fraction_of_recommended', quantity_text=text,
                )])
                self.assertEqual(result.eaten_items, ())

    def test_physical_amount_cannot_be_laundered_through_plan_fraction(self):
        single = MealPlan(DAY, 2, (self.plan.items[0],))
        for text, fraction in (('1 scoop', '1'), ('half a scoop', 'half'),
                               ('half of a ladle', 'half'), ('0.5 cups', '0.5'),
                               ('50% of a scoop', '50%'), ('25 percent of the ladle', '25 percent')):
            with self.subTest(text=text):
                result = self.reconcile('I ate ' + text, [eaten(
                    'item_1', text, relation='fraction_of_recommended', quantity_text=fraction,
                )], plan=single)
                self.assertEqual(result.eaten_items, ())
                self.assert_no_intake()

    def test_durable_single_question_supplies_relative_referent(self):
        result = self.reconcile('half of that', [eaten(
            'item_1', 'half of that', relation='fraction_of_recommended', quantity_text='half of that',
        )], reference='item_1')
        self.assertEqual(result.eaten_items[0].official_servings, Decimal('.875'))

    def test_item_all_cannot_expand_to_other_items(self):
        text = 'I ate all of the spaghetti'
        result = self.reconcile(text, [eaten(f'item_{i}', text) for i in (1, 2, 3)])
        self.assertEqual([r.plan_item for r in result.eaten_items], [self.plan.items[0]])
        self.assertEqual(result.eaten_items[0].official_servings, Decimal('1.75'))

    def test_whole_meal_all_preserves_each_exact_quantity(self):
        text = 'I ate everything you recommended'
        result = self.reconcile(text, [eaten(f'item_{i}', text) for i in (1, 2, 3)])
        self.assertEqual([r.official_servings for r in result.eaten_items], [Decimal('1.75'), Decimal('1'), Decimal('2')])

    def test_unqualified_some_fails_closed(self):
        result = self.reconcile('I ate some of it', [eaten('item_1', 'some of it')])
        self.assertEqual(result.eaten_items, ())
        self.assertTrue(result.clarification_items)

    def test_unambiguous_single_item_all(self):
        single = MealPlan(DAY, 2, (self.plan.items[0],))
        result = self.reconcile('I ate all of it', [eaten('item_1', 'all of it')], plan=single)
        self.assertEqual(result.eaten_items[0].official_servings, Decimal('1.75'))

    def test_none_is_a_skip_not_zero_intake(self):
        text = 'I ate none of the spaghetti'
        result = self.reconcile(text, [skipped('item_1', text)])
        self.assertEqual(result.eaten_items, ())
        self.assertEqual([r.plan_item for r in result.skipped_items], [self.plan.items[0]])
        self.assert_no_intake()

    def test_item_none_cannot_expand_to_other_items(self):
        text = 'I ate none of the spaghetti'
        result = self.reconcile(text, [skipped(f'item_{i}', text) for i in (1, 2, 3)])
        self.assertEqual([r.plan_item for r in result.skipped_items], [self.plan.items[0]])

    def test_attestation_commits_exact_quantity_through_conversation(self):
        self.state.save_meal_plan(self.plan)
        text = 'I ate all of the spaghetti you told me to eat'
        conversation = MealReportConversationOrchestrator(self.state, MealReportReconciler(
            ScriptedInterpreter({text: semantic(planned=[eaten('item_1', text)])}),
            LocalFDFoodResolver(self.catalog), NaturalPortionInterpreter(),
        ))
        result = conversation.process(chat_guid=CHAT_GUID, user_text=text, source_event_id='exact',
                                      context_resolver=CurrentPlanResolver(self.state))
        self.assertEqual(result.outcome, 'applied')
        entries = self.state.get_daily_intake(DAY)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].official_servings, Decimal('1.75'))

    def test_none_commits_skipped_fact_without_intake(self):
        self.state.save_meal_plan(self.plan)
        text = 'I ate none of the spaghetti'
        conversation = MealReportConversationOrchestrator(self.state, MealReportReconciler(
            ScriptedInterpreter({text: semantic(planned=[skipped('item_1', text)])}),
            LocalFDFoodResolver(self.catalog), NaturalPortionInterpreter(),
        ))
        result = conversation.process(chat_guid=CHAT_GUID, user_text=text, source_event_id='none',
                                      context_resolver=CurrentPlanResolver(self.state))
        self.assertEqual(result.outcome, 'applied')
        self.assertEqual(self.state.get_daily_intake(DAY), ())

    def test_ambiguous_none_does_not_skip_a_guessed_item(self):
        result = self.reconcile('I ate none of it', [skipped('item_1', 'none of it')])
        self.assertEqual(result.skipped_items, ())
        self.assertTrue(result.clarification_items)

    def test_existing_official_quantity_paths_remain_authoritative(self):
        for item_id, phrase, food_name, amount in (
            ('item_1', '1 official serving', 'spaghetti', '1'),
            ('item_1', '3.5 oz', 'spaghetti', '.875'),
            ('item_2', '1', 'meatball sub', '1'),
            ('item_3', '3 quarter pieces', 'sweet potatoes', '.75'),
        ):
            with self.subTest(phrase=phrase):
                result = self.reconcile('I ate ' + phrase + ' of ' + food_name, [eaten(
                    item_id, phrase, relation='modified', quantity_text=phrase,
                )])
                self.assertEqual(result.eaten_items[0].official_servings, Decimal(amount))
                self.assertEqual(result.eaten_items[0].quantity_source, 'explicit_deterministic')

    def test_visual_literal_survives_restart_and_duplicate_without_intake(self):
        self.state.save_meal_plan(self.plan)
        for index, (relation, reference) in enumerate((
            ('as_recommended', '1 scoop spaghetti'),
            ('as_recommended', 'spaghetti'),
            ('modified', '1 scoop spaghetti'),
        )):
            text = 'I ate 1 scoop spaghetti'
            response = semantic(scope='partial', planned=[eaten(
                'item_1', reference, relation=relation,
                quantity_text='1 scoop' if relation == 'modified' else None,
            )])
            interpreter = ScriptedInterpreter({text: response})
            reconciler = MealReportReconciler(interpreter, LocalFDFoodResolver(self.catalog), NaturalPortionInterpreter())
            conversation = MealReportConversationOrchestrator(self.state, reconciler)
            args = dict(chat_guid=CHAT_GUID, user_text=text, source_event_id=f'visual-{index}',
                        context_resolver=CurrentPlanResolver(self.state))
            result = conversation.process(**args)
            self.assertEqual(result.outcome, 'clarification_required')
            self.catalog.close()
            self.catalog = OfficialNutritionCatalog(self.path)
            self.state = DurableMealState(self.catalog)
            draft = self.state.load_active_meal_report_draft(CHAT_GUID)
            self.assertEqual(draft.planned_items[0].original_quantity_text, '1 scoop')
            self.assertEqual(draft.planned_items[0].quantity_status, 'unresolved')
            self.assertIsNone(draft.planned_items[0].official_servings)
            self.assertEqual(draft.planned_items[0].plan_item_id, 'item_1')
            self.assert_no_intake()
            replay = MealReportConversationOrchestrator(self.state, MealReportReconciler(
                interpreter, LocalFDFoodResolver(self.catalog), NaturalPortionInterpreter(),
            )).process(**{**args, 'context_resolver': CurrentPlanResolver(self.state)})
            self.assertEqual(replay.outcome, 'replayed')
            self.assert_no_intake()


if __name__ == '__main__':
    unittest.main()
