"""Frozen presentation authority and v13 migration, using isolated SQLite files."""
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from nutrition_optimizer.durable_state import DurableMealState, MealReportApplicationError, _same_plan
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog, NutritionCatalogError
from nutrition_optimizer.fdmealplanner import catalog as catalog_module
from nutrition_optimizer.food_resolution import FoodResolutionRequest, LocalFDFoodResolver
from nutrition_optimizer.meal_optimizer import RecommendedMealItem
from nutrition_optimizer.meal_request import RequestedMealFood
from nutrition_optimizer.durable_state import ScheduledRecommendationAlreadyExistsError
from nutrition_optimizer.meal_report import MealPlan, PlannedMealItem, MealReportReconciler
from nutrition_optimizer.nutrition import Serving, SourceIdentifier
from nutrition_optimizer.portion_interpretation import NaturalPortionInterpreter
from nutrition_optimizer.presentation_binding import PresentationBinding, PresentationKind, freeze_presentation
from nutrition_optimizer.recommendation_rendering import RecommendedPortionRenderer, render_meal_recommendation, format_meal_message
from nutrition_optimizer.serving_presentation import DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS, PresentationCalibrationRegistry
from tests import test_serving_presentation as fixtures
from tests import test_meal_report_outcome_v12 as history
from tests.test_food_resolution import component, mapped, menu_day


class PresentationBindingTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'isolated.sqlite3'
        self.catalog = OfficialNutritionCatalog(self.path)
        self.addCleanup(lambda: self.catalog.close())
        potato = component(62889, 'Baked Sweet Potatoes-Master')
        potato.update(recipePortionSize='1', recipePortionSizeUnit='Each')
        self.catalog.synchronize_fd_refresh(
            mapped(menu_day(fixtures.DAY, 2, 'Lunch', ((potato, 'Homestyle'),))),
            requested_start=fixtures.DAY, requested_end=fixtures.DAY, observed_at=fixtures.OBSERVED_AT)
        self.state = DurableMealState(self.catalog)
        self.food = LocalFDFoodResolver(self.catalog).resolve(
            FoodResolutionRequest('Baked Sweet Potatoes-Master', fixtures.DAY, meal=2))
        item = RecommendedMealItem(self.food.occurrence, self.food.nutrition_record, Decimal('2.00'))
        self.recommendation = fixtures._recommendation(item)
        self.plan = render_meal_recommendation(self.recommendation, RecommendedPortionRenderer()).meal_plan
        self.binding = self.plan.items[0].presentation_binding

    def reopen(self):
        self.catalog.close()
        self.catalog = OfficialNutritionCatalog(self.path)
        self.state = DurableMealState(self.catalog)

    def report(self, plan, phrase='3 quarter pieces'):
        return MealReportReconciler(fixtures._FixedSemanticInterpreter(fixtures._modified_semantic(phrase)),
            fixtures._NoAdditionalFoodResolver(), NaturalPortionInterpreter()).reconcile(plan, 'I ate '+phrase)

    def test_calibrated_render_persists_exact_relationship_and_provenance(self):
        self.assertEqual(self.plan.items[0].recommended_official_servings, Decimal('2.00'))
        self.assertEqual(self.binding.kind, PresentationKind.REVERSIBLE_CALIBRATED)
        calibration = self.binding.calibration
        self.assertEqual(calibration.calibration_id, 'phelps-baked-sweet-potato-quarter')
        self.assertEqual(calibration.calibration_version, 1)
        self.assertEqual(calibration.presentation_units_per_official_serving, Decimal('4'))
        self.assertIn('2026-09-03', calibration.provenance)
        saved = self.state.save_meal_plan(self.plan)
        self.reopen()
        loaded = self.state.load_meal_plan(saved.plan_id)
        self.assertEqual(loaded.plan, self.plan)
        self.assertEqual(self.report(loaded.plan).eaten_items[0].official_servings, Decimal('.75'))

    def test_registry_change_after_restart_does_not_reinterpret_plan(self):
        saved = self.state.save_meal_plan(self.plan)
        newer = replace(self.binding.calibration, calibration_version=2,
                        presentation_units_per_official_serving=Decimal('8'))
        self.reopen()
        registry = PresentationCalibrationRegistry((newer,))
        with patch('nutrition_optimizer.recommendation_rendering.DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS', registry):
            loaded = self.state.load_meal_plan(saved.plan_id)
            self.assertEqual(self.report(loaded.plan).eaten_items[0].official_servings, Decimal('.75'))
            new_plan = render_meal_recommendation(self.recommendation, RecommendedPortionRenderer()).meal_plan
            self.assertEqual(new_plan.items[0].presentation_binding.calibration.calibration_version, 2)
            self.assertNotEqual(new_plan.items[0].natural_quantity_text, loaded.plan.items[0].natural_quantity_text)

    def test_legacy_visual_text_and_legacy_potato_have_no_registry_fallback(self):
        for display in ('~2 scoops', self.plan.items[0].natural_quantity_text):
            item = replace(self.plan.items[0], natural_quantity_text=display, presentation_binding=None)
            legacy = replace(self.plan, items=(item,))
            for phrase in ('1 scoop', '3 quarter pieces'):
                self.assertEqual(self.report(legacy, phrase).eaten_items, ())

    def test_exact_authoritative_roundtrip_has_no_calibrated_ratio(self):
        binding = freeze_presentation(self.food, Decimal('2'), '2 Each', PresentationKind.EXACT_AUTHORITATIVE)
        plan = replace(self.plan, items=(replace(self.plan.items[0], natural_quantity_text='2 Each', presentation_binding=binding),))
        saved = self.state.save_meal_plan(plan)
        self.reopen()
        self.assertEqual(self.state.load_meal_plan(saved.plan_id).plan.items[0].presentation_binding, binding)
        self.assertIsNone(binding.calibration)

    def test_descriptive_only_repetition_cannot_create_intake(self):
        binding = freeze_presentation(self.food, Decimal('2'), 'about a palm-sized portion', PresentationKind.DESCRIPTIVE_ONLY)
        plan = replace(self.plan, items=(replace(self.plan.items[0], natural_quantity_text=binding.display_text, presentation_binding=binding),))
        saved = self.state.save_meal_plan(plan)
        self.reopen()
        self.assertEqual(self.report(self.state.load_meal_plan(saved.plan_id).plan, 'one palm-sized portion').eaten_items, ())
        self.assertEqual(self.state.get_daily_intake(fixtures.DAY), ())
        with self.assertRaises(ValueError):
            replace(binding, calibration=self.binding.calibration)

    def test_display_cannot_change_reverse_relationship(self):
        with self.assertRaises(ValueError):
            replace(self.plan.items[0], natural_quantity_text='100 scoops')
        # Even a consistent Python-created display change cannot alter its ratio.
        binding = replace(self.binding, display_text='100 scoops')
        self.assertIsNone(binding.official_servings_for_phrase(self.food, Decimal('2'), '1 scoop'))
        self.assertEqual(binding.official_servings_for_phrase(self.food, Decimal('2'), '3 quarter pieces'), Decimal('.75'))

    def test_unknown_format_rejected_even_with_valid_checksum(self):
        data = json.loads(self.binding.to_json())
        data['payload']['format_version'] = 2
        data['sha256'] = hashlib.sha256(json.dumps(data['payload'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with self.assertRaises(ValueError):
            PresentationBinding.from_json(json.dumps(data))

    def test_tampered_ratio_checksum_rejected(self):
        data = json.loads(self.binding.to_json())
        data['payload']['calibration']['presentation_units_per_official_serving'] = '100'
        with self.assertRaises(ValueError):
            PresentationBinding.from_json(json.dumps(data))

    def test_food_identity_mismatch_rejected(self):
        identity = SourceIdentifier('component','different')
        occurrence = replace(self.food.occurrence, source_identifier=identity)
        food = replace(self.food, source_identifier=identity, occurrence=occurrence, equivalent_occurrences=(occurrence,))
        with self.assertRaises(ValueError):
            self.binding.validate(food, Decimal('2'), self.binding.display_text)

    def test_official_serving_mismatch_rejected(self):
        record = replace(self.food.nutrition_record, serving=Serving(Decimal('2'), 'Each','2 Each'))
        occurrence = replace(self.food.occurrence, nutrition_record=record)
        food = replace(self.food, nutrition_record=record, occurrence=occurrence, equivalent_occurrences=(occurrence,))
        with self.assertRaises(ValueError):
            self.binding.validate(food, Decimal('2'), self.binding.display_text)

    def test_station_scope_mismatch_rejected(self):
        occurrence = replace(self.food.occurrence, station_concept_id='elsewhere', station_name='Other')
        food = replace(self.food, occurrence=occurrence, equivalent_occurrences=(occurrence,))
        with self.assertRaises(ValueError):
            self.binding.validate(food, Decimal('2'), self.binding.display_text)

    def test_plan_quantity_guard_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.plan.items[0], recommended_official_servings=Decimal('3'))

    def test_parent_display_tampering_fails_closed_on_load(self):
        saved = self.state.save_meal_plan(self.plan)
        with self.catalog._connection:
            self.catalog._connection.execute('UPDATE meal_plan_items SET natural_quantity_text=? WHERE plan_id=?', ('1 scoop',saved.plan_id))
        with self.assertRaises(MealReportApplicationError):
            self.state.load_meal_plan(saved.plan_id)

    def test_immutable_binding_update_rejected(self):
        self.state.save_meal_plan(self.plan)
        with self.assertRaises(sqlite3.IntegrityError):
            self.catalog._connection.execute("UPDATE meal_plan_item_presentation_bindings SET binding_json='{}'")
        self.catalog._connection.rollback()

    def test_immutable_binding_delete_rejected(self):
        self.state.save_meal_plan(self.plan)
        with self.assertRaises(sqlite3.IntegrityError):
            self.catalog._connection.execute('DELETE FROM meal_plan_item_presentation_bindings')
        self.catalog._connection.rollback()

    def test_binding_failure_rolls_back_parent_and_items(self):
        with patch.object(DurableMealState, '_save_presentation_binding', side_effect=sqlite3.OperationalError('injected')):
            with self.assertRaises(MealReportApplicationError):
                self.state.save_meal_plan(self.plan)
        self.assertEqual(self.catalog._connection.execute('SELECT count(*) FROM meal_plans').fetchone()[0], 0)

    def test_scheduled_retry_preserves_binding_and_display(self):
        saved = self.state.save_scheduled_meal_plan(self.plan)
        message = format_meal_message(saved.plan)
        self.reopen()
        with patch('nutrition_optimizer.recommendation_rendering.DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS', PresentationCalibrationRegistry(())):
            dispatch = self.state.load_scheduled_recommendation_dispatch(fixtures.DAY, 2)
            self.assertEqual(dispatch.persisted_plan.plan.items[0].presentation_binding, self.binding)
            self.assertEqual(format_meal_message(dispatch.persisted_plan.plan), message)
            with self.assertRaises(ScheduledRecommendationAlreadyExistsError):
                self.state.save_scheduled_meal_plan(self.plan)
            self.assertEqual(dispatch.persisted_plan.plan_id, saved.plan_id)

    def test_immediate_plan_save_and_guid_retry_freeze_binding(self):
        kwargs = dict(source_event_id='request', chat_guid='chat', requested_foods=(), whole_meal=True,
                      reply_text=format_meal_message(self.plan))
        dispatch = self.state.save_immediate_meal_request_plan(self.plan, **kwargs)
        self.reopen()
        retry = self.state.save_immediate_meal_request_plan(self.plan, **kwargs)
        self.assertEqual(retry.persisted_plan.plan.items[0].presentation_binding, self.binding)
        self.assertEqual(retry.persisted_plan.plan_id, dispatch.persisted_plan.plan_id)

    def test_replacement_owns_new_binding_and_preserves_old(self):
        saved = self.state.save_meal_plan(self.plan)
        newer = replace(self.binding.calibration, calibration_version=2, presentation_units_per_official_serving=Decimal('8'))
        with patch('nutrition_optimizer.recommendation_rendering.DEFAULT_PHELPS_PRESENTATION_CALIBRATIONS', PresentationCalibrationRegistry((newer,))):
            replacement = render_meal_recommendation(self.recommendation, RecommendedPortionRenderer()).meal_plan
        for whole in (False, True):
            with self.subTest(whole_meal=whole):
                dispatch = self.state.save_replacement_meal_plan(saved, replacement,
                    source_event_id='replace'+str(whole), chat_guid='chat', rejected_plan_item_ids=(),
                    whole_meal=whole, request_kind='meal_request',
                    requested_foods=(RequestedMealFood('potato', self.food.source_identifier.kind,
                        self.food.source_identifier.value, self.food.content_signature),),
                    reply_text=format_meal_message(replacement))
                self.assertEqual(self.state.load_meal_plan(saved.plan_id).plan.items[0].presentation_binding,
                                 saved.plan.items[0].presentation_binding)
                self.assertEqual(dispatch.persisted_plan.plan.items[0].presentation_binding.calibration.calibration_version,2)
                saved = dispatch.persisted_plan

    def test_plan_equality_includes_material_binding(self):
        binding = replace(self.binding, calibration=replace(self.binding.calibration, calibration_version=2))
        altered = replace(self.plan, items=(replace(self.plan.items[0], presentation_binding=binding),))
        self.assertFalse(_same_plan(self.plan, altered))
        self.assertTrue(_same_plan(self.plan, replace(self.plan)))


class PresentationMigrationTests(unittest.TestCase):
    # Reuse the existing synthetic history builder, not its migration tests.
    setUp = history.MealReportOutcomeV12MigrationTests.setUp
    _close = history.MealReportOutcomeV12MigrationTests._close
    _plan = history.MealReportOutcomeV12MigrationTests._plan
    _report = staticmethod(history.MealReportOutcomeV12MigrationTests._report)
    _seed_routed_v12_history = history.MealReportOutcomeV12MigrationTests._seed_routed_v12_history

    def v12_fixture(self):
        ids = self._seed_routed_v12_history()
        self.state.cancel_meal_report_draft(self.state.load_meal_report_draft(ids['draft_id']),
            source_event_id='cancel', reply_text='Cancelled', processed_at=history.NOW)
        breakfast = self.state.load_meal_plan('v12-breakfast')
        self.state.save_replacement_meal_plan(breakfast, breakfast.plan,
            source_event_id='replacement', chat_guid='chat', rejected_plan_item_ids=(),
            whole_meal=True, request_kind='meal_request', reply_text='Replacement', created_at=history.NOW)
        self.state.save_scheduled_meal_plan(self._plan(2, 'Chicken'), created_at=history.NOW)
        self.state.save_pending_meal_request(source_event_id='pending', chat_guid='chat',
            service_date=history.DAY, meal=3, requested_foods=(), whole_meal=True,
            reply_text='Pending dinner', created_at=history.NOW)
        c=self.catalog._connection
        with c:
            c.execute("UPDATE meal_plan_items SET natural_quantity_text='~2 scoops'")
            c.execute('DROP TABLE meal_plan_item_presentation_bindings')
            c.execute('PRAGMA user_version=12')
        snapshot=self.snapshot(c)
        self.catalog.close()
        return snapshot

    def snapshot(self,c):
        tables=[r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name != 'meal_plan_item_presentation_bindings'")]
        return {t: sorted([tuple(r) for r in c.execute('SELECT * FROM "'+t+'"')],key=repr) for t in tables}

    def test_fresh_v13_and_reopen(self):
        self.assertEqual(self.catalog._connection.execute('PRAGMA user_version').fetchone()[0],14)
        self.catalog.close()
        self.catalog=OfficialNutritionCatalog(self.path)
        self.assertEqual(self.catalog._connection.execute('SELECT count(*) FROM meal_plan_item_presentation_bindings').fetchone()[0],0)

    def test_v12_history_preserved_byte_values_without_legacy_bindings(self):
        before=self.v12_fixture()
        self.catalog=OfficialNutritionCatalog(self.path)
        c=self.catalog._connection
        self.assertEqual(self.snapshot(c),before)
        self.assertEqual(c.execute('PRAGMA user_version').fetchone()[0],14)
        self.assertEqual(c.execute('SELECT count(*) FROM meal_plan_item_presentation_bindings').fetchone()[0],0)
        self.assertEqual(c.execute('PRAGMA foreign_key_check').fetchall(),[])
        self.assertEqual(c.execute('PRAGMA integrity_check').fetchone()[0],'ok')

    def test_injected_failure_leaves_valid_v12_and_all_history(self):
        before=self.v12_fixture()
        migrate=catalog_module._migrate_presentation_bindings_v13
        def fail(c):
            migrate(c)
            raise sqlite3.OperationalError('injected v13 failure')
        with patch.object(catalog_module,'_migrate_presentation_bindings_v13',side_effect=fail):
            with self.assertRaises(NutritionCatalogError):
                OfficialNutritionCatalog(self.path)
        with sqlite3.connect(self.path) as c:
            self.assertEqual(c.execute('PRAGMA user_version').fetchone()[0],12)
            self.assertEqual(self.snapshot(c),before)
            self.assertIsNone(c.execute("SELECT name FROM sqlite_master WHERE name='meal_plan_item_presentation_bindings'").fetchone())
            self.assertEqual(c.execute('PRAGMA integrity_check').fetchone()[0],'ok')
        self.catalog=OfficialNutritionCatalog(self.path)
        self.assertEqual(self.snapshot(self.catalog._connection),before)
