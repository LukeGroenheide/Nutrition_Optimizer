from __future__ import annotations

import ast
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import unittest

import nutrition_optimizer.application as application
from nutrition_optimizer.application import (
    RecordIntakeCommand,
    RecordIntakeResult,
    execute_command,
)
from nutrition_optimizer.nutrition import (
    DailyLedger,
    IntakeEntry,
    NutritionProvenance,
    NutritionRecord,
    NutrientProfile,
    Serving,
)


def record(
    name: str = "Test official food",
    *,
    protein: Decimal | None = Decimal("27"),
) -> NutritionRecord:
    return NutritionRecord(
        name=name,
        serving=Serving(quantity=Decimal("1"), unit="serving"),
        nutrients=NutrientProfile(
            calories_kcal=Decimal("175"),
            protein_g=protein,
            carbohydrates_g=Decimal("0"),
            fat_g=Decimal("4"),
            sodium_mg=Decimal("303"),
        ),
        provenance=NutritionProvenance(
            provider="test official source",
            retrieved_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
        ),
    )


class ApplicationCommandTests(unittest.TestCase):
    def test_record_intake_adds_exactly_one_entry(self) -> None:
        original = DailyLedger()
        nutrition_record = record()

        result = execute_command(
            original,
            RecordIntakeCommand(nutrition_record, Decimal("1.25")),
        )

        self.assertIsInstance(result, RecordIntakeResult)
        self.assertEqual(len(result.updated_ledger.entries), 1)
        self.assertIs(result.updated_ledger.entries[0].record, nutrition_record)

    def test_decimal_servings_are_preserved(self) -> None:
        result = execute_command(
            DailyLedger(),
            RecordIntakeCommand(record(), Decimal("1.250")),
        )

        self.assertEqual(result.updated_ledger.entries[0].servings, Decimal("1.250"))

    def test_invalid_servings_use_existing_domain_validation(self) -> None:
        for value in (
            Decimal("0"),
            Decimal("-0.5"),
            Decimal("NaN"),
            Decimal("Infinity"),
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    execute_command(
                        DailyLedger(),
                        RecordIntakeCommand(record(), value),
                    )

        with self.assertRaises(TypeError):
            execute_command(  # type: ignore[arg-type]
                DailyLedger(),
                RecordIntakeCommand(record(), 1),  # type: ignore[arg-type]
            )

    def test_record_values_are_not_altered_or_estimated(self) -> None:
        nutrition_record = record(protein=None)

        result = execute_command(
            DailyLedger(),
            RecordIntakeCommand(nutrition_record, Decimal("1.5")),
        )

        entry = result.updated_ledger.entries[0]
        self.assertIs(entry.record, nutrition_record)
        self.assertEqual(
            entry.record.nutrients,
            NutrientProfile(
                calories_kcal=Decimal("175"),
                protein_g=None,
                carbohydrates_g=Decimal("0"),
                fat_g=Decimal("4"),
                sodium_mg=Decimal("303"),
            ),
        )
        self.assertIsNone(result.updated_ledger.total_consumed_nutrients.protein_g)

    def test_original_ledger_is_unchanged(self) -> None:
        existing_entry = IntakeEntry(record("Existing food"), Decimal("1"))
        original = DailyLedger().add_entry(existing_entry)

        result = execute_command(
            original,
            RecordIntakeCommand(record("New food"), Decimal("2")),
        )

        self.assertEqual(original.entries, (existing_entry,))
        self.assertEqual(len(result.updated_ledger.entries), 2)
        self.assertIsNot(result.updated_ledger, original)

    def test_module_has_no_transport_or_interpreter_imports(self) -> None:
        source = Path(application.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported_modules.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )

        self.assertNotIn("nutrition_optimizer.messaging", imported_modules)
        self.assertNotIn("openai", imported_modules)
        self.assertNotIn("BlueBubbles", source)
        self.assertNotIn("OpenAI", source)


if __name__ == "__main__":
    unittest.main()
