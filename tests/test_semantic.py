from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import unittest

import nutrition_optimizer.semantic as semantic
from nutrition_optimizer.nutrition import DailyLedger
from nutrition_optimizer.semantic import (
    DeterministicSemanticInterpreter,
    RecordIntakeIntent,
    SemanticInterpretationError,
    UnsupportedIntent,
    interpret_text,
)


class FakeInterpreter:
    def __init__(self, result: object = None, failure: Exception | None = None) -> None:
        self.result = result
        self.failure = failure

    def interpret(self, user_text: str) -> object:
        del user_text
        if self.failure is not None:
            raise self.failure
        return self.result


class SemanticInterpretationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.interpreter = DeterministicSemanticInterpreter()

    def test_clear_intake_statement_maps_to_structured_intent(self) -> None:
        result = interpret_text(
            self.interpreter,
            "I ate two servings of herb roasted chicken.",
        )

        self.assertEqual(
            result,
            RecordIntakeIntent(
                food_text="herb roasted chicken",
                servings=Decimal("2"),
            ),
        )

    def test_quantity_is_decimal_compatible(self) -> None:
        result = interpret_text(
            self.interpreter,
            "I had 1.25 servings of test food",
        )

        self.assertIsInstance(result, RecordIntakeIntent)
        self.assertEqual(result.servings, Decimal("1.25"))

    def test_missing_quantity_remains_unresolved(self) -> None:
        result = interpret_text(self.interpreter, "I ate herb roasted chicken")

        self.assertIsInstance(result, RecordIntakeIntent)
        self.assertIsNone(result.servings)
        self.assertEqual(result.clarification_required, "servings_required")

    def test_unsupported_request_is_explicit(self) -> None:
        result = interpret_text(self.interpreter, "What should I eat for dinner?")

        self.assertEqual(result, UnsupportedIntent(reason="unsupported_request"))

    def test_fake_interpreter_result_is_accepted_without_domain_execution(self) -> None:
        intent = RecordIntakeIntent("food as stated", Decimal("1"))

        result = interpret_text(FakeInterpreter(result=intent), "private user text")

        self.assertIs(result, intent)
        self.assertEqual(DailyLedger().entries, ())

    def test_malformed_interpreter_output_fails_safely(self) -> None:
        with self.assertRaisesRegex(
            SemanticInterpretationError,
            "invalid result",
        ):
            interpret_text(
                FakeInterpreter(result={"food_text": "invented", "calories": 99}),
                "private user text",
            )

    def test_interpreter_failure_does_not_log_user_text_or_model_output(self) -> None:
        with self.assertNoLogs("nutrition_optimizer.semantic", level="DEBUG"):
            with self.assertRaisesRegex(
                SemanticInterpretationError,
                "semantic interpretation failed",
            ) as raised:
                interpret_text(
                    FakeInterpreter(failure=RuntimeError("private model output")),
                    "private user text",
                )

        self.assertNotIn("private user text", str(raised.exception))
        self.assertNotIn("private model output", str(raised.exception))

    def test_semantic_module_has_no_nutrition_or_transport_dependency(self) -> None:
        module_path = Path(semantic.__file__)
        source = module_path.read_text(encoding="utf-8")

        self.assertNotIn("NutritionRecord", source)
        self.assertNotIn("DailyLedger", source)
        self.assertNotIn("BlueBubbles", source)
        self.assertNotIn("OpenAI", source)


if __name__ == "__main__":
    unittest.main()
