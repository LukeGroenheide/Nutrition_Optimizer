from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import unittest

import nutrition_optimizer.openai_semantic as openai_semantic
from nutrition_optimizer.nutrition import DailyLedger
from nutrition_optimizer.semantic import RecordIntakeIntent, UnsupportedIntent


class FakeResponses:
    def __init__(self, response: object = None, failure: Exception | None = None):
        self.response = response
        self.failure = failure
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        return self.response


class FakeOpenAIClient:
    def __init__(self, response: object = None, failure: Exception | None = None):
        self.responses = FakeResponses(response=response, failure=failure)


def structured_response(**fields: object) -> SimpleNamespace:
    return SimpleNamespace(output_parsed=fields)


class OpenAISemanticInterpreterTests(unittest.TestCase):
    def make_interpreter(
        self,
        response: object = None,
        failure: Exception | None = None,
    ) -> tuple[openai_semantic.OpenAISemanticInterpreter, FakeOpenAIClient]:
        client = FakeOpenAIClient(response=response, failure=failure)
        interpreter = openai_semantic.OpenAISemanticInterpreter(
            "test-only-api-key",
            client=client,
        )
        return interpreter, client

    def test_valid_structured_intake_maps_to_record_intake_intent(self) -> None:
        interpreter, client = self.make_interpreter(
            structured_response(
                intent="record_intake",
                food_text="herb roasted chicken",
                servings="2",
                clarification_required=False,
                reason=None,
            )
        )

        result = interpreter.interpret("I ate two servings of herb roasted chicken")

        self.assertEqual(
            result,
            RecordIntakeIntent(
                food_text="herb roasted chicken",
                servings=Decimal("2"),
            ),
        )
        self.assertEqual(client.responses.calls[0]["model"], "gpt-5.6-luna")
        self.assertIs(
            client.responses.calls[0]["text_format"],
            openai_semantic._StructuredSemanticResult,
        )

    def test_word_or_number_quantity_is_converted_to_exact_decimal(self) -> None:
        interpreter, _ = self.make_interpreter(
            structured_response(
                intent="record_intake",
                food_text="test food",
                servings="1.25",
                clarification_required=False,
                reason=None,
            )
        )

        result = interpreter.interpret("I had one and a quarter servings of test food")

        self.assertIsInstance(result, RecordIntakeIntent)
        self.assertEqual(result.servings, Decimal("1.25"))
        self.assertNotIsInstance(result.servings, float)

    def test_missing_quantity_remains_unresolved(self) -> None:
        interpreter, _ = self.make_interpreter(
            structured_response(
                intent="record_intake",
                food_text="herb roasted chicken",
                servings=None,
                clarification_required=True,
                reason=None,
            )
        )

        result = interpreter.interpret("I ate herb roasted chicken")

        self.assertEqual(
            result,
            RecordIntakeIntent(
                food_text="herb roasted chicken",
                servings=None,
                clarification_required="servings_required",
            ),
        )

    def test_unsupported_result_is_explicit(self) -> None:
        interpreter, _ = self.make_interpreter(
            structured_response(
                intent="unsupported",
                food_text=None,
                servings=None,
                clarification_required=False,
                reason="unsupported_request",
            )
        )

        result = interpreter.interpret("I drove to class")

        self.assertEqual(result, UnsupportedIntent(reason="unsupported_request"))

    def test_malformed_structured_response_fails_safely(self) -> None:
        interpreter, _ = self.make_interpreter(
            structured_response(
                intent="record_intake",
                food_text="test food",
                servings="not-a-number",
                clarification_required=False,
                reason=None,
            )
        )

        with self.assertRaisesRegex(
            openai_semantic.OpenAISemanticResponseError,
            "invalid quantity",
        ) as raised:
            interpreter.interpret("private test text")

        self.assertNotIn("private test text", str(raised.exception))

    def test_refusal_fails_safely(self) -> None:
        interpreter, _ = self.make_interpreter(
            SimpleNamespace(output_parsed=None, refusal="private refusal text")
        )

        with self.assertRaisesRegex(
            openai_semantic.OpenAISemanticRefusalError,
            "refused semantic interpretation",
        ) as raised:
            interpreter.interpret("private test text")

        self.assertNotIn("private refusal text", str(raised.exception))
        self.assertNotIn("private test text", str(raised.exception))

    def test_api_error_fails_safely_without_logging_content(self) -> None:
        interpreter, _ = self.make_interpreter(
            failure=RuntimeError("private model output and test-only secret")
        )

        with self.assertNoLogs("nutrition_optimizer.openai_semantic", level="DEBUG"):
            with self.assertRaisesRegex(
                openai_semantic.OpenAISemanticAPIError,
                "semantic request failed",
            ) as raised:
                interpreter.interpret("private user text")

        self.assertNotIn("private user text", str(raised.exception))
        self.assertNotIn("test-only secret", str(raised.exception))

    def test_invalid_quantities_fail_safely(self) -> None:
        for quantity in ("0", "-1", "NaN", "Infinity", "-Infinity", "", "1.2.3"):
            with self.subTest(quantity=quantity):
                interpreter, _ = self.make_interpreter(
                    structured_response(
                        intent="record_intake",
                        food_text="test food",
                        servings=quantity,
                        clarification_required=False,
                        reason=None,
                    )
                )

                with self.assertRaisesRegex(
                    openai_semantic.OpenAISemanticResponseError,
                    "invalid quantity",
                ):
                    interpreter.interpret("I ate test food")

    def test_runtime_model_override_works(self) -> None:
        response = structured_response(
            intent="unsupported",
            food_text=None,
            servings=None,
            clarification_required=False,
            reason="unsupported_request",
        )
        client = FakeOpenAIClient(response=response)
        interpreter = openai_semantic.OpenAISemanticInterpreter.from_env(
            {
                openai_semantic.OPENAI_API_KEY_ENV_VAR: "test-only-api-key",
                openai_semantic.OPENAI_MODEL_ENV_VAR: "test-model",
            },
            client=client,
        )

        interpreter.interpret("I drove to class")

        self.assertEqual(interpreter.model, "test-model")
        self.assertEqual(client.responses.calls[0]["model"], "test-model")

    def test_missing_api_key_fails_clearly(self) -> None:
        with self.assertRaisesRegex(
            openai_semantic.OpenAISemanticConfigurationError,
            openai_semantic.OPENAI_API_KEY_ENV_VAR,
        ):
            openai_semantic.OpenAISemanticInterpreter.from_env({})

    def test_no_nutrition_ledger_or_messaging_dependency(self) -> None:
        module_path = Path(openai_semantic.__file__)
        source = module_path.read_text(encoding="utf-8")

        self.assertNotIn("NutritionRecord", source)
        self.assertNotIn("DailyLedger", source)
        self.assertNotIn("BlueBubbles", source)
        self.assertNotIn("execute_command", source)
        self.assertEqual(DailyLedger().entries, ())


if __name__ == "__main__":
    unittest.main()
