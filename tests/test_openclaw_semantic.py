from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import subprocess
from typing import Any
from unittest.mock import patch
import unittest

import nutrition_optimizer.openclaw_semantic as openclaw_semantic
from nutrition_optimizer.nutrition import DailyLedger
from nutrition_optimizer.semantic import RecordIntakeIntent, UnsupportedIntent


def openclaw_envelope(
    payload: object,
    *,
    provider: str = "openai",
    model: str = "gpt-5.6-luna",
) -> dict[str, object]:
    return {
        "ok": True,
        "toolName": "llm-task",
        "output": {
            "content": [{"type": "text", "text": json.dumps(payload)}],
            "details": {
                "json": payload,
                "provider": provider,
                "model": model,
            },
        },
        "source": "plugin",
    }


class FakeOpenClawRunner:
    def __init__(
        self,
        *,
        stdout: str | None = None,
        returncode: int = 0,
        stderr: str = "",
        failure: BaseException | None = None,
        payload: object | None = None,
    ) -> None:
        if stdout is None:
            if payload is None:
                payload = {
                    "intent": "record_intake",
                    "food_text": "herb roasted chicken",
                    "servings": "2",
                    "clarification_required": False,
                    "reason": None,
                }
            stdout = json.dumps(openclaw_envelope(payload))
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.failure = failure
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(
        self,
        args: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, kwargs))
        if self.failure is not None:
            raise self.failure
        return subprocess.CompletedProcess(
            args=args,
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


class OpenClawSemanticInterpreterTests(unittest.TestCase):
    def make_interpreter(
        self,
        *,
        payload: object | None = None,
        stdout: str | None = None,
        returncode: int = 0,
        stderr: str = "",
        failure: BaseException | None = None,
    ) -> tuple[openclaw_semantic.OpenClawSemanticInterpreter, FakeOpenClawRunner]:
        runner = FakeOpenClawRunner(
            payload=payload,
            stdout=stdout,
            returncode=returncode,
            stderr=stderr,
            failure=failure,
        )
        interpreter = openclaw_semantic.OpenClawSemanticInterpreter(
            "/test-only/openclaw",
            runner=runner,
        )
        return interpreter, runner

    @staticmethod
    def request_params(runner: FakeOpenClawRunner) -> dict[str, object]:
        args = runner.calls[0][0]
        return json.loads(args[args.index("--params") + 1])

    def test_valid_intake_uses_luna_high_reasoning_and_strict_schema(self) -> None:
        interpreter, runner = self.make_interpreter()

        result = interpreter.interpret("I ate two servings of herb roasted chicken")

        self.assertEqual(
            result,
            RecordIntakeIntent(
                food_text="herb roasted chicken",
                servings=Decimal("2"),
            ),
        )
        params = self.request_params(runner)
        self.assertEqual(params["name"], "llm-task")
        self.assertEqual(params["sessionKey"], "main")
        args = params["args"]
        self.assertIsInstance(args, dict)
        self.assertEqual(args["provider"], "openai")
        self.assertEqual(args["model"], "openai/gpt-5.6-luna")
        self.assertEqual(args["thinking"], "high")
        schema = args["schema"]
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["additionalProperties"], False)
        self.assertEqual(
            schema["required"],
            [
                "intent",
                "food_text",
                "servings",
                "clarification_required",
                "reason",
            ],
        )
        model_prompt = args["prompt"]
        self.assertIsInstance(model_prompt, str)
        self.assertIn("OUTPUT CONTRACT", model_prompt)
        self.assertIn('"intent"', model_prompt)
        self.assertIn('"additionalProperties":false', model_prompt)
        command = runner.calls[0][0]
        self.assertEqual(command[1:4], ["gateway", "call", "tools.invoke"])
        self.assertNotIn("agent", command)

    def test_missing_servings_requires_clarification(self) -> None:
        interpreter, _ = self.make_interpreter(
            payload={
                "intent": "record_intake",
                "food_text": "herb roasted chicken",
                "servings": None,
                "clarification_required": True,
                "reason": None,
            }
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

    def test_unsupported_intent_maps_to_public_result(self) -> None:
        interpreter, _ = self.make_interpreter(
            payload={
                "intent": "unsupported",
                "food_text": None,
                "servings": None,
                "clarification_required": False,
                "reason": "unsupported_request",
            }
        )

        result = interpreter.interpret("What should I eat for dinner?")

        self.assertEqual(result, UnsupportedIntent(reason="unsupported_request"))

    def test_fractional_servings_are_decimal_compatible(self) -> None:
        interpreter, _ = self.make_interpreter(
            payload={
                "intent": "record_intake",
                "food_text": "test food",
                "servings": "1.25",
                "clarification_required": False,
                "reason": None,
            }
        )

        result = interpreter.interpret("I had 1.25 servings of test food")

        self.assertIsInstance(result, RecordIntakeIntent)
        self.assertEqual(result.servings, Decimal("1.25"))
        self.assertNotIsInstance(result.servings, float)

    def test_malformed_json_fails_without_exposing_content(self) -> None:
        interpreter, _ = self.make_interpreter(stdout="{not valid model output")

        with self.assertRaisesRegex(
            openclaw_semantic.OpenClawSemanticResponseError,
            "malformed JSON",
        ) as raised:
            interpreter.interpret("private user text")

        self.assertNotIn("private user text", str(raised.exception))
        self.assertNotIn("not valid model output", str(raised.exception))

    def test_schema_invalid_result_fails_after_envelope_validation(self) -> None:
        interpreter, _ = self.make_interpreter(
            payload={
                "intent": "record_intake",
                "food_text": "test food",
                "servings": "2",
                "clarification_required": False,
                "reason": None,
                "unexpected": "field",
            }
        )

        with self.assertRaisesRegex(
            openclaw_semantic.OpenClawSemanticResponseError,
            "unusable structured output",
        ):
            interpreter.interpret("I ate test food")

    def test_invalid_servings_are_rejected(self) -> None:
        for quantity in ("0", "-1", "NaN", "Infinity", "-Infinity", "", "1.2.3"):
            with self.subTest(quantity=quantity):
                interpreter, _ = self.make_interpreter(
                    payload={
                        "intent": "record_intake",
                        "food_text": "test food",
                        "servings": quantity,
                        "clarification_required": False,
                        "reason": None,
                    }
                )

                with self.assertRaisesRegex(
                    openclaw_semantic.OpenClawSemanticResponseError,
                    "invalid quantity",
                ):
                    interpreter.interpret("I ate test food")

    def test_nonzero_command_failure_is_sanitized(self) -> None:
        interpreter, _ = self.make_interpreter(
            returncode=1,
            stderr="private gateway token and model output",
        )

        with self.assertRaisesRegex(
            openclaw_semantic.OpenClawSemanticInvocationError,
            "request failed",
        ) as raised:
            interpreter.interpret("private user text")

        self.assertNotIn("private gateway token", str(raised.exception))
        self.assertNotIn("private user text", str(raised.exception))

    def test_gateway_error_envelope_is_sanitized(self) -> None:
        interpreter, _ = self.make_interpreter(
            stdout=json.dumps(
                {
                    "ok": False,
                    "toolName": "llm-task",
                    "error": {"message": "private gateway details"},
                }
            )
        )

        with self.assertRaisesRegex(
            openclaw_semantic.OpenClawSemanticInvocationError,
            "tool invocation failed",
        ) as raised:
            interpreter.interpret("private user text")

        self.assertNotIn("private gateway details", str(raised.exception))
        self.assertNotIn("private user text", str(raised.exception))

    def test_llm_task_schema_rejection_is_classified_without_exposing_details(self) -> None:
        interpreter, _ = self.make_interpreter(
            stdout=json.dumps(
                {
                    "ok": False,
                    "toolName": "llm-task",
                    "error": {
                        "message": (
                            "LLM JSON did not match schema: intent is required; "
                            "private model output"
                        )
                    },
                }
            )
        )

        with self.assertRaisesRegex(
            openclaw_semantic.OpenClawSemanticStructuredOutputError,
            "structured output was rejected",
        ) as raised:
            interpreter.interpret("private user text")

        self.assertNotIn("private model output", str(raised.exception))
        self.assertNotIn("private user text", str(raised.exception))

    def test_gateway_connection_failure_is_classified_without_exposing_stderr(self) -> None:
        interpreter, _ = self.make_interpreter(
            returncode=1,
            stderr="ECONNREFUSED private gateway token",
        )

        with self.assertRaisesRegex(
            openclaw_semantic.OpenClawSemanticGatewayUnavailableError,
            "gateway is unavailable",
        ) as raised:
            interpreter.interpret("private user text")

        self.assertNotIn("private gateway token", str(raised.exception))
        self.assertNotIn("private user text", str(raised.exception))

    def test_timeout_is_sanitized(self) -> None:
        interpreter, _ = self.make_interpreter(
            failure=subprocess.TimeoutExpired("openclaw", 65),
        )

        with self.assertRaisesRegex(
            openclaw_semantic.OpenClawSemanticTimeoutError,
            "timed out",
        ):
            interpreter.interpret("private user text")

    def test_executable_unavailable_is_sanitized(self) -> None:
        interpreter, _ = self.make_interpreter(failure=FileNotFoundError())

        with self.assertRaisesRegex(
            openclaw_semantic.OpenClawSemanticConfigurationError,
            "executable is unavailable",
        ):
            interpreter.interpret("private user text")

    def test_unexpected_model_envelope_is_rejected(self) -> None:
        interpreter, _ = self.make_interpreter(
            payload={
                "intent": "record_intake",
                "food_text": "test food",
                "servings": "1",
                "clarification_required": False,
                "reason": None,
            }
        )
        runner = interpreter._runner
        self.assertIsInstance(runner, FakeOpenClawRunner)
        runner.stdout = json.dumps(
            openclaw_envelope(
                {
                    "intent": "record_intake",
                    "food_text": "test food",
                    "servings": "1",
                    "clarification_required": False,
                    "reason": None,
                },
                model="gpt-5.6-sol",
            )
        )

        with self.assertRaisesRegex(
            openclaw_semantic.OpenClawSemanticResponseError,
            "unexpected semantic model",
        ):
            interpreter.interpret("I ate test food")

    def test_api_key_environment_is_not_forwarded_and_is_not_required(self) -> None:
        runner = FakeOpenClawRunner()
        interpreter = openclaw_semantic.OpenClawSemanticInterpreter.from_env(
            {openclaw_semantic.OPENCLAW_PATH_ENV_VAR: "/test-only/openclaw"},
            runner=runner,
        )

        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-only-key",
                "NUTRITION_OPTIMIZER_OPENAI_API_KEY": "test-only-key",
            },
        ):
            interpreter.interpret("I ate test food")

        child_env = runner.calls[0][1]["env"]
        self.assertNotIn("OPENAI_API_KEY", child_env)
        self.assertNotIn("NUTRITION_OPTIMIZER_OPENAI_API_KEY", child_env)
        self.assertNotIn("test-only-key", runner.calls[0][0][-1])

    def test_startup_check_rejects_unavailable_executable(self) -> None:
        interpreter = openclaw_semantic.OpenClawSemanticInterpreter(
            "/test-only/openclaw"
        )

        with patch(
            "nutrition_optimizer.openclaw_semantic.shutil.which",
            return_value=None,
        ):
            with self.assertRaisesRegex(
                openclaw_semantic.OpenClawSemanticConfigurationError,
                "executable is unavailable",
            ):
                interpreter.ensure_executable_available()

    def test_no_ledger_or_messaging_dependency_and_no_ledger_mutation(self) -> None:
        module_path = Path(openclaw_semantic.__file__)
        source = module_path.read_text(encoding="utf-8")

        self.assertNotIn("NutritionRecord", source)
        self.assertNotIn("DailyLedger", source)
        self.assertNotIn("BlueBubbles", source)
        self.assertNotIn("execute_command", source)

        interpreter, _ = self.make_interpreter()
        result = interpreter.interpret("I ate test food")

        self.assertIsInstance(result, RecordIntakeIntent)
        self.assertEqual(DailyLedger().entries, ())


if __name__ == "__main__":
    unittest.main()
