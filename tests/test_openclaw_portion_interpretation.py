"""Offline tests for the scoped OpenClaw/Luna portion matcher."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any
import unittest
from unittest.mock import patch

import nutrition_optimizer.openclaw_portion_interpretation as openclaw_portion
from nutrition_optimizer.portion_interpretation import (
    PortionInterpretationError,
    PortionInterpretationRequest,
    PortionSemanticGatewayUnavailableError,
    PortionSemanticOutputError,
    PortionSemanticTimeoutError,
)
from tests.test_portion_interpretation import make_resolved_food


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
            "details": {
                "json": payload,
                "provider": provider,
                "model": model,
            }
        },
        "source": "plugin",
    }


class FakeOpenClawRunner:
    def __init__(
        self,
        *,
        payload: object | None = None,
        failure: BaseException | None = None,
        stdout: str | None = None,
        returncode: int = 0,
        stderr: str = "",
    ) -> None:
        if payload is None:
            payload = {
                "decision": "estimate",
                "estimated_official_servings": "1.5",
                "confidence": "medium",
                "reason": "bowl_volume",
            }
        self.stdout = stdout or json.dumps(openclaw_envelope(payload))
        self.failure = failure
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, kwargs))
        if self.failure is not None:
            raise self.failure
        return subprocess.CompletedProcess(
            args=args,
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


class OpenClawPortionSemanticMatcherTests(unittest.TestCase):
    def request(self) -> PortionInterpretationRequest:
        return PortionInterpretationRequest(
            resolved_food=make_resolved_food(name="Cereal", serving_unit="Cup"),
            original_quantity_text="a bowl",
        )

    @staticmethod
    def request_params(runner: FakeOpenClawRunner) -> dict[str, object]:
        args = runner.calls[0][0]
        return json.loads(args[args.index("--params") + 1])

    def test_uses_luna_high_reasoning_schema_visible_serving_context_only(self) -> None:
        request = self.request()
        runner = FakeOpenClawRunner()
        matcher = openclaw_portion.OpenClawPortionSemanticMatcher(
            "/test-only/openclaw",
            runner=runner,
        )

        decision = matcher.decide(request)

        self.assertEqual(str(decision.estimated_official_servings), "1.5")
        params = self.request_params(runner)
        self.assertEqual(params["name"], "llm-task")
        args = params["args"]
        self.assertIsInstance(args, dict)
        self.assertEqual(args["provider"], "openai")
        self.assertEqual(args["model"], "openai/gpt-5.6-luna")
        self.assertEqual(args["thinking"], "high")
        schema = args["schema"]
        self.assertIsInstance(schema, dict)
        self.assertEqual(
            schema["required"],
            ["decision", "estimated_official_servings", "confidence", "reason"],
        )
        self.assertFalse(schema["additionalProperties"])
        model_prompt = args["prompt"]
        self.assertIsInstance(model_prompt, str)
        self.assertIn("OUTPUT CONTRACT", model_prompt)
        self.assertIn('"estimated_official_servings"', model_prompt)
        payload = json.loads(args["input"])
        self.assertEqual(payload["official_food_display_name"], "Cereal")
        self.assertEqual(payload["official_serving"], {"quantity": "1", "unit": "Cup", "description": "1 Cup"})
        self.assertEqual(payload["portion_phrase"], "a bowl")
        payload_text = json.dumps(payload)
        for forbidden in (
            "calories",
            "protein",
            "carbohydrates",
            "fat",
            "daily_target",
            "component_identity",
        ):
            self.assertNotIn(forbidden, payload_text)
        self.assertEqual(len(runner.calls), 1)

    def test_invalid_model_estimates_and_unknown_decisions_fail_closed(self) -> None:
        request = self.request()
        invalid_payloads = (
            {
                "decision": "estimate",
                "estimated_official_servings": "0",
                "confidence": "high",
                "reason": "bad",
            },
            {
                "decision": "estimate",
                "estimated_official_servings": "-1",
                "confidence": "high",
                "reason": "bad",
            },
            {
                "decision": "estimate",
                "estimated_official_servings": "NaN",
                "confidence": "high",
                "reason": "bad",
            },
            {
                "decision": "estimate",
                "estimated_official_servings": "Infinity",
                "confidence": "high",
                "reason": "bad",
            },
            {
                "decision": "estimate",
                "estimated_official_servings": "13",
                "confidence": "high",
                "reason": "bad",
            },
            {
                "decision": "unknown",
                "estimated_official_servings": None,
                "confidence": None,
                "reason": "bad",
            },
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                matcher = openclaw_portion.OpenClawPortionSemanticMatcher(
                    "/test-only/openclaw",
                    runner=FakeOpenClawRunner(payload=payload),
                )
                with self.assertRaises(PortionSemanticOutputError):
                    matcher.decide(request)

    def test_gateway_schema_rejection_is_sanitized_as_output_error(self) -> None:
        matcher = openclaw_portion.OpenClawPortionSemanticMatcher(
            "/test-only/openclaw",
            runner=FakeOpenClawRunner(
                stdout=json.dumps(
                    {
                        "ok": False,
                        "toolName": "llm-task",
                        "error": {
                            "message": "LLM JSON did not match schema: private model output"
                        },
                    }
                )
            ),
        )

        with self.assertRaisesRegex(PortionSemanticOutputError, "invalid output") as raised:
            matcher.decide(self.request())

        self.assertNotIn("private model output", str(raised.exception))

    def test_timeout_and_gateway_unavailable_are_distinct_and_sanitized(self) -> None:
        timeout_matcher = openclaw_portion.OpenClawPortionSemanticMatcher(
            "/test-only/openclaw",
            runner=FakeOpenClawRunner(failure=subprocess.TimeoutExpired("openclaw", 1)),
        )
        unavailable_matcher = openclaw_portion.OpenClawPortionSemanticMatcher(
            "/test-only/openclaw",
            runner=FakeOpenClawRunner(
                returncode=1,
                stderr="ECONNREFUSED private gateway token",
            ),
        )

        with self.assertRaises(PortionSemanticTimeoutError):
            timeout_matcher.decide(self.request())
        with self.assertRaisesRegex(
            PortionSemanticGatewayUnavailableError,
            "gateway is unavailable",
        ) as raised:
            unavailable_matcher.decide(self.request())
        self.assertNotIn("private gateway token", str(raised.exception))

    def test_api_key_is_not_required_or_forwarded(self) -> None:
        request = self.request()
        runner = FakeOpenClawRunner()
        matcher = openclaw_portion.OpenClawPortionSemanticMatcher.from_env(
            {"NUTRITION_OPTIMIZER_OPENCLAW_PATH": "/test-only/openclaw"},
            runner=runner,
        )

        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-only-key",
                "NUTRITION_OPTIMIZER_OPENAI_API_KEY": "test-only-key",
            },
        ):
            matcher.decide(request)

        child_env = runner.calls[0][1]["env"]
        self.assertNotIn("OPENAI_API_KEY", child_env)
        self.assertNotIn("NUTRITION_OPTIMIZER_OPENAI_API_KEY", child_env)
        self.assertNotIn("test-only-key", runner.calls[0][0][-1])

    def test_adapter_has_no_catalog_ledger_messaging_or_direct_openai_sdk_dependency(self) -> None:
        source = Path(openclaw_portion.__file__).read_text(encoding="utf-8")

        self.assertNotIn("OfficialNutritionCatalog", source)
        self.assertNotIn("DiningBucket", source)
        self.assertNotIn("DailyLedger", source)
        self.assertNotIn("BlueBubbles", source)
        self.assertNotIn("OpenAI(", source)


if __name__ == "__main__":
    unittest.main()
