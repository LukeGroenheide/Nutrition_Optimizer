"""Offline tests for the scoped OpenClaw/Luna food matcher."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any
import unittest
from unittest.mock import patch

import nutrition_optimizer.openclaw_food_resolution as openclaw_food_resolution
from nutrition_optimizer.food_resolution import (
    FoodResolutionError,
    FoodResolutionRequest,
    FoodSemanticGatewayUnavailableError,
    FoodSemanticOutputError,
    FoodSemanticTimeoutError,
    LocalFDFoodResolver,
)
from tests.test_food_resolution import DAY1, T1, component, mapped, menu_day
from nutrition_optimizer.fdmealplanner import OfficialNutritionCatalog


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
                "decision": "match",
                "component_identity": "7:181:1",
                "reason": "shortened local name",
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


class OpenClawFoodSemanticMatcherTests(unittest.TestCase):
    def one_current_candidate(self):
        from tempfile import TemporaryDirectory

        directory = TemporaryDirectory()
        catalog = OfficialNutritionCatalog(Path(directory.name) / "state" / "nutrition.sqlite3")
        self.addCleanup(catalog.close)
        self.addCleanup(directory.cleanup)
        catalog.synchronize_fd_refresh(
            mapped(
                menu_day(
                    DAY1,
                    3,
                    "Dinner",
                    ((component(1, "Zone Tandoori Chicken", ingredient_statement="chicken, spices"), "ZONE"),),
                )
            ),
            requested_start=DAY1,
            requested_end=DAY1,
            observed_at=T1,
        )
        candidate = LocalFDFoodResolver(catalog).list_current_candidates(
            FoodResolutionRequest("tandoori chicken", DAY1, meal="Dinner", station="ZONE")
        )[0]
        return candidate

    @staticmethod
    def request_params(runner: FakeOpenClawRunner) -> dict[str, object]:
        args = runner.calls[0][0]
        return json.loads(args[args.index("--params") + 1])

    def test_uses_luna_high_reasoning_strict_schema_and_only_supplied_candidate(self) -> None:
        candidate = self.one_current_candidate()
        runner = FakeOpenClawRunner()
        matcher = openclaw_food_resolution.OpenClawFoodSemanticMatcher(
            "/test-only/openclaw",
            runner=runner,
        )

        decision = matcher.decide(
            FoodResolutionRequest("tandoori chicken", DAY1, meal="Dinner", station="ZONE"),
            (candidate,),
        )

        self.assertEqual(decision.component_identity, "7:181:1")
        params = self.request_params(runner)
        self.assertEqual(params["name"], "llm-task")
        args = params["args"]
        self.assertIsInstance(args, dict)
        self.assertEqual(args["provider"], "openai")
        self.assertEqual(args["model"], "openai/gpt-5.6-luna")
        self.assertEqual(args["thinking"], "high")
        schema = args["schema"]
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["required"], ["decision", "component_identity", "reason"])
        self.assertFalse(schema["additionalProperties"])
        payload = json.loads(args["input"])
        self.assertEqual(payload["food_text"], "tandoori chicken")
        self.assertEqual(payload["service_date"], DAY1.isoformat())
        self.assertEqual(payload["candidates"], [candidate.semantic_payload()])
        self.assertNotIn("calories", json.dumps(payload))
        self.assertNotIn("servings", json.dumps(payload))
        self.assertEqual(len(runner.calls), 1)

    def test_malformed_structured_output_is_sanitized(self) -> None:
        candidate = self.one_current_candidate()
        matcher = openclaw_food_resolution.OpenClawFoodSemanticMatcher(
            "/test-only/openclaw",
            runner=FakeOpenClawRunner(
                payload={
                    "decision": "match",
                    "component_identity": "7:181:1",
                    "reason": None,
                }
            ),
        )

        with self.assertRaisesRegex(FoodResolutionError, "invalid output") as raised:
            matcher.decide(FoodResolutionRequest("private wording", DAY1), (candidate,))

        self.assertNotIn("private wording", str(raised.exception))

    def test_timeout_is_sanitized(self) -> None:
        candidate = self.one_current_candidate()
        matcher = openclaw_food_resolution.OpenClawFoodSemanticMatcher(
            "/test-only/openclaw",
            runner=FakeOpenClawRunner(failure=subprocess.TimeoutExpired("openclaw", 1)),
        )

        with self.assertRaisesRegex(FoodSemanticTimeoutError, "timed out") as raised:
            matcher.decide(FoodResolutionRequest("private wording", DAY1), (candidate,))

        self.assertNotIn("private wording", str(raised.exception))

    def test_llm_task_schema_rejection_maps_to_semantic_output_error(self) -> None:
        candidate = self.one_current_candidate()
        matcher = openclaw_food_resolution.OpenClawFoodSemanticMatcher(
            "/test-only/openclaw",
            runner=FakeOpenClawRunner(
                stdout=json.dumps(
                    {
                        "ok": False,
                        "toolName": "llm-task",
                        "error": {
                            "message": "LLM JSON did not match schema: private model text"
                        },
                    }
                )
            ),
        )

        with self.assertRaisesRegex(FoodSemanticOutputError, "invalid output") as raised:
            matcher.decide(FoodResolutionRequest("private wording", DAY1), (candidate,))

        self.assertNotIn("private model text", str(raised.exception))
        self.assertNotIn("private wording", str(raised.exception))

    def test_gateway_unavailable_maps_to_distinct_transport_error(self) -> None:
        candidate = self.one_current_candidate()
        matcher = openclaw_food_resolution.OpenClawFoodSemanticMatcher(
            "/test-only/openclaw",
            runner=FakeOpenClawRunner(
                returncode=1,
                stderr="ECONNREFUSED private gateway token",
            ),
        )

        with self.assertRaisesRegex(
            FoodSemanticGatewayUnavailableError,
            "gateway is unavailable",
        ) as raised:
            matcher.decide(FoodResolutionRequest("private wording", DAY1), (candidate,))

        self.assertNotIn("private gateway token", str(raised.exception))
        self.assertNotIn("private wording", str(raised.exception))

    def test_api_key_is_not_required_or_forwarded(self) -> None:
        candidate = self.one_current_candidate()
        runner = FakeOpenClawRunner()
        matcher = openclaw_food_resolution.OpenClawFoodSemanticMatcher.from_env(
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
            matcher.decide(FoodResolutionRequest("tandoori chicken", DAY1), (candidate,))

        child_env = runner.calls[0][1]["env"]
        self.assertNotIn("OPENAI_API_KEY", child_env)
        self.assertNotIn("NUTRITION_OPTIMIZER_OPENAI_API_KEY", child_env)
        self.assertNotIn("test-only-key", runner.calls[0][0][-1])

    def test_adapter_has_no_diningbucket_or_direct_openai_sdk_dependency(self) -> None:
        source = Path(openclaw_food_resolution.__file__).read_text(encoding="utf-8")

        self.assertNotIn("DiningBucket", source)
        self.assertNotIn("OpenAISemantic", source)
        self.assertNotIn("OpenAI(", source)


if __name__ == "__main__":
    unittest.main()
