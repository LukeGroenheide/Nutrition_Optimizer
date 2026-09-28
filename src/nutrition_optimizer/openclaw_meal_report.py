"""OpenClaw/Luna adapter for structured known-meal report parsing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
import subprocess

from .meal_report import MealPlan
from .meal_report_structured import (
    MEAL_REPORT_STRUCTURED_SCHEMA,
    MEAL_REPORT_SYSTEM_PROMPT,
    MealReportSemanticResult,
    MealReportStructuredValidationError,
    interpret_meal_report_structured_result,
)
from .meal_request_structured import (
    MEAL_REQUEST_STRUCTURED_SCHEMA,
    MEAL_REQUEST_SYSTEM_PROMPT,
    MealRequestSemanticResult,
    MealRequestStructuredValidationError,
    interpret_meal_request_structured_result,
)
from .meal_report_routing import (
    MEAL_REPORT_ROUTING_STRUCTURED_SCHEMA,
    MEAL_REPORT_ROUTING_SYSTEM_PROMPT,
    MealReportRoutingSemanticResult,
    MealReportRoutingStructuredValidationError,
    interpret_meal_report_routing_structured_result,
)
from .openclaw_semantic import (
    DEFAULT_OPENCLAW_PATH,
    DEFAULT_OPENCLAW_SUBPROCESS_TIMEOUT_SECONDS,
    OPENCLAW_PATH_ENV_VAR,
    OpenClawSemanticConfigurationError,
    OpenClawSemanticError,
    OpenClawSemanticGatewayUnavailableError,
    OpenClawSemanticInputError,
    OpenClawSemanticInvocationError,
    OpenClawSemanticResponseError,
    OpenClawSemanticStructuredOutputError,
    OpenClawSemanticTimeoutError,
    _OpenClawStructuredTaskInvoker,
)


__all__ = [
    "OPENCLAW_MEAL_REPORT_PROMPT",
    "OPENCLAW_MEAL_REPORT_ROUTING_PROMPT",
    "OPENCLAW_MEAL_REQUEST_PROMPT",
    "OpenClawMealReportError",
    "OpenClawMealReportInterpreter",
]


OPENCLAW_MEAL_REPORT_PROMPT = (
    f"{MEAL_REPORT_SYSTEM_PROMPT} "
    "Match the supplied JSON Schema exactly. Return arrays even when they are empty."
)
OPENCLAW_MEAL_REPORT_ROUTING_PROMPT = (
    f"{MEAL_REPORT_ROUTING_SYSTEM_PROMPT} "
    "Match the supplied JSON Schema exactly."
)
OPENCLAW_MEAL_REQUEST_PROMPT = (
    f"{MEAL_REQUEST_SYSTEM_PROMPT} "
    "Match the supplied JSON Schema exactly. Return arrays even when they are empty."
)

_CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class OpenClawMealReportError(RuntimeError):
    """Sanitized failure from the meal-report model boundary."""


class OpenClawMealReportInterpreter:
    """Parse only supplied plan references through shared OAuth OpenClaw transport."""

    def __init__(
        self,
        executable_path: str = DEFAULT_OPENCLAW_PATH,
        *,
        runner: _CommandRunner | None = None,
        subprocess_timeout_seconds: float = DEFAULT_OPENCLAW_SUBPROCESS_TIMEOUT_SECONDS,
    ) -> None:
        self._task = _OpenClawStructuredTaskInvoker(
            executable_path,
            runner=runner,
            subprocess_timeout_seconds=subprocess_timeout_seconds,
        )
        self.executable_path = self._task.executable_path
        self.provider = self._task.provider
        self.model = self._task.model
        self.thinking = self._task.thinking
        self.gateway_timeout_ms = self._task.gateway_timeout_ms
        self.subprocess_timeout_seconds = self._task.subprocess_timeout_seconds

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        runner: _CommandRunner | None = None,
    ) -> "OpenClawMealReportInterpreter":
        environment = os.environ if environ is None else environ
        return cls(environment.get(OPENCLAW_PATH_ENV_VAR, DEFAULT_OPENCLAW_PATH), runner=runner)

    def ensure_executable_available(self) -> None:
        self._task.ensure_executable_available()

    def interpret(self, plan: MealPlan, user_text: str) -> MealReportSemanticResult:
        """Return a schema-validated self-contained parse without food resolution."""

        return self._interpret(plan, user_text, conversation_context=None)

    def interpret_with_context(
        self,
        plan: MealPlan,
        user_text: str,
        conversation_context: Mapping[str, object],
    ) -> MealReportSemanticResult:
        """Interpret one turn with compact durable draft context.

        The context consists only of plan-visible food/station facts, resolved
        item identities, and outstanding user-facing questions.  It never
        includes nutrition, official serving multipliers, targets, or model
        generated reasoning.
        """

        if not isinstance(conversation_context, Mapping):
            raise OpenClawMealReportError("meal report conversation context is invalid")
        return self._interpret(plan, user_text, conversation_context=conversation_context)

    def interpret_meal_request(self, user_text: str) -> MealRequestSemanticResult:
        """Classify a no-active-plan message without exposing menu or nutrition."""

        if not isinstance(user_text, str) or not user_text.strip():
            raise OpenClawMealReportError("meal request text must be non-empty")
        input_text = json.dumps(
            {"user_message": user_text},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            payload = self._task.invoke(
                prompt=OPENCLAW_MEAL_REQUEST_PROMPT,
                input_text=input_text,
                schema=MEAL_REQUEST_STRUCTURED_SCHEMA,
            )
        except (
            OpenClawSemanticConfigurationError,
            OpenClawSemanticGatewayUnavailableError,
            OpenClawSemanticInputError,
            OpenClawSemanticInvocationError,
            OpenClawSemanticResponseError,
            OpenClawSemanticStructuredOutputError,
            OpenClawSemanticTimeoutError,
            OpenClawSemanticError,
        ):
            raise OpenClawMealReportError("OpenClaw meal request interpretation failed") from None
        try:
            return interpret_meal_request_structured_result(payload)
        except MealRequestStructuredValidationError:
            raise OpenClawMealReportError(
                "OpenClaw meal request returned invalid output"
            ) from None

    def interpret_report_routing(
        self,
        user_text: str,
    ) -> MealReportRoutingSemanticResult:
        """Recognize explicit report scope without selecting or exposing a plan."""

        if not isinstance(user_text, str) or not user_text.strip():
            raise OpenClawMealReportError("meal report routing text must be non-empty")
        input_text = json.dumps(
            {"user_message": user_text},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            payload = self._task.invoke(
                prompt=OPENCLAW_MEAL_REPORT_ROUTING_PROMPT,
                input_text=input_text,
                schema=MEAL_REPORT_ROUTING_STRUCTURED_SCHEMA,
            )
        except (
            OpenClawSemanticConfigurationError,
            OpenClawSemanticGatewayUnavailableError,
            OpenClawSemanticInputError,
            OpenClawSemanticInvocationError,
            OpenClawSemanticResponseError,
            OpenClawSemanticStructuredOutputError,
            OpenClawSemanticTimeoutError,
            OpenClawSemanticError,
        ):
            raise OpenClawMealReportError(
                "OpenClaw meal report routing interpretation failed"
            ) from None
        try:
            return interpret_meal_report_routing_structured_result(payload)
        except MealReportRoutingStructuredValidationError:
            raise OpenClawMealReportError(
                "OpenClaw meal report routing returned invalid output"
            ) from None

    def _interpret(
        self,
        plan: MealPlan,
        user_text: str,
        *,
        conversation_context: Mapping[str, object] | None,
    ) -> MealReportSemanticResult:
        """Invoke the existing structured OpenClaw task with optional context."""

        if not isinstance(plan, MealPlan):
            raise OpenClawMealReportError("meal plan is invalid")
        if not isinstance(user_text, str) or not user_text.strip():
            raise OpenClawMealReportError("user text must be non-empty")
        payload: dict[str, object] = {
            "plan_items": plan.model_visible_items(),
            "user_message": user_text,
        }
        if conversation_context is not None:
            payload["conversation_context"] = dict(conversation_context)
        input_text = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            payload = self._task.invoke(
                prompt=OPENCLAW_MEAL_REPORT_PROMPT,
                input_text=input_text,
                schema=MEAL_REPORT_STRUCTURED_SCHEMA,
            )
        except (
            OpenClawSemanticConfigurationError,
            OpenClawSemanticGatewayUnavailableError,
            OpenClawSemanticInputError,
            OpenClawSemanticInvocationError,
            OpenClawSemanticResponseError,
            OpenClawSemanticStructuredOutputError,
            OpenClawSemanticTimeoutError,
            OpenClawSemanticError,
        ):
            raise OpenClawMealReportError("OpenClaw meal report interpretation failed") from None
        try:
            return interpret_meal_report_structured_result(payload)
        except MealReportStructuredValidationError:
            raise OpenClawMealReportError(
                "OpenClaw meal report returned invalid output"
            ) from None
