"""OpenClaw-backed implementation of the semantic interpretation boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import errno
import json
import math
import os
import shutil
import subprocess
from typing import Any

from .semantic import SemanticInterpretation
from .semantic_structured import (
    SEMANTIC_SYSTEM_PROMPT,
    SemanticStructuredValidationError,
    interpret_structured_result,
)


__all__ = [
    "DEFAULT_OPENCLAW_PATH",
    "DEFAULT_OPENCLAW_SUBPROCESS_TIMEOUT_SECONDS",
    "OPENCLAW_GATEWAY_TIMEOUT_MS",
    "OPENCLAW_MODEL",
    "OPENCLAW_MODEL_REF",
    "OPENCLAW_PATH_ENV_VAR",
    "OPENCLAW_PROVIDER",
    "OPENCLAW_SEMANTIC_SCHEMA",
    "OPENCLAW_THINKING",
    "OpenClawSemanticConfigurationError",
    "OpenClawSemanticError",
    "OpenClawSemanticGatewayUnavailableError",
    "OpenClawSemanticInputError",
    "OpenClawSemanticInterpreter",
    "OpenClawSemanticInvocationError",
    "OpenClawSemanticResponseError",
    "OpenClawSemanticStructuredOutputError",
    "OpenClawSemanticTimeoutError",
]


DEFAULT_OPENCLAW_PATH = "/home/luke/.openclaw/bin/openclaw"
DEFAULT_OPENCLAW_SUBPROCESS_TIMEOUT_SECONDS = 65.0
OPENCLAW_GATEWAY_TIMEOUT_MS = 60000
OPENCLAW_PATH_ENV_VAR = "NUTRITION_OPTIMIZER_OPENCLAW_PATH"
OPENCLAW_PROVIDER = "openai"
OPENCLAW_MODEL = "gpt-5.6-luna"
OPENCLAW_MODEL_REF = f"{OPENCLAW_PROVIDER}/{OPENCLAW_MODEL}"
OPENCLAW_THINKING = "high"

OPENCLAW_SEMANTIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["record_intake", "unsupported"],
        },
        "food_text": {"type": ["string", "null"]},
        "servings": {"type": ["string", "null"]},
        "clarification_required": {"type": "boolean"},
        "reason": {"type": ["string", "null"]},
    },
    "required": [
        "intent",
        "food_text",
        "servings",
        "clarification_required",
        "reason",
    ],
    "additionalProperties": False,
}

OPENCLAW_SEMANTIC_PROMPT = (
    f"{SEMANTIC_SYSTEM_PROMPT} "
    "Match the supplied JSON Schema exactly. For record_intake, set reason "
    "to null. For unsupported, set food_text and servings to null, set "
    "clarification_required to false, and provide the stable reason "
    "unsupported_request."
)

_BLOCKED_CHILD_ENVIRONMENT = frozenset(
    {
        "OPENAI_API_KEY",
        "NUTRITION_OPTIMIZER_OPENAI_API_KEY",
        "NUTRITION_OPTIMIZER_OPENAI_MODEL",
    }
)
_MISSING = object()
_CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class OpenClawSemanticError(RuntimeError):
    """Base class for sanitized OpenClaw semantic interpreter failures."""


class OpenClawSemanticConfigurationError(OpenClawSemanticError):
    """Raised when the local OpenClaw executable configuration is invalid."""


class OpenClawSemanticInputError(OpenClawSemanticError):
    """Raised when the interpreter receives empty or invalid user text."""


class OpenClawSemanticInvocationError(OpenClawSemanticError):
    """Raised when the local OpenClaw command cannot complete."""


class OpenClawSemanticTimeoutError(OpenClawSemanticInvocationError):
    """Raised when the local OpenClaw command exceeds its timeout."""


class OpenClawSemanticGatewayUnavailableError(OpenClawSemanticInvocationError):
    """Raised when a local gateway connection is clearly unavailable."""


class OpenClawSemanticResponseError(OpenClawSemanticError):
    """Raised when the OpenClaw envelope or semantic result is unusable."""


class OpenClawSemanticStructuredOutputError(OpenClawSemanticResponseError):
    """Raised when ``llm-task`` rejects model output against its schema."""


class _OpenClawStructuredTaskInvoker:
    """Authenticated local OpenClaw ``llm-task`` transport.

    The OpenClaw CLI owns gateway authentication and the ChatGPT/Codex OAuth
    profile.  Narrow adapters supply one prompt, one JSON input, and one strict
    schema; this transport deliberately exposes no general tool surface.
    """

    def __init__(
        self,
        executable_path: str = DEFAULT_OPENCLAW_PATH,
        *,
        runner: _CommandRunner | None = None,
        subprocess_timeout_seconds: float = DEFAULT_OPENCLAW_SUBPROCESS_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(executable_path, str) or not executable_path.strip():
            raise OpenClawSemanticConfigurationError(
                f"{OPENCLAW_PATH_ENV_VAR} must be a non-empty executable path"
            )
        if (
            isinstance(subprocess_timeout_seconds, bool)
            or not isinstance(subprocess_timeout_seconds, (int, float))
            or not math.isfinite(float(subprocess_timeout_seconds))
            or subprocess_timeout_seconds <= 0
        ):
            raise OpenClawSemanticConfigurationError(
                "OpenClaw subprocess timeout must be finite and greater than zero"
            )

        self.executable_path = os.path.expanduser(executable_path.strip())
        self.provider = OPENCLAW_PROVIDER
        self.model = OPENCLAW_MODEL_REF
        self.thinking = OPENCLAW_THINKING
        self.gateway_timeout_ms = OPENCLAW_GATEWAY_TIMEOUT_MS
        self.subprocess_timeout_seconds = float(subprocess_timeout_seconds)
        self._runner = runner or subprocess.run

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        runner: _CommandRunner | None = None,
    ) -> "_OpenClawStructuredTaskInvoker":
        """Build an invoker using only the optional executable override."""

        environment = os.environ if environ is None else environ
        executable_path = environment.get(
            OPENCLAW_PATH_ENV_VAR,
            DEFAULT_OPENCLAW_PATH,
        )
        if not isinstance(executable_path, str) or not executable_path.strip():
            raise OpenClawSemanticConfigurationError(
                f"{OPENCLAW_PATH_ENV_VAR} must be a non-empty executable path"
            )
        return cls(executable_path, runner=runner)

    def invoke(
        self,
        *,
        prompt: str,
        input_text: str,
        schema: Mapping[str, Any],
    ) -> object:
        """Run one schema-constrained task through the configured OAuth runtime."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise OpenClawSemanticInputError("OpenClaw task prompt must be non-empty")
        if not isinstance(input_text, str) or not input_text.strip():
            raise OpenClawSemanticInputError("OpenClaw task input must be non-empty")
        if not isinstance(schema, Mapping):
            raise OpenClawSemanticInputError("OpenClaw task schema must be an object")

        command = self._build_task_command(
            # OpenClaw's llm-task plugin uses ``schema`` for post-generation
            # validation.  Include the same schema in the prompt so the model
            # sees the literal required field names before it generates JSON.
            prompt=_prompt_with_schema_contract(prompt, schema),
            input_text=input_text,
            schema=schema,
        )
        try:
            completed = self._runner(
                command,
                capture_output=True,
                check=False,
                env=_child_environment_without_api_keys(),
                text=True,
                timeout=self.subprocess_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            raise OpenClawSemanticTimeoutError(
                "OpenClaw structured task timed out"
            ) from None
        except FileNotFoundError:
            raise OpenClawSemanticConfigurationError(
                "OpenClaw executable is unavailable"
            ) from None
        except OSError as error:
            if error.errno in {
                errno.ECONNREFUSED,
                errno.ECONNRESET,
                errno.ENOTCONN,
                errno.EPIPE,
            }:
                raise OpenClawSemanticGatewayUnavailableError(
                    "OpenClaw gateway is unavailable"
                ) from None
            raise OpenClawSemanticInvocationError(
                "OpenClaw task request failed"
            ) from None

        if completed.returncode != 0:
            if _looks_like_gateway_unavailable(completed.stderr):
                raise OpenClawSemanticGatewayUnavailableError(
                    "OpenClaw gateway is unavailable"
                ) from None
            raise OpenClawSemanticInvocationError(
                "OpenClaw task request failed"
            )

        return _extract_payload(completed.stdout)

    def ensure_executable_available(self) -> None:
        """Fail clearly before service startup when the CLI cannot be found."""

        if shutil.which(self.executable_path) is None:
            raise OpenClawSemanticConfigurationError(
                "OpenClaw executable is unavailable"
            )

    def _build_task_command(
        self,
        *,
        prompt: str,
        input_text: str,
        schema: Mapping[str, Any],
    ) -> list[str]:
        params = {
            "name": "llm-task",
            "sessionKey": "main",
            "args": {
                "provider": self.provider,
                "model": self.model,
                "thinking": self.thinking,
                "prompt": prompt,
                "input": input_text,
                "schema": dict(schema),
            },
        }
        try:
            serialized_params = json.dumps(params, separators=(",", ":"))
        except (TypeError, ValueError):
            raise OpenClawSemanticInputError("OpenClaw task schema is invalid") from None
        return [
            self.executable_path,
            "gateway",
            "call",
            "tools.invoke",
            "--json",
            "--timeout",
            str(self.gateway_timeout_ms),
            "--params",
            serialized_params,
        ]


class OpenClawSemanticInterpreter(_OpenClawStructuredTaskInvoker):
    """Interpret narrow intake language through local OpenClaw ``llm-task``."""

    def interpret(self, user_text: str) -> SemanticInterpretation:
        """Return a validated semantic result without executing it."""

        if not isinstance(user_text, str) or not user_text.strip():
            raise OpenClawSemanticInputError("user text must be non-empty")

        payload = self.invoke(
            prompt=OPENCLAW_SEMANTIC_PROMPT,
            input_text=user_text,
            schema=OPENCLAW_SEMANTIC_SCHEMA,
        )
        try:
            return interpret_structured_result(payload)
        except SemanticStructuredValidationError as error:
            raise OpenClawSemanticResponseError(
                f"OpenClaw {error}"
            ) from None

    def _build_command(self, user_text: str) -> list[str]:
        """Compatibility helper for existing semantic command-level tests."""

        return self._build_task_command(
            prompt=OPENCLAW_SEMANTIC_PROMPT,
            input_text=user_text,
            schema=OPENCLAW_SEMANTIC_SCHEMA,
        )


def _child_environment_without_api_keys() -> dict[str, str]:
    """Keep provider API-key variables out of the OpenClaw child process."""

    return {
        key: value
        for key, value in os.environ.items()
        if key not in _BLOCKED_CHILD_ENVIRONMENT
    }


def _prompt_with_schema_contract(prompt: str, schema: Mapping[str, Any]) -> str:
    """Make llm-task's post-generation schema visible to the model too."""

    try:
        serialized_schema = json.dumps(
            dict(schema),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        raise OpenClawSemanticInputError("OpenClaw task schema is invalid") from None

    return (
        f"{prompt.rstrip()}\n\n"
        "OUTPUT CONTRACT: Return exactly one JSON object that validates against "
        "the following JSON Schema. Its property names, required fields, allowed "
        "values, nullability, and additionalProperties rule are literal. Do not "
        "rename, omit, or add fields.\n"
        f"JSON_SCHEMA:{serialized_schema}"
    )


def _looks_like_gateway_unavailable(diagnostic: object) -> bool:
    """Classify known local connection failures without exposing diagnostics."""

    if not isinstance(diagnostic, str):
        return False
    normalized = diagnostic.casefold()
    return any(
        marker in normalized
        for marker in (
            "econnrefused",
            "connection refused",
            "econnreset",
            "connection reset",
            "enotconn",
            "not connected",
            "gateway closed",
        )
    )


def _extract_payload(stdout: object) -> object:
    if not isinstance(stdout, str):
        raise OpenClawSemanticResponseError(
            "OpenClaw returned an unexpected response envelope"
        )
    try:
        envelope = json.loads(stdout)
    except (TypeError, json.JSONDecodeError):
        raise OpenClawSemanticResponseError(
            "OpenClaw returned malformed JSON"
        ) from None

    if not isinstance(envelope, Mapping):
        raise OpenClawSemanticResponseError(
            "OpenClaw returned an unexpected response envelope"
        )
    if envelope.get("ok") is not True or envelope.get("toolName") != "llm-task":
        error = envelope.get("error")
        message = error.get("message") if isinstance(error, Mapping) else None
        if isinstance(message, str) and "llm json did not match schema" in message.casefold():
            raise OpenClawSemanticStructuredOutputError(
                "OpenClaw semantic structured output was rejected"
            )
        if _looks_like_gateway_unavailable(message):
            raise OpenClawSemanticGatewayUnavailableError(
                "OpenClaw gateway is unavailable"
            )
        raise OpenClawSemanticInvocationError(
            "OpenClaw semantic tool invocation failed"
        )
    if envelope.get("source") != "plugin":
        raise OpenClawSemanticResponseError(
            "OpenClaw returned an unexpected response envelope"
        )

    output = envelope.get("output")
    if not isinstance(output, Mapping):
        raise OpenClawSemanticResponseError(
            "OpenClaw returned an unexpected response envelope"
        )
    details = output.get("details")
    if not isinstance(details, Mapping):
        raise OpenClawSemanticResponseError(
            "OpenClaw returned an unexpected response envelope"
        )
    if (
        details.get("provider") != OPENCLAW_PROVIDER
        or details.get("model") != OPENCLAW_MODEL
    ):
        raise OpenClawSemanticResponseError(
            "OpenClaw used an unexpected semantic model"
        )
    payload = details.get("json", _MISSING)
    if payload is _MISSING:
        raise OpenClawSemanticResponseError(
            "OpenClaw returned an unexpected response envelope"
        )
    return payload
