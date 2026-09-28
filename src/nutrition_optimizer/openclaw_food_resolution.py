"""OpenClaw/Luna semantic fallback for scoped local FD food candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
import subprocess

from .food_resolution import (
    FoodResolutionCandidate,
    FoodResolutionError,
    FoodResolutionRequest,
    FoodSemanticDecision,
    FoodSemanticGatewayUnavailableError,
    FoodSemanticOutputError,
    FoodSemanticRuntimeUnavailableError,
    FoodSemanticTimeoutError,
    FoodSemanticTransportError,
)
from .food_resolution_structured import (
    FOOD_RESOLUTION_STRUCTURED_SCHEMA,
    FOOD_RESOLUTION_SYSTEM_PROMPT,
    FoodResolutionStructuredValidationError,
    interpret_food_resolution_structured_result,
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
    "OPENCLAW_FOOD_RESOLUTION_PROMPT",
    "OpenClawFoodSemanticMatcher",
]


OPENCLAW_FOOD_RESOLUTION_PROMPT = (
    f"{FOOD_RESOLUTION_SYSTEM_PROMPT} "
    "Match the supplied JSON Schema exactly. For match, provide one allowed "
    "component_identity. For ambiguous and no_match, component_identity must be null."
)

_CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class OpenClawFoodSemanticMatcher:
    """Use the existing local OAuth-backed OpenClaw structured-task transport.

    This adapter receives already-scoped local candidates from
    :class:`LocalFDFoodResolver`; it has no database, live FDMealPlanner, or
    provider API-key dependency of its own.
    """

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
        # Expose read-only configuration facts useful to operators and tests;
        # the invocation remains owned by the shared transport.
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
    ) -> "OpenClawFoodSemanticMatcher":
        """Build from the optional executable override, never an API key."""

        environment = os.environ if environ is None else environ
        executable_path = environment.get(OPENCLAW_PATH_ENV_VAR, DEFAULT_OPENCLAW_PATH)
        return cls(executable_path, runner=runner)

    def ensure_executable_available(self) -> None:
        """Fail clearly before enabling the optional semantic fallback."""

        self._task.ensure_executable_available()

    def decide(
        self,
        request: FoodResolutionRequest,
        candidates: tuple[FoodResolutionCandidate, ...],
    ) -> FoodSemanticDecision:
        """Return a strict decision over only the supplied current candidates."""

        if not isinstance(request, FoodResolutionRequest):
            raise FoodResolutionError("food-resolution request is invalid")
        if not isinstance(candidates, tuple) or not candidates or not all(
            isinstance(candidate, FoodResolutionCandidate) for candidate in candidates
        ):
            raise FoodResolutionError("food-resolution candidates are invalid")

        input_text = json.dumps(
            {
                "food_text": request.food_text,
                "service_date": request.service_date.isoformat(),
                "meal": request.meal,
                "station": request.station,
                "candidates": [candidate.semantic_payload() for candidate in candidates],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            payload = self._task.invoke(
                prompt=OPENCLAW_FOOD_RESOLUTION_PROMPT,
                input_text=input_text,
                schema=FOOD_RESOLUTION_STRUCTURED_SCHEMA,
            )
        except OpenClawSemanticTimeoutError:
            raise FoodSemanticTimeoutError("OpenClaw food resolution timed out") from None
        except OpenClawSemanticGatewayUnavailableError:
            raise FoodSemanticGatewayUnavailableError(
                "OpenClaw food-resolution gateway is unavailable"
            ) from None
        except OpenClawSemanticStructuredOutputError:
            raise FoodSemanticOutputError(
                "OpenClaw food resolution returned invalid output"
            ) from None
        except OpenClawSemanticResponseError:
            raise FoodSemanticOutputError(
                "OpenClaw food resolution returned invalid output"
            ) from None
        except OpenClawSemanticConfigurationError:
            raise FoodSemanticRuntimeUnavailableError(
                "OpenClaw food-resolution runtime is unavailable"
            ) from None
        except OpenClawSemanticInvocationError:
            raise FoodSemanticTransportError("OpenClaw food resolution failed") from None
        except OpenClawSemanticInputError:
            raise FoodResolutionError("OpenClaw food-resolution request is invalid") from None
        except OpenClawSemanticError:
            raise FoodResolutionError("OpenClaw food resolution failed") from None

        try:
            return interpret_food_resolution_structured_result(payload)
        except FoodResolutionStructuredValidationError:
            raise FoodResolutionError("OpenClaw food resolution returned invalid output") from None
