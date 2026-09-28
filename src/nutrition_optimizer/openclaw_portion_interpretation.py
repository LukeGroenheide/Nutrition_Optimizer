"""OpenClaw/Luna semantic fallback for one exact official serving context."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from decimal import Decimal
import json
import os
import subprocess

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
from .portion_interpretation import (
    PortionInterpretationError,
    PortionInterpretationRequest,
    PortionSemanticDecision,
    PortionSemanticGatewayUnavailableError,
    PortionSemanticOutputError,
    PortionSemanticRuntimeUnavailableError,
    PortionSemanticTimeoutError,
    PortionSemanticTransportError,
)
from .portion_interpretation_structured import (
    PORTION_INTERPRETATION_STRUCTURED_SCHEMA,
    PORTION_INTERPRETATION_SYSTEM_PROMPT,
    PortionInterpretationStructuredValidationError,
    interpret_portion_structured_result,
)


__all__ = [
    "OPENCLAW_PORTION_INTERPRETATION_PROMPT",
    "OpenClawPortionSemanticMatcher",
]


OPENCLAW_PORTION_INTERPRETATION_PROMPT = (
    f"{PORTION_INTERPRETATION_SYSTEM_PROMPT} "
    "Match the supplied JSON Schema exactly. For estimate, provide a plain "
    "positive decimal-string multiplier and confidence. For ambiguous and "
    "no_estimate, estimated_official_servings and confidence must be null."
)

_CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class OpenClawPortionSemanticMatcher:
    """Use the existing local OAuth-backed OpenClaw structured-task transport.

    The adapter receives one immutable ``ResolvedFood`` indirectly through a
    request. It has no catalog, FD network, nutrition arithmetic, ledger,
    messaging, or provider API-key dependency.
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
    ) -> "OpenClawPortionSemanticMatcher":
        """Build from the optional executable override, never an API key."""

        environment = os.environ if environ is None else environ
        executable_path = environment.get(OPENCLAW_PATH_ENV_VAR, DEFAULT_OPENCLAW_PATH)
        return cls(executable_path, runner=runner)

    def ensure_executable_available(self) -> None:
        """Fail clearly before enabling the optional semantic matcher."""

        self._task.ensure_executable_available()

    def decide(self, request: PortionInterpretationRequest) -> PortionSemanticDecision:
        """Estimate only a multiplier of the request's fixed official serving."""

        if not isinstance(request, PortionInterpretationRequest):
            raise PortionInterpretationError("portion-interpretation request is invalid")

        record = request.resolved_food.nutrition_record
        serving = record.serving
        input_text = json.dumps(
            {
                "official_food_display_name": record.name,
                "official_serving": {
                    "quantity": _decimal_text(serving.quantity),
                    "unit": serving.unit,
                    "description": _serving_description(serving.quantity, serving.unit, serving.text),
                },
                "portion_phrase": request.original_quantity_text,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            payload = self._task.invoke(
                prompt=OPENCLAW_PORTION_INTERPRETATION_PROMPT,
                input_text=input_text,
                schema=PORTION_INTERPRETATION_STRUCTURED_SCHEMA,
            )
        except OpenClawSemanticTimeoutError:
            raise PortionSemanticTimeoutError("OpenClaw portion interpretation timed out") from None
        except OpenClawSemanticGatewayUnavailableError:
            raise PortionSemanticGatewayUnavailableError(
                "OpenClaw portion-interpretation gateway is unavailable"
            ) from None
        except OpenClawSemanticStructuredOutputError:
            raise PortionSemanticOutputError(
                "OpenClaw portion interpretation returned invalid output"
            ) from None
        except OpenClawSemanticResponseError:
            raise PortionSemanticOutputError(
                "OpenClaw portion interpretation returned invalid output"
            ) from None
        except OpenClawSemanticConfigurationError:
            raise PortionSemanticRuntimeUnavailableError(
                "OpenClaw portion-interpretation runtime is unavailable"
            ) from None
        except OpenClawSemanticInvocationError:
            raise PortionSemanticTransportError(
                "OpenClaw portion interpretation failed"
            ) from None
        except OpenClawSemanticInputError:
            raise PortionInterpretationError(
                "OpenClaw portion-interpretation request is invalid"
            ) from None
        except OpenClawSemanticError:
            raise PortionInterpretationError("OpenClaw portion interpretation failed") from None

        try:
            return interpret_portion_structured_result(payload)
        except PortionInterpretationStructuredValidationError:
            raise PortionSemanticOutputError(
                "OpenClaw portion interpretation returned invalid output"
            ) from None


def _decimal_text(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _serving_description(
    quantity: Decimal | None,
    unit: str | None,
    text: str | None,
) -> str | None:
    if text is not None:
        return text
    if quantity is not None and unit is not None:
        return f"{quantity} {unit}"
    if quantity is not None:
        return str(quantity)
    return unit
