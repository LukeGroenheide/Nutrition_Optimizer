"""OpenClaw/Luna fallback for reverse-direction recommendation rendering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
from .recommendation_rendering import (
    RecommendedPortionRenderRequest,
    RecommendedPortionSemanticResult,
    RecommendedPortionRenderingError,
    RecommendedPortionRenderingOutputError,
    RecommendedPortionRenderingRuntimeUnavailableError,
    RecommendedPortionRenderingTransportError,
    _validate_semantic_rendering,
)
from .physical_quantity import format_physical_amount
from .recommendation_rendering_structured import (
    RECOMMENDED_PORTION_RENDERING_SCHEMA,
    RECOMMENDED_PORTION_RENDERING_SYSTEM_PROMPT,
    RecommendedPortionRenderingStructuredValidationError,
    interpret_recommended_portion_structured_result,
)


__all__ = [
    "OPENCLAW_RECOMMENDATION_RENDERING_PROMPT",
    "OpenClawRecommendedPortionRenderer",
    "OpenClawRecommendedPortionSemanticRenderer",
]


OPENCLAW_RECOMMENDATION_RENDERING_PROMPT = (
    f"{RECOMMENDED_PORTION_RENDERING_SYSTEM_PROMPT} "
    "Match the supplied JSON Schema exactly. Return no fields other than the "
    "three required presentation fields."
)

_CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class OpenClawRecommendedPortionRenderer:
    """Use the shared local OAuth-backed OpenClaw structured-task transport.

    This class is the semantic half of ``RecommendedPortionRenderer``. It has
    no catalog, optimizer, nutrition arithmetic, ledger, persistence, or
    messaging dependency. The outer renderer should be used when deterministic
    count-unit rules should run before Luna.
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
    ) -> "OpenClawRecommendedPortionRenderer":
        """Build from the optional local executable override only."""

        environment = os.environ if environ is None else environ
        return cls(
            environment.get(OPENCLAW_PATH_ENV_VAR, DEFAULT_OPENCLAW_PATH),
            runner=runner,
        )

    def ensure_executable_available(self) -> None:
        """Fail clearly before enabling the optional semantic fallback."""

        self._task.ensure_executable_available()

    def render(
        self,
        request: RecommendedPortionRenderRequest,
    ) -> RecommendedPortionSemanticResult:
        """Render only the fixed physical quantity in the request."""

        if not isinstance(request, RecommendedPortionRenderRequest):
            raise RecommendedPortionRenderingError(
                "recommendation rendering request is invalid"
            )
        quantity = request.physical_quantity
        canonical_quantity: dict[str, object] = {
            "amount": str(quantity.amount),
            "unit": quantity.unit,
            "category": quantity.category,
            "description": format_physical_amount(quantity),
        }
        if quantity.count_unit is not None:
            canonical_quantity["count_unit"] = quantity.count_unit
        input_payload: dict[str, object] = {
            "official_food_display_name": request.official_food_display_name,
            "canonical_physical_quantity": canonical_quantity,
            "approved_descriptive_options": request.approved_descriptive_options,
        }
        input_text = json.dumps(
            input_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            payload = self._task.invoke(
                prompt=OPENCLAW_RECOMMENDATION_RENDERING_PROMPT,
                input_text=input_text,
                schema=RECOMMENDED_PORTION_RENDERING_SCHEMA,
            )
        except OpenClawSemanticTimeoutError:
            raise RecommendedPortionRenderingTransportError(
                "OpenClaw recommendation rendering timed out"
            ) from None
        except OpenClawSemanticGatewayUnavailableError:
            raise RecommendedPortionRenderingTransportError(
                "OpenClaw recommendation-rendering gateway is unavailable"
            ) from None
        except OpenClawSemanticStructuredOutputError:
            raise RecommendedPortionRenderingOutputError(
                "OpenClaw recommendation rendering returned invalid output"
            ) from None
        except OpenClawSemanticResponseError:
            raise RecommendedPortionRenderingOutputError(
                "OpenClaw recommendation rendering returned invalid output"
            ) from None
        except OpenClawSemanticConfigurationError:
            raise RecommendedPortionRenderingRuntimeUnavailableError(
                "OpenClaw recommendation-rendering runtime is unavailable"
            ) from None
        except OpenClawSemanticInvocationError:
            raise RecommendedPortionRenderingTransportError(
                "OpenClaw recommendation rendering failed"
            ) from None
        except OpenClawSemanticInputError:
            raise RecommendedPortionRenderingError(
                "OpenClaw recommendation-rendering request is invalid"
            ) from None
        except OpenClawSemanticError:
            raise RecommendedPortionRenderingError(
                "OpenClaw recommendation rendering failed"
            ) from None

        try:
            result = interpret_recommended_portion_structured_result(payload)
            _validate_semantic_rendering(result, request)
            return result
        except RecommendedPortionRenderingStructuredValidationError:
            raise RecommendedPortionRenderingOutputError(
                "OpenClaw recommendation rendering returned invalid output"
            ) from None
        except RecommendedPortionRenderingOutputError:
            raise


# The longer name makes the direction explicit at composition sites; the
# shorter name is convenient and mirrors the existing OpenClaw adapters.
OpenClawRecommendedPortionSemanticRenderer = OpenClawRecommendedPortionRenderer
