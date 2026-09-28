"""OpenAI-backed implementation of the semantic interpretation boundary.

This module translates user text into the small structured result defined by
``nutrition_optimizer.semantic``. It deliberately stops before nutrition
resolution or ledger execution.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Protocol

from openai import OpenAI

from .semantic import (
    SemanticInterpretation,
)
from .semantic_structured import (
    SEMANTIC_SYSTEM_PROMPT,
    SemanticStructuredValidationError,
    StructuredSemanticResult as _StructuredSemanticResult,
    interpret_structured_result,
)


__all__ = [
    "DEFAULT_OPENAI_MODEL",
    "OPENAI_API_KEY_ENV_VAR",
    "OPENAI_MODEL_ENV_VAR",
    "OpenAISemanticAPIError",
    "OpenAISemanticConfigurationError",
    "OpenAISemanticError",
    "OpenAISemanticInterpreter",
    "OpenAISemanticInputError",
    "OpenAISemanticRefusalError",
    "OpenAISemanticResponseError",
    "SEMANTIC_SYSTEM_PROMPT",
]


OPENAI_API_KEY_ENV_VAR = "NUTRITION_OPTIMIZER_OPENAI_API_KEY"
OPENAI_MODEL_ENV_VAR = "NUTRITION_OPTIMIZER_OPENAI_MODEL"
DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"


class OpenAISemanticError(RuntimeError):
    """Base class for sanitized OpenAI semantic interpreter failures."""


class OpenAISemanticConfigurationError(OpenAISemanticError):
    """Raised when runtime OpenAI configuration is missing or invalid."""


class OpenAISemanticInputError(OpenAISemanticError):
    """Raised when the interpreter receives empty or invalid user text."""


class OpenAISemanticAPIError(OpenAISemanticError):
    """Raised when the OpenAI request cannot be completed."""


class OpenAISemanticRefusalError(OpenAISemanticError):
    """Raised when the model refuses to return a semantic interpretation."""


class OpenAISemanticResponseError(OpenAISemanticError):
    """Raised when a structured response cannot be safely used."""


class _ResponsesAPI(Protocol):
    def parse(self, **kwargs: object) -> object:
        """Return a parsed Structured Outputs response."""


class _OpenAIClient(Protocol):
    responses: _ResponsesAPI


class OpenAISemanticInterpreter:
    """Interpret narrow intake language using OpenAI Structured Outputs.

    The client is injectable so tests and callers can provide an SDK-compatible
    fake without making a network request. Production construction should
    normally use :meth:`from_env`.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        client: _OpenAIClient | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise OpenAISemanticConfigurationError(
                f"{OPENAI_API_KEY_ENV_VAR} must be set"
            )
        if not isinstance(model, str) or not model.strip():
            raise OpenAISemanticConfigurationError(
                f"{OPENAI_MODEL_ENV_VAR} must be non-empty when set"
            )

        self.model = model.strip()
        if client is None:
            try:
                client = OpenAI(api_key=api_key.strip())
            except Exception:
                raise OpenAISemanticConfigurationError(
                    "OpenAI client could not be initialized"
                ) from None
        self._client = client

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        client: _OpenAIClient | None = None,
    ) -> "OpenAISemanticInterpreter":
        """Build an interpreter from the documented runtime environment."""

        environment = os.environ if environ is None else environ
        api_key = environment.get(OPENAI_API_KEY_ENV_VAR)
        if not isinstance(api_key, str) or not api_key.strip():
            raise OpenAISemanticConfigurationError(
                f"{OPENAI_API_KEY_ENV_VAR} must be set"
            )

        model = environment.get(OPENAI_MODEL_ENV_VAR, DEFAULT_OPENAI_MODEL)
        if not isinstance(model, str) or not model.strip():
            raise OpenAISemanticConfigurationError(
                f"{OPENAI_MODEL_ENV_VAR} must be non-empty when set"
            )
        return cls(api_key, model=model, client=client)

    def interpret(self, user_text: str) -> SemanticInterpretation:
        """Return a validated semantic result without executing it."""

        if not isinstance(user_text, str) or not user_text.strip():
            raise OpenAISemanticInputError("user text must be non-empty")

        try:
            response = self._client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": SEMANTIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                text_format=_StructuredSemanticResult,
            )
        except Exception:
            # Do not expose SDK details, credentials, or user text to callers.
            raise OpenAISemanticAPIError(
                "OpenAI semantic request failed"
            ) from None

        return _interpret_structured_response(response)


def _interpret_structured_response(response: object) -> SemanticInterpretation:
    parsed = getattr(response, "output_parsed", _MISSING)
    if parsed is _MISSING:
        raise OpenAISemanticResponseError(
            "OpenAI response did not contain structured output"
        )
    if parsed is None:
        if _response_has_refusal(response):
            raise OpenAISemanticRefusalError(
                "OpenAI refused semantic interpretation"
            )
        raise OpenAISemanticResponseError(
            "OpenAI response did not contain structured output"
        )

    try:
        return interpret_structured_result(parsed)
    except SemanticStructuredValidationError as error:
        raise OpenAISemanticResponseError(
            f"OpenAI {error}"
        ) from None


def _response_has_refusal(response: object) -> bool:
    direct_refusal = getattr(response, "refusal", None)
    if direct_refusal:
        return True

    for output_item in getattr(response, "output", ()) or ():
        if getattr(output_item, "type", None) == "refusal":
            return True
        for content in getattr(output_item, "content", ()) or ():
            if getattr(content, "type", None) == "refusal":
                return True
            if getattr(content, "refusal", None):
                return True

    for choice in getattr(response, "choices", ()) or ():
        message = getattr(choice, "message", None)
        if getattr(message, "refusal", None):
            return True
    return False


_MISSING = object()
