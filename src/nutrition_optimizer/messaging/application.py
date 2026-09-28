"""Application-level handling for inbound Nutrition Optimizer messages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
import logging
import os
from typing import Protocol

from ..semantic import (
    RecordIntakeIntent,
    SemanticInterpretation,
    SemanticInterpreter,
    UnsupportedIntent,
    interpret_text,
)
from .webhook import IncomingMessage


CHAT_GUID_ENV_VAR = "NUTRITION_OPTIMIZER_BLUEBUBBLES_CHAT_GUID"
SENDER_ADDRESS_ENV_VAR = "NUTRITION_OPTIMIZER_BLUEBUBBLES_SENDER_ADDRESS"
SEMANTIC_FAILURE_REPLY_TEXT = (
    "I couldn't interpret that message right now. Please try again."
)
UNSUPPORTED_INTENT_REPLY_TEXT = (
    "I can currently understand food intake messages, but I can't handle that request yet."
)
# Kept as a compatibility alias for callers that imported the old name.  The
# old generic acknowledgment is no longer sent by the application handler.
DEFAULT_REPLY_TEXT = SEMANTIC_FAILURE_REPLY_TEXT

_LOGGER = logging.getLogger(__name__)


class OutboundTextSender(Protocol):
    """The outbound capability required by the application boundary."""

    def send_text(self, chat_guid: str, message: str) -> object:
        """Send text to a chat without exposing transport details here."""


class ApplicationMessageError(RuntimeError):
    """Raised when a valid inbound message cannot be processed."""


@dataclass(frozen=True, slots=True)
class ApplicationMessagingConfig:
    """Runtime identity filters for inbound application messages."""

    expected_chat_guid: str | None = None
    expected_sender_address: str | None = None

    def __post_init__(self) -> None:
        if self.expected_chat_guid is None and self.expected_sender_address is None:
            raise ValueError(
                f"one of {CHAT_GUID_ENV_VAR} or {SENDER_ADDRESS_ENV_VAR} must be set"
            )
        for value, name in (
            (self.expected_chat_guid, CHAT_GUID_ENV_VAR),
            (self.expected_sender_address, SENDER_ADDRESS_ENV_VAR),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when set")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> ApplicationMessagingConfig:
        """Load identity filters and reject an unscoped runtime."""

        source = os.environ if environ is None else environ
        expected_chat_guid = _environment_value(source, CHAT_GUID_ENV_VAR)
        expected_sender_address = _environment_value(source, SENDER_ADDRESS_ENV_VAR)
        if expected_chat_guid is None and expected_sender_address is None:
            raise ValueError(
                f"one of {CHAT_GUID_ENV_VAR} or {SENDER_ADDRESS_ENV_VAR} must be set"
            )
        return cls(expected_chat_guid, expected_sender_address)

    def accepts(self, message: IncomingMessage) -> bool:
        """Return whether an inbound text belongs to this application."""

        if message.is_from_me is not False:
            return False
        if not isinstance(message.text, str) or not message.text.strip():
            return False
        if (
            self.expected_chat_guid is not None
            and message.chat_guid != self.expected_chat_guid
        ):
            return False
        if (
            self.expected_sender_address is not None
            and message.sender_address != self.expected_sender_address
        ):
            return False
        return True


class NutritionOptimizerMessageHandler:
    """Filter inbound messages and reply to validated semantic results."""

    def __init__(
        self,
        outbound_sender: OutboundTextSender,
        config: ApplicationMessagingConfig,
        *,
        semantic_interpreter: SemanticInterpreter,
        logger: logging.Logger | None = None,
    ) -> None:
        if not callable(getattr(outbound_sender, "send_text", None)):
            raise TypeError("outbound_sender must provide send_text")
        if not isinstance(config, ApplicationMessagingConfig):
            raise TypeError("config must be ApplicationMessagingConfig")
        if not callable(getattr(semantic_interpreter, "interpret", None)):
            raise TypeError("semantic_interpreter must provide interpret")

        self.outbound_sender = outbound_sender
        self.config = config
        self.semantic_interpreter = semantic_interpreter
        self._logger = logger or _LOGGER

    def __call__(self, message: IncomingMessage) -> None:
        """Handle one normalized inbound message without exposing its text."""

        self.handle(message)

    def handle(self, message: IncomingMessage) -> bool:
        """Send one reply, returning whether an outbound attempt succeeded."""

        if not self.config.accepts(message):
            return False

        chat_guid = message.chat_guid
        if not isinstance(chat_guid, str) or not chat_guid:
            self._logger.warning(
                "Nutrition Optimizer message ignored (missing chat GUID)"
            )
            return False

        try:
            interpretation = interpret_text(self.semantic_interpreter, message.text)
            reply_text = _reply_for_interpretation(interpretation)
        except Exception as exc:
            # The application boundary intentionally exposes only a fixed
            # fallback.  Do not log user text, model output, or credentials.
            self._logger.warning(
                "Nutrition Optimizer semantic interpretation failed (%s)",
                type(exc).__name__,
            )
            reply_text = SEMANTIC_FAILURE_REPLY_TEXT

        try:
            self.outbound_sender.send_text(chat_guid, reply_text)
        except Exception as exc:
            # Log only the exception type, then raise a sanitized failure so
            # the transport can leave dedupe/cursor state retryable.
            self._logger.warning(
                "Nutrition Optimizer outbound reply failed (%s)",
                type(exc).__name__,
            )
            raise ApplicationMessageError("outbound reply failed") from None
        return True


def _reply_for_interpretation(interpretation: SemanticInterpretation) -> str:
    """Render one validated semantic result without using a language model."""

    if isinstance(interpretation, RecordIntakeIntent):
        if interpretation.servings is None or interpretation.clarification_required is not None:
            return f"How many servings of {interpretation.food_text} did you have?"
        return (
            f"I understood: {interpretation.food_text} — "
            f"{_format_servings(interpretation.servings)} servings."
        )

    if isinstance(interpretation, UnsupportedIntent):
        return UNSUPPORTED_INTENT_REPLY_TEXT

    raise TypeError("semantic interpreter returned an unsupported result")


def _format_servings(servings: Decimal) -> str:
    """Render a validated Decimal quantity without unnecessary trailing zeros."""

    rendered = format(servings, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _environment_value(source: Mapping[str, str], name: str) -> str | None:
    value = source.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string when set")
    return value
