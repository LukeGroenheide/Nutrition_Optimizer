"""Small outbound client for the BlueBubbles REST API."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping
from uuid import uuid4

from curl_cffi import requests


DEFAULT_BASE_URL = "http://127.0.0.1:1234"
DEFAULT_TIMEOUT = 10.0
MESSAGE_TEXT_PATH = "/api/v1/message/text"
MESSAGE_QUERY_PATH = "/api/v1/message/query"
DEFAULT_MESSAGE_QUERY_LIMIT = 100
PASSWORD_ENV_VAR = "NUTRITION_OPTIMIZER_BLUEBUBBLES_PASSWORD"
BASE_URL_ENV_VAR = "NUTRITION_OPTIMIZER_BLUEBUBBLES_URL"


class BlueBubblesError(RuntimeError):
    """Base class for BlueBubbles transport failures."""


class BlueBubblesNetworkError(BlueBubblesError):
    """Raised when the BlueBubbles request cannot be completed."""


class BlueBubblesAPIError(BlueBubblesError):
    """Raised when BlueBubbles returns a non-success HTTP status."""


@dataclass(frozen=True, slots=True)
class SendMessageResult:
    """The useful parts of a successful BlueBubbles text response."""

    status_code: int
    chat_guid: str
    temp_guid: str
    payload: Any


class BlueBubblesClient:
    """Send text messages through a BlueBubbles server."""

    def __init__(
        self,
        password: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not isinstance(password, str) or not password:
            raise ValueError("password must be a non-empty string")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")

        self._password = password
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> BlueBubblesClient:
        """Build a client from explicit environment configuration."""

        source = os.environ if environ is None else environ
        password = source.get(PASSWORD_ENV_VAR)
        if not password:
            raise ValueError(f"{PASSWORD_ENV_VAR} must be set")
        return cls(
            password=password,
            base_url=source.get(BASE_URL_ENV_VAR, DEFAULT_BASE_URL),
            timeout=timeout,
        )

    def send_text(
        self,
        chat_guid: str,
        message: str,
        *,
        temp_guid: str | None = None,
    ) -> SendMessageResult:
        """Send ``message`` to a BlueBubbles chat GUID."""

        _require_non_empty_string(chat_guid, "chat_guid")
        _require_non_empty_string(message, "message")
        if temp_guid is None:
            temp_guid = str(uuid4())
        else:
            _require_non_empty_string(temp_guid, "temp_guid")

        body = {
            "chatGuid": chat_guid,
            "tempGuid": temp_guid,
            "message": message,
            "method": "apple-script",
        }

        try:
            response = requests.post(
                f"{self.base_url}{MESSAGE_TEXT_PATH}",
                params={"password": self._password},
                json=body,
                timeout=self.timeout,
            )
        except Exception as exc:  # curl_cffi exposes several request exception types.
            # Do not chain the original exception: request errors may include the
            # password-bearing URL in their text.
            raise BlueBubblesNetworkError(
                f"BlueBubbles request failed ({type(exc).__name__})"
            ) from None

        status_code = getattr(response, "status_code", None)
        if type(status_code) is not int or not 200 <= status_code < 300:
            raise BlueBubblesAPIError(
                f"BlueBubbles returned HTTP status {status_code!r}"
            )

        try:
            payload = response.json()
        except (AttributeError, TypeError, ValueError):
            payload = None

        return SendMessageResult(
            status_code=status_code,
            chat_guid=chat_guid,
            temp_guid=temp_guid,
            payload=payload,
        )

    def query_messages(
        self,
        after_rowid: int,
        *,
        limit: int = DEFAULT_MESSAGE_QUERY_LIMIT,
        sort: str = "ASC",
    ) -> Any:
        """Query messages newer than ``after_rowid``.

        The query deliberately asks BlueBubbles to filter out messages sent by
        this Mac and to include chat data.  The raw response is returned so the
        polling layer can validate the response shape without making the
        outbound client depend on webhook models.
        """

        if type(after_rowid) is not int or after_rowid < 0:
            raise ValueError("after_rowid must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer between 1 and 1000")
        if sort not in {"ASC", "DESC"}:
            raise ValueError("sort must be ASC or DESC")

        body = {
            "limit": limit,
            "offset": 0,
            "sort": sort,
            "with": ["chat"],
            "where": [
                {
                    "statement": (
                        "message.ROWID > :after_rowid "
                        "AND message.is_from_me = :is_from_me"
                    ),
                    "args": {"after_rowid": after_rowid, "is_from_me": 0},
                }
            ],
        }

        try:
            response = requests.post(
                f"{self.base_url}{MESSAGE_QUERY_PATH}",
                params={"password": self._password},
                json=body,
                timeout=self.timeout,
            )
        except Exception as exc:  # curl_cffi exposes several request exception types.
            raise BlueBubblesNetworkError(
                f"BlueBubbles request failed ({type(exc).__name__})"
            ) from None

        status_code = getattr(response, "status_code", None)
        if type(status_code) is not int or not 200 <= status_code < 300:
            raise BlueBubblesAPIError(
                f"BlueBubbles returned HTTP status {status_code!r}"
            )

        try:
            return response.json()
        except (AttributeError, TypeError, ValueError):
            raise BlueBubblesAPIError("BlueBubbles returned invalid JSON") from None


def _require_non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
