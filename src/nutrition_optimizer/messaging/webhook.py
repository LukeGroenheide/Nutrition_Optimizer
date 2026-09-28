"""BlueBubbles webhook parsing and a small standard-library HTTP receiver."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
import argparse
from dataclasses import dataclass
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition, Lock
from typing import Any


DEFAULT_WEBHOOK_HOST = "0.0.0.0"
DEFAULT_WEBHOOK_PORT = 8765
DEFAULT_RECENT_GUID_CACHE_SIZE = 1024
MAX_WEBHOOK_BODY_BYTES = 1_048_576

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    """The small, source-independent message shape used by integrations."""

    message_guid: str | None
    text: str | None
    sender_address: str | None
    chat_guid: str | None
    is_from_me: bool


MessageHandler = Callable[[IncomingMessage], None]


class RecentMessageGuids:
    """Thread-safe, fixed-size cache for recently handled message GUIDs."""

    def __init__(self, max_size: int = DEFAULT_RECENT_GUID_CACHE_SIZE) -> None:
        if type(max_size) is not int or max_size < 1:
            raise ValueError("max_size must be a positive integer")

        self._max_size = max_size
        self._order: deque[str] = deque()
        self._seen: set[str] = set()
        self._in_flight: set[str] = set()
        self._condition = Condition(Lock())

    def claim(self, message_guid: str) -> bool:
        """Return whether ``message_guid`` is new and remember it if so."""

        if not isinstance(message_guid, str) or not message_guid:
            raise ValueError("message_guid must be a non-empty string")

        with self._condition:
            if message_guid in self._seen or message_guid in self._in_flight:
                return False
            self._remember(message_guid)
            return True

    def dispatch(self, message_guid: str, handler: Callable[[], None]) -> bool:
        """Run ``handler`` once for a GUID, safely across webhook/poll races.

        A failed handler is removed from the in-flight set, allowing a later
        delivery attempt to retry it.  The callback runs outside the condition
        lock, while concurrent deliveries wait for its result.
        """

        if not isinstance(message_guid, str) or not message_guid:
            raise ValueError("message_guid must be a non-empty string")
        if not callable(handler):
            raise TypeError("handler must be callable")

        with self._condition:
            while message_guid in self._in_flight and message_guid not in self._seen:
                self._condition.wait()
            if message_guid in self._seen:
                return False
            self._in_flight.add(message_guid)

        try:
            handler()
        except Exception:
            with self._condition:
                self._in_flight.discard(message_guid)
                self._condition.notify_all()
            raise

        with self._condition:
            self._in_flight.discard(message_guid)
            self._remember(message_guid)
            self._condition.notify_all()
        return True

    def _remember(self, message_guid: str) -> None:
        if len(self._order) >= self._max_size:
            self._seen.remove(self._order.popleft())
        self._order.append(message_guid)
        self._seen.add(message_guid)

    def __len__(self) -> int:
        with self._condition:
            return len(self._order)


def deliver_incoming_message(
    message: IncomingMessage,
    message_handler: MessageHandler,
    recent_message_guids: RecentMessageGuids,
) -> bool:
    """Deliver one message through the shared webhook/polling boundary.

    ``True`` means the handler ran.  ``False`` means the GUID was already
    handled.  A message without a GUID cannot participate in dedupe, but is
    still passed through for compatibility with the existing webhook behavior.
    """

    if not callable(message_handler):
        raise TypeError("message_handler must be callable")
    if not callable(getattr(recent_message_guids, "dispatch", None)):
        raise TypeError("recent_message_guids must provide dispatch")

    if message.message_guid:
        delivered = recent_message_guids.dispatch(
            message.message_guid,
            lambda: message_handler(message),
        )
        if not delivered:
            duplicate_message_diagnostic()
        return delivered

    message_handler(message)
    return True


def parse_webhook_event(payload: object) -> IncomingMessage | None:
    """Parse one BlueBubbles webhook payload, ignoring unrelated events."""

    if not isinstance(payload, Mapping) or payload.get("type") != "new-message":
        return None

    data = payload.get("data")
    if not isinstance(data, Mapping):
        return None

    handle = data.get("handle")
    sender_address = (
        _optional_string(handle.get("address"))
        if isinstance(handle, Mapping)
        else None
    )

    chat_guid = None
    chats = data.get("chats")
    if isinstance(chats, Sequence) and not isinstance(chats, (str, bytes, bytearray)):
        for chat in chats:
            if isinstance(chat, Mapping):
                chat_guid = _optional_string(chat.get("guid"))
                if chat_guid is not None:
                    break

    is_from_me = data.get("isFromMe", False)
    if type(is_from_me) is not bool:
        is_from_me = False

    return IncomingMessage(
        message_guid=_optional_string(data.get("guid")),
        text=_optional_string(data.get("text")),
        sender_address=sender_address,
        chat_guid=chat_guid,
        is_from_me=is_from_me,
    )


class BlueBubblesWebhookServer(ThreadingHTTPServer):
    """Threaded HTTP server carrying a parsed-message callback."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        message_handler: MessageHandler,
        recent_message_guids: RecentMessageGuids | None = None,
    ) -> None:
        if not callable(message_handler):
            raise TypeError("message_handler must be callable")
        self.message_handler = message_handler
        self.recent_message_guids = (
            recent_message_guids
            if recent_message_guids is not None
            else RecentMessageGuids()
        )
        super().__init__(server_address, _WebhookRequestHandler)


def create_webhook_server(
    host: str = DEFAULT_WEBHOOK_HOST,
    port: int = DEFAULT_WEBHOOK_PORT,
    message_handler: MessageHandler | None = None,
    recent_message_guids: RecentMessageGuids | None = None,
) -> BlueBubblesWebhookServer:
    """Create a receiver that invokes ``message_handler`` for new messages."""

    if message_handler is None:
        message_handler = default_message_handler
    return BlueBubblesWebhookServer((host, port), message_handler, recent_message_guids)


def default_message_handler(message: IncomingMessage) -> None:
    """Emit a concise diagnostic without printing message content or addresses."""

    print(
        "BlueBubbles message received "
        f"(guid_present={message.message_guid is not None}, "
        f"chat_present={message.chat_guid is not None}, "
        f"text_present={message.text is not None}, "
        f"is_from_me={message.is_from_me})",
        flush=True,
    )


def duplicate_message_diagnostic() -> None:
    """Report a duplicate without exposing the message GUID or content."""

    print("BlueBubbles duplicate message ignored (GUID already seen)", flush=True)


class _WebhookRequestHandler(BaseHTTPRequestHandler):
    server: BlueBubblesWebhookServer

    def do_POST(self) -> None:
        content_length_header = self.headers.get("Content-Length")
        try:
            content_length = int(content_length_header) if content_length_header else -1
        except ValueError:
            self._send_json(400, {"accepted": False, "error": "invalid content length"})
            return

        if content_length < 0:
            self._send_json(400, {"accepted": False, "error": "content length required"})
            return
        if content_length > MAX_WEBHOOK_BODY_BYTES:
            self._send_json(413, {"accepted": False, "error": "payload too large"})
            return

        raw_body = self.rfile.read(content_length)
        if len(raw_body) != content_length:
            self._send_json(400, {"accepted": False, "error": "incomplete request body"})
            return

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"accepted": False, "error": "invalid JSON"})
            return

        message = parse_webhook_event(payload)
        if message is None:
            # Acknowledge valid JSON that is not a usable new-message event so
            # BlueBubbles does not needlessly retry unrelated webhook traffic.
            self._send_json(200, {"accepted": False})
            return

        try:
            delivered = deliver_incoming_message(
                message,
                self.server.message_handler,
                self.server.recent_message_guids,
            )
        except Exception:
            _LOGGER.exception("BlueBubbles webhook message handler failed")
            self._send_json(500, {"accepted": False, "error": "handler failed"})
            return

        if message.message_guid and not delivered:
            self._send_json(200, {"accepted": True, "duplicate": True})
            return
        self._send_json(200, {"accepted": True})

    def _send_json(self, status_code: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Keep routine webhook access out of stdout diagnostics."""


def main(argv: list[str] | None = None) -> int:
    """Run the standalone webhook receiver until interrupted."""

    parser = argparse.ArgumentParser(description="Run the BlueBubbles webhook receiver")
    parser.add_argument("--host", default=DEFAULT_WEBHOOK_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_WEBHOOK_PORT)
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=15.0,
        help="BlueBubbles history poll interval in seconds (default: 15)",
    )
    parser.add_argument(
        "--cursor-path",
        default=None,
        help="durable BlueBubbles ROWID cursor path (default: runtime state path)",
    )
    args = parser.parse_args(argv)

    # Import lazily so ``python -m ...webhook`` remains free of module
    # pre-import warnings and argument help remains available without runtime
    # configuration.
    from .bluebubbles import BlueBubblesClient
    from .meal_workflow import open_production_meal_report_runtime
    from .polling import BlueBubblesPollingWorker
    from ..openclaw_semantic import (
        OpenClawSemanticConfigurationError,
    )

    meal_report_runtime = None
    try:
        client = BlueBubblesClient.from_env()
        meal_report_runtime = open_production_meal_report_runtime(client)
    except ValueError as exc:
        parser.error(str(exc))
    except OpenClawSemanticConfigurationError as exc:
        parser.error(str(exc))

    assert meal_report_runtime is not None
    application_handler = meal_report_runtime.handler
    recent_message_guids = RecentMessageGuids()
    server = create_webhook_server(
        host=args.host,
        port=args.port,
        message_handler=application_handler,
        recent_message_guids=recent_message_guids,
    )
    polling_worker = BlueBubblesPollingWorker(
        client,
        application_handler,
        recent_message_guids=recent_message_guids,
        cursor_path=args.cursor_path,
        interval=args.poll_interval,
    )
    polling_worker.start()
    _LOGGER.info("BlueBubbles history polling enabled (interval=%ss)", args.poll_interval)

    print(f"BlueBubbles webhook listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if polling_worker is not None:
            polling_worker.stop(timeout=5)
        server.server_close()
        meal_report_runtime.close()
    return 0


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


if __name__ == "__main__":
    raise SystemExit(main())
