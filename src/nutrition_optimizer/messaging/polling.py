"""BlueBubbles message-history polling fallback.

The poller is intentionally independent of SQLite and Messages.app.  It uses
BlueBubbles' authenticated history API to recover rows for which the normal
database watcher/webhook path did not emit an event.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import tempfile
from threading import Event, Thread, current_thread
from typing import Any

from .bluebubbles import BlueBubblesClient
from .webhook import (
    IncomingMessage,
    MessageHandler,
    RecentMessageGuids,
    deliver_incoming_message,
)


DEFAULT_POLL_INTERVAL = 15.0
DEFAULT_POLL_LIMIT = 100
CURSOR_PATH_ENV_VAR = "NUTRITION_OPTIMIZER_BLUEBUBBLES_CURSOR_PATH"
XDG_STATE_HOME_ENV_VAR = "XDG_STATE_HOME"
DEFAULT_STATE_DIRECTORY_NAME = "nutrition_optimizer"
DEFAULT_CURSOR_FILENAME = "bluebubbles-message-cursor.json"

_LOGGER = logging.getLogger(__name__)


class PollingResponseError(RuntimeError):
    """Raised when BlueBubbles returns an unusable message-query response."""


class CursorStateError(RuntimeError):
    """Raised when the durable polling cursor cannot be read or written."""


@dataclass(frozen=True, slots=True)
class QueriedMessage:
    """One validated message-query row and its database row identifier."""

    original_rowid: int
    message: IncomingMessage | None


def default_cursor_path(environ: Mapping[str, str] | None = None) -> Path:
    """Return the configurable, non-repository runtime cursor location."""

    source = os.environ if environ is None else environ
    configured = source.get(CURSOR_PATH_ENV_VAR)
    if configured:
        return Path(configured).expanduser()

    state_home = source.get(XDG_STATE_HOME_ENV_VAR)
    if state_home:
        return Path(state_home).expanduser() / DEFAULT_STATE_DIRECTORY_NAME / DEFAULT_CURSOR_FILENAME

    return Path.home() / ".local" / "state" / DEFAULT_STATE_DIRECTORY_NAME / DEFAULT_CURSOR_FILENAME


class RowIDCursorStore:
    """Atomically persist one non-decreasing SQLite ``ROWID`` cursor."""

    def __init__(self, path: str | os.PathLike[str] | Path) -> None:
        self.path = Path(path).expanduser()
        self._value: int | None = None
        self._loaded = False

    @property
    def value(self) -> int | None:
        if not self._loaded:
            self.load()
        return self._value

    def load(self) -> int | None:
        """Load the cursor, returning ``None`` when no state exists."""

        if self._loaded:
            return self._value

        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._loaded = True
            self._value = None
            return None
        except OSError as exc:
            raise CursorStateError("unable to read BlueBubbles cursor") from exc

        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise CursorStateError("BlueBubbles cursor is not valid JSON") from exc

        if not isinstance(payload, Mapping):
            raise CursorStateError("BlueBubbles cursor must be a JSON object")
        value = payload.get("rowid")
        if type(value) is not int or value < 0:
            raise CursorStateError("BlueBubbles cursor rowid is invalid")

        self._loaded = True
        self._value = value
        return value

    def initialize(self, rowid: int) -> int:
        """Set an initial high-water mark only when no cursor exists."""

        _validate_rowid(rowid)
        current = self.load()
        if current is not None:
            return current
        self._write(rowid)
        self._value = rowid
        self._loaded = True
        return rowid

    def advance(self, rowid: int) -> int:
        """Persist ``rowid`` unless it would regress the cursor."""

        _validate_rowid(rowid)
        current = self.load()
        if current is not None and rowid <= current:
            return current
        self._write(rowid)
        self._value = rowid
        self._loaded = True
        return rowid

    def _write(self, rowid: int) -> None:
        parent = self.path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                dir=parent,
                text=True,
            )
            try:
                os.fchmod(file_descriptor, 0o600)
                with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
                    file_descriptor = -1
                    json.dump({"rowid": rowid}, stream, separators=(",", ":"))
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, self.path)
                temporary_name = ""
                directory_descriptor = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            finally:
                if file_descriptor != -1:
                    os.close(file_descriptor)
                if temporary_name:
                    try:
                        os.unlink(temporary_name)
                    except FileNotFoundError:
                        pass
        except OSError as exc:
            raise CursorStateError("unable to persist BlueBubbles cursor") from exc


class BlueBubblesPollingWorker:
    """Poll BlueBubbles history and deliver missed inbound messages."""

    def __init__(
        self,
        client: BlueBubblesClient,
        message_handler: MessageHandler,
        *,
        recent_message_guids: RecentMessageGuids | None = None,
        cursor_path: str | os.PathLike[str] | Path | None = None,
        cursor_store: RowIDCursorStore | None = None,
        interval: float = DEFAULT_POLL_INTERVAL,
        limit: int = DEFAULT_POLL_LIMIT,
    ) -> None:
        if not isinstance(client, BlueBubblesClient) and not callable(
            getattr(client, "query_messages", None)
        ):
            raise TypeError("client must provide query_messages")
        if not callable(message_handler):
            raise TypeError("message_handler must be callable")
        if recent_message_guids is not None and not callable(
            getattr(recent_message_guids, "dispatch", None)
        ):
            raise TypeError("recent_message_guids must provide dispatch")
        if cursor_path is not None and cursor_store is not None:
            raise ValueError("provide cursor_path or cursor_store, not both")
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or interval <= 0:
            raise ValueError("interval must be a positive number")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer between 1 and 1000")

        self.client = client
        self.message_handler = message_handler
        self.recent_message_guids = (
            recent_message_guids
            if recent_message_guids is not None
            else RecentMessageGuids()
        )
        self.cursor_store = cursor_store or RowIDCursorStore(
            cursor_path if cursor_path is not None else default_cursor_path()
        )
        self.interval = float(interval)
        self.limit = limit
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._initialized = False

    @property
    def cursor(self) -> int | None:
        return self.cursor_store.value

    def poll_once(self) -> int:
        """Run one poll, returning the number of newly handled messages."""

        if not self._initialized:
            cursor = self.cursor_store.load()
            if cursor is None:
                self._initialize_high_water_mark()
                self._initialized = True
                return 0
            self._initialized = True

        cursor = self.cursor_store.value
        if cursor is None:
            cursor = 0

        payload = self.client.query_messages(
            cursor,
            limit=self.limit,
            sort="ASC",
        )
        rows = parse_message_query_payload(payload)
        handled = 0
        for row in rows:
            current_cursor = self.cursor_store.value
            if current_cursor is not None and row.original_rowid <= current_cursor:
                continue

            if row.message is not None:
                # A duplicate is considered successfully accepted: advancing
                # past it lets a retry recover after a cursor-write failure.
                deliver_incoming_message(
                    row.message,
                    self.message_handler,
                    self.recent_message_guids,
                )
                handled += 1

            # This happens only after the handler succeeds (or the row was
            # intentionally ignored as non-inbound).
            self.cursor_store.advance(row.original_rowid)
        return handled

    def start(self) -> None:
        """Start the daemon polling thread if it is not already running."""

        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(
            target=self._run,
            name="bluebubbles-polling",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Stop the polling thread, if running."""

        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join(timeout)
            if thread.is_alive():
                return
        self._thread = None

    def _initialize_high_water_mark(self) -> None:
        """Skip existing history on first startup."""

        payload = self.client.query_messages(0, limit=1, sort="DESC")
        rows = parse_message_query_payload(payload)
        high_water = max((row.original_rowid for row in rows), default=0)
        self.cursor_store.initialize(high_water)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception as exc:
                # Keep the worker alive and never include the password-bearing
                # request URL or response body in diagnostics.
                _LOGGER.warning(
                    "BlueBubbles polling failed (%s); retrying",
                    type(exc).__name__,
                )
            self._stop_event.wait(self.interval)


def parse_message_query_payload(payload: object) -> list[QueriedMessage]:
    """Validate and normalize one BlueBubbles message-query response."""

    if not isinstance(payload, Mapping):
        raise PollingResponseError("message query response is not an object")
    data = payload.get("data")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        raise PollingResponseError("message query response data is not a list")

    rows: list[QueriedMessage] = []
    for item in data:
        if not isinstance(item, Mapping):
            raise PollingResponseError("message query row is not an object")
        rowid = item.get("originalROWID")
        if type(rowid) is not int or rowid < 0:
            raise PollingResponseError("message query row has an invalid ROWID")

        is_from_me = item.get("isFromMe")
        if type(is_from_me) is not bool:
            raise PollingResponseError("message query row has an invalid direction")

        # The server query includes this predicate.  Treat a row marked from
        # this Mac as safely irrelevant rather than ever risking delivery of an
        # outbound message.
        message = None
        if not is_from_me:
            message = _parse_incoming_message(item)
        rows.append(QueriedMessage(original_rowid=rowid, message=message))

    rows.sort(key=lambda row: row.original_rowid)
    return rows


def _parse_incoming_message(item: Mapping[str, Any]) -> IncomingMessage:
    handle = item.get("handle")
    sender_address = (
        handle.get("address") if isinstance(handle, Mapping) else None
    )
    if not isinstance(sender_address, str):
        sender_address = None

    chat_guid = None
    chats = item.get("chats")
    if isinstance(chats, Sequence) and not isinstance(chats, (str, bytes, bytearray)):
        for chat in chats:
            if isinstance(chat, Mapping) and isinstance(chat.get("guid"), str):
                chat_guid = chat["guid"]
                break
    elif isinstance(item.get("chat"), Mapping) and isinstance(item["chat"].get("guid"), str):
        chat_guid = item["chat"]["guid"]

    text = item.get("text")
    return IncomingMessage(
        message_guid=item.get("guid") if isinstance(item.get("guid"), str) else None,
        text=text if isinstance(text, str) else None,
        sender_address=sender_address,
        chat_guid=chat_guid,
        is_from_me=False,
    )


def _validate_rowid(rowid: int) -> None:
    if type(rowid) is not int or rowid < 0:
        raise ValueError("rowid must be a non-negative integer")
