from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nutrition_optimizer.messaging.bluebubbles import BlueBubblesClient
from nutrition_optimizer.messaging.polling import (
    BlueBubblesPollingWorker,
    PollingResponseError,
    RowIDCursorStore,
    parse_message_query_payload,
)
from nutrition_optimizer.messaging.webhook import (
    RecentMessageGuids,
    deliver_incoming_message,
    parse_webhook_event,
)


def message_row(
    rowid: int,
    guid: str,
    *,
    is_from_me: bool = False,
    text: str = "hello",
) -> dict[str, object]:
    return {
        "originalROWID": rowid,
        "guid": guid,
        "text": text,
        "isFromMe": is_from_me,
        "handle": {"address": "sender@example.test"},
        "chats": [{"guid": "chat-guid"}],
    }


class FakeQueryClient:
    def __init__(self, *payloads: object) -> None:
        self.payloads = deque(payloads)
        self.calls: list[tuple[int, int, str]] = []

    def query_messages(self, after_rowid: int, *, limit: int, sort: str) -> object:
        self.calls.append((after_rowid, limit, sort))
        if not self.payloads:
            raise AssertionError("unexpected extra query")
        payload = self.payloads.popleft()
        if isinstance(payload, BaseException):
            raise payload
        return payload


class MessageQueryClientTests(unittest.TestCase):
    def test_query_constructs_incremental_inbound_request(self) -> None:
        class FakeResponse:
            status_code = 200

            def json(self) -> dict[str, object]:
                return {"data": []}

        with patch(
            "nutrition_optimizer.messaging.bluebubbles.requests.post",
            return_value=FakeResponse(),
        ) as post:
            result = BlueBubblesClient("runtime-secret").query_messages(39, limit=25)

        self.assertEqual(result, {"data": []})
        post.assert_called_once_with(
            "http://127.0.0.1:1234/api/v1/message/query",
            params={"password": "runtime-secret"},
            json={
                "limit": 25,
                "offset": 0,
                "sort": "ASC",
                "with": ["chat"],
                "where": [
                    {
                        "statement": (
                            "message.ROWID > :after_rowid "
                            "AND message.is_from_me = :is_from_me"
                        ),
                        "args": {"after_rowid": 39, "is_from_me": 0},
                    }
                ],
            },
            timeout=10.0,
        )

    def test_malformed_json_is_an_api_failure(self) -> None:
        class FakeResponse:
            status_code = 200

            def json(self) -> object:
                raise ValueError("not json")

        with patch(
            "nutrition_optimizer.messaging.bluebubbles.requests.post",
            return_value=FakeResponse(),
        ):
            with self.assertRaisesRegex(Exception, "invalid JSON"):
                BlueBubblesClient("runtime-secret").query_messages(0)


class CursorStoreTests(unittest.TestCase):
    def test_cursor_persists_and_does_not_regress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state" / "cursor.json"
            store = RowIDCursorStore(path)
            self.assertIsNone(store.load())
            store.advance(40)
            store.advance(39)

            reloaded = RowIDCursorStore(path)
            self.assertEqual(reloaded.load(), 40)
            self.assertEqual(json.loads(path.read_text()), {"rowid": 40})
            self.assertNotIn("hello", path.read_text())


class PollingWorkerTests(unittest.TestCase):
    def make_worker(
        self,
        directory: str,
        client: FakeQueryClient,
        received: list,
        *,
        cursor: int = 39,
        recent: RecentMessageGuids | None = None,
    ) -> BlueBubblesPollingWorker:
        cursor_store = RowIDCursorStore(Path(directory) / "cursor.json")
        cursor_store.initialize(cursor)
        return BlueBubblesPollingWorker(
            client,
            received.append,
            recent_message_guids=recent,
            cursor_store=cursor_store,
        )

    def test_multiple_messages_are_processed_in_rowid_order(self) -> None:
        client = FakeQueryClient(
            {"data": [message_row(42, "guid-42"), message_row(40, "guid-40")]}
        )
        with tempfile.TemporaryDirectory() as directory:
            received: list = []
            worker = self.make_worker(directory, client, received)
            self.assertEqual(worker.poll_once(), 2)
            self.assertEqual([item.message_guid for item in received], ["guid-40", "guid-42"])
            self.assertEqual(worker.cursor, 42)
            self.assertEqual(client.calls, [(39, 100, "ASC")])

    def test_startup_high_water_mark_does_not_replay_history(self) -> None:
        client = FakeQueryClient({"data": [message_row(39, "old-guid")]})
        with tempfile.TemporaryDirectory() as directory:
            received: list = []
            worker = BlueBubblesPollingWorker(
                client,
                received.append,
                cursor_path=Path(directory) / "cursor.json",
            )
            self.assertEqual(worker.poll_once(), 0)
            self.assertEqual(received, [])
            self.assertEqual(worker.cursor, 39)
            self.assertEqual(client.calls, [(0, 1, "DESC")])

            client.payloads.append({"data": [message_row(40, "new-guid")]})
            self.assertEqual(worker.poll_once(), 1)
            self.assertEqual([item.message_guid for item in received], ["new-guid"])

    def test_cursor_is_reloaded_on_worker_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cursor.json"
            first_client = FakeQueryClient({"data": [message_row(40, "guid-40")]})
            first_received: list = []
            first = BlueBubblesPollingWorker(
                first_client,
                first_received.append,
                cursor_path=path,
            )
            first.cursor_store.initialize(39)
            first.poll_once()

            second_client = FakeQueryClient({"data": []})
            second_received: list = []
            second = BlueBubblesPollingWorker(
                second_client,
                second_received.append,
                cursor_path=path,
            )
            second.poll_once()
            self.assertEqual(second_client.calls, [(40, 100, "ASC")])
            self.assertEqual(second_received, [])

    def test_webhook_then_poll_same_guid_is_handled_once(self) -> None:
        recent = RecentMessageGuids()
        received: list = []
        webhook_message = parse_webhook_event(
            {"type": "new-message", "data": {"guid": "same-guid", "isFromMe": False}}
        )
        assert webhook_message is not None
        self.assertTrue(deliver_incoming_message(webhook_message, received.append, recent))

        client = FakeQueryClient({"data": [message_row(40, "same-guid")]})
        with tempfile.TemporaryDirectory() as directory:
            worker = self.make_worker(directory, client, received, recent=recent)
            self.assertEqual(worker.poll_once(), 1)
        self.assertEqual([item.message_guid for item in received], ["same-guid"])

    def test_poll_then_webhook_same_guid_is_handled_once(self) -> None:
        recent = RecentMessageGuids()
        received: list = []
        client = FakeQueryClient({"data": [message_row(40, "same-guid")]})
        with tempfile.TemporaryDirectory() as directory:
            worker = self.make_worker(directory, client, received, recent=recent)
            worker.poll_once()

        webhook_message = parse_webhook_event(
            {"type": "new-message", "data": {"guid": "same-guid", "isFromMe": False}}
        )
        assert webhook_message is not None
        self.assertFalse(deliver_incoming_message(webhook_message, received.append, recent))
        self.assertEqual([item.message_guid for item in received], ["same-guid"])

    def test_messages_from_me_are_ignored_but_cursor_advances(self) -> None:
        client = FakeQueryClient({"data": [message_row(40, "outbound", is_from_me=True)]})
        with tempfile.TemporaryDirectory() as directory:
            received: list = []
            worker = self.make_worker(directory, client, received)
            self.assertEqual(worker.poll_once(), 0)
            self.assertEqual(received, [])
            self.assertEqual(worker.cursor, 40)

    def test_api_failure_leaves_cursor_safe_and_next_poll_retries(self) -> None:
        client = FakeQueryClient(
            RuntimeError("network down"),
            {"data": [message_row(40, "guid-40")]},
        )
        with tempfile.TemporaryDirectory() as directory:
            received: list = []
            worker = self.make_worker(directory, client, received)
            with self.assertRaisesRegex(RuntimeError, "network down"):
                worker.poll_once()
            self.assertEqual(worker.cursor, 39)
            self.assertEqual(worker.poll_once(), 1)
            self.assertEqual(worker.cursor, 40)

    def test_partial_processing_failure_does_not_skip_following_rows(self) -> None:
        client = FakeQueryClient(
            {"data": [message_row(40, "guid-40"), message_row(41, "guid-41")]},
            {"data": [message_row(40, "guid-40"), message_row(41, "guid-41")]},
        )
        calls: list[str] = []

        def handler(message) -> None:
            calls.append(message.message_guid)
            if message.message_guid == "guid-41" and calls.count("guid-41") == 1:
                raise RuntimeError("handler failed")

        with tempfile.TemporaryDirectory() as directory:
            cursor_store = RowIDCursorStore(Path(directory) / "cursor.json")
            cursor_store.initialize(39)
            worker = BlueBubblesPollingWorker(client, handler, cursor_store=cursor_store)
            with self.assertRaisesRegex(RuntimeError, "handler failed"):
                worker.poll_once()
            self.assertEqual(worker.cursor, 40)
            self.assertEqual(worker.poll_once(), 1)
            self.assertEqual(worker.cursor, 41)
        self.assertEqual(calls, ["guid-40", "guid-41", "guid-41"])

    def test_malformed_response_leaves_cursor_unchanged(self) -> None:
        client = FakeQueryClient({"data": [{"guid": "missing-rowid"}]})
        with tempfile.TemporaryDirectory() as directory:
            received: list = []
            worker = self.make_worker(directory, client, received)
            with self.assertRaises(PollingResponseError):
                worker.poll_once()
            self.assertEqual(worker.cursor, 39)
            self.assertEqual(received, [])


class QueryPayloadParsingTests(unittest.TestCase):
    def test_inbound_query_row_includes_sender_and_chat(self) -> None:
        rows = parse_message_query_payload({"data": [message_row(40, "guid-40")]})
        self.assertEqual(rows[0].message.sender_address, "sender@example.test")
        self.assertEqual(rows[0].message.chat_guid, "chat-guid")

    def test_non_list_data_is_rejected(self) -> None:
        with self.assertRaises(PollingResponseError):
            parse_message_query_payload({"data": {}})


if __name__ == "__main__":
    unittest.main()
