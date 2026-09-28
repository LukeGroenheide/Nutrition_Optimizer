from __future__ import annotations

from collections import deque
from decimal import Decimal
from pathlib import Path
import tempfile
from threading import Event, Thread
import unittest

import nutrition_optimizer.messaging.application as application_module
from nutrition_optimizer.messaging.application import (
    ApplicationMessageError,
    CHAT_GUID_ENV_VAR,
    SEMANTIC_FAILURE_REPLY_TEXT,
    SENDER_ADDRESS_ENV_VAR,
    UNSUPPORTED_INTENT_REPLY_TEXT,
    ApplicationMessagingConfig,
    NutritionOptimizerMessageHandler,
)
from nutrition_optimizer.messaging.polling import (
    BlueBubblesPollingWorker,
    RowIDCursorStore,
)
from nutrition_optimizer.messaging.webhook import (
    IncomingMessage,
    RecentMessageGuids,
    deliver_incoming_message,
    parse_webhook_event,
)
from nutrition_optimizer.nutrition import DailyLedger
from nutrition_optimizer.semantic import RecordIntakeIntent, UnsupportedIntent


class FakeOutboundSender:
    def __init__(self, *, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.calls: list[tuple[str, str]] = []

    def send_text(self, chat_guid: str, message: str) -> object:
        self.calls.append((chat_guid, message))
        if self.failure is not None:
            raise self.failure
        return object()


class FakeSemanticInterpreter:
    def __init__(
        self,
        result: RecordIntakeIntent | UnsupportedIntent | None = None,
        *,
        failure: BaseException | None = None,
        started: Event | None = None,
        release: Event | None = None,
    ) -> None:
        self.result = result or RecordIntakeIntent(
            food_text="herb roasted chicken",
            servings=Decimal("2"),
        )
        self.failure = failure
        self.started = started
        self.release = release
        self.calls: list[str] = []

    def interpret(self, user_text: str):
        self.calls.append(user_text)
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            self.release.wait(timeout=2)
        if self.failure is not None:
            raise self.failure
        return self.result


class FakeQueryClient:
    def __init__(self, *payloads: object) -> None:
        self.payloads = deque(payloads)

    def query_messages(self, after_rowid: int, *, limit: int, sort: str) -> object:
        del after_rowid, limit, sort
        return self.payloads.popleft()


def inbound_message(
    *,
    text: str | None = "test text",
    is_from_me: bool = False,
    sender_address: str | None = "sender@example.test",
    chat_guid: str | None = "test-chat-guid",
    message_guid: str | None = "test-message-guid",
) -> IncomingMessage:
    return IncomingMessage(
        message_guid=message_guid,
        text=text,
        sender_address=sender_address,
        chat_guid=chat_guid,
        is_from_me=is_from_me,
    )


class ApplicationMessagingConfigTests(unittest.TestCase):
    def test_requires_at_least_one_runtime_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, CHAT_GUID_ENV_VAR):
            ApplicationMessagingConfig.from_env({})

    def test_loads_chat_and_sender_filters_from_environment(self) -> None:
        config = ApplicationMessagingConfig.from_env(
            {
                CHAT_GUID_ENV_VAR: "test-chat-guid",
                SENDER_ADDRESS_ENV_VAR: "sender@example.test",
            }
        )

        self.assertEqual(config.expected_chat_guid, "test-chat-guid")
        self.assertEqual(config.expected_sender_address, "sender@example.test")


class NutritionOptimizerMessageHandlerTests(unittest.TestCase):
    def make_handler(
        self,
        sender: FakeOutboundSender,
        interpreter: FakeSemanticInterpreter | None = None,
    ) -> NutritionOptimizerMessageHandler:
        return NutritionOptimizerMessageHandler(
            sender,
            ApplicationMessagingConfig(
                expected_chat_guid="test-chat-guid",
                expected_sender_address="sender@example.test",
            ),
            semantic_interpreter=interpreter or FakeSemanticInterpreter(),
        )

    def test_resolved_intake_produces_one_deterministic_acknowledgment(self) -> None:
        sender = FakeOutboundSender()
        interpreter = FakeSemanticInterpreter()
        handler = self.make_handler(sender, interpreter)

        handler(inbound_message(text="I ate two servings of herb roasted chicken"))

        self.assertEqual(
            sender.calls,
            [
                (
                    "test-chat-guid",
                    "I understood: herb roasted chicken — 2 servings.",
                )
            ],
        )
        self.assertEqual(interpreter.calls, ["I ate two servings of herb roasted chicken"])

    def test_fractional_servings_are_rendered_deterministically(self) -> None:
        sender = FakeOutboundSender()
        handler = self.make_handler(
            sender,
            FakeSemanticInterpreter(
                RecordIntakeIntent(
                    food_text="rice",
                    servings=Decimal("1.250"),
                )
            ),
        )

        handler(inbound_message(text="I ate rice"))

        self.assertEqual(sender.calls[0][1], "I understood: rice — 1.25 servings.")

    def test_missing_servings_produces_clarification(self) -> None:
        sender = FakeOutboundSender()
        handler = self.make_handler(
            sender,
            FakeSemanticInterpreter(
                RecordIntakeIntent(
                    food_text="herb roasted chicken",
                    servings=None,
                    clarification_required="servings_required",
                )
            ),
        )

        handler(inbound_message(text="I ate herb roasted chicken"))

        self.assertEqual(
            sender.calls,
            [
                (
                    "test-chat-guid",
                    "How many servings of herb roasted chicken did you have?",
                )
            ],
        )

    def test_unsupported_intent_produces_deterministic_capability_reply(self) -> None:
        sender = FakeOutboundSender()
        handler = self.make_handler(
            sender,
            FakeSemanticInterpreter(UnsupportedIntent(reason="unsupported_request")),
        )

        handler(inbound_message(text="What should I eat tonight?"))

        self.assertEqual(sender.calls, [("test-chat-guid", UNSUPPORTED_INTENT_REPLY_TEXT)])

    def test_semantic_failure_uses_safe_fallback_and_handler_remains_usable(self) -> None:
        sender = FakeOutboundSender()
        handler = self.make_handler(
            sender,
            FakeSemanticInterpreter(failure=RuntimeError("private model output")),
        )

        with self.assertLogs(
            "nutrition_optimizer.messaging.application",
            level="WARNING",
        ) as logs:
            self.assertTrue(handler.handle(inbound_message(text="private user text")))
            self.assertTrue(handler.handle(inbound_message(message_guid="second-guid")))

        output = "\n".join(logs.output)
        self.assertIn("semantic interpretation failed", output)
        self.assertNotIn("private user text", output)
        self.assertNotIn("private model output", output)
        self.assertEqual(
            sender.calls,
            [
                ("test-chat-guid", SEMANTIC_FAILURE_REPLY_TEXT),
                ("test-chat-guid", SEMANTIC_FAILURE_REPLY_TEXT),
            ],
        )

    def test_interpreter_is_not_called_for_ignored_messages(self) -> None:
        sender = FakeOutboundSender()
        interpreter = FakeSemanticInterpreter()
        handler = self.make_handler(sender, interpreter)

        handler(inbound_message(is_from_me=True))
        handler(inbound_message(chat_guid="other-test-chat-guid"))
        handler(inbound_message(sender_address="other-sender@example.test"))
        handler(inbound_message(text=None))
        handler(inbound_message(text="   "))

        self.assertEqual(interpreter.calls, [])
        self.assertEqual(sender.calls, [])

    def test_from_me_message_is_ignored(self) -> None:
        sender = FakeOutboundSender()
        handler = self.make_handler(sender)
        handler(inbound_message(is_from_me=True))
        self.assertEqual(sender.calls, [])

    def test_wrong_chat_or_sender_is_ignored(self) -> None:
        sender = FakeOutboundSender()
        handler = self.make_handler(sender)

        handler(inbound_message(chat_guid="other-test-chat-guid"))
        handler(inbound_message(sender_address="other-sender@example.test"))

        self.assertEqual(sender.calls, [])

    def test_missing_or_blank_text_is_ignored(self) -> None:
        sender = FakeOutboundSender()
        handler = self.make_handler(sender)

        handler(inbound_message(text=None))
        handler(inbound_message(text="   "))

        self.assertEqual(sender.calls, [])

    def test_webhook_and_polling_duplicate_produce_one_reply(self) -> None:
        sender = FakeOutboundSender()
        interpreter = FakeSemanticInterpreter()
        handler = self.make_handler(sender, interpreter)
        recent = RecentMessageGuids()
        webhook_message = parse_webhook_event(
            {
                "type": "new-message",
                "data": {
                    "guid": "test-message-guid",
                    "text": "same test text",
                    "isFromMe": False,
                    "handle": {"address": "sender@example.test"},
                    "chats": [{"guid": "test-chat-guid"}],
                },
            }
        )
        assert webhook_message is not None
        self.assertTrue(deliver_incoming_message(webhook_message, handler, recent))

        query_client = FakeQueryClient(
            {
                "data": [
                    {
                        "originalROWID": 42,
                        "guid": "test-message-guid",
                        "text": "same test text",
                        "isFromMe": False,
                        "handle": {"address": "sender@example.test"},
                        "chats": [{"guid": "test-chat-guid"}],
                    }
                ]
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            cursor_store = RowIDCursorStore(Path(directory) / "cursor.json")
            cursor_store.initialize(41)
            worker = BlueBubblesPollingWorker(
                query_client,
                handler,
                recent_message_guids=recent,
                cursor_store=cursor_store,
            )
            self.assertEqual(worker.poll_once(), 1)

        self.assertEqual(
            sender.calls,
            [("test-chat-guid", "I understood: herb roasted chicken — 2 servings.")],
        )
        self.assertEqual(interpreter.calls, ["same test text"])

    def test_slow_semantic_interpreter_still_runs_once_for_concurrent_duplicate(self) -> None:
        sender = FakeOutboundSender()
        started = Event()
        release = Event()
        interpreter = FakeSemanticInterpreter(started=started, release=release)
        handler = self.make_handler(sender, interpreter)
        recent = RecentMessageGuids()
        message = inbound_message(text="slow test text")
        results: list[bool] = []

        first = Thread(
            target=lambda: results.append(deliver_incoming_message(message, handler, recent)),
            daemon=True,
        )
        second = Thread(
            target=lambda: results.append(deliver_incoming_message(message, handler, recent)),
            daemon=True,
        )
        first.start()
        self.assertTrue(started.wait(timeout=1))
        second.start()
        release.set()
        first.join(timeout=2)
        second.join(timeout=2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(interpreter.calls, ["slow test text"])
        self.assertEqual(
            sender.calls,
            [("test-chat-guid", "I understood: herb roasted chicken — 2 servings.")],
        )

    def test_outbound_failure_raises_sanitized_error_without_leaking_content(self) -> None:
        sender = FakeOutboundSender(failure=RuntimeError("test-only-password"))
        handler = self.make_handler(sender)

        with self.assertLogs(
            "nutrition_optimizer.messaging.application",
            level="WARNING",
        ) as logs:
            with self.assertRaisesRegex(ApplicationMessageError, "outbound reply failed"):
                handler.handle(inbound_message(text="private test text"))

        output = "\n".join(logs.output)
        self.assertIn("outbound reply failed", output)
        self.assertNotIn("private test text", output)
        self.assertNotIn("test-only-password", output)

    def test_outbound_failure_does_not_consume_guid_or_cursor_and_retries(self) -> None:
        sender = FakeOutboundSender(failure=RuntimeError("test-only-password"))
        handler = self.make_handler(sender)
        recent = RecentMessageGuids()
        query_payload = {
            "data": [
                {
                    "originalROWID": 42,
                    "guid": "test-message-guid",
                    "text": "retry test text",
                    "isFromMe": False,
                    "handle": {"address": "sender@example.test"},
                    "chats": [{"guid": "test-chat-guid"}],
                }
            ]
        }
        query_client = FakeQueryClient(query_payload, query_payload)

        with tempfile.TemporaryDirectory() as directory:
            cursor_store = RowIDCursorStore(Path(directory) / "cursor.json")
            cursor_store.initialize(41)
            worker = BlueBubblesPollingWorker(
                query_client,
                handler,
                recent_message_guids=recent,
                cursor_store=cursor_store,
            )

            with self.assertRaises(ApplicationMessageError):
                worker.poll_once()
            self.assertEqual(worker.cursor, 41)
            self.assertEqual(len(recent), 0)

            sender.failure = None
            self.assertEqual(worker.poll_once(), 1)
            self.assertEqual(worker.cursor, 42)

        self.assertEqual(len(recent), 1)
        self.assertEqual(
            sender.calls,
            [
                ("test-chat-guid", "I understood: herb roasted chicken — 2 servings."),
                ("test-chat-guid", "I understood: herb roasted chicken — 2 servings."),
            ],
        )

    def test_handler_does_not_mutate_or_fabricate_nutrition_state(self) -> None:
        sender = FakeOutboundSender()
        ledger = DailyLedger()
        handler = self.make_handler(sender)

        handler(inbound_message())

        self.assertEqual(ledger.entries, ())
        self.assertIsInstance(handler.semantic_interpreter.result, RecordIntakeIntent)
        source = Path(application_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("NutritionRecord", source)
        self.assertNotIn("DailyLedger", source)
        self.assertNotIn("execute_command", source)

    def test_application_has_no_api_key_dependency(self) -> None:
        source = Path(application_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("OPENAI_API_KEY", source)


if __name__ == "__main__":
    unittest.main()
