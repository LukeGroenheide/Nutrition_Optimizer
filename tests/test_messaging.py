from __future__ import annotations

from http.client import HTTPConnection
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch
from uuid import UUID

from nutrition_optimizer.messaging.bluebubbles import (
    BASE_URL_ENV_VAR,
    PASSWORD_ENV_VAR,
    BlueBubblesAPIError,
    BlueBubblesClient,
)
from nutrition_optimizer.messaging.webhook import (
    RecentMessageGuids,
    create_webhook_server,
    parse_webhook_event,
)


class BlueBubblesClientTests(unittest.TestCase):
    def test_send_text_constructs_request(self) -> None:
        class FakeResponse:
            status_code = 201

            def json(self) -> dict[str, str]:
                return {"status": "sent"}

        with patch(
            "nutrition_optimizer.messaging.bluebubbles.requests.post",
            return_value=FakeResponse(),
        ) as post:
            result = BlueBubblesClient(
                password="runtime-secret",
                base_url="http://example.test:1234/",
                timeout=7.5,
            ).send_text("chat-guid", "hello", temp_guid="temp-guid")

        post.assert_called_once_with(
            "http://example.test:1234/api/v1/message/text",
            params={"password": "runtime-secret"},
            json={
                "chatGuid": "chat-guid",
                "tempGuid": "temp-guid",
                "message": "hello",
                "method": "apple-script",
            },
            timeout=7.5,
        )
        self.assertEqual(result.status_code, 201)
        self.assertEqual(result.payload, {"status": "sent"})

    @patch("nutrition_optimizer.messaging.bluebubbles.uuid4")
    def test_send_text_generates_temp_guid_when_omitted(self, uuid4_mock) -> None:
        uuid4_mock.return_value = UUID("12345678-1234-5678-1234-567812345678")

        class FakeResponse:
            status_code = 200

            def json(self) -> None:
                return None

        with patch(
            "nutrition_optimizer.messaging.bluebubbles.requests.post",
            return_value=FakeResponse(),
        ) as post:
            result = BlueBubblesClient("runtime-secret").send_text("chat-guid", "hello")

        self.assertEqual(result.temp_guid, "12345678-1234-5678-1234-567812345678")
        self.assertEqual(post.call_args.kwargs["json"]["tempGuid"], result.temp_guid)
        uuid4_mock.assert_called_once_with()

    def test_from_env_requires_password_and_supports_url(self) -> None:
        client = BlueBubblesClient.from_env(
            {
                PASSWORD_ENV_VAR: "runtime-secret",
                BASE_URL_ENV_VAR: "http://example.test",
            }
        )

        self.assertEqual(client.base_url, "http://example.test")
        with self.assertRaisesRegex(ValueError, PASSWORD_ENV_VAR):
            BlueBubblesClient.from_env({})

    def test_non_success_response_raises_api_error(self) -> None:
        class FakeResponse:
            status_code = 401

        with patch(
            "nutrition_optimizer.messaging.bluebubbles.requests.post",
            return_value=FakeResponse(),
        ):
            with self.assertRaises(BlueBubblesAPIError):
                BlueBubblesClient("runtime-secret").send_text("chat-guid", "hello")


class WebhookParsingTests(unittest.TestCase):
    def test_parses_observed_new_message_shape(self) -> None:
        message = parse_webhook_event(
            {
                "type": "new-message",
                "data": {
                    "guid": "message-guid",
                    "text": "Webhook test",
                    "isFromMe": False,
                    "handle": {"address": "sender@example.test", "service": "iMessage"},
                    "chats": [{"guid": "chat-guid", "chatIdentifier": "identifier"}],
                },
            }
        )

        self.assertIsNotNone(message)
        assert message is not None
        self.assertEqual(message.message_guid, "message-guid")
        self.assertEqual(message.text, "Webhook test")
        self.assertEqual(message.sender_address, "sender@example.test")
        self.assertEqual(message.chat_guid, "chat-guid")
        self.assertFalse(message.is_from_me)

    def test_distinguishes_messages_from_me(self) -> None:
        for is_from_me in (False, True):
            with self.subTest(is_from_me=is_from_me):
                message = parse_webhook_event(
                    {"type": "new-message", "data": {"isFromMe": is_from_me}}
                )
                self.assertIsNotNone(message)
                assert message is not None
                self.assertEqual(message.is_from_me, is_from_me)

    def test_ignores_unrelated_and_malformed_events(self) -> None:
        self.assertIsNone(parse_webhook_event({"type": "updated-chat", "data": {}}))
        self.assertIsNone(parse_webhook_event({"type": "new-message", "data": []}))
        self.assertIsNone(parse_webhook_event(None))

        minimal = parse_webhook_event({"type": "new-message", "data": {}})
        self.assertIsNotNone(minimal)
        assert minimal is not None
        self.assertIsNone(minimal.message_guid)
        self.assertIsNone(minimal.text)
        self.assertFalse(minimal.is_from_me)


class RecentMessageGuidTests(unittest.TestCase):
    def test_cache_is_bounded_and_old_entries_fall_out(self) -> None:
        cache = RecentMessageGuids(max_size=2)

        self.assertTrue(cache.claim("guid-1"))
        self.assertTrue(cache.claim("guid-2"))
        self.assertFalse(cache.claim("guid-1"))
        self.assertEqual(len(cache), 2)

        self.assertTrue(cache.claim("guid-3"))
        self.assertEqual(len(cache), 2)
        self.assertTrue(cache.claim("guid-1"))


class WebhookServerTests(unittest.TestCase):
    def test_duplicate_is_acknowledged_but_handler_runs_once(self) -> None:
        received = []
        server = create_webhook_server(
            host="127.0.0.1",
            port=0,
            message_handler=received.append,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]

            def post(payload: dict[str, object]) -> tuple[int, dict[str, object]]:
                connection = HTTPConnection("127.0.0.1", port, timeout=2)
                try:
                    connection.request(
                        "POST",
                        "/",
                        body=json.dumps(payload),
                        headers={"Content-Type": "application/json"},
                    )
                    response = connection.getresponse()
                    return response.status, json.loads(response.read())
                finally:
                    connection.close()

            first = {
                "type": "new-message",
                "data": {"guid": "message-guid", "text": "hello", "isFromMe": True},
            }
            self.assertEqual(post(first), (200, {"accepted": True}))
            self.assertEqual(
                post(first),
                (200, {"accepted": True, "duplicate": True}),
            )
            self.assertEqual(
                post(
                    {
                        "type": "new-message",
                        "data": {"guid": "distinct-guid", "text": "other"},
                    }
                ),
                (200, {"accepted": True}),
            )
            self.assertEqual(
                post({"type": "new-message", "data": {"text": "no guid"}}),
                (200, {"accepted": True}),
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(
            [message.message_guid for message in received],
            ["message-guid", "distinct-guid", None],
        )
        self.assertTrue(received[0].is_from_me)


class WebhookModuleExecutionTests(unittest.TestCase):
    def test_module_execution_has_no_runpy_preimport_warning(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        source_path = str(repository_root / "src")
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (source_path, environment.get("PYTHONPATH")) if value
        )

        result = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::RuntimeWarning",
                "-m",
                "nutrition_optimizer.messaging.webhook",
                "--help",
            ],
            cwd=repository_root,
            env=environment,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("found in sys.modules", result.stderr)


if __name__ == "__main__":
    unittest.main()
