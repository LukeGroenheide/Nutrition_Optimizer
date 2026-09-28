"""BlueBubbles messaging transport primitives."""

from importlib import import_module
from typing import Any

from .bluebubbles import (
    BASE_URL_ENV_VAR,
    PASSWORD_ENV_VAR,
    BlueBubblesAPIError,
    BlueBubblesClient,
    BlueBubblesError,
    BlueBubblesNetworkError,
    DEFAULT_BASE_URL,
    DEFAULT_MESSAGE_QUERY_LIMIT,
    DEFAULT_TIMEOUT,
    MESSAGE_QUERY_PATH,
    SendMessageResult,
)

_APPLICATION_EXPORTS = frozenset(
    {
        "ApplicationMessageError",
        "ApplicationMessagingConfig",
        "CHAT_GUID_ENV_VAR",
        "DEFAULT_REPLY_TEXT",
        "SEMANTIC_FAILURE_REPLY_TEXT",
        "UNSUPPORTED_INTENT_REPLY_TEXT",
        "NutritionOptimizerMessageHandler",
        "OutboundTextSender",
        "SENDER_ADDRESS_ENV_VAR",
    }
)

_WEBHOOK_EXPORTS = frozenset(
    {
        "BlueBubblesWebhookServer",
        "DEFAULT_RECENT_GUID_CACHE_SIZE",
        "DEFAULT_WEBHOOK_HOST",
        "DEFAULT_WEBHOOK_PORT",
        "IncomingMessage",
        "RecentMessageGuids",
        "create_webhook_server",
        "default_message_handler",
        "deliver_incoming_message",
        "parse_webhook_event",
    }
)

_POLLING_EXPORTS = frozenset(
    {
        "BlueBubblesPollingWorker",
        "CURSOR_PATH_ENV_VAR",
        "CursorStateError",
        "DEFAULT_POLL_INTERVAL",
        "DEFAULT_POLL_LIMIT",
        "PollingResponseError",
        "RowIDCursorStore",
        "QueriedMessage",
        "default_cursor_path",
        "parse_message_query_payload",
    }
)

_MEAL_WORKFLOW_EXPORTS = frozenset(
    {
        "AMBIGUOUS_ACTIVE_MEAL_PLAN_REPLY_TEXT",
        "ActiveMealPlanContextResolver",
        "ConversationalMealReportMessageHandler",
        "ManualMealRecommendationDelivery",
        "MessageScopedMealReportMessageHandler",
        "MealPlanDeliveryError",
        "MealPlanResendError",
        "MealRecommendationReplacementDeliveryError",
        "MealReportContextError",
        "MealReportMessageHandler",
        "MealWorkflowMessageError",
        "NO_ACTIVE_MEAL_PLAN_REPLY_TEXT",
        "ProductionMealReportRuntime",
        "open_production_meal_report_runtime",
        "send_manual_recommendation",
    }
)

_RECOMMENDATION_SCHEDULER_EXPORTS = frozenset(
    {
        "run_production_scheduler_once",
        "scheduler_status",
    }
)


def __getattr__(name: str) -> Any:
    """Load webhook exports lazily so ``python -m ...webhook`` stays clean."""

    if name in _APPLICATION_EXPORTS:
        application = import_module(f"{__name__}.application")
        value = getattr(application, name)
        globals()[name] = value
        return value
    if name in _WEBHOOK_EXPORTS:
        webhook = import_module(f"{__name__}.webhook")
        value = getattr(webhook, name)
        globals()[name] = value
        return value
    if name in _POLLING_EXPORTS:
        polling = import_module(f"{__name__}.polling")
        value = getattr(polling, name)
        globals()[name] = value
        return value
    if name in _MEAL_WORKFLOW_EXPORTS:
        meal_workflow = import_module(f"{__name__}.meal_workflow")
        value = getattr(meal_workflow, name)
        globals()[name] = value
        return value
    if name in _RECOMMENDATION_SCHEDULER_EXPORTS:
        scheduler = import_module(f"{__name__}.recommendation_scheduler")
        value = getattr(scheduler, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BASE_URL_ENV_VAR",
    "CHAT_GUID_ENV_VAR",
    "ApplicationMessageError",
    "ApplicationMessagingConfig",
    "AMBIGUOUS_ACTIVE_MEAL_PLAN_REPLY_TEXT",
    "ActiveMealPlanContextResolver",
    "ConversationalMealReportMessageHandler",
    "BlueBubblesAPIError",
    "BlueBubblesClient",
    "BlueBubblesError",
    "BlueBubblesNetworkError",
    "BlueBubblesPollingWorker",
    "BlueBubblesWebhookServer",
    "DEFAULT_BASE_URL",
    "DEFAULT_MESSAGE_QUERY_LIMIT",
    "DEFAULT_POLL_INTERVAL",
    "DEFAULT_POLL_LIMIT",
    "DEFAULT_RECENT_GUID_CACHE_SIZE",
    "DEFAULT_REPLY_TEXT",
    "SEMANTIC_FAILURE_REPLY_TEXT",
    "DEFAULT_TIMEOUT",
    "DEFAULT_WEBHOOK_HOST",
    "DEFAULT_WEBHOOK_PORT",
    "IncomingMessage",
    "CURSOR_PATH_ENV_VAR",
    "CursorStateError",
    "MESSAGE_QUERY_PATH",
    "ManualMealRecommendationDelivery",
    "MessageScopedMealReportMessageHandler",
    "MealPlanDeliveryError",
    "MealPlanResendError",
    "MealRecommendationReplacementDeliveryError",
    "MealReportContextError",
    "MealReportMessageHandler",
    "MealWorkflowMessageError",
    "NO_ACTIVE_MEAL_PLAN_REPLY_TEXT",
    "NutritionOptimizerMessageHandler",
    "OutboundTextSender",
    "PASSWORD_ENV_VAR",
    "RecentMessageGuids",
    "RowIDCursorStore",
    "PollingResponseError",
    "ProductionMealReportRuntime",
    "QueriedMessage",
    "SendMessageResult",
    "SENDER_ADDRESS_ENV_VAR",
    "UNSUPPORTED_INTENT_REPLY_TEXT",
    "create_webhook_server",
    "default_message_handler",
    "default_cursor_path",
    "deliver_incoming_message",
    "parse_webhook_event",
    "parse_message_query_payload",
    "open_production_meal_report_runtime",
    "send_manual_recommendation",
    "run_production_scheduler_once",
    "scheduler_status",
]
