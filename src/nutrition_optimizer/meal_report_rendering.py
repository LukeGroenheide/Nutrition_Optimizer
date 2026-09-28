"""Deterministic, concise presentation of reconciled meal-report outcomes."""

from __future__ import annotations

from .meal_identity import meal_name_for_display
from .meal_report import ReconciledMealReport


__all__ = [
    "format_meal_report_clarification",
    "format_meal_report_confirmation",
]


def format_meal_report_confirmation(report: ReconciledMealReport) -> str:
    """Render a safe applied report without adding any semantic interpretation."""

    if not isinstance(report, ReconciledMealReport):
        raise TypeError("report must be a ReconciledMealReport")
    if report.clarification_items:
        raise ValueError("a clarification-required report cannot be confirmed")

    if (
        len(report.eaten_items) == len(report.plan.items)
        and not report.skipped_items
        and not report.unplanned_items
    ):
        return (
            "Got it — I logged everything from your "
            f"{meal_name_for_display(report.plan.meal)} plan."
        )

    lines: list[str] = []
    if report.eaten_items:
        lines.append("I understood that as:")
        lines.extend(f"- {item.plan_item.display_name}" for item in report.eaten_items)
    if report.skipped_items:
        if lines:
            lines.append("")
        lines.append("Skipped:")
        lines.extend(f"- {item.plan_item.display_name}" for item in report.skipped_items)
    if report.unplanned_items:
        if lines:
            lines.append("")
        lines.append("Also logged (unplanned):")
        lines.extend(f"- {item.food_text}" for item in report.unplanned_items)
    if report.unspecified_items:
        if lines:
            lines.append("")
        lines.append("I left the other planned items unlogged.")
    if not lines:
        return "Got it — I left the planned items unlogged."
    return "\n".join(lines)


def format_meal_report_clarification(report: ReconciledMealReport) -> str:
    """Render structured clarification needs without inferring an intake outcome."""

    if not isinstance(report, ReconciledMealReport):
        raise TypeError("report must be a ReconciledMealReport")
    if not report.clarification_items:
        raise ValueError("report has no clarification items")

    lines = ["I need a little clarification before I log that:"]
    for item in report.clarification_items:
        if item.plan_item is not None:
            lines.append(
                f"- How much {item.plan_item.display_name} did you eat? "
                f"I had recommended {item.plan_item.natural_quantity_text}."
            )
        elif item.food_text is not None:
            if item.reason == "unplanned_food_ambiguous":
                lines.append(f"- Which current menu item did you mean by {item.food_text}?")
            elif item.reason == "unplanned_food_unresolved":
                lines.append(
                    f"- I couldn't match {item.food_text} to one current menu item. "
                    "What was it?"
                )
            else:
                lines.append(f"- How much {item.food_text} did you eat?")
        elif item.original_user_phrase is not None:
            lines.append(f"- Please clarify: {item.original_user_phrase}.")
        else:
            lines.append("- Please restate the unresolved part of the meal report.")
    return "\n".join(lines)
