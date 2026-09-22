from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from models.agent import Agent
from models.schedule import Schedule, ScheduleStatus, ScheduleType


def schedules_keyboard(schedules: list[Schedule]) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                text=f"Open: {schedule.name[:40]} [{schedule.status.value.lower()}]",
                callback_data=f"schedule:show:{schedule.id}",
            )
        ]
        for schedule in schedules
    ]
    rows.append([InlineKeyboardButton(text="Create schedule", callback_data="schedule:new")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def schedule_details_keyboard(schedule: Schedule) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if schedule.status == ScheduleStatus.ACTIVE:
        rows.append([InlineKeyboardButton(text="Disable", callback_data=f"schedule:disable:{schedule.id}")])
    elif schedule.status == ScheduleStatus.DISABLED:
        rows.append([InlineKeyboardButton(text="Enable", callback_data=f"schedule:enable:{schedule.id}")])
    if schedule.status != ScheduleStatus.ARCHIVED:
        rows.append([InlineKeyboardButton(text="Archive", callback_data=f"schedule:archive:{schedule.id}")])
    rows.append([InlineKeyboardButton(text="Back to schedules", callback_data="schedule:list")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def schedule_agent_keyboard(agents: list[Agent], wizard_id: str) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text=agent.name[:50], callback_data=f"schedule:agent:{agent.id}:{wizard_id}")]
        for agent in agents
    ]
    rows.append([InlineKeyboardButton(text="Cancel", callback_data=f"schedule:create:cancel:{wizard_id}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def schedule_type_keyboard(wizard_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Once", callback_data=f"schedule:type:{ScheduleType.ONCE.value}:{wizard_id}")],
            [InlineKeyboardButton(text="Interval", callback_data=f"schedule:type:{ScheduleType.INTERVAL.value}:{wizard_id}")],
            [InlineKeyboardButton(text="Cron", callback_data=f"schedule:type:{ScheduleType.CRON.value}:{wizard_id}")],
            [InlineKeyboardButton(text="Cancel", callback_data=f"schedule:create:cancel:{wizard_id}")],
        ]
    )


def schedule_confirmation_keyboard(wizard_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Confirm schedule", callback_data=f"schedule:create:confirm:{wizard_id}")],
            [
                InlineKeyboardButton(text="Edit expression", callback_data=f"schedule:edit:expression:{wizard_id}"),
                InlineKeyboardButton(text="Edit timezone", callback_data=f"schedule:edit:timezone:{wizard_id}"),
            ],
            [InlineKeyboardButton(text="Cancel", callback_data=f"schedule:create:cancel:{wizard_id}")],
        ]
    )
