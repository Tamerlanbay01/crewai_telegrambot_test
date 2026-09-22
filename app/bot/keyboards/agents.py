from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from models.agent import Agent
from models.system_agent import RuntimeSystemAgent


def agents_keyboard(
    agents: list[Agent], system_agents: list[RuntimeSystemAgent]
) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text=f"Open: {agent.name[:40]}", callback_data=f"agent:show:{agent.id}")]
        for agent in agents
    ]
    rows.append([InlineKeyboardButton(text="Create agent", callback_data="agent:new")])
    for agent in system_agents:
        action = "disable" if agent.enabled else "enable"
        state = "enabled" if agent.enabled else "disabled"
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{agent.key}: {state}",
                    callback_data=f"system:{action}:{agent.key}",
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def agent_details_keyboard(agent: Agent) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Archive", callback_data=f"agent:archive:{agent.id}")],
            [InlineKeyboardButton(text="Back to agents", callback_data="agent:list")],
        ]
    )


def agent_spawn_keyboard(wizard_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Yes", callback_data=f"agent:spawn:yes:{wizard_id}"),
                InlineKeyboardButton(text="No", callback_data=f"agent:spawn:no:{wizard_id}"),
            ],
            [InlineKeyboardButton(text="Cancel", callback_data=f"agent:create:cancel:{wizard_id}")],
        ]
    )


def agent_confirmation_keyboard(wizard_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Create agent", callback_data=f"agent:create:confirm:{wizard_id}"),
                InlineKeyboardButton(text="Cancel", callback_data=f"agent:create:cancel:{wizard_id}"),
            ]
        ]
    )
