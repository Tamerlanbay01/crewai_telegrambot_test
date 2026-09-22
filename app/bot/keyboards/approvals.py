from uuid import UUID

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def approval_keyboard(approval_id: UUID) -> InlineKeyboardMarkup:
    approval_key = str(approval_id)
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Approve",
                    callback_data=f"approval:approve:{approval_key}",
                ),
                InlineKeyboardButton(
                    text="Reject",
                    callback_data=f"approval:reject:{approval_key}",
                ),
            ]
        ]
    )
