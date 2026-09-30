from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def crew_design_keyboard(wizard_id: str, *, can_create: bool) -> InlineKeyboardMarkup:
    buttons = []
    if can_create:
        buttons.append(InlineKeyboardButton(
            text="Create crew", callback_data=f"crew:design:confirm:{wizard_id}",
        ))
    buttons.append(InlineKeyboardButton(
        text="Cancel", callback_data=f"crew:design:cancel:{wizard_id}",
    ))
    return InlineKeyboardMarkup(inline_keyboard=[buttons])
