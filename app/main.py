"""Local MVP entry point: run with ``python app/main.py`` for Telegram polling."""

import asyncio
import logging
from collections.abc import Callable

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage
from sqlalchemy.ext.asyncio import AsyncSession

from bot.middlewares.database import DatabaseSessionMiddleware
from bot.router import router
from core.config import config

logger = logging.getLogger(__name__)


def create_dispatcher(session_factory: Callable[[], AsyncSession]) -> Dispatcher:
    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher.update.outer_middleware(DatabaseSessionMiddleware(session_factory))
    dispatcher.include_router(router)
    return dispatcher


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not config.telegram.token:
        raise RuntimeError("Set TELEGRAM_TOKEN before starting the Telegram bot")
    if not config.database.url:
        raise RuntimeError("Set DATABASE_URL before starting the Telegram bot")

    from database.database import close_database, session_factory
    from services.system_agent import SystemAgentService

    bot: Bot | None = None
    try:
        bot = Bot(
            token=config.telegram.token,
            default=DefaultBotProperties(parse_mode=None),
        )
        dispatcher = create_dispatcher(session_factory)
        async with session_factory() as session:
            await SystemAgentService(session).ensure_initial_templates()
        logger.info("System-agent templates are ready; starting long polling")
        await dispatcher.start_polling(bot, close_bot_session=False)
    finally:
        if bot is not None:
            await bot.session.close()
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
