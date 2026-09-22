"""Create the current metadata schema for a fresh local Compose database."""

from __future__ import annotations

import asyncio
import importlib

from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import create_async_engine

from core.config import config
from database.base import Base


ENTITY_MODULES = (
    "database.entities.agent",
    "database.entities.agent_connection",
    "database.entities.agent_prompt",
    "database.entities.agent_run",
    "database.entities.agent_run_event",
    "database.entities.agent_skill",
    "database.entities.approval",
    "database.entities.chat",
    "database.entities.memory",
    "database.entities.memory_candidate",
    "database.entities.memory_extraction_run",
    "database.entities.message",
    "database.entities.permission",
    "database.entities.schedule",
    "database.entities.skill",
    "database.entities.skill_execution",
    "database.entities.system_agent_template",
    "database.entities.user",
    "database.entities.user_agent_override",
)


def _register_entities() -> None:
    for module_name in ENTITY_MODULES:
        importlib.import_module(module_name)


async def initialize_schema() -> None:
    if not config.database.url:
        raise RuntimeError("DATABASE_URL must be configured")
    _register_entities()
    for attempt in range(30):
        engine = create_async_engine(config.database.url, pool_pre_ping=True)
        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            return
        except OperationalError:
            if attempt == 29:
                raise
            await asyncio.sleep(2)
        finally:
            await engine.dispose()


if __name__ == "__main__":
    asyncio.run(initialize_schema())
