"""Persistence operations for system-agent templates and user overrides."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.entities.system_agent_template import SystemAgentTemplateEntity
from database.entities.user_agent_override import UserAgentOverrideEntity
from models.system_agent import (
    SystemAgentTemplate,
    SystemAgentTemplateCreate,
    UserAgentOverride,
    UserAgentOverrideCreate,
    UserAgentOverrideUpdate,
)


class SystemAgentRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_template(
        self, *, key: str, version: int | None = None
    ) -> SystemAgentTemplate | None:
        statement = select(SystemAgentTemplateEntity).where(
            SystemAgentTemplateEntity.key == key
        )
        if version is not None:
            statement = statement.where(SystemAgentTemplateEntity.version == version)
        else:
            statement = statement.order_by(SystemAgentTemplateEntity.version.desc()).limit(1)
        entity = (await self._session.execute(statement)).scalar_one_or_none()
        return SystemAgentTemplate.model_validate(entity) if entity else None

    async def list_templates(self, *, enabled_only: bool = False) -> list[SystemAgentTemplate]:
        statement = select(SystemAgentTemplateEntity).order_by(
            SystemAgentTemplateEntity.key, SystemAgentTemplateEntity.version.desc()
        )
        if enabled_only:
            statement = statement.where(SystemAgentTemplateEntity.enabled.is_(True))
        return [
            SystemAgentTemplate.model_validate(entity)
            for entity in (await self._session.execute(statement)).scalars().all()
        ]

    async def list_latest_templates(self, *, enabled_only: bool = False) -> list[SystemAgentTemplate]:
        templates = await self.list_templates(enabled_only=enabled_only)
        latest: dict[str, SystemAgentTemplate] = {}
        for template in templates:
            latest.setdefault(template.key, template)
        return list(latest.values())

    async def create_template(self, data: SystemAgentTemplateCreate) -> SystemAgentTemplate:
        entity = SystemAgentTemplateEntity(**data.model_dump())
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return SystemAgentTemplate.model_validate(entity)

    async def get_override(
        self, *, user_id: int, template_id: UUID
    ) -> UserAgentOverride | None:
        entity = (
            await self._session.execute(
                select(UserAgentOverrideEntity).where(
                    UserAgentOverrideEntity.user_id == user_id,
                    UserAgentOverrideEntity.template_id == template_id,
                )
            )
        ).scalar_one_or_none()
        return UserAgentOverride.model_validate(entity) if entity else None

    async def create_override(self, data: UserAgentOverrideCreate) -> UserAgentOverride:
        entity = UserAgentOverrideEntity(**data.model_dump())
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return UserAgentOverride.model_validate(entity)

    async def update_override(
        self, *, user_id: int, template_id: UUID, data: UserAgentOverrideUpdate
    ) -> UserAgentOverride | None:
        result = await self._session.execute(
            select(UserAgentOverrideEntity).where(
                UserAgentOverrideEntity.user_id == user_id,
                UserAgentOverrideEntity.template_id == template_id,
            )
        )
        entity = result.scalar_one_or_none()
        if entity is None:
            return None
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(entity, field, value)
        await self._session.flush()
        await self._session.refresh(entity)
        return UserAgentOverride.model_validate(entity)
