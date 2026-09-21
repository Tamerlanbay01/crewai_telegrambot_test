"""Repository operations for permissions."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.entities.permission import PermissionEntity
from models.permission import (
    ActionClass,
    Permission,
    PermissionCreate,
    PermissionSubjectType,
)


class PermissionRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_id(self, permission_id: UUID) -> Permission | None:
        entity = (
            await self._session.execute(
                select(PermissionEntity).where(PermissionEntity.id == permission_id)
            )
        ).scalar_one_or_none()
        return Permission.model_validate(entity) if entity else None

    async def get_exact(
        self,
        *,
        user_id: int,
        subject_type: PermissionSubjectType,
        subject_id: str,
        action_class: ActionClass,
        resource_scope: str,
    ) -> Permission | None:
        entity = (
            await self._session.execute(
                select(PermissionEntity).where(
                    PermissionEntity.user_id == user_id,
                    PermissionEntity.subject_type == subject_type,
                    PermissionEntity.subject_id == subject_id,
                    PermissionEntity.action_class == action_class,
                    PermissionEntity.resource_scope == resource_scope,
                )
            )
        ).scalar_one_or_none()
        return Permission.model_validate(entity) if entity else None

    async def list_for_action(
        self,
        *,
        user_id: int,
        subject_type: PermissionSubjectType,
        subject_id: str,
        action_class: ActionClass,
    ) -> list[Permission]:
        entities = (
            await self._session.execute(
                select(PermissionEntity).where(
                    PermissionEntity.user_id == user_id,
                    PermissionEntity.subject_type == subject_type,
                    PermissionEntity.subject_id == subject_id,
                    PermissionEntity.action_class == action_class,
                )
            )
        ).scalars().all()
        return [Permission.model_validate(entity) for entity in entities]

    async def list_for_subject(
        self,
        *,
        user_id: int,
        subject_type: PermissionSubjectType,
        subject_id: str,
    ) -> list[Permission]:
        entities = (
            await self._session.execute(
                select(PermissionEntity).where(
                    PermissionEntity.user_id == user_id,
                    PermissionEntity.subject_type == subject_type,
                    PermissionEntity.subject_id == subject_id,
                )
            )
        ).scalars().all()
        return [Permission.model_validate(entity) for entity in entities]

    async def create(self, data: PermissionCreate) -> Permission:
        entity = PermissionEntity(**data.model_dump())
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return Permission.model_validate(entity)

    async def set_decision(
        self,
        *,
        permission_id: UUID,
        allowed: bool,
        revoked_at: datetime | None,
    ) -> Permission | None:
        entity = (
            await self._session.execute(
                select(PermissionEntity).where(PermissionEntity.id == permission_id)
            )
        ).scalar_one_or_none()
        if entity is None:
            return None
        entity.allowed = allowed
        entity.revoked_at = revoked_at
        await self._session.flush()
        await self._session.refresh(entity)
        return Permission.model_validate(entity)
