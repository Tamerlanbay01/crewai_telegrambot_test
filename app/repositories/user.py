from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.entities.user import UserEntity
from models.user import (
    User,
    UserCreate,
    UserUpdate,
)


class UserRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_id(
        self,
        user_id: int,
    ) -> User | None:
        entity = await self._get_entity_by_id(user_id)

        if entity is None:
            return None

        return User.model_validate(entity)

    async def get_by_telegram_id(
        self,
        telegram_id: int,
    ) -> User | None:
        result = await self._session.execute(
            select(UserEntity)
            .where(
                UserEntity.telegram_id == telegram_id
            )
        )

        entity = result.scalar_one_or_none()

        if entity is None:
            return None

        return User.model_validate(entity)

    async def create(
        self,
        data: UserCreate,
    ) -> User:
        entity = UserEntity(
            **data.model_dump()
        )

        self._session.add(entity)

        await self._session.flush()
        await self._session.refresh(entity)

        return User.model_validate(entity)

    async def update(
        self,
        user_id: int,
        data: UserUpdate,
    ) -> User | None:
        entity = await self._get_entity_by_id(user_id)

        if entity is None:
            return None

        changes = data.model_dump(
            exclude_unset=True
        )

        for field, value in changes.items():
            setattr(entity, field, value)

        await self._session.flush()
        await self._session.refresh(entity)

        return User.model_validate(entity)

    async def delete(
        self,
        user_id: int,
    ) -> bool:
        entity = await self._get_entity_by_id(user_id)

        if entity is None:
            return False

        await self._session.delete(entity)
        await self._session.flush()

        return True

    async def _get_entity_by_id(
        self,
        user_id: int,
    ) -> UserEntity | None:
        result = await self._session.execute(
            select(UserEntity)
            .where(UserEntity.id == user_id)
        )

        return result.scalar_one_or_none()