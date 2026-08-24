from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User, UserCreate, UserUpdate
from repositories.user import UserRepository


class UserService:
    def __init__(self, session: AsyncSession):
        self._session = session
        self._users = UserRepository(session)

    async def get_or_create(
        self,
        *,
        telegram_id: int,
        username: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
    ) -> User:
        user = await self._users.get_by_telegram_id(
            telegram_id
        )

        if user is not None:
            return user

        user = await self._users.create(
            UserCreate(
                telegram_id=telegram_id,
                username=username,
                first_name=first_name,
                last_name=last_name,
            )
        )

        await self._session.commit()

        return user

    async def update_profile(
        self,
        *,
        user_id: int,
        username: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
    ) -> User:
        user = await self._users.update(
            user_id,
            UserUpdate(
                username=username,
                first_name=first_name,
                last_name=last_name,
            ),
        )

        if user is None:
            raise LookupError(
                f"User not found: {user_id}"
            )

        await self._session.commit()

        return user