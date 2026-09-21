"""Backend authority for agent permissions."""

from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from models.permission import ActionClass, Permission, PermissionCreate, PermissionSubjectType
from repositories.permission import PermissionRepository


class PermissionDeniedError(PermissionError):
    """Raised when an agent has no matching allowed backend permission."""


class PermissionService:
    def __init__(self, session: AsyncSession):
        self._session = session
        self._permissions = PermissionRepository(session)

    async def check(
        self,
        *,
        user_id: int,
        subject_type: PermissionSubjectType,
        subject_id: str,
        action_class: ActionClass,
        resource: str,
    ) -> bool:
        permissions = await self._permissions.list_for_action(
            user_id=user_id,
            subject_type=subject_type,
            subject_id=subject_id,
            action_class=action_class,
        )
        matches = [
            permission
            for permission in permissions
            if self._scope_matches(permission.resource_scope, resource)
        ]
        if any(not permission.allowed for permission in matches):
            return False
        return any(permission.allowed for permission in matches)

    async def require(
        self,
        *,
        user_id: int,
        subject_type: PermissionSubjectType,
        subject_id: str,
        action_class: ActionClass,
        resource: str,
    ) -> None:
        if not await self.check(
            user_id=user_id,
            subject_type=subject_type,
            subject_id=subject_id,
            action_class=action_class,
            resource=resource,
        ):
            raise PermissionDeniedError(
                f"Permission denied for {subject_type.value}:{subject_id} "
                f"to {action_class.value} {resource}"
            )

    async def list_allowed_scopes(
        self,
        *,
        user_id: int,
        subject_type: PermissionSubjectType,
        subject_id: str,
    ) -> list[str]:
        permissions = await self._permissions.list_for_subject(
            user_id=user_id,
            subject_type=subject_type,
            subject_id=subject_id,
        )
        return [
            f"{permission.action_class.value}:{permission.resource_scope}"
            for permission in permissions
            if permission.allowed
        ]

    async def grant(
        self,
        *,
        user_id: int,
        subject_type: PermissionSubjectType,
        subject_id: str,
        action_class: ActionClass,
        resource_scope: str,
    ) -> Permission:
        clean_subject = subject_id.strip()
        clean_scope = resource_scope.strip()
        if not clean_subject or not clean_scope:
            raise ValueError("Permission subject and resource scope cannot be empty")
        existing = await self._permissions.get_exact(
            user_id=user_id,
            subject_type=subject_type,
            subject_id=clean_subject,
            action_class=action_class,
            resource_scope=clean_scope,
        )
        if existing is None:
            permission = await self._permissions.create(
                PermissionCreate(
                    user_id=user_id,
                    subject_type=subject_type,
                    subject_id=clean_subject,
                    action_class=action_class,
                    resource_scope=clean_scope,
                )
            )
        else:
            permission = await self._permissions.set_decision(
                permission_id=existing.id, allowed=True, revoked_at=None
            )
            assert permission is not None
        await self._session.commit()
        return permission

    async def revoke(
        self,
        *,
        user_id: int,
        subject_type: PermissionSubjectType,
        subject_id: str,
        action_class: ActionClass,
        resource_scope: str,
    ) -> Permission:
        existing = await self._permissions.get_exact(
            user_id=user_id,
            subject_type=subject_type,
            subject_id=subject_id,
            action_class=action_class,
            resource_scope=resource_scope,
        )
        if existing is None:
            raise LookupError("Permission not found")
        permission = await self._permissions.set_decision(
            permission_id=existing.id,
            allowed=False,
            revoked_at=datetime.now(timezone.utc),
        )
        assert permission is not None
        await self._session.commit()
        return permission

    @staticmethod
    def _scope_matches(scope: str, resource: str) -> bool:
        if scope == "*" or scope == resource:
            return True
        if scope.endswith("*"):
            return resource.startswith(scope[:-1])
        return False
