"""Storage port for immutable skill-package objects."""

from collections.abc import Sequence
from typing import Protocol

from models.skill import SkillPackageFile


class SkillStorageError(RuntimeError):
    """An object-storage operation could not be completed."""


class SkillStorageConflictError(SkillStorageError):
    """An immutable package prefix already contains different content."""


class SkillStorage(Protocol):
    async def upload_package(
        self, *, prefix: str, files: Sequence[SkillPackageFile], checksum: str
    ) -> None: ...

    async def get_file(self, *, prefix: str, path: str, max_bytes: int) -> bytes: ...

    async def list_files(self, *, prefix: str) -> list[str]: ...

    async def delete_version(self, *, prefix: str) -> None: ...

    async def exists(self, *, prefix: str) -> bool: ...

    async def get_package_checksum(self, *, prefix: str) -> str | None: ...
