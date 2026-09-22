"""Async wrapper around a deliberately small S3-compatible storage surface."""

import asyncio
import mimetypes
from collections.abc import Sequence
from typing import Any

from core.config import S3Config
from integrations.storage.skill_storage import (
    SkillStorageConflictError,
    SkillStorageError,
)
from models.skill import SkillPackageFile


class S3SkillStorage:
    """S3/MinIO adapter using boto3 calls off the event loop.

    ``boto3`` is imported lazily so unit tests and installations which do not use
    package storage do not need the optional dependency merely to import the app.
    """

    def __init__(self, config: S3Config) -> None:
        if not config.endpoint:
            raise SkillStorageError("S3_ENDPOINT must be configured for skill package storage")
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on deployment extras
            raise SkillStorageError(
                "Install requirements-storage.txt to enable S3 skill package storage"
            ) from exc

        endpoint = config.endpoint
        if "://" not in endpoint:
            endpoint = f"{'https' if config.use_ssl else 'http'}://{endpoint}"
        self._bucket = config.skill_bucket
        self._client = boto3.session.Session(
            aws_access_key_id=config.access_key or None,
            aws_secret_access_key=config.secret_key or None,
            region_name=config.region,
        ).client("s3", endpoint_url=endpoint, use_ssl=config.use_ssl)

    async def upload_package(
        self, *, prefix: str, files: Sequence[SkillPackageFile], checksum: str
    ) -> None:
        existing = await self.get_package_checksum(prefix=prefix)
        if existing is not None:
            if existing == checksum:
                return
            raise SkillStorageConflictError(f"Package prefix already exists: {prefix}")
        if await self.exists(prefix=prefix):
            # Without manifest.json there is no commit marker, so every object
            # currently under the prefix is an incomplete, safely replaceable upload.
            await self.delete_version(prefix=prefix)

        # manifest.json is written last: it is the commit marker that carries the checksum.
        for item in sorted(files, key=lambda value: (value.path == "manifest.json", value.path)):
            await self._call(
                self._client.put_object,
                Bucket=self._bucket,
                Key=f"{prefix}{item.path}",
                Body=item.content,
                ContentType=mimetypes.guess_type(item.path)[0] or "application/octet-stream",
                Metadata={"package-checksum": checksum},
            )

    async def get_file(self, *, prefix: str, path: str, max_bytes: int) -> bytes:
        def read() -> bytes:
            response = self._client.get_object(Bucket=self._bucket, Key=f"{prefix}{path}")
            length = response.get("ContentLength")
            if isinstance(length, int) and length > max_bytes:
                raise SkillStorageError(f"Skill package file exceeds {max_bytes} bytes")
            content = response["Body"].read(max_bytes + 1)
            if len(content) > max_bytes:
                raise SkillStorageError(f"Skill package file exceeds {max_bytes} bytes")
            return content

        try:
            return await asyncio.to_thread(read)
        except SkillStorageError:
            raise
        except Exception as exc:
            raise SkillStorageError(f"Unable to read skill package file {path}") from exc

    async def list_files(self, *, prefix: str) -> list[str]:
        files: list[str] = []
        continuation: str | None = None
        try:
            while True:
                params: dict[str, Any] = {"Bucket": self._bucket, "Prefix": prefix}
                if continuation is not None:
                    params["ContinuationToken"] = continuation
                response = await self._call(self._client.list_objects_v2, **params)
                files.extend(
                    item["Key"][len(prefix) :]
                    for item in response.get("Contents", [])
                    if item["Key"].startswith(prefix) and item["Key"] != prefix
                )
                if not response.get("IsTruncated"):
                    return sorted(files)
                continuation = response.get("NextContinuationToken")
                if not continuation:
                    return sorted(files)
        except Exception as exc:
            if isinstance(exc, SkillStorageError):
                raise
            raise SkillStorageError(f"Unable to list skill package files under {prefix}") from exc

    async def delete_version(self, *, prefix: str) -> None:
        files = await self.list_files(prefix=prefix)
        try:
            for start in range(0, len(files), 1000):
                batch = files[start : start + 1000]
                await self._call(
                    self._client.delete_objects,
                    Bucket=self._bucket,
                    Delete={"Objects": [{"Key": f"{prefix}{path}"} for path in batch], "Quiet": True},
                )
        except Exception as exc:
            if isinstance(exc, SkillStorageError):
                raise
            raise SkillStorageError(f"Unable to delete skill package {prefix}") from exc

    async def exists(self, *, prefix: str) -> bool:
        try:
            response = await self._call(
                self._client.list_objects_v2, Bucket=self._bucket, Prefix=prefix, MaxKeys=1
            )
            return bool(response.get("Contents"))
        except Exception as exc:
            raise SkillStorageError(f"Unable to inspect skill package {prefix}") from exc

    async def get_package_checksum(self, *, prefix: str) -> str | None:
        try:
            response = await asyncio.to_thread(
                self._client.head_object, Bucket=self._bucket, Key=f"{prefix}manifest.json"
            )
            value = response.get("Metadata", {}).get("package-checksum")
            return value if isinstance(value, str) else None
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if str(code) in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise SkillStorageError(f"Unable to inspect skill package checksum for {prefix}") from exc

    @staticmethod
    async def _call(function: Any, /, *args: Any, **kwargs: Any) -> Any:
        try:
            return await asyncio.to_thread(function, *args, **kwargs)
        except Exception as exc:
            raise SkillStorageError("S3-compatible storage operation failed") from exc
