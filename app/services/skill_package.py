"""Validation and immutable persistence rules for skill packages."""

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Callable

from core.config import S3Config
from integrations.storage.skill_storage import SkillStorage
from models.skill import (
    PreparedSkillPackage,
    SkillOwnerType,
    SkillPackage,
    SkillPackageFile,
)


class SkillPackageValidationError(ValueError):
    """The uploaded package does not meet the safe, stable package contract."""


class SkillPackageService:
    """Keeps package-format concerns outside the skill ownership service."""

    def __init__(self, *, storage_provider: Callable[[], SkillStorage], config: S3Config) -> None:
        self._storage_provider = storage_provider
        self._config = config

    def validate(self, *, package: SkillPackage, key: str, version: int) -> PreparedSkillPackage:
        files: dict[str, SkillPackageFile] = {}
        package_bytes = 0
        for item in package.canonical_files():
            self.validate_path(item.path)
            if item.path in files:
                raise SkillPackageValidationError(f"Duplicate package file: {item.path}")
            if len(item.content) > self._config.max_skill_file_bytes:
                raise SkillPackageValidationError(
                    f"Package file {item.path} exceeds MAX_SKILL_FILE_BYTES"
                )
            package_bytes += len(item.content)
            if package_bytes > self._config.max_skill_package_bytes:
                raise SkillPackageValidationError("Package exceeds MAX_SKILL_PACKAGE_BYTES")
            files[item.path] = item

        if "SKILL.md" not in files:
            raise SkillPackageValidationError("Skill package must include SKILL.md")
        if "manifest.json" not in files:
            raise SkillPackageValidationError("Skill package must include manifest.json")
        try:
            manifest_value = json.loads(files["manifest.json"].content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SkillPackageValidationError("manifest.json must be valid UTF-8 JSON") from exc
        if not isinstance(manifest_value, dict):
            raise SkillPackageValidationError("manifest.json must contain a JSON object")
        self.validate_manifest(
            manifest_value,
            key=key,
            version=version,
            package_files=set(files),
        )
        ordered = tuple(files[path] for path in sorted(files))
        return PreparedSkillPackage(
            files=ordered,
            manifest=manifest_value,
            checksum=self.package_checksum(ordered),
        )

    async def upload(self, *, prefix: str, package: PreparedSkillPackage) -> None:
        await self._storage_provider().upload_package(
            prefix=prefix, files=package.files, checksum=package.checksum
        )

    @staticmethod
    def validate_path(path: str) -> None:
        if not path or "\\" in path or "\x00" in path or re.match(r"^[A-Za-z]:", path):
            raise SkillPackageValidationError("Package paths must be relative POSIX paths")
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
            raise SkillPackageValidationError("Package paths cannot escape the package root")
        if path in {"SKILL.md", "manifest.json"}:
            return
        if not candidate.parts or candidate.parts[0] not in {"src", "assets", "resources"}:
            raise SkillPackageValidationError(
                "Optional package files must be under src/, assets/, or resources/"
            )

    @staticmethod
    def package_checksum(files: tuple[SkillPackageFile, ...]) -> str:
        digest = hashlib.sha256()
        for item in files:
            encoded_path = item.path.encode("utf-8")
            digest.update(len(encoded_path).to_bytes(4, "big"))
            digest.update(encoded_path)
            digest.update(len(item.content).to_bytes(8, "big"))
            digest.update(item.content)
        return digest.hexdigest()

    @staticmethod
    def object_prefix(*, owner_type: SkillOwnerType, owner_user_id: int | None, key: str, version: int) -> str:
        if owner_type == SkillOwnerType.SYSTEM:
            return f"system/{key}/v{version}/"
        if owner_user_id is None:
            raise SkillPackageValidationError("User skill package requires an owner")
        return f"users/{owner_user_id}/{key}/v{version}/"

    def storage_uri(self, *, prefix: str) -> str:
        return f"s3://{self._config.skill_bucket}/{prefix}"

    @staticmethod
    def validate_manifest(
        manifest: dict[str, object],
        *,
        key: str,
        version: int,
        package_files: set[str] | None,
    ) -> None:
        if manifest.get("key") != key:
            raise SkillPackageValidationError("manifest.key must match Skill.key")
        if manifest.get("version") != version:
            raise SkillPackageValidationError("manifest.version must match Skill.version")
        unsupported_execution_fields = {
            "docker_image",
            "image",
            "network",
            "privileged",
            "mounts",
            "host_paths",
        }
        unsupported = unsupported_execution_fields.intersection(manifest)
        if unsupported:
            raise SkillPackageValidationError(
                f"Manifest execution fields are backend-controlled: {sorted(unsupported)[0]}"
            )

        runtime = manifest.get("runtime")
        if runtime is not None and runtime != "python":
            raise SkillPackageValidationError("Only the python skill runtime is supported")
        entrypoint = manifest.get("entrypoint")
        if entrypoint is not None and not isinstance(entrypoint, str):
            raise SkillPackageValidationError("manifest.entrypoint must be a string or null")
        if entrypoint is None:
            if runtime is not None:
                raise SkillPackageValidationError(
                    "Instructional skills must not declare a runtime"
                )
        else:
            if runtime != "python":
                raise SkillPackageValidationError(
                    "Executable skills must declare runtime=python"
                )
            try:
                SkillPackageService.validate_path(entrypoint)
            except SkillPackageValidationError as exc:
                raise SkillPackageValidationError(
                    "Executable entrypoint must be a safe relative path"
                ) from exc
            if not entrypoint.startswith("src/") or not entrypoint.endswith(".py"):
                raise SkillPackageValidationError(
                    "Executable entrypoint must be a Python file under src/"
                )
            if package_files is not None and entrypoint not in package_files:
                raise SkillPackageValidationError("Executable entrypoint is missing from the package")
        external_side_effect = manifest.get("external_side_effect", False)
        if not isinstance(external_side_effect, bool):
            raise SkillPackageValidationError("manifest.external_side_effect must be boolean")
        description = manifest.get("description")
        if description is not None and not isinstance(description, str):
            raise SkillPackageValidationError("manifest.description must be a string")
        listed_files = manifest.get("files")
        if listed_files is not None and (
            not isinstance(listed_files, list) or not all(isinstance(item, str) for item in listed_files)
        ):
            raise SkillPackageValidationError("manifest.files must be a list of strings")
        if listed_files is not None and package_files is not None:
            actual_files = package_files - {"SKILL.md", "manifest.json"}
            if len(listed_files) != len(set(listed_files)) or set(listed_files) != actual_files:
                raise SkillPackageValidationError(
                    "manifest.files must match the package's optional file paths"
                )
