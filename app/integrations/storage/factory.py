"""Construction of the configured skill-package storage adapter."""

from core.config import S3Config
from integrations.storage.s3 import S3SkillStorage
from integrations.storage.skill_storage import SkillStorage


def create_skill_storage(config: S3Config) -> SkillStorage:
    return S3SkillStorage(config)
