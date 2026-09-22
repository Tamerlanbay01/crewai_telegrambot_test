"""Factory for the configured skill sandbox implementation."""

from core.config import SandboxConfig
from integrations.sandbox.docker import DockerSkillSandbox
from integrations.sandbox.executor import SkillSandbox


def create_skill_sandbox(config: SandboxConfig) -> SkillSandbox:
    return DockerSkillSandbox(image=config.python_image)
