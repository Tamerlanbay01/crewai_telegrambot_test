"""Isolated execution adapters for backend-owned executable skills."""

from integrations.sandbox.executor import SkillSandbox
from integrations.sandbox.fake import FakeSkillSandbox

__all__ = ["FakeSkillSandbox", "SkillSandbox"]
