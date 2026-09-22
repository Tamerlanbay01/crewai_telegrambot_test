"""Application service for resolving reusable system agents."""

from sqlalchemy.ext.asyncio import AsyncSession

from models.system_agent import (
    RuntimeSystemAgent,
    SystemAgentTemplate,
    SystemAgentTemplateCreate,
    UserAgentOverride,
    UserAgentOverrideCreate,
    UserAgentOverrideUpdate,
)
from repositories.system_agent import SystemAgentRepository


INITIAL_TEMPLATES = (
    SystemAgentTemplateCreate(
        key="information",
        role="Information Specialist",
        goal="Find, verify, and summarize information for the user.",
        backstory="You provide concise, well-sourced research assistance.",
    ),
    SystemAgentTemplateCreate(
        key="mail",
        role="Mail Assistant",
        goal="Help the user understand and prepare email-related work.",
        backstory="You prepare drafts and organize mail work without sending messages yourself.",
    ),
    SystemAgentTemplateCreate(
        key="calendar",
        role="Calendar Assistant",
        goal="Help the user plan and organize calendar-related work.",
        backstory="You propose calendar changes without performing external side effects.",
    ),
)

INTERNAL_SYSTEM_AGENT_KEYS = frozenset({"memory"})
MEMORY_TEMPLATE = SystemAgentTemplateCreate(
    key="memory",
    role="Memory Curator",
    goal="Extract only durable, useful memory candidates from completed runs.",
    backstory="You propose structured candidates and never access persistence directly.",
)


class SystemAgentService:
    def __init__(self, session: AsyncSession):
        self._session = session
        self._templates = SystemAgentRepository(session)

    async def ensure_initial_templates(self) -> list[SystemAgentTemplate]:
        """Create the built-in v1 templates once; safe to call during startup."""
        templates: list[SystemAgentTemplate] = []
        for definition in INITIAL_TEMPLATES:
            template = await self._templates.get_template(
                key=definition.key, version=definition.version
            )
            if template is None:
                template = await self._templates.create_template(definition)
            templates.append(template)
        await self._session.commit()
        return templates

    async def ensure_memory_template(self) -> SystemAgentTemplate:
        """Create the hidden pipeline-only Memory Curator lazily and idempotently."""
        template = await self._templates.get_template(
            key=MEMORY_TEMPLATE.key, version=MEMORY_TEMPLATE.version
        )
        if template is None:
            template = await self._templates.create_template(MEMORY_TEMPLATE)
            await self._session.commit()
        return template

    async def resolve_for_user(
        self, *, user_id: int, key: str
    ) -> RuntimeSystemAgent | None:
        template = await self._templates.get_template(key=key)
        if template is None or not template.enabled:
            return None
        override = await self._templates.get_override(
            user_id=user_id, template_id=template.id
        )
        if override is not None and not override.enabled:
            return None
        return self._resolve(template, override)

    async def list_available_for_user(self, *, user_id: int) -> list[RuntimeSystemAgent]:
        return [
            agent
            for agent in await self.list_for_user(user_id=user_id)
            if agent.enabled and agent.key not in INTERNAL_SYSTEM_AGENT_KEYS
        ]

    async def list_for_user(self, *, user_id: int) -> list[RuntimeSystemAgent]:
        """List configured system agents, including those disabled by an override."""
        resolved: list[RuntimeSystemAgent] = []
        for template in await self._templates.list_latest_templates(enabled_only=False):
            if template.key in INTERNAL_SYSTEM_AGENT_KEYS:
                continue
            override = await self._templates.get_override(
                user_id=user_id, template_id=template.id
            )
            resolved.append(self._resolve(template, override))
        return resolved

    async def set_override(
        self, *, user_id: int, key: str, data: UserAgentOverrideUpdate
    ) -> UserAgentOverride:
        template = await self._templates.get_template(key=key)
        if template is None:
            raise LookupError(f"System agent template not found: {key}")
        override = await self._templates.update_override(
            user_id=user_id, template_id=template.id, data=data
        )
        if override is None:
            override = await self._templates.create_override(
                UserAgentOverrideCreate(
                    user_id=user_id,
                    template_id=template.id,
                    **data.model_dump(exclude_unset=True),
                )
            )
        await self._session.commit()
        return override

    @staticmethod
    def _resolve(
        template: SystemAgentTemplate, override: UserAgentOverride | None
    ) -> RuntimeSystemAgent:
        return RuntimeSystemAgent(
            key=template.key,
            version=template.version,
            name=(override.custom_name if override and override.custom_name else template.role),
            role=template.role,
            goal=template.goal,
            backstory=template.backstory,
            custom_instructions=(override.custom_instructions if override else None),
            allowed_skills=template.allowed_skills,
            default_skills=template.default_skills,
            memory_policy=(override.memory_policy if override else None),
            enabled=(template.enabled and (override.enabled if override else True)),
        )
