import asyncio
from collections.abc import Awaitable, Callable
import importlib.util
import unittest
from uuid import uuid4

SQLALCHEMY_AVAILABLE = importlib.util.find_spec("sqlalchemy") is not None

if SQLALCHEMY_AVAILABLE:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from database.base import Base
    from database.entities.agent import AgentEntity
    from database.entities.agent_connection import AgentConnectionEntity
    from database.entities.agent_prompt import AgentPromptVersionEntity
    from database.entities.agent_run import AgentRunEntity
    from database.entities.agent_run_event import AgentRunEventEntity
    from database.entities.chat import ChatEntity
    from database.entities.message import MessageEntity
    from database.entities.system_agent_template import SystemAgentTemplateEntity
    from database.entities.user import UserEntity
    from database.entities.user_agent_override import UserAgentOverrideEntity
    from models.agent_run import AgentRunStatus
    from models.agent import AgentStatus, AgentUpdate
    from models.system_agent import UserAgentOverrideUpdate
    from services.agent import AgentService
    from services.agent_run import AgentRunService
    from services.system_agent import SystemAgentService


async def _with_session(test: Callable[[AsyncSession], Awaitable[None]]) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            session.add_all(
                [
                    UserEntity(id=1, telegram_id=101),
                    UserEntity(id=2, telegram_id=202),
                ]
            )
            await session.commit()
            await test(session)
    finally:
        await engine.dispose()


@unittest.skipUnless(
    SQLALCHEMY_AVAILABLE and importlib.util.find_spec("aiosqlite"),
    "SQLAlchemy with aiosqlite is not installed",
)
class AgentFoundationTests(unittest.TestCase):
    def test_agent_domain_rules(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            service = AgentService(session)
            primary = await service.ensure_primary_agent(user_id=1)
            assert (await service.ensure_primary_agent(user_id=1)).id == primary.id

            first = await service.create_user_agent(
                user_id=1, name="Python", role="Python expert", goal="Explain async"
            )
            second = await service.create_user_agent(
                user_id=1, name="Research", role="Researcher", goal="Find facts"
            )
            assert (await service.get_current_prompt(user_id=1, agent_id=first.id)).version == 1
            assert [connection.child_agent_id for connection in await service.list_connections(
                user_id=1, parent_agent_id=primary.id
            )] == [first.id, second.id]

            duplicate = await service.connect_agents(
                user_id=1, parent_agent_id=primary.id, child_agent_id=first.id
            )
            assert duplicate.child_agent_id == first.id
            with self.assertRaisesRegex(ValueError, "primary"):
                await service.connect_agents(user_id=1, parent_agent_id=first.id, child_agent_id=second.id)
            with self.assertRaisesRegex(ValueError, "itself"):
                await service.connect_agents(user_id=1, parent_agent_id=first.id, child_agent_id=first.id)
            with self.assertRaisesRegex(ValueError, "empty"):
                await service.update_prompt(user_id=1, agent_id=first.id, role=" ", goal="valid")

            updated_prompt = await service.update_prompt(
                user_id=1, agent_id=first.id, role="Python architect", goal="Explain async well"
            )
            assert updated_prompt.version == 2
            with self.assertRaisesRegex(ValueError, "Primary"):
                await service.archive_agent(user_id=1, agent_id=primary.id)
            with self.assertRaises(LookupError):
                await service.get_agent(user_id=2, agent_id=first.id)
            await service.archive_agent(user_id=1, agent_id=second.id)
            disabled = await service._agents.update(
                first.id, AgentUpdate(status=AgentStatus.DISABLED)
            )
            assert disabled is not None
            await session.commit()
            assert await service.list_active_agents(1) == [primary]

        asyncio.run(_with_session(scenario))


    def test_system_agents_are_templates_with_user_overrides(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            service = SystemAgentService(session)
            templates = await service.ensure_initial_templates()
            assert {template.key for template in templates} == {"information", "mail", "calendar"}
            assert len(await service.ensure_initial_templates()) == 3

            override = await service.set_override(
                user_id=1,
                key="information",
                data=UserAgentOverrideUpdate(
                    custom_name="My Researcher", custom_instructions="Use bullet points"
                ),
            )
            resolved = await service.resolve_for_user(user_id=1, key="information")
            assert override.user_id == 1
            assert resolved is not None
            assert resolved.name == "My Researcher"
            assert resolved.custom_instructions == "Use bullet points"

        asyncio.run(_with_session(scenario))


    def test_agent_run_state_machine_and_tenant_isolation(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            agents = AgentService(session)
            agent = await agents.create_user_agent(
                user_id=1, name="Python", role="Python expert", goal="Explain async"
            )
            own_chat_id, other_chat_id = uuid4(), uuid4()
            own_message_id, other_message_id = uuid4(), uuid4()
            session.add_all(
                [
                    ChatEntity(id=own_chat_id, user_id=1, title="Own"),
                    ChatEntity(id=other_chat_id, user_id=2, title="Other"),
                    MessageEntity(id=own_message_id, chat_id=own_chat_id, role="user", content="Hello"),
                    MessageEntity(id=other_message_id, chat_id=other_chat_id, role="user", content="Other"),
                ]
            )
            await session.commit()
            runs = AgentRunService(session)
            run = await runs.create_run(
                user_id=1,
                starting_agent_id=agent.id,
                model_name="test-model",
                chat_id=own_chat_id,
                message_id=own_message_id,
            )
            assert run.status == AgentRunStatus.CREATED
            assert run.chat_id == own_chat_id
            assert run.message_id == own_message_id
            with self.assertRaises(LookupError):
                await runs.create_run(
                    user_id=1, starting_agent_id=agent.id, model_name="test-model", chat_id=other_chat_id
                )
            with self.assertRaisesRegex(ValueError, "Message does not belong"):
                await runs.create_run(
                    user_id=1,
                    starting_agent_id=agent.id,
                    model_name="test-model",
                    chat_id=own_chat_id,
                    message_id=other_message_id,
                )
            with self.assertRaisesRegex(ValueError, "Invalid"):
                await runs.wait_for_approval(user_id=1, run_id=run.id)
            with self.assertRaises(LookupError):
                await runs.get_run(user_id=2, run_id=run.id)

            assert (await runs.start(user_id=1, run_id=run.id)).status == AgentRunStatus.RUNNING
            assert (await runs.wait_for_approval(user_id=1, run_id=run.id)).status == AgentRunStatus.WAITING_APPROVAL
            assert (await runs.resume(user_id=1, run_id=run.id)).status == AgentRunStatus.RUNNING
            completed = await runs.complete(
                user_id=1, run_id=run.id, result_metadata={"answer": "done"}, usage={"tokens": 10}
            )
            assert completed.status == AgentRunStatus.COMPLETED
            assert completed.completed_at is not None
            with self.assertRaisesRegex(ValueError, "Invalid"):
                await runs.start(user_id=1, run_id=run.id)
            assert len(await runs.list_events(user_id=1, run_id=run.id)) == 5

            failed = await runs.create_run(user_id=1, starting_agent_id=agent.id, model_name="test-model")
            await runs.start(user_id=1, run_id=failed.id)
            assert (await runs.fail(user_id=1, run_id=failed.id, error="runtime error")).status == AgentRunStatus.FAILED
            cancelled = await runs.create_run(user_id=1, starting_agent_id=agent.id, model_name="test-model")
            assert (await runs.cancel(user_id=1, run_id=cancelled.id)).status == AgentRunStatus.CANCELLED

        asyncio.run(_with_session(scenario))
