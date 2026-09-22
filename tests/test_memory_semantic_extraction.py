from __future__ import annotations

import asyncio
import unittest
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import UUID
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import config
from database.base import Base
from database.entities.agent import AgentEntity
from database.entities.agent_connection import AgentConnectionEntity
from database.entities.agent_prompt import AgentPromptVersionEntity
from database.entities.agent_run import AgentRunEntity
from database.entities.agent_run_event import AgentRunEventEntity
from database.entities.agent_skill import AgentSkillEntity
from database.entities.memory import MemoryEntity
from database.entities.memory_candidate import MemoryCandidateEntity
from database.entities.memory_extraction_run import MemoryExtractionRunEntity
from database.entities.message import MessageEntity
from database.entities.skill import SkillEntity
from database.entities.system_agent_template import SystemAgentTemplateEntity
from database.entities.user import UserEntity
from database.entities.user_agent_override import UserAgentOverrideEntity
from integrations.memory.index import MemorySearchHit
from models.memory import (
    MemoryCandidate,
    MemoryCandidateDecision,
    MemoryExtractionResult,
    MemoryExtractionStatus,
    MemoryScope,
    MemoryType,
)
from models.runtime import AgentRuntimeRequest, AgentRuntimeStatus, DelegationDecision, DelegationType, RuntimeStep
from repositories.memory_extraction import MemoryExtractionRepository
from repositories.agent_run import AgentRunRepository
from services.agent import AgentService
from services.agent_run import AgentRunService
from services.agent_runtime import AgentRuntimeService
from services.memory_index import MemoryIndexService
from services.memory import MemoryService
from services.memory_extraction import MemoryExtractionService
from services.memory_processor import MemoryExtractionProcessor
from services.system_agent import SystemAgentService


async def _with_session(test: Callable[[AsyncSession], Awaitable[None]]) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            session.add(UserEntity(id=1, telegram_id=101))
            await session.commit()
            await test(session)
    finally:
        await engine.dispose()


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.texts.append(text)
        return [0.1, 0.2, 0.3]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]


class FakeMemorySearchIndex:
    def __init__(self) -> None:
        self.ensure_calls = 0
        self.upserts: list[UUID] = []
        self.deleted: list[UUID] = []
        self.hits: list[MemorySearchHit] = []
        self.fail_upsert = False
        self.fail_delete = False
        self.last_search: dict[str, object] | None = None

    async def ensure_collection(self) -> None:
        self.ensure_calls += 1

    async def upsert(self, memory, vector: list[float]) -> None:
        if self.fail_upsert:
            raise RuntimeError("Qdrant unavailable")
        self.upserts.append(memory.id)

    async def delete(self, memory_id: UUID) -> None:
        if self.fail_delete:
            raise RuntimeError("Qdrant unavailable")
        self.deleted.append(memory_id)

    async def search(self, vector: list[float], **kwargs) -> list[MemorySearchHit]:
        self.last_search = kwargs
        return list(self.hits)


class FakeMemoryAgent:
    def __init__(self, candidates: list[MemoryCandidate]) -> None:
        self.candidates = candidates
        self.calls = 0

    async def extract(self, context) -> MemoryExtractionResult:
        self.calls += 1
        return MemoryExtractionResult(candidates=self.candidates)


class FailingMemoryAgent:
    async def extract(self, context) -> MemoryExtractionResult:
        raise RuntimeError("Memory model unavailable")


class SequencedMemoryAgent:
    def __init__(self, outcomes: list[Exception | list[MemoryCandidate]]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def extract(self, context) -> MemoryExtractionResult:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return MemoryExtractionResult(candidates=outcome)


class RespondingRuntime:
    async def run(self, context, request) -> RuntimeStep:
        return RuntimeStep(
            decision=DelegationDecision(type=DelegationType.RESPOND),
            content="Completed response.",
        )


class SemanticMemoryAndExtractionTests(unittest.TestCase):
    def test_only_persistent_scopes_are_indexed_and_index_failure_is_nonfatal(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            agents = AgentService(session)
            primary = await agents.ensure_primary_agent(user_id=1)
            run = await AgentRunService(session).create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            embeddings = FakeEmbeddingProvider()
            index = FakeMemorySearchIndex()
            memory = MemoryService(
                session,
                embedding_provider=embeddings,
                search_index=index,
            )

            global_memory = await memory.put(
                user_id=1,
                scope=MemoryScope.USER_GLOBAL,
                memory_type=MemoryType.PREFERENCE,
                key="python_package_manager",
                content="Prefer uv for Python projects.",
            )
            private_memory = await memory.put(
                user_id=1,
                scope=MemoryScope.AGENT_PRIVATE,
                agent_id=primary.id,
                memory_type=MemoryType.PROJECT_CONTEXT,
                key="imports",
                content="Use root-based imports.",
            )
            await memory.put(
                user_id=1,
                scope=MemoryScope.CREW_SHARED,
                run_id=run.id,
                key="sources",
                content="Temporary research sources.",
            )
            self.assertEqual(index.upserts, [global_memory.id, private_memory.id])
            self.assertEqual(len(embeddings.texts), 2)

            index.fail_upsert = True
            stored_despite_index_failure = await memory.put(
                user_id=1,
                scope=MemoryScope.USER_GLOBAL,
                key="delegation",
                content="Keep delegation backend-controlled.",
            )
            self.assertIsNotNone(
                await memory.get(user_id=1, memory_id=stored_despite_index_failure.id)
            )
            index.fail_upsert = False
            self.assertTrue(await memory.delete(user_id=1, memory_id=global_memory.id))
            self.assertIn(global_memory.id, index.deleted)
            index.fail_delete = True
            self.assertTrue(await memory.delete(user_id=1, memory_id=private_memory.id))
            self.assertIsNone(await memory.get(user_id=1, memory_id=private_memory.id))

        asyncio.run(_with_session(scenario))

    def test_semantic_results_are_post_validated_and_never_cross_agent_boundary(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            agents = AgentService(session)
            primary = await agents.ensure_primary_agent(user_id=1)
            worker = await agents.create_user_agent(
                user_id=1, name="Worker", role="Worker", goal="Work"
            )
            run = await AgentRunService(session).create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            index = FakeMemorySearchIndex()
            memory = MemoryService(
                session,
                embedding_provider=FakeEmbeddingProvider(),
                search_index=index,
            )
            global_memory = await memory.put(
                user_id=1,
                scope=MemoryScope.USER_GLOBAL,
                key="python_package_manager",
                content="Prefer uv for Python projects.",
            )
            worker_private = await memory.put(
                user_id=1,
                scope=MemoryScope.AGENT_PRIVATE,
                agent_id=worker.id,
                key="worker_only",
                content="Do not expose this to primary.",
            )
            index.hits = [
                MemorySearchHit(memory_id=worker_private.id, score=0.99),
                MemorySearchHit(memory_id=global_memory.id, score=0.98),
                MemorySearchHit(memory_id=UUID("00000000-0000-0000-0000-000000000001"), score=0.97),
            ]

            result = await memory.search_runtime_memory(
                user_id=1,
                query="How should I manage Python environments?",
                run_id=run.id,
                agent_id=primary.id,
                limit=10,
            )
            self.assertEqual([item.memory_id for item in result], [global_memory.id])
            self.assertEqual(index.last_search["user_id"], 1)
            self.assertEqual(index.last_search["agent_id"], primary.id)

        asyncio.run(_with_session(scenario))

    def test_completed_run_is_extracted_once_and_policy_updates_by_key(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            agents = AgentService(session)
            primary = await agents.ensure_primary_agent(user_id=1)
            runs = AgentRunService(session)
            run = await runs.create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            await AgentRunRepository(session).update_runtime_state(
                run_id=run.id,
                usage={},
                checkpoint={"original_message": "I now prefer uv for Python projects."},
            )
            await session.commit()
            await runs.start(user_id=1, run_id=run.id)
            await runs.complete(
                user_id=1,
                run_id=run.id,
                result_metadata={"content": "Understood."},
            )
            await MemoryService(session).put(
                user_id=1,
                scope=MemoryScope.USER_GLOBAL,
                key="python_package_manager",
                content="Prefer pip for Python projects.",
            )
            await SystemAgentService(session).ensure_initial_templates()
            extractor = FakeMemoryAgent(
                [
                    MemoryCandidate(
                        memory_type=MemoryType.PREFERENCE,
                        scope=MemoryScope.USER_GLOBAL,
                        key="python_package_manager",
                        content="Prefer uv for Python projects.",
                        confidence=0.98,
                        reason="The user stated a durable tooling preference.",
                    ),
                    MemoryCandidate(
                        memory_type=MemoryType.FACT,
                        scope=MemoryScope.USER_GLOBAL,
                        key="trivial",
                        content="thanks",
                        confidence=0.99,
                        reason="Must be rejected as trivial.",
                    ),
                    MemoryCandidate(
                        memory_type=MemoryType.GOAL,
                        scope=MemoryScope.USER_GLOBAL,
                        key="low_confidence",
                        content="Build a production platform.",
                        confidence=0.2,
                        reason="Below policy threshold.",
                    ),
                    MemoryCandidate(
                        memory_type=MemoryType.FACT,
                        scope=MemoryScope.CREW_SHARED,
                        key="invalid_long_term_scope",
                        content="A runtime scratch value.",
                        confidence=0.99,
                        reason="Memory Agent may not persist runtime scopes.",
                    ),
                    MemoryCandidate(
                        memory_type=MemoryType.PROJECT_CONTEXT,
                        scope=MemoryScope.AGENT_PRIVATE,
                        agent_id=UUID("deadbeef-dead-beef-dead-beefdeadbeef"),
                        key="foreign_agent",
                        content="Unsafe arbitrary agent target.",
                        confidence=0.99,
                        reason="Target must have participated in this user's run.",
                    ),
                ]
            )
            service = MemoryExtractionService(session, extractor=extractor)

            extraction = await service.ensure_pending(run_id=run.id)
            self.assertEqual(extraction.status, MemoryExtractionStatus.PENDING)
            processor = MemoryExtractionProcessor(session, extractor=extractor)
            await processor.process_pending()
            repeated = await processor.process_pending()
            extraction = await MemoryExtractionRepository(session).get_by_agent_run(run.id)
            self.assertEqual(extractor.calls, 1)
            self.assertEqual(repeated, [])
            self.assertEqual(extraction.status, MemoryExtractionStatus.COMPLETED)
            self.assertEqual(
                (
                    await MemoryService(session).list(
                        user_id=1, scope=MemoryScope.USER_GLOBAL
                    )
                )[0].content,
                "Prefer uv for Python projects.",
            )
            candidates = (
                await session.execute(
                    select(MemoryCandidateEntity).where(
                        MemoryCandidateEntity.extraction_run_id == extraction.id
                    )
                )
            ).scalars().all()
            decisions = {candidate.key: candidate.decision for candidate in candidates}
            self.assertEqual(decisions["python_package_manager"], MemoryCandidateDecision.UPDATED)
            self.assertEqual(decisions["trivial"], MemoryCandidateDecision.REJECTED)
            self.assertEqual(decisions["low_confidence"], MemoryCandidateDecision.IGNORED)
            self.assertEqual(
                decisions["invalid_long_term_scope"], MemoryCandidateDecision.REJECTED
            )
            self.assertEqual(decisions["foreign_agent"], MemoryCandidateDecision.REJECTED)
            self.assertIsNotNone(
                await MemoryExtractionRepository(session).get_by_agent_run(run.id)
            )

        asyncio.run(_with_session(scenario))

    def test_reindex_uses_batch_embeddings_for_only_active_persistent_memory(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            primary = await AgentService(session).ensure_primary_agent(user_id=1)
            run = await AgentRunService(session).create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            source_memory = MemoryService(session)
            global_memory = await source_memory.put(
                user_id=1,
                scope=MemoryScope.USER_GLOBAL,
                key="package_manager",
                content="Prefer uv.",
            )
            private_memory = await source_memory.put(
                user_id=1,
                scope=MemoryScope.AGENT_PRIVATE,
                agent_id=primary.id,
                key="imports",
                content="Use root imports.",
            )
            await source_memory.put(
                user_id=1,
                scope=MemoryScope.CREW_SHARED,
                run_id=run.id,
                key="scratch",
                content="Do not index this.",
            )
            embeddings = FakeEmbeddingProvider()
            index = FakeMemorySearchIndex()
            rebuilt = await MemoryIndexService(
                session,
                embedding_provider=embeddings,
                search_index=index,
            ).reindex_user(user_id=1)
            self.assertEqual(rebuilt, 2)
            self.assertEqual(index.upserts, [global_memory.id, private_memory.id])
            self.assertEqual(len(embeddings.texts), 2)

        asyncio.run(_with_session(scenario))

    def test_runtime_completion_always_records_memory_processing_for_eligible_turn(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            primary = await AgentService(session).ensure_primary_agent(user_id=1)
            run = await AgentRunService(session).create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            extractor = FakeMemoryAgent([])
            result = await AgentRuntimeService(
                session,
                runtime=RespondingRuntime(),
                memory_extractor=extractor,
            ).execute(
                AgentRuntimeRequest(
                    run_id=run.id,
                    user_id=1,
                    agent_id=primary.id,
                    message="Thanks!",
                )
            )
            extraction = await MemoryExtractionRepository(session).get_by_agent_run(run.id)
            self.assertEqual(result.status, AgentRuntimeStatus.COMPLETED)
            self.assertEqual(extractor.calls, 0)
            self.assertIsNotNone(extraction)
            self.assertEqual(extraction.status, MemoryExtractionStatus.PENDING)
            await MemoryExtractionProcessor(session, extractor=extractor).process_pending()
            extraction = await MemoryExtractionRepository(session).get_by_agent_run(run.id)
            self.assertEqual(extractor.calls, 1)
            self.assertEqual(extraction.status, MemoryExtractionStatus.COMPLETED)

        asyncio.run(_with_session(scenario))

    def test_memory_agent_failure_does_not_fail_the_completed_agent_run(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            primary = await AgentService(session).ensure_primary_agent(user_id=1)
            run = await AgentRunService(session).create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            result = await AgentRuntimeService(
                session,
                runtime=RespondingRuntime(),
                memory_extractor=FailingMemoryAgent(),
            ).execute(
                AgentRuntimeRequest(
                    run_id=run.id,
                    user_id=1,
                    agent_id=primary.id,
                    message="Please remember I use uv.",
                )
            )
            persisted_run = await AgentRunService(session).get_run(user_id=1, run_id=run.id)
            extraction = await MemoryExtractionRepository(session).get_by_agent_run(run.id)
            self.assertEqual(result.status, AgentRuntimeStatus.COMPLETED)
            self.assertEqual(persisted_run.status.value, "completed")
            self.assertIsNotNone(extraction)
            self.assertEqual(extraction.status, MemoryExtractionStatus.PENDING)
            await MemoryExtractionProcessor(
                session, extractor=FailingMemoryAgent()
            ).process_pending()
            extraction = await MemoryExtractionRepository(session).get_by_agent_run(run.id)
            self.assertEqual(extraction.status, MemoryExtractionStatus.FAILED)

        asyncio.run(_with_session(scenario))

    def test_command_run_is_not_eligible_for_memory_extraction(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            primary = await AgentService(session).ensure_primary_agent(user_id=1)
            run = await AgentRunService(session).create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            extractor = FakeMemoryAgent([])
            result = await AgentRuntimeService(
                session,
                runtime=RespondingRuntime(),
                memory_extractor=extractor,
            ).execute(
                AgentRuntimeRequest(
                    run_id=run.id,
                    user_id=1,
                    agent_id=primary.id,
                    message="/help memory",
                )
            )
            self.assertEqual(result.status, AgentRuntimeStatus.COMPLETED)
            self.assertEqual(extractor.calls, 0)
            self.assertIsNone(await MemoryExtractionRepository(session).get_by_agent_run(run.id))

        asyncio.run(_with_session(scenario))

    def test_processor_retries_transient_failure_after_backoff(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            primary = await AgentService(session).ensure_primary_agent(user_id=1)
            run = await AgentRunService(session).create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            result = await AgentRuntimeService(
                session,
                runtime=RespondingRuntime(),
            ).execute(
                AgentRuntimeRequest(
                    run_id=run.id,
                    user_id=1,
                    agent_id=primary.id,
                    message="I prefer uv.",
                )
            )
            self.assertEqual(result.status, AgentRuntimeStatus.COMPLETED)
            base = datetime(2026, 9, 22, tzinfo=timezone.utc)
            extractor = SequencedMemoryAgent([RuntimeError("temporary timeout"), []])
            processor = MemoryExtractionProcessor(session, extractor=extractor)
            with patch.object(config, "memory_extraction_retry_base_seconds", 30):
                await processor.process_pending(now=base)
                extraction = await MemoryExtractionRepository(session).get_by_agent_run(run.id)
                self.assertEqual(extraction.status, MemoryExtractionStatus.FAILED)
                self.assertEqual(extraction.attempt_count, 1)
                self.assertEqual(
                    extraction.next_retry_at.replace(tzinfo=timezone.utc),
                    base + timedelta(seconds=30),
                )

                self.assertEqual(await processor.process_pending(now=base + timedelta(seconds=29)), [])
                self.assertEqual(extractor.calls, 1)
                await processor.process_pending(now=base + timedelta(seconds=30))

            extraction = await MemoryExtractionRepository(session).get_by_agent_run(run.id)
            self.assertEqual(extractor.calls, 2)
            self.assertEqual(extraction.status, MemoryExtractionStatus.COMPLETED)
            self.assertEqual(extraction.attempt_count, 2)
            self.assertIsNone(extraction.next_retry_at)

        asyncio.run(_with_session(scenario))

    def test_terminal_failure_stops_auto_retry_and_manual_retry_resets_it(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            primary = await AgentService(session).ensure_primary_agent(user_id=1)
            run = await AgentRunService(session).create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            await AgentRuntimeService(session, runtime=RespondingRuntime()).execute(
                AgentRuntimeRequest(
                    run_id=run.id,
                    user_id=1,
                    agent_id=primary.id,
                    message="Remember the project convention.",
                )
            )
            base = datetime(2026, 9, 22, tzinfo=timezone.utc)
            failing = SequencedMemoryAgent([RuntimeError("outage")])
            processor = MemoryExtractionProcessor(session, extractor=failing)
            with patch.object(config, "memory_extraction_max_attempts", 1):
                await processor.process_pending(now=base)
                extraction = await MemoryExtractionRepository(session).get_by_agent_run(run.id)
                self.assertEqual(extraction.status, MemoryExtractionStatus.FAILED)
                self.assertIsNone(extraction.next_retry_at)
                self.assertEqual(
                    await processor.process_pending(now=base + timedelta(days=1)), []
                )
                reset = await processor.retry(
                    extraction_run_id=extraction.id, now=base + timedelta(days=1)
                )
                self.assertEqual(reset.status, MemoryExtractionStatus.PENDING)
                self.assertEqual(reset.attempt_count, 0)

                successful = MemoryExtractionProcessor(
                    session, extractor=SequencedMemoryAgent([[]])
                )
                await successful.process_pending(now=base + timedelta(days=1))

            extraction = await MemoryExtractionRepository(session).get_by_agent_run(run.id)
            self.assertEqual(extraction.status, MemoryExtractionStatus.COMPLETED)

        asyncio.run(_with_session(scenario))

    def test_expired_lease_is_reclaimed_but_live_lease_is_not(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            primary = await AgentService(session).ensure_primary_agent(user_id=1)
            run = await AgentRunService(session).create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            await AgentRuntimeService(session, runtime=RespondingRuntime()).execute(
                AgentRuntimeRequest(
                    run_id=run.id,
                    user_id=1,
                    agent_id=primary.id,
                    message="Store this durable preference.",
                )
            )
            extraction = await MemoryExtractionRepository(session).get_by_agent_run(run.id)
            base = datetime(2026, 9, 22, tzinfo=timezone.utc)
            repository = MemoryExtractionRepository(session)
            first_claim = await repository.claim_by_id(
                extraction_run_id=extraction.id,
                now=base,
                claim_token=uuid4(),
                claim_expires_at=base + timedelta(seconds=60),
            )
            await session.commit()
            self.assertIsNotNone(first_claim)
            self.assertIsNone(
                await repository.claim_by_id(
                    extraction_run_id=extraction.id,
                    now=base + timedelta(seconds=1),
                    claim_token=uuid4(),
                    claim_expires_at=base + timedelta(seconds=61),
                )
            )
            await session.commit()

            extractor = SequencedMemoryAgent([[]])
            processor = MemoryExtractionProcessor(session, extractor=extractor)
            self.assertEqual(await processor.process_pending(now=base + timedelta(seconds=59)), [])
            self.assertEqual(extractor.calls, 0)
            await processor.process_pending(now=base + timedelta(seconds=61))

            extraction = await MemoryExtractionRepository(session).get_by_agent_run(run.id)
            self.assertEqual(extractor.calls, 1)
            self.assertEqual(extraction.status, MemoryExtractionStatus.COMPLETED)
            self.assertEqual(extraction.attempt_count, 2)

        asyncio.run(_with_session(scenario))

    def test_retry_resumes_persisted_candidates_without_second_memory_agent_call(self) -> None:
        async def scenario(session: AsyncSession) -> None:
            primary = await AgentService(session).ensure_primary_agent(user_id=1)
            run = await AgentRunService(session).create_run(
                user_id=1, starting_agent_id=primary.id, model_name="test-model"
            )
            await AgentRuntimeService(session, runtime=RespondingRuntime()).execute(
                AgentRuntimeRequest(
                    run_id=run.id,
                    user_id=1,
                    agent_id=primary.id,
                    message="I prefer uv for Python projects.",
                )
            )
            extraction = await MemoryExtractionRepository(session).get_by_agent_run(run.id)
            base = datetime(2026, 9, 22, tzinfo=timezone.utc)
            claim_token = uuid4()
            claim = await MemoryExtractionRepository(session).claim_by_id(
                extraction_run_id=extraction.id,
                now=base,
                claim_token=claim_token,
                claim_expires_at=base + timedelta(seconds=60),
            )
            await session.commit()
            extractor = SequencedMemoryAgent(
                [
                    [
                        MemoryCandidate(
                            memory_type=MemoryType.PREFERENCE,
                            scope=MemoryScope.USER_GLOBAL,
                            key="python_package_manager",
                            content="Prefer uv for Python projects.",
                            confidence=0.99,
                            reason="Explicit stable preference.",
                        )
                    ]
                ]
            )
            await MemoryExtractionService(session, extractor=extractor).process_claimed(
                extraction_run_id=claim.id,
                claim_token=claim_token,
                now=base,
            )
            self.assertEqual(extractor.calls, 1)
            self.assertEqual(
                len(await MemoryService(session).list(user_id=1, scope=MemoryScope.USER_GLOBAL)),
                1,
            )

            resumed = MemoryExtractionProcessor(
                session, extractor=SequencedMemoryAgent([RuntimeError("must not run")])
            )
            await resumed.process_pending(now=base + timedelta(seconds=61))
            extraction = await MemoryExtractionRepository(session).get_by_agent_run(run.id)
            self.assertEqual(extraction.status, MemoryExtractionStatus.COMPLETED)
            self.assertEqual(
                len(await MemoryService(session).list(user_id=1, scope=MemoryScope.USER_GLOBAL)),
                1,
            )

        asyncio.run(_with_session(scenario))
