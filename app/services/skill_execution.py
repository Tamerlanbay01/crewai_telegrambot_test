"""Backend-owned orchestration for verified Docker-isolated skill execution."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import SandboxConfig, config
from database.entities.skill_execution import SkillExecutionEntity  # noqa: F401 - metadata registration
from integrations.sandbox.executor import SkillSandbox
from integrations.sandbox.factory import create_skill_sandbox
from models.agent_run import AgentRunEventCreate, AgentRunEventType, AgentRunStatus
from models.runtime import RuntimeSkillDefinition
from models.skill import (
    ExecutableSkillDefinition,
    MaterializedSkillPackage,
    SandboxLimits,
    SkillExecution,
    SkillExecutionCreate,
    SkillExecutionRequest,
    SkillExecutionResult,
    SkillExecutionStatus,
)
from repositories.agent_run import AgentRunRepository
from repositories.skill_execution import SkillExecutionRepository
from services.skill import SkillExecutionDeniedError, SkillService
from services.skill_package import SkillPackageService, SkillPackageValidationError

logger = logging.getLogger(__name__)


class SkillExecutionService:
    """Materialize, execute, persist, and audit exactly one skill invocation."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        skill_service: SkillService | None = None,
        sandbox: SkillSandbox | None = None,
        sandbox_config: SandboxConfig | None = None,
    ) -> None:
        self._session = session
        self._skills = skill_service or SkillService(session)
        self._sandbox_config = sandbox_config or config.sandbox
        self._sandbox = sandbox or create_skill_sandbox(self._sandbox_config)
        self._executions = SkillExecutionRepository(session)
        self._runs = AgentRunRepository(session)

    async def resolve_executable_skill(self, **kwargs) -> ExecutableSkillDefinition:
        return await self._skills.resolve_executable_skill(**kwargs)

    async def execute(
        self,
        request: SkillExecutionRequest,
        *,
        runtime_skill_catalog: Sequence[RuntimeSkillDefinition] = (),
    ) -> SkillExecutionResult:
        definition = await self._resolve_for_request(request, runtime_skill_catalog)
        if definition is None:
            return SkillExecutionResult(
                status=SkillExecutionStatus.DENIED,
                error="Skill is not executable for this agent",
            )

        run = await self._runs.get_by_id(request.run_id)
        if run is None or run.user_id != request.user_id:
            return SkillExecutionResult(
                status=SkillExecutionStatus.DENIED,
                error="Skill execution run was not found",
            )
        if run.status == AgentRunStatus.CANCELLED:
            return SkillExecutionResult(
                status=SkillExecutionStatus.CANCELLED,
                error="Agent run was cancelled before skill execution",
            )
        if run.status != AgentRunStatus.RUNNING:
            return SkillExecutionResult(
                status=SkillExecutionStatus.DENIED,
                error="Skill execution requires a running agent run",
            )

        idempotency_key = request.idempotency_key or self._idempotency_key(request, definition)
        existing = await self._executions.get_by_idempotency_key(idempotency_key)
        if existing is not None:
            if existing.status in {
                SkillExecutionStatus.SUCCEEDED,
                SkillExecutionStatus.FAILED,
                SkillExecutionStatus.TIMED_OUT,
                SkillExecutionStatus.CANCELLED,
                SkillExecutionStatus.DENIED,
                SkillExecutionStatus.SANDBOX_ERROR,
            }:
                return self._result_from_persisted(existing)
            return SkillExecutionResult(
                status=SkillExecutionStatus.FAILED,
                error="Skill execution is already in progress",
            )

        execution = await self._executions.create(
            SkillExecutionCreate(
                id=request.execution_id or uuid4(),
                run_id=request.run_id,
                user_id=request.user_id,
                skill_id=definition.skill_id,
                skill_version=definition.version,
                requesting_subject_type=request.requesting_subject_type,
                requesting_subject_id=request.requesting_subject_id,
                idempotency_key=idempotency_key,
                arguments_sanitized=self._sanitize(request.arguments),
            )
        )
        # Commit PENDING before any storage or Docker call.
        await self._session.commit()
        started_at = datetime.now(timezone.utc)
        await self._executions.mark_running(execution_id=execution.id, started_at=started_at)
        await self._append_event(
            run_id=request.run_id,
            event_type=AgentRunEventType.SKILL_EXECUTION_STARTED,
            execution=execution,
            status=SkillExecutionStatus.RUNNING,
        )
        await self._session.commit()

        result: SkillExecutionResult
        try:
            prepared = await self._skills.get_verified_package(definition)
            with tempfile.TemporaryDirectory(prefix="skill-exec-") as temporary_root:
                materialized = self._materialize(
                    prepared=prepared,
                    definition=definition,
                    temporary_root=Path(temporary_root),
                )
                latest_run = await self._runs.get_by_id(request.run_id)
                if latest_run is None or latest_run.status == AgentRunStatus.CANCELLED:
                    result = SkillExecutionResult(
                        status=SkillExecutionStatus.CANCELLED,
                        error="Agent run was cancelled before skill execution",
                    )
                else:
                    # Close the read transaction opened by the run inspection
                    # before crossing the external Docker boundary.
                    await self._session.commit()
                    result = await self._execute_in_sandbox(
                        package=materialized,
                        entrypoint=definition.entrypoint,
                        arguments=request.arguments,
                        run_id=request.run_id,
                        skill_id=definition.skill_id,
                        skill_version=definition.version,
                    )
        except SkillExecutionDeniedError:
            result = SkillExecutionResult(
                status=SkillExecutionStatus.DENIED,
                error="Skill is not executable for this agent",
            )
        except SkillPackageValidationError:
            result = SkillExecutionResult(
                status=SkillExecutionStatus.SANDBOX_ERROR,
                error="Skill package verification failed",
            )
        except Exception:
            logger.exception(
                "Skill execution failed: run=%s skill=%s version=%s",
                request.run_id,
                definition.skill_id,
                definition.version,
            )
            await self._session.rollback()
            result = SkillExecutionResult(
                status=SkillExecutionStatus.SANDBOX_ERROR,
                error="Skill sandbox failed",
            )

        result = self._bounded_result(result)
        completed_at = datetime.now(timezone.utc)
        persisted = await self._executions.update_result(
            execution_id=execution.id,
            result=result,
            started_at=started_at,
            completed_at=completed_at,
            stdout_preview=self._preview(result.stdout, self._sandbox_config.max_stdout_bytes),
            stderr_preview=self._preview(result.stderr, self._sandbox_config.max_stderr_bytes),
            error=result.error,
        )
        if persisted is None:
            raise LookupError(f"Skill execution not found: {execution.id}")
        await self._append_event(
            run_id=request.run_id,
            event_type=self._event_type_for(result.status),
            execution=execution,
            status=result.status,
        )
        await self._session.commit()
        return result

    async def _resolve_for_request(
        self,
        request: SkillExecutionRequest,
        runtime_skill_catalog: Sequence[RuntimeSkillDefinition],
    ) -> ExecutableSkillDefinition | None:
        try:
            return await self._skills.resolve_executable_skill(
                user_id=request.user_id,
                requesting_subject_type=request.requesting_subject_type,
                requesting_subject_id=request.requesting_subject_id,
                skill_key=request.skill_key,
                skill_version=request.skill_version,
                skill_id=request.skill_id,
                runtime_skill_catalog=runtime_skill_catalog,
            )
        except SkillExecutionDeniedError:
            return None

    async def _execute_in_sandbox(
        self,
        *,
        package: MaterializedSkillPackage,
        entrypoint: str,
        arguments: dict[str, object],
        run_id: UUID,
        skill_id: UUID,
        skill_version: int,
    ) -> SkillExecutionResult:
        kwargs: dict[str, object] = {
            "package": package,
            "entrypoint": entrypoint,
            "arguments": arguments,
            "limits": self._limits(),
        }
        signature = inspect.signature(self._sandbox.execute)
        if "environment" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            kwargs["environment"] = {
                "SKILL_RUN_ID": str(run_id),
                "SKILL_ID": str(skill_id),
                "SKILL_VERSION": str(skill_version),
            }
        return await self._sandbox.execute(**kwargs)

    @staticmethod
    def _idempotency_key(
        request: SkillExecutionRequest, definition: ExecutableSkillDefinition
    ) -> str:
        payload = {
            "run_id": str(request.run_id),
            "user_id": request.user_id,
            "subject_type": request.requesting_subject_type.value,
            "subject_id": request.requesting_subject_id,
            "skill_id": str(definition.skill_id),
            "skill_version": definition.version,
            "arguments": request.arguments,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _materialize(
        *,
        prepared,
        definition: ExecutableSkillDefinition,
        temporary_root: Path,
    ) -> MaterializedSkillPackage:
        root = temporary_root / "skill"
        work = temporary_root / "work"
        output = temporary_root / "output"
        root.mkdir(parents=True, exist_ok=True)
        work.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        root_resolved = root.resolve()
        for item in prepared.files:
            SkillPackageService.validate_path(item.path)
            target = root.joinpath(*PurePosixPath(item.path).parts)
            if root_resolved not in target.resolve().parents:
                raise SkillPackageValidationError("Skill package path escaped its workspace")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.content)
        return MaterializedSkillPackage(
            skill_id=definition.skill_id,
            key=definition.key,
            version=definition.version,
            root_path=str(root),
            entrypoint=definition.entrypoint,
            checksum=prepared.checksum,
            work_path=str(work),
            output_path=str(output),
        )

    def _limits(self) -> SandboxLimits:
        return SandboxLimits(
            timeout_seconds=self._sandbox_config.timeout_seconds,
            memory_mb=self._sandbox_config.memory_mb,
            cpus=self._sandbox_config.cpus,
            pids_limit=self._sandbox_config.pids_limit,
            max_stdout_bytes=self._sandbox_config.max_stdout_bytes,
            max_stderr_bytes=self._sandbox_config.max_stderr_bytes,
        )

    @classmethod
    def _sanitize(cls, value: dict[str, object]) -> dict[str, object]:
        sensitive = (
            "authorization",
            "password",
            "secret",
            "token",
            "apikey",
            "privatekey",
            "credential",
            "bearer",
            "cookie",
        )

        def redact(item: object) -> object:
            if isinstance(item, dict):
                clean: dict[str, object] = {}
                for key, nested in item.items():
                    normalized = "".join(char for char in str(key).casefold() if char.isalnum())
                    clean[str(key)] = (
                        "[REDACTED]"
                        if any(fragment in normalized for fragment in sensitive)
                        else redact(nested)
                    )
                return clean
            if isinstance(item, list):
                return [redact(nested) for nested in item]
            return item

        result = redact(value)
        return result if isinstance(result, dict) else {}

    @staticmethod
    def _preview(value: str, limit: int) -> str:
        return value.encode("utf-8")[: min(limit, 4096)].decode("utf-8", errors="replace")

    def _bounded_result(self, result: SkillExecutionResult) -> SkillExecutionResult:
        stdout = result.stdout.encode("utf-8")
        stderr = result.stderr.encode("utf-8")
        stdout_limit = self._sandbox_config.max_stdout_bytes
        stderr_limit = self._sandbox_config.max_stderr_bytes
        stdout_truncated = len(stdout) > stdout_limit
        stderr_truncated = len(stderr) > stderr_limit
        return result.model_copy(
            update={
                "stdout": stdout[:stdout_limit].decode("utf-8", errors="replace"),
                "stderr": stderr[:stderr_limit].decode("utf-8", errors="replace"),
                "truncated": result.truncated or stdout_truncated or stderr_truncated,
            }
        )

    @staticmethod
    def _event_type_for(status: SkillExecutionStatus) -> AgentRunEventType:
        if status == SkillExecutionStatus.SUCCEEDED:
            return AgentRunEventType.SKILL_EXECUTION_COMPLETED
        if status == SkillExecutionStatus.TIMED_OUT:
            return AgentRunEventType.SKILL_EXECUTION_TIMED_OUT
        return AgentRunEventType.SKILL_EXECUTION_FAILED

    async def _append_event(
        self,
        *,
        run_id: UUID,
        event_type: AgentRunEventType,
        execution: SkillExecution,
        status: SkillExecutionStatus,
    ) -> None:
        await self._runs.append_event(
            AgentRunEventCreate(
                run_id=run_id,
                event_type=event_type,
                payload={
                    "execution_id": str(execution.id),
                    "skill_id": str(execution.skill_id),
                    "skill_version": execution.skill_version,
                    "status": status.value,
                },
            )
        )

    @staticmethod
    def _result_from_persisted(execution: SkillExecution) -> SkillExecutionResult:
        output: object | None = None
        try:
            decoded = json.loads(execution.stdout_preview)
            output = decoded.get("result") if isinstance(decoded, dict) and "result" in decoded else decoded
        except (TypeError, json.JSONDecodeError):
            pass
        return SkillExecutionResult(
            status=execution.status,
            exit_code=execution.exit_code,
            stdout=execution.stdout_preview,
            stderr=execution.stderr_preview,
            output=output,
            duration_ms=execution.duration_ms or 0,
            error=execution.error,
        )
