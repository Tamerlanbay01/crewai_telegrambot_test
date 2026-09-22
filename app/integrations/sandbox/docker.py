"""Docker CLI adapter for the controlled Python skill runtime."""

import asyncio
import json
import os
from collections.abc import Mapping
from pathlib import Path
from time import monotonic
from uuid import uuid4

from models.skill import MaterializedSkillPackage, SandboxLimits, SkillExecutionResult, SkillExecutionStatus


class DockerSkillSandbox:
    """Run a package in a pinned, networkless, non-root Docker container."""

    ENVIRONMENT_ALLOWLIST = frozenset({"SKILL_RUN_ID", "SKILL_ID", "SKILL_VERSION"})

    def __init__(self, *, image: str, docker_binary: str = "docker", user: str = "65532:65532") -> None:
        if not image or image.endswith(":latest") or image == "latest":
            raise ValueError("Skill sandbox image must be explicitly pinned")
        self._image = image
        self._docker_binary = docker_binary
        self._user = user

    def build_command(
        self,
        *,
        package: MaterializedSkillPackage,
        entrypoint: str,
        limits: SandboxLimits,
        container_name: str,
        environment: Mapping[str, str] | None = None,
    ) -> list[str]:
        if not entrypoint.startswith("src/") or not entrypoint.endswith(".py") or ".." in Path(entrypoint).parts:
            raise ValueError("Sandbox entrypoint must be a Python file under src/")
        root = Path(package.root_path)
        work = Path(package.work_path) if package.work_path else root.parent / "work"
        output = Path(package.output_path) if package.output_path else root.parent / "output"
        command = [
            self._docker_binary,
            "run",
            "--rm",
            "--interactive",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(limits.pids_limit),
            "--memory",
            f"{limits.memory_mb}m",
            "--cpus",
            str(limits.cpus),
            "--tmpfs",
            "/tmp",
            "--user",
            self._user,
            "--mount",
            f"type=bind,source={root},target=/skill,readonly",
            "--mount",
            f"type=bind,source={work},target=/work",
            "--mount",
            f"type=bind,source={output},target=/output",
            "--workdir",
            "/work",
        ]
        for key in sorted(self.ENVIRONMENT_ALLOWLIST):
            value = (environment or {}).get(key)
            if value is not None:
                command.extend(["--env", f"{key}={value}"])
        command.extend([self._image, "python", f"/skill/{entrypoint}"])
        return command

    # Kept as a small test seam for callers that need to inspect the exact
    # argv without starting Docker.
    _build_command = build_command

    async def execute(
        self,
        *,
        package: MaterializedSkillPackage,
        entrypoint: str,
        arguments: dict[str, object],
        limits: SandboxLimits,
        environment: Mapping[str, str] | None = None,
    ) -> SkillExecutionResult:
        root = Path(package.root_path)
        work = Path(package.work_path) if package.work_path else root.parent / "work"
        output = Path(package.output_path) if package.output_path else root.parent / "output"
        work.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        container_name = f"skill-exec-{uuid4().hex}"
        command = self.build_command(
            package=package,
            entrypoint=entrypoint,
            limits=limits,
            container_name=container_name,
            environment=environment,
        )
        payload = json.dumps({"arguments": arguments}, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        started = monotonic()
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._host_environment(),
            )
            assert process.stdin is not None
            process.stdin.write(payload)
            await process.stdin.drain()
            process.stdin.close()
            assert process.stdout is not None and process.stderr is not None
            try:
                (stdout, stdout_truncated), (stderr, stderr_truncated) = await asyncio.wait_for(
                    asyncio.gather(
                        self._read_bounded(process.stdout, limits.max_stdout_bytes),
                        self._read_bounded(process.stderr, limits.max_stderr_bytes),
                    ),
                    timeout=limits.timeout_seconds,
                )
                exit_code = await process.wait()
            except asyncio.TimeoutError:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
                await self._remove_container(container_name)
                return SkillExecutionResult(
                    status=SkillExecutionStatus.TIMED_OUT,
                    exit_code=None,
                    stdout="",
                    stderr="",
                    duration_ms=self._duration_ms(started),
                    error="Skill execution timed out",
                )

            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")
            truncated = stdout_truncated or stderr_truncated
            if exit_code == 125:
                return SkillExecutionResult(
                    status=SkillExecutionStatus.SANDBOX_ERROR,
                    exit_code=exit_code,
                    stdout=stdout_text,
                    stderr=stderr_text,
                    duration_ms=self._duration_ms(started),
                    truncated=truncated,
                    error="Skill sandbox is unavailable",
                )
            if exit_code != 0:
                return SkillExecutionResult(
                    status=SkillExecutionStatus.FAILED,
                    exit_code=exit_code,
                    stdout=stdout_text,
                    stderr=stderr_text,
                    duration_ms=self._duration_ms(started),
                    truncated=truncated,
                    error="Skill exited with a non-zero status",
                )
            try:
                decoded = json.loads(stdout_text)
            except json.JSONDecodeError:
                return SkillExecutionResult(
                    status=SkillExecutionStatus.FAILED,
                    exit_code=exit_code,
                    stdout=stdout_text,
                    stderr=stderr_text,
                    duration_ms=self._duration_ms(started),
                    truncated=truncated,
                    error="Skill stdout was not valid JSON",
                )
            if isinstance(decoded, dict) and decoded.get("ok") is False:
                return SkillExecutionResult(
                    status=SkillExecutionStatus.FAILED,
                    exit_code=exit_code,
                    stdout=stdout_text,
                    stderr=stderr_text,
                    output=decoded.get("result"),
                    duration_ms=self._duration_ms(started),
                    truncated=truncated,
                    error="Skill returned a failed result",
                )
            result_output = decoded.get("result") if isinstance(decoded, dict) and "result" in decoded else decoded
            return SkillExecutionResult(
                status=SkillExecutionStatus.SUCCEEDED,
                exit_code=exit_code,
                stdout=stdout_text,
                stderr=stderr_text,
                output=result_output,
                duration_ms=self._duration_ms(started),
                truncated=truncated,
            )
        except FileNotFoundError:
            return SkillExecutionResult(
                status=SkillExecutionStatus.SANDBOX_ERROR,
                duration_ms=self._duration_ms(started),
                error="Skill sandbox is unavailable",
            )
        except Exception:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            await self._remove_container(container_name)
            return SkillExecutionResult(
                status=SkillExecutionStatus.SANDBOX_ERROR,
                duration_ms=self._duration_ms(started),
                error="Skill sandbox failed",
            )

    @staticmethod
    async def _read_bounded(
        stream: asyncio.StreamReader, limit: int
    ) -> tuple[bytes, bool]:
        buffer = bytearray()
        truncated = False
        while True:
            chunk = await stream.read(64 * 1024)
            if not chunk:
                break
            remaining = limit - len(buffer)
            if remaining > 0:
                buffer.extend(chunk[:remaining])
            if len(chunk) > max(remaining, 0):
                truncated = True
        return bytes(buffer), truncated

    async def _remove_container(self, container_name: str) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                self._docker_binary,
                "rm",
                "-f",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=self._host_environment(),
            )
            await asyncio.wait_for(process.wait(), timeout=5)
        except Exception:
            # The original bounded result is more useful than leaking cleanup
            # implementation details to the caller.
            return

    @staticmethod
    def _duration_ms(started: float) -> int:
        return max(0, int((monotonic() - started) * 1000))

    @staticmethod
    def _host_environment() -> dict[str, str]:
        # Docker receives only executable lookup and explicit daemon connection
        # settings. Application secrets are never inherited by the CLI process.
        environment = {"PATH": os.environ.get("PATH", "")}
        for key in ("DOCKER_HOST", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
            value = os.environ.get(key)
            if value is not None:
                environment[key] = value
        return environment
