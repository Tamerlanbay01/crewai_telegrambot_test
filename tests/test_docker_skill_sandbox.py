from uuid import uuid4

from integrations.sandbox.docker import DockerSkillSandbox
from models.skill import MaterializedSkillPackage, SandboxLimits


def test_build_command_keeps_stdin_interactive(tmp_path) -> None:
    package = MaterializedSkillPackage(
        skill_id=uuid4(),
        key="live-smoke",
        version=1,
        root_path=str(tmp_path / "skill"),
        entrypoint="src/main.py",
        checksum="0" * 64,
        work_path=str(tmp_path / "work"),
        output_path=str(tmp_path / "output"),
    )
    limits = SandboxLimits(
        timeout_seconds=10,
        memory_mb=64,
        cpus=0.25,
        pids_limit=8,
        max_stdout_bytes=1024,
        max_stderr_bytes=1024,
    )

    command = DockerSkillSandbox(image="python:3.12.7-slim").build_command(
        package=package,
        entrypoint="src/main.py",
        limits=limits,
        container_name="skill-exec-test",
    )

    assert "--interactive" in command


def test_host_environment_allows_docker_connection_without_app_secrets(monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_HOST", "tcp://docker:2375")
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret")

    environment = DockerSkillSandbox._host_environment()

    assert environment["DOCKER_HOST"] == "tcp://docker:2375"
    assert "DATABASE_URL" not in environment
