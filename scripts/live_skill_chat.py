"""Live S3 + LLM chat-handler acceptance scenario, with isolated local SQL data.

Run: .venv/Scripts/python.exe scripts/live_skill_chat.py
Telegram delivery is recorded locally; application handlers/services are real.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

os.environ.setdefault("CREWAI_TELEMETRY_ENABLED", "false")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.skill_events import SkillUsedEvent
from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from bot.handlers.agents import decide_agent_design
from bot.handlers.crews import decide_crew_design
from bot.handlers.messages import handle_text_message, start_command
from bot.handlers.common import report_error as original_report_error
from core.config import config
from database.base import Base
from database.bootstrap import _register_entities
from database.entities.agent_run import AgentRunEntity
from database.entities.agent_run_event import AgentRunEventEntity
from database.entities.user import UserEntity
from models.agent import AgentKind
from models.agent_run import AgentRunStatus
from models.permission import ActionClass, PermissionSubjectType
from models.skill import SkillPackage, SkillPackageFile
from models.tool import ToolDefinition
from services.agent import AgentService
from services.chat import ChatService
from services.crew import CrewService
from services.permission import PermissionService
from services.skill import SkillService
from services.tool_executor import RegisteredTool, ToolRegistry


class LocalChat:
    """Only the outbound Telegram transport is replaced by a local recorder."""

    def __init__(self, report: dict, telegram_id: int):
        self.report = report
        self.from_user = SimpleNamespace(
            id=telegram_id, username="live_skill_validation", first_name="Live Test",
            last_name=None,
        )
        self.chat = SimpleNamespace(id=telegram_id)
        self.text = ""
        self.bot = SimpleNamespace(send_chat_action=self.send_chat_action)

    async def send_chat_action(self, **kwargs):
        pass

    async def answer(self, text: str = "", **kwargs):
        self.report["dialogue"].append({"role": "assistant", "content": text})
        print("ASSISTANT:", text, flush=True)


async def main(*, crew_mode: bool = False) -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    nonce = uuid4().hex[:12]
    directory = Path("reports") / f"live_{'crew' if crew_mode else 'skill'}_chat_{stamp}_{nonce}"
    directory.mkdir(parents=True)
    report = {"started_at": stamp, "dialogue": [], "stages": [],
              "skill_events": [], "tool_events": [], "s3_reads": [],
              "transport": "local chat transport; real Telegram handlers",
              "database": "isolated SQLite", "llm": "configured real endpoint",
              "storage": "configured real S3", "success": False}
    engine = create_async_engine(f"sqlite+aiosqlite:///{directory.as_posix()}/database.sqlite")
    _register_entities()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    storage = MemoryStorage()
    user_id = 100000 + int(nonce[:6], 16)
    chat = LocalChat(report, user_id)
    state = FSMContext(storage=storage, key=StorageKey(
        bot_id=1, chat_id=user_id, user_id=user_id,
    ))
    marker = f"RULE-{nonce}"
    skill_key = f"live-quote-{nonce}"
    agent_name = f"Расчётчик закупок {nonce}"
    crew_name = f"Команда расчёта закупок {nonce}"
    skill = None
    skills = None
    skill_paths: set[str] = set()
    started = time.monotonic()

    def on_skill(_source, event):
        report["skill_events"].append({"name": event.skill_name,
                                        "disclosure_level": event.disclosure_level})
        if event.skill_path:
            skill_paths.add(str(event.skill_path))

    def on_tool(_source, event):
        report["tool_events"].append({"name": event.tool_name})

    async def say(text, session):
        chat.text = text
        report["dialogue"].append({"role": "user", "content": text})
        print("USER:", text, flush=True)
        await asyncio.wait_for(handle_text_message(chat, session, state), timeout=180)

    async def observe_error(target, error, operation):
        detail = str(error)
        for secret in (config.s3.access_key, config.s3.secret_key, config.llm.api_key):
            if secret:
                detail = detail.replace(secret, "[REDACTED]")
        report.setdefault("handler_errors", []).append({
            "operation": operation, "type": type(error).__name__, "message": detail[:2000],
        })
        print("HANDLER ERROR:", type(error).__name__, detail[:1000], flush=True)
        await original_report_error(target, error, operation)

    try:
        async with session_factory() as session:
            session.add(UserEntity(id=user_id, telegram_id=user_id))
            await session.commit()
            await start_command(chat, session)
            primary = await AgentService(session).ensure_primary_agent(user_id=user_id)
            resource = f"quotes:{skill_key}"
            await PermissionService(session).grant(
                user_id=user_id, subject_type=PermissionSubjectType.PERSISTENT_AGENT,
                subject_id=str(primary.id), action_class=ActionClass.READ,
                resource_scope=resource,
            )
            quotes = {
                "A-17": {"quantity": 7, "unit_price": 1250},
                "B-24": {"quantity": 4, "unit_price": 1875},
            }
            body = (
                "# Расчёт закупок по специальному регламенту\n"
                "Для расчёта сначала получи исходные данные через backend tool read_quote.\n"
                "Аргумент arguments должен содержать quote_id с номером заявки пользователя.\n"
                "Считай: quantity * unit_price, затем скидка ровно 13%, затем сбор ровно 370 тенге.\n"
                "Округли итог до двух знаков после запятой. Сбор добавляется после скидки.\n"
                f"В ответе обязательно укажи контрольную метку {marker}, номер заявки, "
                "исходную сумму, скидку, сбор и итог в тенге. Не выдумывай исходные данные.\n"
            )
            skills = SkillService(session)
            skill = await skills.upload_user_skill(
                user_id=user_id, key=skill_key, name="Регламент расчёта закупок",
                description="Расчёт суммы закупки по заявке из S3 через read_quote с внутренним регламентом.",
                required_permissions=[f"read:{resource}"],
                package=SkillPackage(
                    skill_md=body,
                    manifest={"key": skill_key, "version": 1, "entrypoint": None,
                              "files": ["resources/quotes.json"],
                              "tools": [{"id": "read_quote", "resource": resource}]},
                    files=[SkillPackageFile(path="resources/quotes.json",
                                            content=json.dumps(quotes).encode("utf-8"))],
                ),
            )
            report["skill_id"] = str(skill.id)
            report["s3_package_verified"] = await skills.verify_skill_storage(
                user_id=user_id, skill_id=skill.id,
            )
            assert report["s3_package_verified"]
            report["stages"].append("real_s3_package_uploaded_and_verified")
            print("STAGE: real S3 package uploaded and verified", flush=True)

            async def read_quote(arguments):
                quote_id = arguments.get("quote_id")
                data = await skills.get_package_file(
                    user_id=user_id, skill_id=skill.id, path="resources/quotes.json",
                )
                values = json.loads(data)
                if quote_id not in values:
                    raise ValueError("Unknown test quote")
                report["s3_reads"].append({"quote_id": quote_id, "source": "real S3"})
                print("BACKEND READ: real S3 quote", quote_id, flush=True)
                return {"quote_id": quote_id, **values[quote_id]}

            registry = ToolRegistry([RegisteredTool(ToolDefinition(
                name="read_quote", action_class=ActionClass.READ, resource_prefix="quotes:",
            ), read_quote)])
            with (patch("services.assistant.create_tool_executor", return_value=registry),
                  patch("bot.handlers.messages.report_error", side_effect=observe_error),
                  patch("bot.handlers.agents.report_error", side_effect=observe_error),
                  patch("bot.handlers.crews.report_error", side_effect=observe_error),
                  crewai_event_bus.scoped_handlers()):
                crewai_event_bus.on(SkillUsedEvent)(on_skill)
                crewai_event_bus.on(ToolUsageFinishedEvent)(on_tool)
                await say("Привет! Мне нужен помощник для расчёта закупок. Пока просто поздоровайся.", session)
                report["stages"].append("real_llm_greeting")
                await say(
                    (f"Создай команду с именем «{crew_name}» для расчёта закупок по заявкам. "
                     "Включи расчётчика с указанным Skill и редактора итогового отчёта. "
                     "Сохрани последовательные задачи: получить данные и рассчитать заявку, затем оформить отчёт. "
                     "Редактор должен сохранить расчёт и контрольную метку из результата расчётчика. "
                     if crew_mode else
                     f"Создай агента с именем «{agent_name}» для расчёта закупок по заявкам. ") +
                    f"Используй доступный Skill {skill_key} — «Регламент расчёта закупок». "
                    "Выбери этот Skill из каталога и запроси только его необходимое READ permission. "
                    "Агент должен загружать native Skill перед расчётом и получать данные через read_quote. "
                    "Другие инструменты и subagents не нужны, пользователь будет давать номер заявки.",
                    session,
                )
                proposal_data = await state.get_data()
                assert "blueprint" in proposal_data, "Chat design did not produce a blueprint"
                report["blueprint"] = proposal_data["blueprint"]
                selected = (
                    [item for member in proposal_data["blueprint"]["agents"] for item in member["skill_ids"]]
                    if crew_mode else proposal_data["blueprint"]["skill_ids"]
                )
                assert str(skill.id) in selected, "LLM did not select the S3 skill"
                confirmation = "Подтверждаю создание команды." if crew_mode else "Подтверждаю создание агента."
                report["dialogue"].append({"role": "user", "content": confirmation})
                print("USER:", confirmation, flush=True)

                async def callback_answer(*args, **kwargs):
                    pass

                callback = SimpleNamespace(
                    data=f"{'crew' if crew_mode else 'agent'}:design:confirm:{proposal_data['wizard_id']}",
                    from_user=chat.from_user, message=chat, answer=callback_answer,
                )
                if crew_mode:
                    await decide_crew_design(callback, state, session)
                    crews = await CrewService(session).list(user_id=user_id)
                    assert len(crews) == 1
                    report["crew_id"] = str(crews[0].id)
                    crew_name = crews[0].name
                else:
                    await decide_agent_design(callback, state, session)
                agents = await AgentService(session).list_active_agents(user_id)
                workers = [item for item in agents if item.kind == AgentKind.USER]
                worker = None
                for candidate in workers:
                    candidate_skills = await skills.list_agent_skills(user_id=user_id, agent_id=candidate.id)
                    if any(item.id == skill.id for item in candidate_skills):
                        worker = candidate
                        break
                assert worker is not None, "Created agents have no assigned test skill"
                report["agent_id"] = str(worker.id)
                assigned = await skills.list_agent_skills(user_id=user_id, agent_id=worker.id)
                assert any(item.id == skill.id for item in assigned)
                report["stages"].append(f"chat_created_{'crew' if crew_mode else 'agent'}_with_llm_selected_skill")
                for quote_id, expected in [("A-17", "7982.50"), ("B-24", "6895")]:
                    before = len(report["dialogue"])
                    await say(
                        (f"Запусти сохранённую команду «{crew_name}» для заявки {quote_id}. "
                         "Выбери её из Available crews через RUN_CREW и верни итоговый отчёт команды. "
                         if crew_mode else f"Передай агенту «{worker.name}» заявку {quote_id}. ") +
                        "Пусть расчётчик применит свой Skill "
                        "«Регламент расчёта закупок», получит исходные данные через read_quote "
                        "и рассчитает итог. Верни его расчёт и контрольную метку без изменений.", session,
                    )
                    answers = "\n".join(row["content"] for row in report["dialogue"][before:]
                                        if row["role"] == "assistant")
                    assert marker in answers, "Hidden skill-only marker did not reach the chat response"
                    normalized = answers.replace(" ", "").replace("\u00a0", "").replace(",", ".")
                    assert expected in normalized or expected.removesuffix(".50") + ".5" in normalized, "Incorrect quote calculation"
                    report["stages"].append(f"chat_completed_{quote_id}_with_skill")
                runs = (await session.execute(select(AgentRunEntity))).scalars().all()
                report["runs"] = [{"id": str(run.id), "status": run.status.value,
                                   "usage": run.usage,
                                   "delegations": run.checkpoint.get("delegation_state", [])}
                                  for run in runs]
                assert all(run.status == AgentRunStatus.COMPLETED for run in runs)
                if crew_mode:
                    events = (await session.execute(select(AgentRunEventEntity))).scalars().all()
                    report["crew_events"] = [
                        {"type": event.event_type.value, "payload": event.payload}
                        for event in events if event.payload.get("delegation_type") == "run_crew"
                    ]
                    assert sum(row["type"] == "delegation_completed" for row in report["crew_events"]) == 2
                assert {item["quote_id"] for item in report["s3_reads"]} == {"A-17", "B-24"}
                assert any(item["disclosure_level"] == 2 for item in report["skill_events"])
                assert report["tool_events"] and any(row["name"] == "load_skill" for row in report["tool_events"])
                report["temporary_skills_removed"] = all(not Path(path).exists() for path in skill_paths)
                assert report["temporary_skills_removed"]
                report["success"] = True
    except Exception as error:
        text = str(error)
        for secret in (config.s3.access_key, config.s3.secret_key, config.llm.api_key):
            if secret:
                text = text.replace(secret, "[REDACTED]")
        report["error"] = {"type": type(error).__name__, "message": text[:2000]}
        print("SCENARIO FAILED:", type(error).__name__, text[:500], flush=True)
    finally:
        async with session_factory() as inspection_session:
            runs = (await inspection_session.execute(select(AgentRunEntity))).scalars().all()
            report["runs"] = [{"id": str(run.id), "status": run.status.value,
                               "error": run.error, "usage": run.usage,
                               "delegations": run.checkpoint.get("delegation_state", [])}
                              for run in runs]
        if skill is not None:
            async with session_factory() as cleanup_session:
                cleanup_skills = SkillService(cleanup_session)
                await cleanup_skills.archive_skill(user_id=user_id, skill_id=skill.id)
                await cleanup_skills.delete_skill_version_storage(skill_id=skill.id)
                report["test_s3_package_removed"] = True
        await storage.close()
        await engine.dispose()
        report["duration_seconds"] = round(time.monotonic() - started, 2)
        destination = directory / "report.json"
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("REPORT:", destination, flush=True)
        print("SUCCESS:", report["success"], flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(crew_mode="--crew" in sys.argv)))
