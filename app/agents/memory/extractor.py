"""The Memory Agent produces structured candidates and has no storage access."""

import json
from typing import Protocol

import httpx

from core.config import LLMConfig
from models.memory import MemoryExtractionContext, MemoryExtractionResult


class MemoryExtractionAgent(Protocol):
    async def extract(self, context: MemoryExtractionContext) -> MemoryExtractionResult:
        ...


class OpenAICompatibleMemoryAgent:
    """Internal Memory Curator backed by the platform's OpenAI-compatible LLM API."""

    def __init__(self, settings: LLMConfig, *, instructions: str):
        if not settings.base_url or not settings.model:
            raise ValueError("Memory Agent requires VLLM_URL and VLLM_MODEL")
        self._settings = settings
        self._instructions = instructions

    async def extract(self, context: MemoryExtractionContext) -> MemoryExtractionResult:
        headers = {"Content-Type": "application/json"}
        if self._settings.api_key:
            headers["Authorization"] = f"Bearer {self._settings.api_key}"
        async with httpx.AsyncClient(timeout=self._settings.timeout) as client:
            response = await client.post(
                self._endpoint(),
                headers=headers,
                json={
                    "model": self._settings.model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": self._system_prompt()},
                        {
                            "role": "user",
                            "content": json.dumps(context.model_dump(mode="json")),
                        },
                    ],
                },
            )
            response.raise_for_status()
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Memory Agent returned an unexpected completion shape") from exc
        if not isinstance(content, str):
            raise ValueError("Memory Agent returned non-text structured output")
        return MemoryExtractionResult.model_validate_json(self._strip_fence(content))

    def _system_prompt(self) -> str:
        return (
            f"You are the platform's internal Memory Curator. {self._instructions}\n"
            "Return only JSON matching {\"candidates\": [...]}. Each candidate must contain "
            "memory_type, scope, agent_id (or null), key, content, confidence, reason, and "
            "explicit_user_request. Extract only durable user preferences, facts, goals, "
            "project conventions, or decisions. Never extract greetings, acknowledgements, "
            "temporary tool output, secrets, speculation, or a duplicate of existing memory. "
            "Use USER_GLOBAL for stable user-level memory. Use AGENT_PRIVATE only when the "
            "context provides the exact eligible persistent agent id. Never use CREW_SHARED "
            "or RUN_EPHEMERAL. You only propose candidates; you cannot store them."
        )

    def _endpoint(self) -> str:
        base_url = self._settings.base_url.rstrip("/")
        return base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"

    @staticmethod
    def _strip_fence(content: str) -> str:
        stripped = content.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            return stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return stripped
