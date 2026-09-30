"""Factory for the OpenAI-compatible LLM used by CrewAI."""

from typing import Any

from core.config import config


def create_crewai_llm() -> Any:
    from crewai import LLM

    model = config.llm.model or "openai/gpt-4o-mini"
    if "/" not in model:
        model = f"openai/{model}"
    options: dict[str, object] = {
        "model": model,
        "temperature": config.llm.temperature,
        "timeout": config.llm.timeout,
        "max_tokens": config.llm.max_token,
    }
    if config.llm.api_key:
        options["api_key"] = config.llm.api_key
    elif config.llm.base_url:
        # OpenAI-compatible servers may intentionally require no authentication.
        # CrewAI still requires a nonempty transport value; prevent implicit use
        # of an unrelated OPENAI_API_KEY from the process environment.
        options["api_key"] = "not-required"
    if config.llm.base_url:
        options["base_url"] = config.llm.base_url
    return LLM(**options)
