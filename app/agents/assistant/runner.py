"""Public assistant runtime backed by the dynamic CrewAI implementation."""

from agents.assistant.crewai.factory import DynamicCrewAIFactory
from agents.assistant.crewai.runtime import DynamicCrewAIRuntime


class CrewAIAssistant(DynamicCrewAIRuntime):
    def __init__(self, *, factory: DynamicCrewAIFactory | None = None) -> None:
        super().__init__(factory=factory)
