"""Application facing adapter for structured agent and crew design."""

from typing import Any, TypeVar

from pydantic import ValidationError

from agents.factory.crewai.crew import AgentFactoryCrew
from models.agent_factory import AgentBlueprint, AgentFactoryInput, CrewBlueprint


BlueprintT = TypeVar("BlueprintT", AgentBlueprint, CrewBlueprint)


class FactoryOutputError(ValueError):
    """CrewAI returned no usable structured blueprint."""


class AgentFactoryRunner:
    def __init__(self, *, factory: AgentFactoryCrew | None = None) -> None:
        self._factory = factory or AgentFactoryCrew()

    async def design_agent(self, request: AgentFactoryInput) -> AgentBlueprint:
        output = await self._factory.build(request).kickoff_async()
        return self._parse(output, AgentBlueprint)

    async def design_crew(self, request: AgentFactoryInput) -> CrewBlueprint:
        output = await self._factory.build(request, crew=True).kickoff_async()
        return self._parse(output, CrewBlueprint)

    @staticmethod
    def _parse(output: Any, model: type[BlueprintT]) -> BlueprintT:
        try:
            if output.pydantic is not None:
                return model.model_validate(output.pydantic)
            if output.json_dict is not None:
                return model.model_validate(output.json_dict)
            return model.model_validate_json(output.raw)
        except (AttributeError, TypeError, ValueError, ValidationError) as exc:
            raise FactoryOutputError("Agent design output is malformed") from exc
