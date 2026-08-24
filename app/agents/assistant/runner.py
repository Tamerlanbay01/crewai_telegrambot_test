from models.message import Message
from agents.assistant.crewai.crew import AssistantCrew


class CrewAIAssistant:
    async def run(
        self,
        *,
        message: str,
        history: list[Message],
    ) -> str:
        crew = AssistantCrew().crew()

        result = await crew.kickoff_async(
            inputs={
                "message": message,
                "history": self._format_history(history),
            }
        )

        return result.raw

    @staticmethod
    def _format_history(
        history: list[Message],
    ) -> str:
        return "\n".join(
            f"{item.role}: {item.content}"
            for item in history
        )