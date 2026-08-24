from typing import Protocol

from models.message import Message


class AssistantAgent(Protocol):
    async def run(
        self,
        *,
        message: str,
        history: list[Message],
    ) -> str:
        ...