"""Guard actual Crew LLM requests, including conversion and forced-final calls."""

import threading
from typing import Any

from crewai import BaseLLM
from pydantic import PrivateAttr

from models.runtime import RuntimeBudgetExceededError


class LLMCallBudget:
    def __init__(self, limit: int) -> None:
        self.limit = max(0, limit)
        self.used = 0
        self._lock = threading.Lock()

    def consume(self) -> None:
        with self._lock:
            if self.used >= self.limit:
                raise RuntimeBudgetExceededError("Runtime budget exceeded: crew llm_calls")
            self.used += 1


class BudgetedLLM(BaseLLM):
    """Use the configured provider while gating each inference request."""

    _delegate: BaseLLM = PrivateAttr()
    _budget: LLMCallBudget = PrivateAttr()

    def __init__(self, delegate: BaseLLM, budget: LLMCallBudget) -> None:
        super().__init__(model=delegate.model, temperature=delegate.temperature)
        self._delegate = delegate
        self._budget = budget

    def supports_function_calling(self) -> bool:
        capability = getattr(self._delegate, "supports_function_calling", None)
        return bool(capability()) if callable(capability) else False

    def supports_stop_words(self) -> bool:
        return self._delegate.supports_stop_words()

    def supports_multimodal(self) -> bool:
        return self._delegate.supports_multimodal()

    def get_context_window_size(self) -> int:
        return self._delegate.get_context_window_size()

    def call(self, messages, **kwargs: Any):
        self._budget.consume()
        self._delegate.stop = list(self.stop)
        return self._delegate.call(messages, **kwargs)

    async def acall(self, messages, **kwargs: Any):
        self._budget.consume()
        self._delegate.stop = list(self.stop)
        return await self._delegate.acall(messages, **kwargs)
