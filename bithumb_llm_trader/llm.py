"""LLM client abstractions used by the trading engine."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional

from .config import LLMConfig, StrategyConfig
from .decision import DecisionParser, TradeDecision
from .prompts import build_trading_prompt


class LLMClient(ABC):
    """Abstract base class describing a minimal language model client."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a textual completion for ``prompt``."""


class OpenAIChatClient(LLMClient):
    """Adapter around the ``openai`` Python package (Responses API)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided via argument or OPENAI_API_KEY env variable")
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The 'openai' package is required for OpenAIChatClient. Install it via 'pip install openai'."
            ) from exc
        self._client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        model = kwargs.get("model", "gpt-4o-mini")
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_output_tokens", 512)
        response = self._client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        return response.output_text  # type: ignore[return-value]


class LLMDecisionMaker:
    """High-level helper that turns market context into a :class:`TradeDecision`."""

    def __init__(
        self,
        client: LLMClient,
        parser: Optional[DecisionParser] = None,
        llm_config: Optional[LLMConfig] = None,
    ) -> None:
        self.client = client
        self.parser = parser or DecisionParser()
        self.llm_config = llm_config or LLMConfig()

    def decide(
        self,
        market_data: Dict[str, Any],
        account_state: Dict[str, Any],
        config: StrategyConfig,
        history: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> TradeDecision:
        prompt = build_trading_prompt(market_data, account_state, config, history)
        response = self.client.generate(
            prompt,
            model=self.llm_config.model,
            temperature=self.llm_config.temperature,
            max_output_tokens=self.llm_config.max_output_tokens,
        )
        return self.parser.parse(response)


__all__ = ["LLMClient", "LLMDecisionMaker", "OpenAIChatClient"]
