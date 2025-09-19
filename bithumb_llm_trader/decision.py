"""Utilities for interpreting and validating LLM trade decisions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Optional


class DecisionParseError(ValueError):
    """Raised when the LLM output cannot be converted into a decision."""


class Action(str, Enum):
    """Enumeration of supported trade actions."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

    @classmethod
    def from_text(cls, value: str) -> "Action":
        upper = value.strip().upper()
        try:
            return cls[upper]
        except KeyError as exc:
            raise DecisionParseError(f"Unsupported action: {value}") from exc


@dataclass(slots=True)
class TradeDecision:
    """Structured representation of a trading decision."""

    action: Action
    confidence: float
    amount: float
    target_price: Optional[float] = None
    reasoning: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    raw_response: Optional[str] = None

    def validate(self) -> "TradeDecision":
        if not 0 <= self.confidence <= 1:
            raise DecisionParseError("Confidence must be between 0 and 1")
        if self.amount < 0:
            raise DecisionParseError("Trade amount cannot be negative")
        if self.target_price is not None and self.target_price <= 0:
            raise DecisionParseError("Target price must be positive when provided")
        return self

    def with_adjustments(
        self,
        *,
        amount: Optional[float] = None,
        target_price: Optional[float] = None,
        reasoning: Optional[str] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> "TradeDecision":
        return replace(
            self,
            amount=self.amount if amount is None else amount,
            target_price=self.target_price if target_price is None else target_price,
            reasoning=self.reasoning if reasoning is None else reasoning,
            stop_loss=self.stop_loss if stop_loss is None else stop_loss,
            take_profit=self.take_profit if take_profit is None else take_profit,
        )

    @classmethod
    def hold(cls, confidence: float, reasoning: str, raw_response: Optional[str] = None) -> "TradeDecision":
        return cls(
            action=Action.HOLD,
            confidence=confidence,
            amount=0.0,
            reasoning=reasoning,
            raw_response=raw_response,
        )


class DecisionParser:
    """Parses the raw text output of an LLM into :class:`TradeDecision`."""

    JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

    def parse(self, text: str) -> TradeDecision:
        candidate = self._extract_json_block(text)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise DecisionParseError("LLM response did not contain valid JSON") from exc
        return self._decision_from_dict(data, text)

    def _extract_json_block(self, text: str) -> str:
        # Support Markdown code fences surrounding the JSON payload
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced:
            return fenced.group(1)
        match = self.JSON_PATTERN.search(text)
        if not match:
            raise DecisionParseError("Could not locate a JSON object in LLM response")
        return match.group(0)

    def _decision_from_dict(self, data: Dict[str, Any], raw: str) -> TradeDecision:
        if "action" not in data:
            raise DecisionParseError("Decision JSON must include an 'action' field")
        action = Action.from_text(str(data["action"]))
        confidence = float(data.get("confidence", 0))
        amount = float(data.get("amount", 0))
        target_price = data.get("target_price")
        target_value = float(target_price) if target_price is not None else None
        reasoning = str(data.get("reasoning", "")).strip()
        stop_loss = data.get("stop_loss")
        take_profit = data.get("take_profit")
        decision = TradeDecision(
            action=action,
            confidence=confidence,
            amount=amount,
            target_price=target_value,
            reasoning=reasoning,
            stop_loss=float(stop_loss) if stop_loss is not None else None,
            take_profit=float(take_profit) if take_profit is not None else None,
            raw_response=raw,
        )
        return decision.validate()


__all__ = ["Action", "DecisionParseError", "DecisionParser", "TradeDecision"]
