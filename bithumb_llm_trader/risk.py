"""Risk management utilities for the trading engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .config import RiskConfig
from .decision import Action, TradeDecision
from .utils import safe_float


@dataclass(slots=True)
class RiskManager:
    """Applies sizing and confidence constraints to trade decisions."""

    config: RiskConfig

    def apply(
        self,
        decision: TradeDecision,
        market_data: Dict[str, Any],
        account_state: Dict[str, Any],
    ) -> TradeDecision:
        if decision.action is Action.HOLD:
            return decision
        if decision.confidence < self.config.min_confidence:
            return TradeDecision.hold(
                confidence=decision.confidence,
                reasoning=(
                    f"Decision rejected by risk manager: confidence {decision.confidence:.2f} "
                    f"below threshold {self.config.min_confidence:.2f}."
                ),
                raw_response=decision.raw_response,
            )

        price = self._resolve_price(decision, market_data)
        if price <= 0:
            return TradeDecision.hold(
                confidence=decision.confidence,
                reasoning="Decision rejected: unable to determine valid execution price.",
                raw_response=decision.raw_response,
            )

        if decision.action is Action.BUY:
            return self._assess_buy(decision, price, account_state)
        if decision.action is Action.SELL:
            return self._assess_sell(decision, price, account_state)
        return decision

    # ------------------------------------------------------------------
    def _assess_buy(
        self, decision: TradeDecision, price: float, account_state: Dict[str, Any]
    ) -> TradeDecision:
        available_cash = safe_float(account_state.get("balance_payment_currency"))
        max_by_cash = available_cash / price if price else 0.0
        max_by_value = self.config.max_trade_value / price if price else 0.0
        allowed = min(
            max(decision.amount, 0.0),
            self.config.max_position_size,
            max_by_cash,
            max_by_value,
        )
        if allowed <= 0:
            return TradeDecision.hold(
                confidence=decision.confidence,
                reasoning="Insufficient funds to execute BUY order.",
                raw_response=decision.raw_response,
            )
        adjusted = decision.with_adjustments(amount=allowed)
        return self._attach_protection_levels(adjusted, price)

    def _assess_sell(
        self, decision: TradeDecision, price: float, account_state: Dict[str, Any]
    ) -> TradeDecision:
        available_asset = safe_float(account_state.get("balance_order_currency"))
        allowed = min(
            max(decision.amount, 0.0),
            self.config.max_position_size,
            available_asset,
        )
        if allowed <= 0:
            return TradeDecision.hold(
                confidence=decision.confidence,
                reasoning="No inventory available to SELL.",
                raw_response=decision.raw_response,
            )
        adjusted = decision.with_adjustments(amount=allowed)
        return self._attach_protection_levels(adjusted, price)

    def _resolve_price(self, decision: TradeDecision, market_data: Dict[str, Any]) -> float:
        if decision.target_price:
            return decision.target_price
        ticker = market_data.get("ticker", {})
        candidates = [
            ticker.get("closing_price"),
            ticker.get("closePrice"),
            ticker.get("price"),
        ]
        for value in candidates:
            price = safe_float(value, -1)
            if price > 0:
                return price
        return -1.0

    def _attach_protection_levels(self, decision: TradeDecision, entry_price: float) -> TradeDecision:
        if entry_price <= 0:
            return decision
        if decision.action is Action.BUY:
            stop_loss = entry_price * (1 - self.config.stop_loss_pct)
            take_profit = entry_price * (1 + self.config.take_profit_pct)
        else:
            stop_loss = entry_price * (1 + self.config.stop_loss_pct)
            take_profit = entry_price * (1 - self.config.take_profit_pct)
        reasoning = decision.reasoning or ""
        reasoning = reasoning + " | Risk controls applied"
        return decision.with_adjustments(
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning.strip(),
        )


__all__ = ["RiskManager"]
