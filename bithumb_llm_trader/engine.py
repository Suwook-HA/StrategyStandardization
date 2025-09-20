"""Trading engine orchestrating API calls, LLM inference and risk controls."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from .api_client import BithumbAPI, BithumbAPIError
from .config import StrategyConfig
from .decision import Action, TradeDecision
from .llm import LLMDecisionMaker
from .risk import RiskManager
from .utils import extract_balance, format_units, safe_float


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TradingEngine:
    """Coordinates the different components of the automated trading stack."""

    api: BithumbAPI
    decision_maker: LLMDecisionMaker
    config: StrategyConfig
    risk_manager: RiskManager = field(init=False)
    history: List[Dict[str, Any]] = field(default_factory=list, init=False)
    max_history: int = 50

    def __post_init__(self) -> None:
        self.risk_manager = RiskManager(self.config.risk)

    # ------------------------------------------------------------------
    def fetch_market_state(self) -> Dict[str, Any]:
        pair = self.config.trading_pair
        ticker = self.api.get_ticker(pair.order_currency, pair.payment_currency)
        orderbook = self.api.get_orderbook(pair.order_currency, pair.payment_currency)
        return {
            "ticker": ticker.get("data", ticker),
            "orderbook": orderbook.get("data", orderbook),
        }

    def fetch_account_state(self) -> Dict[str, Any]:
        pair = self.config.trading_pair
        response = self.api.get_balance(pair.order_currency, pair.payment_currency)
        data = response.get("data", response)
        balance_order = self._extract_balance(data, pair.order_currency)
        balance_payment = self._extract_balance(data, pair.payment_currency)
        return {
            "balance_order_currency": balance_order,
            "balance_payment_currency": balance_payment,
            "raw": data,
        }

    def run_once(self) -> TradeDecision:
        market_state = self.fetch_market_state()
        account_state = self.fetch_account_state()
        decision = self.decision_maker.decide(
            market_state, account_state, self.config, self.history
        )
        adjusted = self.risk_manager.apply(decision, market_state, account_state)
        executed = self._execute(adjusted, market_state, account_state)
        self._record_history(executed)
        return executed

    # ------------------------------------------------------------------
    def _execute(
        self,
        decision: TradeDecision,
        market_state: Dict[str, Any],
        account_state: Dict[str, Any],
    ) -> TradeDecision:
        if decision.action is Action.HOLD or decision.amount <= 0:
            logger.info("No trade executed: %s", decision.reasoning)
            return decision

        price = decision.target_price or safe_float(market_state["ticker"].get("closing_price"))
        if price <= 0:
            logger.warning("Execution aborted: invalid price derived from market data.")
            return TradeDecision.hold(
                confidence=decision.confidence,
                reasoning="Execution aborted due to invalid price.",
                raw_response=decision.raw_response,
            )

        if self.config.dry_run:
            logger.info(
                "Dry-run mode: %s %s units at %s", decision.action.value, decision.amount, price
            )
            return decision.with_adjustments(target_price=price)

        order_type = "bid" if decision.action is Action.BUY else "ask"
        units = format_units(decision.amount)
        pair = self.config.trading_pair
        try:
            response = self.api.place_order(
                order_type=order_type,
                order_currency=pair.order_currency,
                payment_currency=pair.payment_currency,
                units=units,
                price=format_units(price),
            )
            logger.info("Order response: %s", response)
        except BithumbAPIError as exc:
            logger.error("Order placement failed: %s", exc)
            return TradeDecision.hold(
                confidence=decision.confidence,
                reasoning=f"Order rejected by exchange: {exc}",
                raw_response=decision.raw_response,
            )
        return decision.with_adjustments(target_price=price)

    def _extract_balance(self, data: Dict[str, Any], currency: str) -> float:
        return extract_balance(data, currency)

    def _record_history(self, decision: TradeDecision) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": decision.action.value,
            "amount": decision.amount,
            "price": decision.target_price,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
        }
        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]


__all__ = ["TradingEngine"]
