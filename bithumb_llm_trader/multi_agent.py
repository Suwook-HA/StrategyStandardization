"""Multi-agent orchestration layer for portfolio level automation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .api_client import BithumbAPI, BithumbAPIError
from .config import StrategyConfig
from .decision import Action, TradeDecision
from .llm import LLMDecisionMaker
from .risk import RiskManager
from .utils import extract_balance, format_units, safe_float

logger = logging.getLogger(__name__)


def _pair_key(config: StrategyConfig) -> str:
    pair = config.trading_pair
    return f"{pair.order_currency}/{pair.payment_currency}"


def _decision_exposure(decision: TradeDecision, reference_price: float) -> float:
    if decision.action is Action.HOLD or decision.amount <= 0:
        return 0.0
    price = decision.target_price if decision.target_price is not None else reference_price
    price = safe_float(price, -1)
    if price <= 0:
        return 0.0
    direction = 1 if decision.action is Action.BUY else -1
    return direction * price * decision.amount


@dataclass(slots=True)
class StrategyCycleResult:
    """Outcome of a single strategy within a multi-agent cycle."""

    name: str
    market_state: Dict[str, Any]
    account_state: Dict[str, Any]
    llm_decision: TradeDecision
    final_decision: TradeDecision
    order_response: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class PortfolioCycleResult:
    """Aggregated result for an entire multi-asset portfolio run."""

    timestamp: str
    strategy_results: List[StrategyCycleResult]
    cash_by_currency: Dict[str, float]
    positions_by_pair: Dict[str, float]
    position_values_by_pair: Dict[str, float]
    total_cash: float
    total_positions_value: float
    net_exposure_change: float


@dataclass(slots=True)
class StrategyAgentBundle:
    """Container bundling all agents required for a single strategy."""

    config: StrategyConfig
    decision_maker: LLMDecisionMaker
    name: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    risk_manager: RiskManager = field(init=False)

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = _pair_key(self.config)
        self.risk_manager = RiskManager(self.config.risk)


class MarketDataAgent:
    """Collects market and account state required by the strategy agents."""

    def __init__(self, api: BithumbAPI):
        self.api = api

    def gather(self, config: StrategyConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pair = config.trading_pair
        ticker = self.api.get_ticker(pair.order_currency, pair.payment_currency)
        orderbook = self.api.get_orderbook(pair.order_currency, pair.payment_currency)
        balance = self.api.get_balance(pair.order_currency, pair.payment_currency)
        ticker_data = ticker.get("data", ticker)
        orderbook_data = orderbook.get("data", orderbook)
        balance_data = balance.get("data", balance)
        account_state = {
            "balance_order_currency": extract_balance(balance_data, pair.order_currency),
            "balance_payment_currency": extract_balance(balance_data, pair.payment_currency),
            "raw": balance_data,
        }
        market_state = {"ticker": ticker_data, "orderbook": orderbook_data}
        return market_state, account_state


class StrategyAgent:
    """Requests trade ideas from the LLM layer."""

    def decide(
        self,
        decision_maker: LLMDecisionMaker,
        market_state: Dict[str, Any],
        account_state: Dict[str, Any],
        config: StrategyConfig,
        history: Iterable[Dict[str, Any]],
    ) -> TradeDecision:
        return decision_maker.decide(market_state, account_state, config, history)


class RiskAgent:
    """Applies per-strategy risk management."""

    def assess(
        self,
        risk_manager: RiskManager,
        decision: TradeDecision,
        market_state: Dict[str, Any],
        account_state: Dict[str, Any],
    ) -> TradeDecision:
        return risk_manager.apply(decision, market_state, account_state)


class ExecutionAgent:
    """Handles live trading or dry-run execution for a strategy."""

    def __init__(self, api: BithumbAPI):
        self.api = api

    def execute(
        self,
        config: StrategyConfig,
        decision: TradeDecision,
        market_state: Dict[str, Any],
    ) -> Tuple[TradeDecision, Optional[Dict[str, Any]]]:
        if decision.action is Action.HOLD or decision.amount <= 0:
            logger.info("Portfolio execution agent skipping trade: %s", decision.reasoning)
            return decision, None

        ticker = market_state.get("ticker", {})
        price = decision.target_price or safe_float(ticker.get("closing_price"))
        if price <= 0:
            logger.warning("Portfolio execution aborted: invalid price derived from market data.")
            return (
                TradeDecision.hold(
                    confidence=decision.confidence,
                    reasoning="Execution aborted due to invalid price.",
                    raw_response=decision.raw_response,
                ),
                None,
            )

        if config.dry_run:
            logger.info(
                "Dry-run portfolio trade: %s %s units at %s",
                decision.action.value,
                decision.amount,
                price,
            )
            return decision.with_adjustments(target_price=price), None

        order_type = "bid" if decision.action is Action.BUY else "ask"
        pair = config.trading_pair
        try:
            response = self.api.place_order(
                order_type=order_type,
                order_currency=pair.order_currency,
                payment_currency=pair.payment_currency,
                units=format_units(decision.amount),
                price=format_units(price),
            )
        except BithumbAPIError as exc:
            logger.error("Portfolio order placement failed: %s", exc)
            return (
                TradeDecision.hold(
                    confidence=decision.confidence,
                    reasoning=f"Order rejected by exchange: {exc}",
                    raw_response=decision.raw_response,
                ),
                None,
            )
        return decision.with_adjustments(target_price=price), response


class MultiAgentPortfolioManager:
    """Coordinates several strategy agents to manage a portfolio."""

    def __init__(
        self,
        api: BithumbAPI,
        strategies: Sequence[StrategyAgentBundle],
        *,
        max_history: int = 50,
    ) -> None:
        self.api = api
        self.strategies = list(strategies)
        self.market_agent = MarketDataAgent(api)
        self.strategy_agent = StrategyAgent()
        self.risk_agent = RiskAgent()
        self.execution_agent = ExecutionAgent(api)
        self.max_history = max_history

    def run_cycle(self) -> PortfolioCycleResult:
        timestamp = datetime.now(timezone.utc).isoformat()
        strategy_results: List[StrategyCycleResult] = []
        cash_by_currency: Dict[str, float] = {}
        positions_by_pair: Dict[str, float] = {}
        position_values_by_pair: Dict[str, float] = {}
        net_exposure_change = 0.0

        for bundle in self.strategies:
            market_state, account_state = self.market_agent.gather(bundle.config)
            llm_decision = self.strategy_agent.decide(
                bundle.decision_maker,
                market_state,
                account_state,
                bundle.config,
                bundle.history,
            )
            final_decision = self.risk_agent.assess(
                bundle.risk_manager, llm_decision, market_state, account_state
            )
            executed_decision, order_response = self.execution_agent.execute(
                bundle.config, final_decision, market_state
            )
            self._record_history(bundle, executed_decision)
            pair_name = bundle.name or _pair_key(bundle.config)
            strategy_results.append(
                StrategyCycleResult(
                    name=pair_name,
                    market_state=market_state,
                    account_state=account_state,
                    llm_decision=llm_decision,
                    final_decision=executed_decision,
                    order_response=order_response,
                )
            )

            payment_currency = bundle.config.trading_pair.payment_currency
            cash_by_currency.setdefault(
                payment_currency, safe_float(account_state.get("balance_payment_currency"))
            )
            positions_by_pair[pair_name] = safe_float(account_state.get("balance_order_currency"))
            ticker_price = safe_float(market_state.get("ticker", {}).get("closing_price"))
            position_values_by_pair[pair_name] = (
                positions_by_pair[pair_name] * ticker_price
            )
            net_exposure_change += _decision_exposure(executed_decision, ticker_price)

        total_cash = sum(cash_by_currency.values())
        total_positions_value = sum(position_values_by_pair.values())
        return PortfolioCycleResult(
            timestamp=timestamp,
            strategy_results=strategy_results,
            cash_by_currency=cash_by_currency,
            positions_by_pair=positions_by_pair,
            position_values_by_pair=position_values_by_pair,
            total_cash=total_cash,
            total_positions_value=total_positions_value,
            net_exposure_change=net_exposure_change,
        )

    def _record_history(self, bundle: StrategyAgentBundle, decision: TradeDecision) -> None:
        bundle.history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": decision.action.value,
                "amount": decision.amount,
                "price": decision.target_price,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
            }
        )
        if len(bundle.history) > self.max_history:
            bundle.history[:] = bundle.history[-self.max_history :]


__all__ = [
    "StrategyCycleResult",
    "PortfolioCycleResult",
    "StrategyAgentBundle",
    "MultiAgentPortfolioManager",
]
