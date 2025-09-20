from __future__ import annotations

from typing import List

import pytest

from bithumb_llm_trader.config import (
    APIConfig,
    RiskConfig,
    StrategyConfig,
    TradingPairConfig,
)
from bithumb_llm_trader.decision import Action
from bithumb_llm_trader.llm import LLMClient, LLMDecisionMaker
from bithumb_llm_trader.multi_agent import (
    MultiAgentPortfolioManager,
    StrategyAgentBundle,
)


class QueueLLM(LLMClient):
    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self.prompts: List[str] = []

    def generate(self, prompt: str, **kwargs) -> str:  # type: ignore[override]
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("No scripted responses remaining")
        return self._responses.pop(0)


class DummyPortfolioAPI:
    def __init__(self) -> None:
        self.orders = []
        self.tickers = {
            ("BTC", "KRW"): {"closing_price": "1000000"},
            ("ETH", "KRW"): {"closing_price": "5000000"},
        }
        self.orderbooks = {
            ("BTC", "KRW"): {"bids": [], "asks": []},
            ("ETH", "KRW"): {"bids": [], "asks": []},
        }
        self.balances = {
            "available_btc": "0.8",
            "available_eth": "2.5",
            "available_krw": "6000000",
        }

    def get_ticker(self, order_currency: str, payment_currency: str):
        return {"data": self.tickers[(order_currency, payment_currency)]}

    def get_orderbook(self, order_currency: str, payment_currency: str):
        return {"data": self.orderbooks[(order_currency, payment_currency)]}

    def get_balance(self, order_currency: str, payment_currency: str):
        return {"data": self.balances}

    def place_order(
        self,
        order_type: str,
        order_currency: str,
        payment_currency: str,
        units: str,
        price: str,
    ):
        order = {
            "type": order_type,
            "order_currency": order_currency,
            "payment_currency": payment_currency,
            "units": units,
            "price": price,
        }
        self.orders.append(order)
        return {"status": "0000", "data": {"order_id": str(len(self.orders))}}


@pytest.fixture
def api():
    return DummyPortfolioAPI()


def make_strategy_config(
    order_currency: str,
    *,
    dry_run: bool,
    max_trade_value: float = 10_000_000,
    max_position_size: float = 1.0,
) -> StrategyConfig:
    return StrategyConfig(
        api=APIConfig(api_key="key", api_secret="secret"),
        trading_pair=TradingPairConfig(order_currency=order_currency, payment_currency="KRW"),
        risk=RiskConfig(
            max_trade_value=max_trade_value,
            max_position_size=max_position_size,
            min_confidence=0.5,
        ),
        dry_run=dry_run,
    )


def test_multi_agent_cycle_aggregates_results(api):
    btc_config = make_strategy_config("BTC", dry_run=True, max_position_size=0.5)
    eth_config = make_strategy_config("ETH", dry_run=True, max_position_size=1.0)

    btc_bundle = StrategyAgentBundle(
        config=btc_config,
        decision_maker=LLMDecisionMaker(
            QueueLLM(
                [
                    '{"action": "BUY", "confidence": 0.9, "amount": 0.3, "target_price": 1010000}',
                ]
            ),
            llm_config=btc_config.llm,
        ),
    )
    eth_bundle = StrategyAgentBundle(
        config=eth_config,
        decision_maker=LLMDecisionMaker(
            QueueLLM(
                [
                    '{"action": "SELL", "confidence": 0.8, "amount": 1.5}',
                ]
            ),
            llm_config=eth_config.llm,
        ),
    )

    manager = MultiAgentPortfolioManager(api, [btc_bundle, eth_bundle])
    result = manager.run_cycle()

    assert {r.name for r in result.strategy_results} == {"BTC/KRW", "ETH/KRW"}
    btc_result = next(r for r in result.strategy_results if r.name == "BTC/KRW")
    eth_result = next(r for r in result.strategy_results if r.name == "ETH/KRW")

    assert btc_result.llm_decision.action is Action.BUY
    assert btc_result.final_decision.amount == pytest.approx(0.3)
    assert eth_result.final_decision.amount == pytest.approx(1.0)  # limited by max_position_size

    assert result.cash_by_currency["KRW"] == pytest.approx(6_000_000.0)
    assert result.positions_by_pair["BTC/KRW"] == pytest.approx(0.8)
    assert result.positions_by_pair["ETH/KRW"] == pytest.approx(2.5)
    assert result.position_values_by_pair["BTC/KRW"] == pytest.approx(800_000.0)
    assert result.position_values_by_pair["ETH/KRW"] == pytest.approx(12_500_000.0)
    assert result.net_exposure_change == pytest.approx(0.3 * 1_010_000 - 1.0 * 5_000_000)

    assert len(btc_bundle.history) == 1
    assert len(eth_bundle.history) == 1


def test_live_execution_places_orders(api):
    config = make_strategy_config("BTC", dry_run=False, max_position_size=0.4)
    bundle = StrategyAgentBundle(
        config=config,
        decision_maker=LLMDecisionMaker(
            QueueLLM(['{"action": "BUY", "confidence": 0.9, "amount": 1.0, "target_price": 1000000}']),
            llm_config=config.llm,
        ),
    )

    manager = MultiAgentPortfolioManager(api, [bundle])
    result = manager.run_cycle()

    assert len(api.orders) == 1
    order = api.orders[0]
    assert order["type"] == "bid"
    assert float(order["units"]) == pytest.approx(0.4)  # capped by risk manager
    assert order["price"] == "1000000"

    cycle_result = result.strategy_results[0]
    assert cycle_result.final_decision.action is Action.BUY
    assert cycle_result.final_decision.amount == pytest.approx(0.4)
    assert result.net_exposure_change == pytest.approx(0.4 * 1_000_000)
