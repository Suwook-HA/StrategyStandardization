"""Integration tests for the trading engine orchestration."""

from __future__ import annotations

from bithumb_llm_trader.config import APIConfig, RiskConfig, StrategyConfig
from bithumb_llm_trader.decision import Action
from bithumb_llm_trader.engine import TradingEngine
from bithumb_llm_trader.llm import LLMClient, LLMDecisionMaker


class DummyAPI:
    def __init__(self) -> None:
        self.orders = []

    def get_ticker(self, order_currency: str, payment_currency: str):
        return {"data": {"closing_price": "1000000"}}

    def get_orderbook(self, order_currency: str, payment_currency: str):
        return {
            "data": {
                "bids": [{"price": "999000", "quantity": "1"}],
                "asks": [{"price": "1001000", "quantity": "1"}],
            }
        }

    def get_balance(self, order_currency: str, payment_currency: str):
        return {
            "data": {
                "available_btc": "0.8",
                "available_krw": "5000000",
            }
        }

    def place_order(
        self,
        order_type: str,
        order_currency: str,
        payment_currency: str,
        units: str,
        price: str,
    ):
        self.orders.append({
            "type": order_type,
            "order_currency": order_currency,
            "payment_currency": payment_currency,
            "units": units,
            "price": price,
        })
        return {"status": "0000", "data": {"order_id": "123"}}


class DummyLLM(LLMClient):
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts = []

    def generate(self, prompt: str, **kwargs) -> str:  # type: ignore[override]
        self.prompts.append(prompt)
        return self.response


def test_engine_dry_run_skips_order():
    api = DummyAPI()
    config = StrategyConfig(
        api=APIConfig(api_key="key", api_secret="secret"),
        dry_run=True,
        risk=RiskConfig(max_trade_value=10_000_000, max_position_size=1.0),
    )
    llm_client = DummyLLM('{"action": "BUY", "confidence": 0.9, "amount": 0.5}')
    decision_maker = LLMDecisionMaker(llm_client)
    engine = TradingEngine(api, decision_maker, config)

    result = engine.run_once()

    assert result.action is Action.BUY
    assert api.orders == []  # dry run should not call place_order
    assert len(engine.history) == 1
    assert llm_client.prompts  # prompt should be generated


def test_engine_places_order_when_live():
    api = DummyAPI()
    config = StrategyConfig(
        api=APIConfig(api_key="key", api_secret="secret"),
        dry_run=False,
        risk=RiskConfig(max_trade_value=10_000_000, max_position_size=0.4),
    )
    llm_client = DummyLLM('{"action": "SELL", "confidence": 0.9, "amount": 0.5, "target_price": 1000000}')
    decision_maker = LLMDecisionMaker(llm_client)
    engine = TradingEngine(api, decision_maker, config)

    result = engine.run_once()

    assert result.action is Action.SELL
    assert api.orders  # order placed
    order = api.orders[0]
    assert order["type"] == "ask"
    # Amount limited by risk manager to max_position_size 0.4
    assert float(order["units"]) == 0.4
    assert result.target_price == 1000000
