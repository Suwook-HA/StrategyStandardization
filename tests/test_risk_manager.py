"""Unit tests for the risk manager."""

from __future__ import annotations

import pytest

from bithumb_llm_trader.config import RiskConfig
from bithumb_llm_trader.decision import Action, TradeDecision
from bithumb_llm_trader.risk import RiskManager


@pytest.fixture
def market_state():
    return {"ticker": {"closing_price": "1000000"}}


@pytest.fixture
def account_state():
    return {"balance_order_currency": 0.5, "balance_payment_currency": 3000000}


def test_buy_adjusted_by_cash_limits(market_state, account_state):
    manager = RiskManager(RiskConfig(max_trade_value=1_000_000, max_position_size=1))
    decision = TradeDecision(
        action=Action.BUY,
        confidence=0.9,
        amount=5,
        target_price=1_000_000,
        reasoning="Test"
    )
    adjusted = manager.apply(decision, market_state, account_state)
    assert adjusted.action is Action.BUY
    # Limited by max_trade_value (1M / price 1M -> 1 unit)
    assert adjusted.amount == pytest.approx(1.0)
    assert adjusted.stop_loss == pytest.approx(980000.0)
    assert adjusted.take_profit == pytest.approx(1030000.0)


def test_sell_rejected_without_inventory(market_state):
    manager = RiskManager(RiskConfig())
    account_state = {"balance_order_currency": 0.0, "balance_payment_currency": 0.0}
    decision = TradeDecision(action=Action.SELL, confidence=0.8, amount=1.0)
    adjusted = manager.apply(decision, market_state, account_state)
    assert adjusted.action is Action.HOLD
    assert "No inventory" in adjusted.reasoning


def test_low_confidence_rejected(market_state, account_state):
    manager = RiskManager(RiskConfig(min_confidence=0.7))
    decision = TradeDecision(action=Action.BUY, confidence=0.5, amount=0.1)
    adjusted = manager.apply(decision, market_state, account_state)
    assert adjusted.action is Action.HOLD
    assert "confidence" in adjusted.reasoning
