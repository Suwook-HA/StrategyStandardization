"""Tests for parsing LLM generated trade decisions."""

from __future__ import annotations

import pytest

from bithumb_llm_trader.decision import Action, DecisionParseError, DecisionParser


def test_parse_json_in_code_block():
    parser = DecisionParser()
    response = """
    Trading analysis...
    ```json
    {
        "action": "buy",
        "confidence": 0.82,
        "amount": 0.0125,
        "target_price": 91500000,
        "reasoning": "Bullish momentum with strong support",
        "stop_loss": 90000000,
        "take_profit": 94000000
    }
    ```
    """
    decision = parser.parse(response)
    assert decision.action is Action.BUY
    assert decision.confidence == pytest.approx(0.82)
    assert decision.amount == pytest.approx(0.0125)
    assert decision.target_price == pytest.approx(91500000)
    assert decision.stop_loss == pytest.approx(90000000)
    assert decision.take_profit == pytest.approx(94000000)


def test_parse_invalid_json():
    parser = DecisionParser()
    with pytest.raises(DecisionParseError):
        parser.parse("no json here")


def test_reject_invalid_action():
    parser = DecisionParser()
    text = '{"action": "long", "confidence": 0.9, "amount": 1}'
    with pytest.raises(DecisionParseError):
        parser.parse(text)
