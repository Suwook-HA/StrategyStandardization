"""Prompt templates used to instruct the LLM."""

from __future__ import annotations

import textwrap
from typing import Any, Dict, Iterable, Optional

from .config import StrategyConfig


def _format_orderbook(orderbook: Dict[str, Any], depth: int = 5) -> str:
    bids = orderbook.get("bids") or orderbook.get("bid") or []
    asks = orderbook.get("asks") or orderbook.get("ask") or []
    bid_lines = [f"{idx+1}. {b['price']} ({b.get('quantity') or b.get('amount')})" for idx, b in enumerate(bids[:depth])]
    ask_lines = [f"{idx+1}. {a['price']} ({a.get('quantity') or a.get('amount')})" for idx, a in enumerate(asks[:depth])]
    return "\n".join(
        ["Top Asks:"] + ask_lines + ["", "Top Bids:"] + bid_lines
    )


def build_trading_prompt(
    market_data: Dict[str, Any],
    account_state: Dict[str, Any],
    config: StrategyConfig,
    history: Optional[Iterable[Dict[str, Any]]] = None,
) -> str:
    ticker = market_data.get("ticker", {})
    orderbook = market_data.get("orderbook", {})
    last_actions = list(history or [])[-3:]
    history_block = "\n".join(
        f"- {item['timestamp']}: {item['action']} ({item['details']})" for item in last_actions
    ) or "- No previous trades"
    template = config.prompt_template or textwrap.dedent(
        """
        You are an autonomous crypto trading strategist operating on the Bithumb exchange.
        Analyse the provided market data and account balances to decide whether to BUY, SELL or HOLD
        the pair {order_currency}/{payment_currency}. Your answer must be a single JSON object with the following fields:
        - action: one of BUY, SELL, HOLD
        - confidence: value between 0 and 1
        - amount: number of {order_currency} units to trade
        - target_price: desired execution price in {payment_currency} (optional)
        - reasoning: concise explanation for the decision
        - stop_loss: optional stop loss price in {payment_currency}
        - take_profit: optional take profit price in {payment_currency}

        Ensure the JSON is valid and does not include additional commentary.
        """
    ).strip()

    order_currency = config.trading_pair.order_currency
    payment_currency = config.trading_pair.payment_currency
    prompt = template.format(order_currency=order_currency, payment_currency=payment_currency)

    price_info = textwrap.dedent(
        f"""
        Market snapshot:
        - Closing price: {ticker.get('closing_price')}
        - 24h change (%): {ticker.get('fluctate_rate_24H')}
        - 24h volume: {ticker.get('acc_trade_value_24H')}

        Orderbook (top levels):
        {_format_orderbook(orderbook)}
        """
    ).strip()

    balance_info = textwrap.dedent(
        f"""
        Account:
        - Available {order_currency}: {account_state.get('balance_order_currency')}
        - Available {payment_currency}: {account_state.get('balance_payment_currency')}
        - Position value limit: {config.risk.max_trade_value} {payment_currency}
        - Position size limit: {config.risk.max_position_size} {order_currency}
        """
    ).strip()

    risk_info = textwrap.dedent(
        f"""
        Risk constraints:
        - Minimum confidence to trade: {config.risk.min_confidence}
        - Stop loss tolerance: {config.risk.stop_loss_pct*100:.2f}%
        - Take profit target: {config.risk.take_profit_pct*100:.2f}%
        Trade history:
        {history_block}
        """
    ).strip()

    sections = [prompt, price_info, balance_info, risk_info]
    return "\n\n".join(sections)


__all__ = ["build_trading_prompt"]
