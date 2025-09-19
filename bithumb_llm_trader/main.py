"""Command line entry point for running the trading engine."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .api_client import BithumbAPI
from .config import StrategyConfig, load_config
from .engine import TradingEngine
from .llm import LLMDecisionMaker, OpenAIChatClient


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def summarize_decision(decision: Any) -> str:
    return json.dumps(
        {
            "action": decision.action.value,
            "amount": decision.amount,
            "price": decision.target_price,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
        },
        ensure_ascii=False,
        indent=2,
    )


def run_once(config: StrategyConfig, openai_api_key: str | None) -> None:
    api = BithumbAPI(
        api_key=config.api.api_key,
        api_secret=config.api.api_secret,
        base_url=config.api.base_url,
        timeout=config.api.timeout,
    )
    llm_client = OpenAIChatClient(api_key=openai_api_key)
    decision_maker = LLMDecisionMaker(llm_client, llm_config=config.llm)
    engine = TradingEngine(api, decision_maker, config)
    decision = engine.run_once()
    print(summarize_decision(decision))


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-driven automated trader for Bithumb")
    parser.add_argument("config", type=Path, help="Path to strategy configuration file")
    parser.add_argument("--openai-api-key", dest="openai_api_key", help="Override OpenAI API key")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    configure_logging(args.verbose)
    config = load_config(args.config)
    run_once(config, args.openai_api_key)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
