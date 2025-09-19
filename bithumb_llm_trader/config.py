"""Configuration helpers for the LLM-driven Bithumb trading system."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for older versions
    tomllib = None  # type: ignore[assignment]


@dataclass(slots=True)
class APIConfig:
    """Holds credentials and networking parameters for the Bithumb API."""

    api_key: str
    api_secret: str
    base_url: str = "https://api.bithumb.com"
    timeout: float = 10.0


@dataclass(slots=True)
class TradingPairConfig:
    """Represents the trading pair configuration for the strategy."""

    order_currency: str = "BTC"
    payment_currency: str = "KRW"


@dataclass(slots=True)
class RiskConfig:
    """Parameters that constrain order sizing and manage downside risk."""

    max_trade_value: float = 1_000_000.0  # in payment currency (KRW)
    max_position_size: float = 0.2  # maximum units of the asset to trade in one order
    stop_loss_pct: float = 0.02  # 2% below the entry price
    take_profit_pct: float = 0.03  # 3% above the entry price
    min_confidence: float = 0.55  # minimum LLM confidence required to trade


@dataclass(slots=True)
class LLMConfig:
    """Configuration for the language model that generates trade decisions."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_output_tokens: int = 512


@dataclass(slots=True)
class StrategyConfig:
    """Top-level configuration object consumed by the trading engine."""

    api: APIConfig
    trading_pair: TradingPairConfig = field(default_factory=TradingPairConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    dry_run: bool = True
    prompt_template: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyConfig":
        """Construct :class:`StrategyConfig` from a plain dictionary."""

        api_cfg = APIConfig(**data["api"])
        trading_cfg = TradingPairConfig(**data.get("trading_pair", {}))
        risk_cfg = RiskConfig(**data.get("risk", {}))
        llm_cfg = LLMConfig(**data.get("llm", {}))
        dry_run = data.get("dry_run", True)
        prompt_template = data.get("prompt_template")
        return cls(
            api=api_cfg,
            trading_pair=trading_cfg,
            risk=risk_cfg,
            llm=llm_cfg,
            dry_run=dry_run,
            prompt_template=prompt_template,
        )


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyYAML is required to load YAML configuration files. Install it via 'pip install pyyaml'."
        ) from exc
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_toml(path: Path) -> Dict[str, Any]:
    if tomllib is None:  # pragma: no cover - Python < 3.11 fallback
        raise RuntimeError("TOML configuration files require Python 3.11 or newer.")
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_config(path: Union[str, os.PathLike[str]]) -> StrategyConfig:
    """Load configuration data from JSON, YAML, or TOML files."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".json":
        data = _load_json(file_path)
    elif suffix in {".yml", ".yaml"}:
        data = _load_yaml(file_path)
    elif suffix == ".toml":
        data = _load_toml(file_path)
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported configuration format: {suffix}")

    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a dictionary at the top level.")

    return StrategyConfig.from_dict(data)
