"""Utility helpers shared across trading components."""

from __future__ import annotations

from typing import Any, Dict


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert ``value`` to ``float`` while swallowing type errors.

    Parameters
    ----------
    value:
        Arbitrary numeric-like value.
    default:
        Value returned when conversion fails.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_units(value: float) -> str:
    """Format a float amount suitable for API submission."""

    formatted = f"{value:.8f}".rstrip("0").rstrip(".")
    return formatted or "0"


def extract_balance(data: Dict[str, Any], currency: str) -> float:
    """Extract balance information for ``currency`` from an API payload."""

    keys = [
        f"available_{currency.lower()}",
        f"available_{currency.upper()}",
        f"available_{currency}",
        currency.lower(),
        currency.upper(),
        currency,
    ]
    for key in keys:
        if key in data:
            return safe_float(data[key])
    return 0.0


__all__ = ["safe_float", "format_units", "extract_balance"]
