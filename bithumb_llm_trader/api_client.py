"""HTTP client for interacting with the Bithumb REST API."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple


class BithumbAPIError(RuntimeError):
    """Exception raised when the Bithumb API reports an error."""

    def __init__(self, message: str, payload: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.payload = payload or {}


class Transport(Protocol):
    """Protocol for pluggable HTTP transports."""

    def request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        data: Optional[bytes],
        timeout: float,
    ) -> Tuple[int, bytes]:
        """Perform an HTTP request and return a status code with a body."""


class UrllibTransport:
    """Default transport implementation built on top of :mod:`urllib`."""

    def request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        data: Optional[bytes],
        timeout: float,
    ) -> Tuple[int, bytes]:
        request = urllib.request.Request(url=url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.getcode(), response.read()
        except urllib.error.HTTPError as exc:  # pragma: no cover - network failure path
            return exc.code, exc.read()


@dataclass(slots=True)
class BithumbAPI:
    """Thin wrapper around Bithumb's REST endpoints."""

    api_key: str
    api_secret: str
    base_url: str = "https://api.bithumb.com"
    timeout: float = 10.0
    transport: Transport = UrllibTransport()

    def _nonce(self) -> str:
        return str(int(time.time() * 1000))

    def _sign(self, endpoint: str, body: str, nonce: str) -> str:
        message = endpoint.encode() + b"\0" + body.encode() + b"\0" + nonce.encode()
        signature = hmac.new(self.api_secret.encode(), message, hashlib.sha512).digest()
        return base64.b64encode(signature).decode()

    def _public_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = urllib.parse.urljoin(self.base_url, endpoint)
        if params:
            query = urllib.parse.urlencode(params)
            url = f"{url}?{query}"
        status, body = self.transport.request("GET", url, {}, None, self.timeout)
        return self._parse_response(status, body)

    def _private_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        body = urllib.parse.urlencode(params)
        nonce = self._nonce()
        signature = self._sign(endpoint, body, nonce)
        headers = {
            "Api-Key": self.api_key,
            "Api-Sign": signature,
            "Api-Nonce": nonce,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        url = urllib.parse.urljoin(self.base_url, endpoint)
        status, response = self.transport.request(
            "POST", url, headers, body.encode(), self.timeout
        )
        return self._parse_response(status, response)

    def _parse_response(self, status: int, body: bytes) -> Dict[str, Any]:
        if status >= 400:
            raise BithumbAPIError(f"HTTP error from Bithumb: {status}")
        text = body.decode() if body else "{}"
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
            raise BithumbAPIError("Invalid JSON response", {"body": text}) from exc
        status_code = payload.get("status")
        if status_code not in (None, "0000"):
            raise BithumbAPIError(payload.get("message", "Unknown error"), payload)
        return payload

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------
    def get_ticker(self, order_currency: str, payment_currency: str) -> Dict[str, Any]:
        endpoint = f"/public/ticker/{order_currency}_{payment_currency}"
        return self._public_request(endpoint)

    def get_orderbook(self, order_currency: str, payment_currency: str) -> Dict[str, Any]:
        endpoint = f"/public/orderbook/{order_currency}_{payment_currency}"
        return self._public_request(endpoint)

    def get_recent_transactions(
        self, order_currency: str, payment_currency: str
    ) -> Dict[str, Any]:
        endpoint = f"/public/recent_transactions/{order_currency}_{payment_currency}"
        return self._public_request(endpoint)

    # ------------------------------------------------------------------
    # Private endpoints
    # ------------------------------------------------------------------
    def get_balance(self, order_currency: str, payment_currency: str) -> Dict[str, Any]:
        params = {
            "currency": order_currency,
            "payment_currency": payment_currency,
        }
        return self._private_request("/info/balance", params)

    def place_order(
        self,
        order_type: str,
        order_currency: str,
        payment_currency: str,
        units: str,
        price: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "order_currency": order_currency,
            "payment_currency": payment_currency,
            "units": units,
            "type": order_type,
        }
        if price is not None:
            params["price"] = price
        return self._private_request("/trade/place", params)

    def cancel_order(
        self,
        order_id: str,
        order_currency: str,
        payment_currency: str,
        order_type: str,
    ) -> Dict[str, Any]:
        params = {
            "type": order_type,
            "order_id": order_id,
            "order_currency": order_currency,
            "payment_currency": payment_currency,
        }
        return self._private_request("/trade/cancel", params)

    def get_open_orders(self, order_currency: str, payment_currency: str) -> Dict[str, Any]:
        params = {
            "order_currency": order_currency,
            "payment_currency": payment_currency,
        }
        return self._private_request("/info/orders", params)


__all__ = ["BithumbAPI", "BithumbAPIError", "Transport", "UrllibTransport"]
