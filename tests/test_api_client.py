"""Tests for the low level Bithumb API client."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

from bithumb_llm_trader.api_client import BithumbAPI


class RecordingTransport:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = []

    def request(self, method, url, headers, data, timeout):  # type: ignore[override]
        self.calls.append({
            "method": method,
            "url": url,
            "headers": headers,
            "data": data,
            "timeout": timeout,
        })
        return 200, json.dumps(self.payload).encode()


def test_signature_matches_reference():
    api = BithumbAPI(api_key="key", api_secret="secret")
    signature = api._sign("/info/balance", "currency=BTC", "1234")
    expected = base64.b64encode(
        hmac.new(b"secret", b"/info/balance\x00currency=BTC\x001234", hashlib.sha512).digest()
    ).decode()
    assert signature == expected


def test_private_request_sends_auth_headers():
    transport = RecordingTransport({"status": "0000", "data": {"available_btc": "1"}})
    api = BithumbAPI(api_key="key", api_secret="secret", transport=transport)

    response = api.get_balance("BTC", "KRW")

    call = transport.calls[0]
    assert call["headers"]["Api-Key"] == "key"
    assert "Api-Sign" in call["headers"]
    assert response["data"]["available_btc"] == "1"
