# live_trading/kalshi_client.py
from __future__ import annotations

from urllib.parse import urlparse
import requests
from live_trading.kalshi_auth import KalshiAuth


class KalshiClient:
    """
    base_url can be either:
      - https://demo-api.kalshi.co
      - https://demo-api.kalshi.co/trade-api/v2

    We will:
      - build request URL correctly
      - SIGN the full request path (including /trade-api/v2) as required by docs :contentReference[oaicite:5]{index=5}
    """

    def __init__(self, base_url: str, auth: KalshiAuth, timeout: int = 20):
        parsed = urlparse(base_url)
        self.scheme = parsed.scheme or "https"
        self.netloc = parsed.netloc or parsed.path  # if someone passed without scheme
        self.base_path = parsed.path.rstrip("/")    # "" or "/trade-api/v2"
        if self.base_path == "":
            # default to trade-api/v2 if not included
            self.base_path = "/trade-api/v2"
        self.auth = auth
        self.timeout = timeout

    def _full_url(self, path: str) -> str:
        return f"{self.scheme}://{self.netloc}{self.base_path}{path}"

    def _signing_path(self, path: str) -> str:
        # must match the path component of the URL being requested
        return f"{self.base_path}{path}"

    def _req(self, method: str, path: str, params=None, json=None):
        url = self._full_url(path)
        signing_path = self._signing_path(path)
        headers = self.auth.headers(method, signing_path)

        r = requests.request(method, url, params=params, json=json, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
    
    def get_orders(self, status="resting", limit=200, cursor=None):
        params = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._req("GET", "/portfolio/orders", params=params)



    # -------- Portfolio ----------
    def get_balance(self):
        return self._req("GET", "/portfolio/balance")  # :contentReference[oaicite:6]{index=6}

    def get_positions(self, limit: int = 200, cursor: str | None = None):
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._req("GET", "/portfolio/positions", params=params)  # :contentReference[oaicite:7]{index=7}

    # -------- Markets ----------
    def get_markets(self, limit: int = 100, cursor: str | None = None, status: str | None = None):
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        return self._req("GET", "/markets", params=params)

    def get_market(self, market_ticker: str):
        return self._req("GET", f"/markets/{market_ticker}")

    def get_orderbook(self, market_ticker: str):
        return self._req("GET", f"/markets/{market_ticker}/orderbook")  # :contentReference[oaicite:8]{index=8}
    
        def get_orders(self, status: str = "resting", limit: int = 200, cursor: str | None = None):
            params = {"limit": limit, "status": status}
        if cursor:
            params["cursor"] = cursor
        return self._req("GET", "/portfolio/orders", params=params)


    # -------- Orders ----------
    def create_order_yes(self, market_ticker: str, action: str, count: int, yes_price: int):
        body = {
            "ticker": market_ticker,
            "action": action,     # "buy" or "sell"
            "side": "yes",
            "count": int(count),
            "type": "limit",
            "yes_price": int(yes_price),  # cents 1..99
        }
        return self._req("POST", "/portfolio/orders", json=body)  # :contentReference[oaicite:9]{index=9}
    

        

