# live_trading/kalshi_auth.py
from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


@dataclass
class KalshiAuth:
    api_key_id: str
    private_key_path: str

    def _load_private_key(self):
        with open(self.private_key_path, "rb") as f:
            key_data = f.read()
        try:
            return serialization.load_pem_private_key(key_data, password=None)
        except ValueError:
            return serialization.load_der_private_key(key_data, password=None)

    def sign(self, method: str, path: str, timestamp_ms: str) -> str:
        # Docs: signature over timestamp + method + path :contentReference[oaicite:3]{index=3}
        msg = f"{timestamp_ms}{method.upper()}{path}".encode("utf-8")
        priv = self._load_private_key()
        sig = priv.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("utf-8")

    def headers(self, method: str, full_path: str) -> dict:
        # Docs: timestamp is milliseconds :contentReference[oaicite:4]{index=4}
        ts_ms = str(int(time.time() * 1000))
        sig = self.sign(method, full_path, ts_ms)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        }
