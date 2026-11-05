"""Kalshi API-key request signing (RSA-PSS, SHA-256).

The signed message is ``<timestamp_ms><METHOD><path>``, where ``path`` excludes the host and the
query string (for the WebSocket handshake it is ``/trade-api/ws/v2``). The key comes only from
``.env`` (``KALSHI_API_KEY_ID``, ``KALSHI_PRIVATE_KEY``). It is never logged or stored.
"""

import base64
import time

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from albert.config import secret


def load_private_key(text: str) -> rsa.RSAPrivateKey:
    """Load a PEM key, a PEM with literal ``\\n`` escapes, or a bare base64 PKCS#1 body."""
    text = text.strip().replace("\\n", "\n")
    if not text.startswith("-----"):
        body = "".join(text.split())
        base64.b64decode(body, validate=True)  # fail early on a mangled value
        lines = "\n".join(body[i:i + 64] for i in range(0, len(body), 64))
        text = f"-----BEGIN RSA PRIVATE KEY-----\n{lines}\n-----END RSA PRIVATE KEY-----\n"
    key = serialization.load_pem_private_key(text.encode(), password=None)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise ValueError("KALSHI_PRIVATE_KEY is not an RSA key")
    return key


class KalshiSigner:
    def __init__(self, key_id: str, key: rsa.RSAPrivateKey) -> None:
        self.key_id = key_id
        self._key = key

    @classmethod
    def from_env(cls) -> "KalshiSigner":
        key_id, pem = secret("KALSHI_API_KEY_ID"), secret("KALSHI_PRIVATE_KEY")
        if not key_id or not pem:
            raise RuntimeError("KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY must be set in .env")
        return cls(key_id, load_private_key(pem))

    def headers(self, method: str, path: str, *, now_ms: int | None = None) -> dict[str, str]:
        ts = str(now_ms if now_ms is not None else int(time.time() * 1000))
        sig = self._key.sign(
            f"{ts}{method.upper()}{path.split('?', 1)[0]}".encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }
