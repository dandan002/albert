"""Kalshi API-key request signing: Ed25519 (Kalshi's default for new keys) or RSA-PSS SHA-256.

The signed message is ``<timestamp_ms><METHOD><path>``, where ``path`` excludes the host and the
query string (for the WebSocket handshake it is ``/trade-api/ws/v2``). The key comes only from
``.env`` (``KALSHI_API_KEY_ID``, ``KALSHI_PRIVATE_KEY``; demo: ``KALSHI_DEMO_*``). It is never
logged or stored.
"""

import base64
import time

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from albert.config import secret


PrivateKey = rsa.RSAPrivateKey | ed25519.Ed25519PrivateKey


def load_private_key(text: str) -> PrivateKey:
    """Load an RSA or Ed25519 private key: PEM (also with literal ``\\n`` escapes), bare base64
    DER (PKCS#8, or PKCS#1 for RSA), or a bare 32-byte Ed25519 seed in base64 or hex."""
    text = text.strip().replace("\\n", "\n")
    if text.startswith("-----"):
        key = serialization.load_pem_private_key(text.encode(), password=None)
    else:
        body = "".join(text.split())
        if len(body) == 64 and all(c in "0123456789abcdefABCDEF" for c in body):
            raw = bytes.fromhex(body)
        else:
            raw = base64.b64decode(body + "=" * (-len(body) % 4), validate=True)
        if len(raw) == 32:
            key = ed25519.Ed25519PrivateKey.from_private_bytes(raw)
        else:
            try:
                key = serialization.load_der_private_key(raw, password=None)
            except ValueError:
                lines = "\n".join(body[i:i + 64] for i in range(0, len(body), 64))
                pem = f"-----BEGIN RSA PRIVATE KEY-----\n{lines}\n-----END RSA PRIVATE KEY-----\n"
                key = serialization.load_pem_private_key(pem.encode(), password=None)
    if not isinstance(key, (rsa.RSAPrivateKey, ed25519.Ed25519PrivateKey)):
        raise ValueError("Kalshi private key must be RSA or Ed25519")
    return key


class KalshiSigner:
    def __init__(self, key_id: str, key: PrivateKey) -> None:
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
        msg = f"{ts}{method.upper()}{path.split('?', 1)[0]}".encode()
        if isinstance(self._key, ed25519.Ed25519PrivateKey):
            sig = self._key.sign(msg)
        else:
            sig = self._key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                                  salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }
