from __future__ import annotations

import hashlib
from urllib.parse import urlsplit, urlunsplit


def normalize_fetch_url(url: str) -> str:
    value = url.strip()
    parts = urlsplit(value)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"unsupported or invalid URL: {url!r}")
    host = parts.hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = parts.port
    default_port = (parts.scheme.lower() == "http" and port == 80) or (
        parts.scheme.lower() == "https" and port == 443
    )
    netloc = host if port is None or default_port else f"{host}:{port}"
    if parts.username or parts.password:
        raise ValueError("URLs containing credentials are not supported")
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), netloc, path, parts.query, ""))


def domain_from_url(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def origin_from_url(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = parts.port
    default_port = (parts.scheme == "http" and port == 80) or (
        parts.scheme == "https" and port == 443
    )
    netloc = host if port is None or default_port else f"{host}:{port}"
    return urlunsplit((parts.scheme, netloc, "", "", ""))


def fetch_key(url: str) -> str:
    return hashlib.sha256(normalize_fetch_url(url).encode("utf-8")).hexdigest()


def is_pdf_like(url: str) -> bool:
    return urlsplit(url).path.lower().endswith(".pdf")

