from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DomainRequestAdapter(Protocol):
    """Protocol for domain-specific request adaptation, header customization, and bot detection."""

    def adapt_request(self, url: str) -> tuple[str, dict[str, str] | None]:
        """Optionally rewrite the request URL and return custom request headers."""
        ...

    def is_bot_challenge(self, final_url: str | None, status: int | None = None) -> bool:
        """Check whether the response URL/status represents an anti-bot challenge."""
        ...
