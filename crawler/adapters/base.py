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

    @property
    def respect_robots(self) -> bool:
        """Whether to enforce robots.txt. Defaults to True."""
        ...

    def can_recover_task(
        self,
        status: str,
        http_status: int | None,
        bytes_downloaded: int | None,
        error_type: str | None,
    ) -> bool:
        """Check whether a previously failed/empty task should be recovered for retry."""
        ...
