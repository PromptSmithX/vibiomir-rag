from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from crawler.cleaner import is_bot_challenge_url

_MOBILE_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


class Ask39RequestAdapter:
    """Strategy adapter for ask.39.net: routes to mobile CDN mirror and detects verification."""

    respect_robots: bool = True

    def adapt_request(self, url: str) -> tuple[str, dict[str, str] | None]:
        try:
            parsed = urlsplit(url)
        except ValueError:
            return url, None
        host = (parsed.hostname or "").lower()
        if host in {"ask.39.net", "wapask.39.net"} and parsed.path.startswith("/question/"):
            mobile_url = urlunsplit(("https", "wapask.39.net", parsed.path, parsed.query, ""))
            return mobile_url, _MOBILE_HEADERS
        return url, None

    def is_bot_challenge(self, final_url: str | None, status: int | None = None) -> bool:
        return is_bot_challenge_url(final_url)

    def can_recover_task(
        self,
        status: str,
        http_status: int | None,
        bytes_downloaded: int | None,
        error_type: str | None,
    ) -> bool:
        return status == "EMPTY_CONTENT" and bytes_downloaded == 2287
