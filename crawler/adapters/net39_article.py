from __future__ import annotations

import re
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


class Net39ArticleAdapter:
    """Strategy adapter for 39.net article subdomains (woman.39.net, pf.39.net).

    Routes requests to official mobile CDN mirror (m.39.net) with mobile headers
    to bypass OpenResty desktop WAF captcha challenge and avoid redirection loops.
    """

    respect_robots: bool = True

    def adapt_request(self, url: str) -> tuple[str, dict[str, str] | None]:
        try:
            parsed = urlsplit(url)
        except ValueError:
            return url, _MOBILE_HEADERS

        host = (parsed.hostname or "").lower()
        if host.endswith(".39.net"):
            subdomain = host.split(".")[0]
            if subdomain not in {"ask", "wapask", "m", "image"}:
                match = re.match(r"^/a/(?:\d+/)?([a-zA-Z0-9_]+)\.html$", parsed.path)
                if match:
                    article_id = match.group(1)
                    mobile_url = urlunsplit(
                        ("https", "m.39.net", f"/{subdomain}/a_{article_id}.html", "", "")
                    )
                    return mobile_url, _MOBILE_HEADERS
        return url, _MOBILE_HEADERS

    def is_bot_challenge(self, final_url: str | None, status: int | None = None) -> bool:
        return is_bot_challenge_url(final_url)

    def can_recover_task(
        self,
        status: str,
        http_status: int | None,
        bytes_downloaded: int | None,
        error_type: str | None,
    ) -> bool:
        if status == "NEEDS_JS" and error_type == "BOT_CHALLENGE":
            return True
        if status == "EMPTY_CONTENT" and bytes_downloaded == 2287:
            return True
        return False
