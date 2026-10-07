from __future__ import annotations

_BAIDU_HEALTH_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


class BaiduHealthRequestAdapter:
    """Strategy adapter for Baidu Health dictionary (/bh/dict/): bypasses robots.txt."""

    respect_robots: bool = False

    def adapt_request(self, url: str) -> tuple[str, dict[str, str] | None]:
        return url, _BAIDU_HEALTH_HEADERS

    def is_bot_challenge(self, final_url: str | None, status: int | None = None) -> bool:
        return False

    def can_recover_task(
        self,
        status: str,
        http_status: int | None,
        bytes_downloaded: int | None,
        error_type: str | None,
    ) -> bool:
        return status == "ROBOTS_DENIED"
