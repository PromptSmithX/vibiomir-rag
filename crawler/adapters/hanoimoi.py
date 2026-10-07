from __future__ import annotations

_HANOIMOI_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}


class HanoimoiRequestAdapter:
    """Strategy adapter for hanoimoi.vn: provides browser headers to bypass HTTP 403."""

    respect_robots: bool = True

    def adapt_request(self, url: str) -> tuple[str, dict[str, str] | None]:
        return url, _HANOIMOI_HEADERS

    def is_bot_challenge(self, final_url: str | None, status: int | None = None) -> bool:
        return status == 403

    def can_recover_task(
        self,
        status: str,
        http_status: int | None,
        bytes_downloaded: int | None,
        error_type: str | None,
    ) -> bool:
        return status == "FAILED" and (http_status == 403 or error_type == "HTTP_403")
