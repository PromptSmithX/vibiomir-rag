from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from urllib import robotparser

from crawler.utils.urls import origin_from_url


class RobotsUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class RobotsRule:
    parser: robotparser.RobotFileParser | None
    expires_at: float
    deny_all: bool = False
    crawl_delay: float | None = None


class RobotsCache:
    def __init__(
        self,
        session,
        user_agent: str,
        ttl_seconds: int = 86_400,
        timeout_seconds: float = 10,
    ) -> None:
        self.session = session
        self.user_agent = user_agent
        self.agent_token = user_agent.split("/", 1)[0]
        self.ttl_seconds = ttl_seconds
        self.timeout_seconds = timeout_seconds
        self._rules: dict[str, RobotsRule] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> tuple[bool, float | None]:
        origin = origin_from_url(url)
        rule = self._rules.get(origin)
        if rule is None or rule.expires_at <= time.monotonic():
            lock = self._locks.setdefault(origin, asyncio.Lock())
            async with lock:
                rule = self._rules.get(origin)
                if rule is None or rule.expires_at <= time.monotonic():
                    rule = await self._fetch(origin)
                    self._rules[origin] = rule
        if rule.deny_all:
            return False, rule.crawl_delay
        if rule.parser is None:
            return True, rule.crawl_delay
        return rule.parser.can_fetch(self.agent_token, url), rule.crawl_delay

    async def _fetch(self, origin: str) -> RobotsRule:
        try:
            import aiohttp
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("aiohttp is required; run `uv sync`") from exc
        url = f"{origin}/robots.txt"
        try:
            async with self.session.get(
                url,
                allow_redirects=True,
                max_redirects=5,
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
            ) as response:
                client_error_allows = 400 <= response.status < 500 and response.status not in {
                    401,
                    403,
                }
                if response.status == 404 or client_error_allows:
                    return RobotsRule(None, time.monotonic() + self.ttl_seconds)
                if response.status in {401, 403}:
                    return RobotsRule(None, time.monotonic() + self.ttl_seconds, deny_all=True)
                if response.status >= 500:
                    raise RobotsUnavailable(f"robots.txt returned HTTP {response.status}")
                raw = await response.content.read(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise RobotsUnavailable("robots.txt exceeds 1 MiB")
                text = raw.decode(response.charset or "utf-8", errors="replace")
        except RobotsUnavailable:
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise RobotsUnavailable(f"cannot fetch robots.txt: {exc}") from exc

        parser = robotparser.RobotFileParser(url)
        parser.parse(text.splitlines())
        delay = parser.crawl_delay(self.agent_token)
        if delay is None:
            delay = parser.crawl_delay("*")
        return RobotsRule(
            parser,
            time.monotonic() + self.ttl_seconds,
            crawl_delay=float(delay) if delay is not None else None,
        )
