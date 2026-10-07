from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import time
from collections import defaultdict
from urllib.parse import urljoin

from crawler.adapters import get_adapter
from crawler.config import CrawlerRuntimeConfig
from crawler.models import CrawlStatus, FetchOutcome, FetchTarget
from crawler.robots import RobotsCache, RobotsUnavailable
from crawler.utils.urls import domain_from_url, normalize_fetch_url

LOGGER = logging.getLogger(__name__)
REDIRECT_STATUSES = {301, 302, 303, 307, 308}
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
HTML_TYPES = {"text/html", "application/xhtml+xml", "text/plain"}
HTML_DOCUMENT_TYPES = {"text/html", "application/xhtml+xml"}


class ByteBudget:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.available = capacity
        self.condition = asyncio.Condition()

    async def acquire(self, amount: int) -> int:
        reserved = min(max(1, amount), self.capacity)
        async with self.condition:
            await self.condition.wait_for(lambda: self.available >= reserved)
            self.available -= reserved
        return reserved

    async def release(self, amount: int) -> None:
        async with self.condition:
            self.available = min(self.capacity, self.available + amount)
            self.condition.notify_all()


class DomainLimiter:
    def __init__(self, concurrency: int, overrides: dict[str, int] | None = None) -> None:
        self.concurrency = concurrency
        self.overrides = {domain.lower(): limit for domain, limit in (overrides or {}).items()}
        if any(limit < 1 for limit in self.overrides.values()):
            raise ValueError("domain concurrency overrides must be positive")
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._last_request: dict[str, float] = {}
        self._delay_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def semaphore(self, domain: str) -> asyncio.Semaphore:
        normalized = domain.lower()
        if normalized not in self._semaphores:
            limit = self.overrides.get(normalized, self.concurrency)
            self._semaphores[normalized] = asyncio.Semaphore(limit)
        return self._semaphores[normalized]

    async def respect_delay(self, domain: str, delay: float | None) -> None:
        if not delay:
            return
        async with self._delay_locks[domain]:
            remaining = delay - (time.monotonic() - self._last_request.get(domain, 0.0))
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last_request[domain] = time.monotonic()


def _content_type(header: str | None, prefix: bytes) -> str:
    declared = (header or "").split(";", 1)[0].strip().lower()
    if prefix.startswith(b"%PDF-"):
        return "application/pdf"
    stripped = prefix.lstrip().lower()
    if stripped.startswith((b"<!doctype html", b"<html")):
        return "text/html"
    if declared in HTML_DOCUMENT_TYPES:
        return declared
    if declared == "text/plain":
        return declared
    return declared or "application/octet-stream"


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return time.time() + max(0, int(value.strip()))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.astimezone()
            return parsed.timestamp()
        except (TypeError, ValueError, OverflowError):
            return None


class Fetcher:
    def __init__(
        self,
        config: CrawlerRuntimeConfig,
        response_budget: ByteBudget,
        domain_concurrency_overrides: dict[str, int] | None = None,
    ) -> None:
        self.config = config
        self.response_budget = response_budget
        self.session = None
        self.robots: RobotsCache | None = None
        self.limiter = DomainLimiter(config.per_domain_concurrency, domain_concurrency_overrides)
        self._domain_cooldown_until: dict[str, float] = {}

    async def __aenter__(self) -> Fetcher:
        try:
            import aiohttp
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("aiohttp is required; run `uv sync`") from exc
        timeout = aiohttp.ClientTimeout(
            total=self.config.total_timeout_seconds,
            connect=self.config.connect_timeout_seconds,
        )
        connector = aiohttp.TCPConnector(
            limit=self.config.global_concurrency,
            limit_per_host=self.config.per_domain_concurrency,
            ttl_dns_cache=300,
        )
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": self.config.user_agent, "Accept-Encoding": "gzip, deflate"},
            auto_decompress=True,
        )
        self.robots = RobotsCache(
            self.session,
            self.config.user_agent,
            self.config.robots_cache_seconds,
            self.config.robots_timeout_seconds,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self.session:
            await self.session.close()

    def backoff_time(self, attempts: int) -> float:
        base = self.config.retry_base_seconds * (2 ** max(0, attempts - 1))
        return time.time() + base + random.uniform(0, base * 0.25)

    async def fetch(self, target: FetchTarget) -> tuple[FetchOutcome, int]:
        active_seconds = [0.0]
        outcome, reserved = await self._fetch_impl(target, active_seconds)
        outcome.fetch_seconds = active_seconds[0]
        return outcome, reserved

    async def _fetch_impl(
        self, target: FetchTarget, active_seconds: list[float]
    ) -> tuple[FetchOutcome, int]:
        try:
            import aiohttp
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("aiohttp is required; run `uv sync`") from exc
        if self.session is None or self.robots is None:
            raise RuntimeError("Fetcher must be used as an async context manager")

        current_url = target.request_url
        domain = domain_from_url(current_url)
        adapter = get_adapter(domain)
        request_headers = None
        if adapter is not None:
            current_url, request_headers = adapter.adapt_request(current_url)
            domain = domain_from_url(current_url)

        reserved = 0
        try:
            for _redirect in range(self.config.max_redirects + 1):
                try:
                    current_url = normalize_fetch_url(current_url)
                except ValueError as exc:
                    return (
                        FetchOutcome(
                            target,
                            CrawlStatus.FAILED,
                            final_url=current_url,
                            error_type="INVALID_URL",
                            error=str(exc),
                        ),
                        reserved,
                    )
                if adapter is not None and adapter.is_bot_challenge(current_url, None):
                    retry_at = self.backoff_time(target.attempts)
                    domain = domain_from_url(current_url)
                    self._domain_cooldown_until[domain] = max(
                        self._domain_cooldown_until.get(domain, 0.0), retry_at
                    )
                    return (
                        FetchOutcome(
                            target,
                            CrawlStatus.FAILED,
                            final_url=current_url,
                            retryable=True,
                            retry_at=retry_at,
                            error_type="BOT_CHALLENGE",
                            error="Bot verification challenge at final URL",
                        ),
                        reserved,
                    )
                domain = domain_from_url(current_url)
                cooldown = self._domain_cooldown_until.get(domain, 0.0) - time.time()
                if cooldown > 0:
                    await asyncio.sleep(cooldown)
                robots_started = time.monotonic()
                try:
                    allowed, crawl_delay = await self.robots.allowed(current_url)
                except RobotsUnavailable as exc:
                    return (
                        FetchOutcome(
                            target,
                            CrawlStatus.FAILED,
                            final_url=current_url,
                            retryable=True,
                            retry_at=self.backoff_time(target.attempts),
                            error_type="ROBOTS_UNAVAILABLE",
                            error=str(exc),
                        ),
                        reserved,
                    )
                finally:
                    active_seconds[0] += time.monotonic() - robots_started
                if not allowed:
                    return (
                        FetchOutcome(
                            target,
                            CrawlStatus.ROBOTS_DENIED,
                            final_url=current_url,
                            error_type="ROBOTS_DENIED",
                            error="robots.txt disallows this URL",
                        ),
                        reserved,
                    )

                await self.limiter.respect_delay(domain, crawl_delay)
                async with self.limiter.semaphore(domain):
                    request_started = time.monotonic()
                    try:
                        response = await self.session.get(
                            current_url, headers=request_headers, allow_redirects=False
                        )
                    except (aiohttp.ClientError, TimeoutError) as exc:
                        return (
                            FetchOutcome(
                                target,
                                CrawlStatus.FAILED,
                                final_url=current_url,
                                retryable=True,
                                retry_at=self.backoff_time(target.attempts),
                                error_type="NETWORK_ERROR",
                                error=f"{type(exc).__name__}: {exc}"[:2000],
                            ),
                            reserved,
                        )
                    finally:
                        active_seconds[0] += time.monotonic() - request_started
                    async with response:
                        status = response.status
                        if status in REDIRECT_STATUSES:
                            location = response.headers.get("Location")
                            if not location:
                                return (
                                    FetchOutcome(
                                        target,
                                        CrawlStatus.FAILED,
                                        final_url=current_url,
                                        http_status=status,
                                        error_type="INVALID_REDIRECT",
                                        error="redirect response has no Location header",
                                    ),
                                    reserved,
                                )
                            current_url = urljoin(current_url, location)
                            domain = domain_from_url(current_url)
                            adapter = get_adapter(domain)
                            if adapter is not None:
                                current_url, request_headers = adapter.adapt_request(current_url)
                                domain = domain_from_url(current_url)
                            else:
                                request_headers = None
                            continue
                        if status in RETRYABLE_STATUSES:
                            retry_at = _retry_after(response.headers.get("Retry-After")) or (
                                self.backoff_time(target.attempts)
                            )
                            if status == 429:
                                self._domain_cooldown_until[domain] = max(
                                    self._domain_cooldown_until.get(domain, 0.0), retry_at
                                )
                            return (
                                FetchOutcome(
                                    target,
                                    CrawlStatus.FAILED,
                                    final_url=current_url,
                                    http_status=status,
                                    retryable=True,
                                    retry_at=retry_at,
                                    error_type=f"HTTP_{status}",
                                    error=f"HTTP {status}",
                                ),
                                reserved,
                            )
                        if not 200 <= status < 300:
                            return (
                                FetchOutcome(
                                    target,
                                    CrawlStatus.FAILED,
                                    final_url=current_url,
                                    http_status=status,
                                    error_type=f"HTTP_{status}",
                                    error=f"HTTP {status}",
                                ),
                                reserved,
                            )

                        read_started = time.monotonic()
                        try:
                            prefix = await response.content.read(8192)
                        finally:
                            active_seconds[0] += time.monotonic() - read_started
                        kind = _content_type(response.headers.get("Content-Type"), prefix)
                        if kind == "application/pdf":
                            limit = self.config.max_pdf_bytes
                        elif kind in HTML_TYPES:
                            limit = self.config.max_html_bytes
                        else:
                            return (
                                FetchOutcome(
                                    target,
                                    CrawlStatus.UNSUPPORTED,
                                    final_url=current_url,
                                    content_type=kind,
                                    http_status=status,
                                    bytes_downloaded=len(prefix),
                                    error_type="UNSUPPORTED_CONTENT_TYPE",
                                    error=f"unsupported content type: {kind}",
                                ),
                                reserved,
                            )
                        declared_size = response.content_length
                        if declared_size is not None and declared_size > limit:
                            return (
                                FetchOutcome(
                                    target,
                                    CrawlStatus.TOO_LARGE,
                                    final_url=current_url,
                                    content_type=kind,
                                    http_status=status,
                                    bytes_downloaded=0,
                                    error_type="CONTENT_LENGTH_LIMIT",
                                    error=f"declared size {declared_size} exceeds {limit}",
                                ),
                                reserved,
                            )

                        # Reserve the decompressed upper bound. Content-Length may describe
                        # compressed bytes and therefore cannot safely size the live buffer.
                        reserved = await self.response_budget.acquire(limit)
                        chunks = [prefix]
                        downloaded = len(prefix)
                        if downloaded > limit:
                            return (
                                FetchOutcome(
                                    target,
                                    CrawlStatus.TOO_LARGE,
                                    final_url=current_url,
                                    content_type=kind,
                                    http_status=status,
                                    bytes_downloaded=downloaded,
                                    error_type="RESPONSE_SIZE_LIMIT",
                                    error=f"response exceeds {limit} bytes",
                                ),
                                reserved,
                            )
                        body_started = time.monotonic()
                        try:
                            async for chunk in response.content.iter_chunked(64 * 1024):
                                downloaded += len(chunk)
                                if downloaded > limit:
                                    return (
                                        FetchOutcome(
                                            target,
                                            CrawlStatus.TOO_LARGE,
                                            final_url=current_url,
                                            content_type=kind,
                                            http_status=status,
                                            bytes_downloaded=downloaded,
                                            error_type="RESPONSE_SIZE_LIMIT",
                                            error=f"response exceeds {limit} bytes",
                                        ),
                                        reserved,
                                    )
                                chunks.append(chunk)
                        finally:
                            active_seconds[0] += time.monotonic() - body_started
                        return (
                            FetchOutcome(
                                target,
                                CrawlStatus.SUCCESS,
                                final_url=current_url,
                                content_type=kind,
                                content=b"".join(chunks),
                                http_status=status,
                                bytes_downloaded=downloaded,
                            ),
                            reserved,
                        )
            return (
                FetchOutcome(
                    target,
                    CrawlStatus.FAILED,
                    final_url=current_url,
                    error_type="TOO_MANY_REDIRECTS",
                    error=f"exceeded {self.config.max_redirects} redirects",
                ),
                reserved,
            )
        except (aiohttp.ClientError, TimeoutError) as exc:
            return (
                FetchOutcome(
                    target,
                    CrawlStatus.FAILED,
                    final_url=current_url,
                    retryable=True,
                    retry_at=self.backoff_time(target.attempts),
                    error_type="NETWORK_ERROR",
                    error=f"{type(exc).__name__}: {exc}"[:2000],
                ),
                reserved,
            )
        except Exception as exc:
            LOGGER.exception("unexpected fetch error", extra={"fetch_key": target.fetch_key})
            return (
                FetchOutcome(
                    target,
                    CrawlStatus.FAILED,
                    final_url=current_url,
                    retryable=True,
                    retry_at=self.backoff_time(target.attempts),
                    error_type="FETCH_ERROR",
                    error=f"{type(exc).__name__}: {exc}"[:2000],
                ),
                reserved,
            )
