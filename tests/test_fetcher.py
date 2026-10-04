import time

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402

from crawler.config import CrawlerRuntimeConfig  # noqa: E402
from crawler.fetcher import ByteBudget, Fetcher  # noqa: E402
from crawler.models import CrawlStatus, FetchTarget  # noqa: E402


def runtime_config(**overrides) -> CrawlerRuntimeConfig:
    values = {
        "global_concurrency": 4,
        "per_domain_concurrency": 2,
        "connect_timeout_seconds": 2,
        "total_timeout_seconds": 5,
        "max_attempts": 3,
        "max_redirects": 3,
        "max_html_bytes": 1024,
        "max_pdf_bytes": 2048,
        "user_agent": "ViBioMIRResearchCrawler/1.0",
        "retry_base_seconds": 0.01,
        "robots_cache_seconds": 60,
        "robots_timeout_seconds": 2,
    }
    values.update(overrides)
    return CrawlerRuntimeConfig(**values)


async def start_server(handler):
    app = web.Application()

    async def robots(_request):
        return web.Response(text="User-agent: *\nAllow: /")

    app.router.add_get("/robots.txt", robots)
    app.router.add_get("/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


@pytest.mark.asyncio
async def test_fetcher_streams_html_and_holds_byte_budget() -> None:
    async def handler(_request):
        return web.Response(text="<html><main><p>medical article</p></main></html>")

    runner, base = await start_server(handler)
    budget = ByteBudget(4096)
    try:
        async with Fetcher(runtime_config(), budget) as fetcher:
            target = FetchTarget("key", f"{base}/article", "127.0.0.1", 1)
            outcome, reserved = await fetcher.fetch(target)
            assert outcome.status == CrawlStatus.SUCCESS
            assert outcome.content_type == "text/html"
            assert b"medical article" in outcome.content
            assert reserved > 0
            await budget.release(reserved)
            assert budget.available == budget.capacity
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_fetcher_marks_429_retryable() -> None:
    async def handler(_request):
        return web.Response(status=429, headers={"Retry-After": "2"})

    runner, base = await start_server(handler)
    try:
        async with Fetcher(runtime_config(), ByteBudget(4096)) as fetcher:
            target = FetchTarget("key", f"{base}/limited", "127.0.0.1", 1)
            before = time.time()
            outcome, reserved = await fetcher.fetch(target)
            assert reserved == 0
            assert outcome.retryable is True
            assert outcome.http_status == 429
            assert outcome.retry_at >= before + 1.5
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_fetcher_stops_oversized_response() -> None:
    async def handler(_request):
        return web.Response(body=b"<html>" + b"x" * 2000 + b"</html>")

    runner, base = await start_server(handler)
    budget = ByteBudget(4096)
    try:
        async with Fetcher(runtime_config(max_html_bytes=512), budget) as fetcher:
            target = FetchTarget("key", f"{base}/large", "127.0.0.1", 1)
            outcome, reserved = await fetcher.fetch(target)
            assert outcome.status == CrawlStatus.TOO_LARGE
            if reserved:
                await budget.release(reserved)
    finally:
        await runner.cleanup()
