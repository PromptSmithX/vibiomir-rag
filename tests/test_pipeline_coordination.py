import asyncio
from types import SimpleNamespace

import pytest

from crawler.pipeline import CrawlPipeline


class BufferedManifest:
    """A terminal result is buffered while SQLite still reports FETCHING."""

    def claim_targets(self, _limit):
        return []

    def active_count(self):
        raise AssertionError("coordinator must not wait on persisted FETCHING state")

    def unfinished_count(self):
        return 1

    def next_retry_delay(self):
        return None


@pytest.mark.asyncio
async def test_coordinator_flushes_when_only_writer_buffer_remains() -> None:
    pipeline = CrawlPipeline.__new__(CrawlPipeline)
    pipeline.config = SimpleNamespace(
        queues=SimpleNamespace(fetch=1), crawler=SimpleNamespace(global_concurrency=0)
    )
    pipeline.manifest = BufferedManifest()
    pipeline.stop_requested = asyncio.Event()
    pipeline.metrics_stop = asyncio.Event()
    pipeline.fetch_queue = asyncio.Queue()
    pipeline.extract_queue = asyncio.Queue()
    pipeline.write_queue = asyncio.Queue()
    pipeline.extraction_workers = 0
    pipeline.in_flight = 0
    pipeline.progress = None

    async def writer():
        item = await pipeline.write_queue.get()
        pipeline.write_queue.task_done()
        assert item is None

    writer_task = asyncio.create_task(writer())
    await asyncio.wait_for(pipeline._coordinate(writer_task, None), timeout=1)

    assert writer_task.done()
    assert pipeline.metrics_stop.is_set()


def test_in_flight_counter_rejects_double_completion() -> None:
    pipeline = CrawlPipeline.__new__(CrawlPipeline)
    pipeline.in_flight = 1

    pipeline._target_finished()

    with pytest.raises(RuntimeError, match="underflow"):
        pipeline._target_finished()
