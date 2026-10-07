import asyncio

import pytest

from crawler.domain_queue import DomainDispatchQueue
from crawler.models import FetchTarget


def target(domain: str, index: int) -> FetchTarget:
    return FetchTarget(
        f"{domain}-{index}", f"https://{domain}/{index}", domain, 1
    )


@pytest.mark.asyncio
async def test_domain_queue_dispatches_round_robin() -> None:
    queue = DomainDispatchQueue(maxsize=8, per_domain_limit=3)
    for item in (
        target("a.example", 1),
        target("a.example", 2),
        target("b.example", 1),
        target("b.example", 2),
    ):
        await queue.put(item)

    claimed = [await queue.get() for _ in range(4)]

    assert [item.domain for item in claimed if item is not None] == [
        "a.example",
        "b.example",
        "a.example",
        "b.example",
    ]
    for item in claimed:
        assert item is not None
        await queue.task_done(item.domain)
    await queue.join()


@pytest.mark.asyncio
async def test_domain_queue_holds_domain_at_active_limit() -> None:
    queue = DomainDispatchQueue(maxsize=8, per_domain_limit=2)
    for item in (
        target("slow.example", 1),
        target("slow.example", 2),
        target("slow.example", 3),
        target("fast.example", 1),
    ):
        await queue.put(item)

    first = await queue.get()
    second = await queue.get()
    third = await queue.get()
    assert first is not None and second is not None and third is not None
    assert [first.domain, second.domain, third.domain] == [
        "slow.example",
        "fast.example",
        "slow.example",
    ]

    blocked = asyncio.create_task(queue.get())
    await asyncio.sleep(0)
    assert not blocked.done()

    await queue.task_done("slow.example")
    released = await asyncio.wait_for(blocked, timeout=1)
    assert released is not None
    assert released.domain == "slow.example"

    for item in (second, third, released):
        await queue.task_done(item.domain)
    await queue.join()


@pytest.mark.asyncio
async def test_closed_domain_queue_wakes_idle_workers() -> None:
    queue = DomainDispatchQueue(maxsize=1, per_domain_limit=1)
    waiter = asyncio.create_task(queue.get())

    await queue.close()

    assert await asyncio.wait_for(waiter, timeout=1) is None
