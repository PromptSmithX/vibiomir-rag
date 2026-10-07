from __future__ import annotations

import asyncio
from collections import Counter, defaultdict, deque

from crawler.models import FetchTarget


class DomainDispatchQueue:
    """Bounded round-robin queue that limits active fetches per domain."""

    def __init__(self, maxsize: int, per_domain_limit: int) -> None:
        if maxsize < 1 or per_domain_limit < 1:
            raise ValueError("queue limits must be positive")
        self.maxsize = maxsize
        self.per_domain_limit = per_domain_limit
        self._queues: dict[str, deque[FetchTarget]] = defaultdict(deque)
        self._domains: deque[str] = deque()
        self._active: Counter[str] = Counter()
        self._queued = 0
        self._unfinished = 0
        self._closed = False
        self._condition = asyncio.Condition()

    async def put(self, target: FetchTarget) -> None:
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._queued < self.maxsize or self._closed
            )
            if self._closed:
                raise RuntimeError("cannot add targets to a closed domain queue")
            queue = self._queues[target.domain]
            if not queue:
                self._domains.append(target.domain)
            queue.append(target)
            self._queued += 1
            self._unfinished += 1
            self._condition.notify_all()

    async def get(self) -> FetchTarget | None:
        async with self._condition:
            while True:
                for _ in range(len(self._domains)):
                    domain = self._domains.popleft()
                    queue = self._queues[domain]
                    if self._active[domain] < self.per_domain_limit:
                        target = queue.popleft()
                        self._queued -= 1
                        self._active[domain] += 1
                        if queue:
                            self._domains.append(domain)
                        else:
                            del self._queues[domain]
                        self._condition.notify_all()
                        return target
                    self._domains.append(domain)
                if self._closed and self._queued == 0:
                    return None
                await self._condition.wait()

    async def task_done(self, domain: str) -> None:
        async with self._condition:
            if self._active[domain] <= 0 or self._unfinished <= 0:
                raise RuntimeError("domain queue task counter underflow")
            self._active[domain] -= 1
            if self._active[domain] == 0:
                del self._active[domain]
            self._unfinished -= 1
            self._condition.notify_all()

    async def join(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self._unfinished == 0)

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def active_by_domain(self) -> dict[str, int]:
        return dict(self._active)

