from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass
from typing import TextIO

from colorama import Fore, Style, just_fix_windows_console

from crawler.models import CrawlStatus, FetchTarget

PROGRESS_MODES = ("auto", "full", "bar", "off")

_ACTIVE_PROGRESS: CrawlProgress | None = None


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 / 1024:.1f} MB"


def _truncate(value: str, width: int) -> str:
    if width <= 1:
        return "…"
    if len(value) <= width:
        return value
    return value[: width - 1] + "…"


def emit_console_line(message: str, stream: TextIO | None = None) -> None:
    """Write a log line without corrupting an active progress bar."""
    if _ACTIVE_PROGRESS is not None and _ACTIVE_PROGRESS.stream is (stream or sys.stderr):
        _ACTIVE_PROGRESS.external_line(message)
        return
    target = stream or sys.stderr
    target.write(message.rstrip("\n") + "\n")
    target.flush()


@dataclass(slots=True)
class ProgressEvent:
    target: FetchTarget
    status: str
    http_status: int | None
    bytes_downloaded: int
    elapsed_seconds: float
    retry: bool = False


class CrawlProgress:
    def __init__(
        self,
        mode: str,
        total: int,
        max_attempts: int,
        *,
        stream: TextIO | None = None,
        is_terminal: bool | None = None,
        width: int | None = None,
    ) -> None:
        if mode not in PROGRESS_MODES:
            raise ValueError(f"unknown progress mode: {mode}")
        self.stream = stream or sys.stderr
        terminal = self.stream.isatty() if is_terminal is None else is_terminal
        self.mode = "full" if mode == "auto" and terminal else mode
        if self.mode == "auto":
            self.mode = "off"
        self.total = max(0, total)
        self.max_attempts = max_attempts
        self.width = width
        self.completed = 0
        self.active = 0
        self.latest_url = ""
        self.phase = "crawling"
        self.started_at = time.monotonic()
        self._target_started: dict[str, float] = {}
        self._bar_drawn = False

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def show_lines(self) -> bool:
        return self.mode == "full"

    def __enter__(self) -> CrawlProgress:
        global _ACTIVE_PROGRESS
        if self.enabled:
            just_fix_windows_console()
            _ACTIVE_PROGRESS = self
            self._render_bar()
        return self

    def __exit__(self, *_: object) -> None:
        global _ACTIVE_PROGRESS
        if self.enabled:
            self._render_bar()
            self.stream.write("\n")
            self.stream.flush()
        if _ACTIVE_PROGRESS is self:
            _ACTIVE_PROGRESS = None

    def target_started(self, target: FetchTarget) -> None:
        if not self.enabled:
            return
        self.active += 1
        self.latest_url = target.request_url
        self._target_started[target.fetch_key] = time.monotonic()
        self._render_bar()

    def retry(self, target: FetchTarget, error_type: str | None) -> None:
        if not self.enabled:
            return
        elapsed = self._finish_attempt(target)
        event = ProgressEvent(
            target=target,
            status=error_type or "RETRY_WAIT",
            http_status=None,
            bytes_downloaded=0,
            elapsed_seconds=elapsed,
            retry=True,
        )
        if self.show_lines:
            self._print_event(event)
        self._render_bar()

    def terminal(
        self,
        target: FetchTarget,
        status: CrawlStatus,
        http_status: int | None,
        bytes_downloaded: int,
    ) -> None:
        if not self.enabled:
            return
        elapsed = self._finish_attempt(target)
        self.completed += 1
        event = ProgressEvent(
            target=target,
            status=status.value,
            http_status=http_status,
            bytes_downloaded=bytes_downloaded,
            elapsed_seconds=elapsed,
        )
        if self.show_lines:
            self._print_event(event)
        self._render_bar()

    def external_line(self, message: str) -> None:
        self._clear_bar()
        self.stream.write(message.rstrip("\n") + "\n")
        self.stream.flush()
        self._render_bar()

    def set_phase(self, phase: str) -> None:
        if not self.enabled:
            return
        self.phase = phase
        self._render_bar()

    def _finish_attempt(self, target: FetchTarget) -> float:
        self.active = max(0, self.active - 1)
        started = self._target_started.pop(target.fetch_key, self.started_at)
        return max(0.0, time.monotonic() - started)

    def _terminal_width(self) -> int:
        if self.width is not None:
            return self.width
        return shutil.get_terminal_size(fallback=(120, 24)).columns

    def _clear_bar(self) -> None:
        if self._bar_drawn:
            self.stream.write("\r\x1b[2K")
            self._bar_drawn = False

    def _status_color(self, event: ProgressEvent) -> str:
        if event.retry or event.status == CrawlStatus.RETRY_WAIT.value:
            return Fore.YELLOW
        if event.status == CrawlStatus.SUCCESS.value:
            return Fore.GREEN
        if event.status == CrawlStatus.ROBOTS_DENIED.value:
            return Fore.YELLOW
        return Fore.RED

    def _print_event(self, event: ProgressEvent) -> None:
        self._clear_bar()
        position = f"{self.completed:05d}/{self.total:05d}"
        if event.retry:
            position = " " * len(position)
        status = (
            f"RETRY {event.target.attempts}/{self.max_attempts}"
            if event.retry
            else event.status
        )
        http = str(event.http_status) if event.http_status is not None else "-"
        size = _format_bytes(event.bytes_downloaded) if event.bytes_downloaded else "-"
        prefix = (
            f"{position}  {status:<18} {http:>3}  "
            f"{event.elapsed_seconds:>6.2f}s  {size:>9}  {event.target.domain:<28}  "
        )
        url = _truncate(event.target.request_url, max(12, self._terminal_width() - len(prefix)))
        self.stream.write(
            f"{self._status_color(event)}{prefix}{url}{Style.RESET_ALL}\n"
        )
        self.stream.flush()

    def _render_bar(self) -> None:
        if not self.enabled:
            return
        self._clear_bar()
        elapsed = max(0.001, time.monotonic() - self.started_at)
        rate = self.completed / elapsed
        remaining = max(0, self.total - self.completed)
        eta = remaining / rate if rate > 0 else 0
        ratio = self.completed / self.total if self.total else 1.0
        bar_width = 24
        filled = min(bar_width, int(ratio * bar_width))
        bar = "█" * filled + "░" * (bar_width - filled)
        latest_width = max(12, self._terminal_width() - 82)
        latest = _truncate(self.latest_url, latest_width) if self.latest_url else "waiting"
        line = (
            f"{Fore.CYAN}[{bar}] {self.completed:,}/{self.total:,} "
            f"{rate:5.1f} url/s ETA {eta:6.0f}s active={self.active:<3} "
            f"{self.phase:<9} {latest}{Style.RESET_ALL}"
        )
        self.stream.write("\r" + _truncate(line, self._terminal_width()))
        self.stream.flush()
        self._bar_drawn = True
