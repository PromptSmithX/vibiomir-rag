import re

from curation.base import BaseDomainRule


class VOV2Rule(BaseDomainRule):
    """Cleaning rule for vov2.vov.vn."""

    TAIL_CUTOFF_TRIGGERS = [
        re.compile(
            r"^#+\s*(tin cùng chuyên mục|tin liên quan|tin mới nhất|tin mới|xem nhiều|đọc nhiều)",
            re.IGNORECASE,
        ),
        re.compile(r"^#+\s*(chương trình liên quan|podcast|nghe lại)", re.IGNORECASE),
    ]

    def clean(self, text: str, title: str | None = None) -> str:
        if not text:
            return ""

        lines = text.split("\n")

        # Explicit cutoff
        cutoff_index = len(lines)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if any(p.search(stripped) for p in self.TAIL_CUTOFF_TRIGGERS):
                cutoff_index = i
                break

        lines = lines[:cutoff_index]

        # Scan backwards: VOV2 frequently ends with a cluster of 3-6 lines of `#### <News Headline>`
        # that are unrelated site-wide news recommendations.
        while lines:
            last = lines[-1].strip()
            if not last:
                lines.pop()
                continue
            if last.startswith("#### "):
                lines.pop()
                continue
            break

        while lines and not lines[-1].strip():
            lines.pop()

        return "\n".join(lines)
