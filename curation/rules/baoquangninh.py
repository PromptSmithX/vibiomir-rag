import re

from curation.base import BaseDomainRule


class BaoQuangNinhRule(BaseDomainRule):
    """Cleaning rule for baoquangninh.vn."""

    TAIL_CUTOFF_TRIGGERS = [
        re.compile(r"^#+\s*từ khóa\s*:?", re.IGNORECASE),
        re.compile(r"^#+\s*tin liên quan", re.IGNORECASE),
        re.compile(r"^#+\s*ý kiến bạn đọc", re.IGNORECASE),
    ]

    AUTHOR_SIGNATURE_PATTERN = re.compile(
        r"^\*\*([A-ZÀ-Ỹa-zà-ỹ\s]+(\s*-\s*[A-ZÀ-Ỹa-zà-ỹ\s]+)?)\*\*$"
    )

    def clean(self, text: str, title: str | None = None) -> str:
        if not text:
            return ""

        lines = text.split("\n")

        # Explicit cutoff
        cutoff_index = len(lines)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if any(p.match(stripped) for p in self.TAIL_CUTOFF_TRIGGERS):
                cutoff_index = i
                break

        lines = lines[:cutoff_index]

        # Strip trailing author signature or tags
        while lines:
            last = lines[-1].strip()
            if not last:
                lines.pop()
                continue
            if self.AUTHOR_SIGNATURE_PATTERN.match(last) and len(last) < 60:
                lines.pop()
                continue
            if last.startswith("- ") and len(last) < 40:
                lines.pop()
                continue
            break

        while lines and not lines[-1].strip():
            lines.pop()

        return "\n".join(lines)
