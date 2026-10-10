import re

from curation.base import BaseDomainRule


class NhanDanRule(BaseDomainRule):
    """Cleaning rule for nhandan.vn."""

    TAIL_CUTOFF_TRIGGERS = [
        re.compile(r"^#+\s*(tin liên quan|tin cùng chuyên mục|đọc thêm)", re.IGNORECASE),
        re.compile(r"^#+\s*từ khóa\s*:?", re.IGNORECASE),
        re.compile(r"^#+\s*bình luận", re.IGNORECASE),
    ]

    AUTHOR_LINE_PATTERN = re.compile(
        r"^(theo\s+|bài và ảnh:\s*|nguồn:\s*|phóng viên\s*|nhân dân\s*|báo nhân dân).*",
        re.IGNORECASE,
    )

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

        # Scan backwards for author/source signature lines
        while lines:
            last = lines[-1].strip()
            if not last:
                lines.pop()
                continue
            if self.AUTHOR_LINE_PATTERN.match(last) and len(last) < 80:
                lines.pop()
                continue
            if last.startswith("- ") and len(last) < 40:
                lines.pop()
                continue
            break

        while lines and not lines[-1].strip():
            lines.pop()

        return "\n".join(lines)
