import re

from curation.base import BaseDomainRule


class LongChauRule(BaseDomainRule):
    """Cleaning rule for tiemchunglongchau.com.vn."""

    TAIL_CUTOFF_TRIGGERS = [
        re.compile(
            r"^#+\s*(bài viết liên quan|chủ đề liên quan|tiêm chủng liên quan|nội dung liên quan)",
            re.IGNORECASE,
        ),
        re.compile(r"^#+\s*(thông tin tiêm chủng|hệ thống trung tâm tiêm chủng)", re.IGNORECASE),
    ]

    VIEW_COUNT_PATTERN = re.compile(r"^\d+\s*lượt\s*xem", re.IGNORECASE)

    def clean(self, text: str, title: str | None = None) -> str:
        if not text:
            return ""

        lines = text.split("\n")

        # Check explicit tail cutoff trigger
        cutoff_index = len(lines)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if any(p.search(stripped) for p in self.TAIL_CUTOFF_TRIGGERS):
                cutoff_index = i
                break

        lines = lines[:cutoff_index]

        # Scan backwards to strip trailing carousel: (heading followed by view count)
        # e.g., '### Sinh thiết phôi...', '0 lượt xem'
        while lines:
            last = lines[-1].strip()
            if not last:
                lines.pop()
                continue
            if self.VIEW_COUNT_PATTERN.match(last):
                lines.pop()
                # If preceding line was a header (e.g. ### ...), pop it as well
                if lines and lines[-1].strip().startswith("#"):
                    lines.pop()
                continue
            if last.startswith("### ") and any(
                self.VIEW_COUNT_PATTERN.match(cand.strip()) for cand in lines[-3:]
            ):
                lines.pop()
                continue
            break

        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Drop standalone view counts if any leaked inline
            if self.VIEW_COUNT_PATTERN.match(stripped):
                continue
            cleaned.append(line)

        while cleaned and not cleaned[-1].strip():
            cleaned.pop()

        return "\n".join(cleaned)
