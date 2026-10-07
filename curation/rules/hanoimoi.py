import re

from curation.base import BaseDomainRule


class HanoiMoiRule(BaseDomainRule):
    """Cleaning rule for hanoimoi.vn."""

    COPYRIGHT_PATTERN = re.compile(
        r"^\(\*\)\s*Không sao chép dưới mọi hình thức.*Cơ quan Báo.*Hà Nội",
        re.IGNORECASE,
    )

    TAIL_CUTOFF_TRIGGERS = [
        re.compile(r"^#+\s*(tin liên quan|tin cùng chuyên mục|đọc thêm)", re.IGNORECASE),
        re.compile(r"^#+\s*từ khóa", re.IGNORECASE),
    ]

    def clean(self, text: str, title: str | None = None) -> str:
        if not text:
            return ""

        lines = text.split("\n")

        # Strip copyright notice and cutoff triggers
        cutoff_index = len(lines)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if self.COPYRIGHT_PATTERN.search(stripped) or any(
                p.search(stripped) for p in self.TAIL_CUTOFF_TRIGGERS
            ):
                cutoff_index = min(cutoff_index, i)
                break

        lines = lines[:cutoff_index]

        # Scan backwards to strip trailing tag bullets (e.g. '- Đào Hồng Lan', '- Trách nhiệm')
        while lines:
            last = lines[-1].strip()
            if not last:
                lines.pop()
                continue
            if last.startswith("- ") and len(last) < 60:
                lines.pop()
                continue
            break

        while lines and not lines[-1].strip():
            lines.pop()

        return "\n".join(lines)
