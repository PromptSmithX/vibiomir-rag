import re

from curation.base import BaseDomainRule


class VietNamNetRule(BaseDomainRule):
    """Cleaning rule for vietnamnet.vn."""

    TAIL_CUTOFF_TRIGGERS = [
        re.compile(r"^#+\s*Tin cùng chuyên mục", re.IGNORECASE),
        re.compile(r"^#+\s*Tin mới", re.IGNORECASE),
        re.compile(r"^#+\s*Tin nổi bật", re.IGNORECASE),
        re.compile(r"^#+\s*Bình luận", re.IGNORECASE),
    ]

    def clean(self, text: str, title: str | None = None) -> str:
        if not text:
            return ""

        lines = text.split("\n")

        # 1. Clean head breadcrumbs (short bullet lines before the first `# ` header)
        first_h1_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("# "):
                first_h1_idx = i
                break

        if first_h1_idx is not None and first_h1_idx > 0:
            # Check if all lines before first_h1_idx are breadcrumbs or empty
            pre_lines = lines[:first_h1_idx]
            if all(not item.strip() or item.strip().startswith("- ") for item in pre_lines):
                lines = lines[first_h1_idx:]

        # 2. Clean tail triggers
        cutoff_index = len(lines)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if any(p.match(stripped) for p in self.TAIL_CUTOFF_TRIGGERS):
                cutoff_index = i
                break

        lines = lines[:cutoff_index]

        # 3. Clean trailing related tags/bullets
        while lines:
            last = lines[-1].strip()
            if not last:
                lines.pop()
                continue
            if last.lower() in ("xem thêm về:", "xem thêm:", "- xem thêm về:", "- xem thêm:"):
                lines.pop()
                continue
            if last.startswith("- ") and len(last) < 60:
                lines.pop()
                continue
            if last.startswith("### ") and len(last) < 40:
                lines.pop()
                continue
            break

        while lines and not lines[-1].strip():
            lines.pop()

        return "\n".join(lines)
