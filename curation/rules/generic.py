import re

from curation.base import BaseDomainRule


class GenericDomainRule(BaseDomainRule):
    """Generic fallback rule providing safe boilerplate removal for any domain."""

    # Common generic footer noise triggers
    GENERIC_TAIL_TRIGGERS = [
        re.compile(
            r"^#+\s*(chia sẻ|share|bình luận|comment|bài viết liên quan|"
            r"tin liên quan|tin cùng chuyên mục)",
            re.IGNORECASE,
        ),
        re.compile(
            r"^#+\s*(theo dõi chúng tôi|kết nối với chúng tôi|thông tin liên hệ)",
            re.IGNORECASE,
        ),
        re.compile(r"^\s*©\s*(copyright|bản quyền).*", re.IGNORECASE),
    ]

    def clean(self, text: str, title: str | None = None) -> str:
        if not text:
            return ""

        lines = text.split("\n")
        cleaned_lines = []

        # Find first generic tail trigger
        cutoff_index = len(lines)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if any(p.search(stripped) for p in self.GENERIC_TAIL_TRIGGERS):
                cutoff_index = i
                break

        lines = lines[:cutoff_index]

        # Filter inline repetitive junk
        for line in lines:
            stripped = line.strip()
            # Ignore standalone social share text
            if stripped.lower() in ("chia sẻ", "share", "like", "tweet", "in bài"):
                continue
            cleaned_lines.append(line)

        # Strip trailing empty lines or separators
        while cleaned_lines:
            tail = cleaned_lines[-1].strip()
            if not tail or tail in ("---", "___", "***"):
                cleaned_lines.pop()
            else:
                break

        return "\n".join(cleaned_lines)
