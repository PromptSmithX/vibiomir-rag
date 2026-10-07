import re

from curation.base import BaseDomainRule


class TamAnhHospitalRule(BaseDomainRule):
    """Cleaning rule for tamanhhospital.vn."""

    TAIL_CUTOFF_TRIGGERS = [
        re.compile(r"^#+\s*ĐỐI TÁC BẢO HIỂM", re.IGNORECASE),
        re.compile(r"^Bước 1:\s*Bấm vào nút.*Thêm Bệnh viện Đa khoa Tâm Anh", re.IGNORECASE),
        re.compile(r"^#+\s*ĐẶT LỊCH KHÁM", re.IGNORECASE),
        re.compile(r"^#+\s*HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH", re.IGNORECASE),
        re.compile(
            r"^#+\s*(bài viết cùng chủ đề|bài viết liên quan|chuyên mục liên quan)",
            re.IGNORECASE,
        ),
    ]

    INLINE_DROP_PATTERNS = [
        re.compile(r"^Bước \d+:\s*.*Bệnh viện Đa khoa Tâm Anh.*", re.IGNORECASE),
        re.compile(r"^Khi ô vuông chuyển thành dấu tích xanh là hoàn thành.*", re.IGNORECASE),
        re.compile(r"^\*?\s*Hotline:\s*\d+.*", re.IGNORECASE),
    ]

    def clean(self, text: str, title: str | None = None) -> str:
        if not text:
            return ""

        lines = text.split("\n")

        # Cut off from the first tail trigger
        cutoff_index = len(lines)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if any(p.search(stripped) for p in self.TAIL_CUTOFF_TRIGGERS):
                cutoff_index = i
                break

        lines = lines[:cutoff_index]

        # Filter inline lines
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if any(p.match(stripped) for p in self.INLINE_DROP_PATTERNS):
                continue
            cleaned.append(line)

        # Strip trailing empty lines
        while cleaned and not cleaned[-1].strip():
            cleaned.pop()

        return "\n".join(cleaned)
