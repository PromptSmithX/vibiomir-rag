
from curation.base import BaseDomainRule


class BaiduHealthRule(BaseDomainRule):
    """Cleaning rule for www.baidu.com (Baidu Health)."""

    TOC_BULLETS = {
        "- 概述",
        "- 病因",
        "- 症状",
        "- 就医",
        "- 检查",
        "- 诊断",
        "- 治疗",
        "- 预后",
        "- 预防",
        "- 日常",
    }

    def clean(self, text: str, title: str | None = None) -> str:
        if not text:
            return ""

        lines = text.split("\n")

        # Strip audit banner at head
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if "百度健康医典原创内容" in stripped and "三审三校" in stripped:
                continue
            if stripped in self.TOC_BULLETS:
                continue
            cleaned.append(line)

        while cleaned and not cleaned[0].strip():
            cleaned.pop(0)

        while cleaned and not cleaned[-1].strip():
            cleaned.pop()

        return "\n".join(cleaned)
