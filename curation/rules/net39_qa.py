
from curation.base import BaseDomainRule


class Ask39QARule(BaseDomainRule):
    """Cleaning rule for ask.39.net Q&A articles."""

    # Trailing utility menu bullets to remove
    NAV_BULLETS = {
        "- 就医助手",
        "- 药品通",
        "- 疾病百科",
        "- 咨讯诊疗",
        "- 资讯诊疗",
        "- 药品",
        "- 检查",
        "- 症状",
        "- 医院",
        "- 找医生",
        "- 39问医生",
        "- 39找医生",
        "- 39健康网",
        "- 找医院",
        "- 问答",
    }

    def clean(self, text: str, title: str | None = None) -> str:
        if not text:
            return ""

        lines = text.split("\n")

        # Strip trailing nav bullets
        while lines:
            last = lines[-1].strip()
            if not last:
                lines.pop()
                continue
            if last in self.NAV_BULLETS:
                lines.pop()
                continue
            break

        cleaned = []
        for line in lines:
            stripped = line.strip()
            # If line is exactly one of nav bullets that leaked inline
            if stripped in self.NAV_BULLETS:
                continue
            cleaned.append(line)

        while cleaned and not cleaned[-1].strip():
            cleaned.pop()

        return "\n".join(cleaned)
