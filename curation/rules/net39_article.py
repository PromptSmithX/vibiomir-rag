import re

from curation.base import BaseDomainRule


class Net39ArticleRule(BaseDomainRule):
    """Cleaning rule for woman.39.net and pf.39.net articles."""

    # Matches hospital recommendation entries at the tail
    HOSPITAL_LINE_PATTERN = re.compile(
        r"^-\s+.+?(医院|妇幼保健院|诊所|中心)\s+(一级|二级|三级)(甲等|乙等|丙等)?.*",
        re.IGNORECASE,
    )

    TAIL_CUTOFF_TRIGGERS = [
        re.compile(r"^#+\s*(推荐医院|相关医院|就医推荐|相关阅读|相关推荐)", re.IGNORECASE),
    ]

    def clean(self, text: str, title: str | None = None) -> str:
        if not text:
            return ""

        lines = text.split("\n")

        # Cut off from explicit recommendation heading
        cutoff_index = len(lines)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if any(p.search(stripped) for p in self.TAIL_CUTOFF_TRIGGERS):
                cutoff_index = i
                break

        lines = lines[:cutoff_index]

        # Scan backwards to strip trailing hospital ad list
        while lines:
            last = lines[-1].strip()
            if not last:
                lines.pop()
                continue
            if self.HOSPITAL_LINE_PATTERN.match(last):
                lines.pop()
                continue
            # Some entries don't have level: e.g. '- xxx医院'
            if last.startswith("- ") and "医院" in last and len(last) < 45:
                lines.pop()
                continue
            break

        while lines and not lines[-1].strip():
            lines.pop()

        return "\n".join(lines)
