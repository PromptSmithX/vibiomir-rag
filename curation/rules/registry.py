
from curation.base import BaseDomainRule
from curation.rules.baidu import BaiduHealthRule
from curation.rules.baoquangninh import BaoQuangNinhRule
from curation.rules.generic import GenericDomainRule
from curation.rules.hanoimoi import HanoiMoiRule
from curation.rules.longchau import LongChauRule
from curation.rules.net39_article import Net39ArticleRule
from curation.rules.net39_qa import Ask39QARule
from curation.rules.nhandan import NhanDanRule
from curation.rules.tamanh import TamAnhHospitalRule
from curation.rules.vietnamnet import VietNamNetRule
from curation.rules.vov2 import VOV2Rule


class RuleRegistry:
    """Registry maintaining domain-to-rule mappings using Strategy Pattern."""

    def __init__(self) -> None:
        self._generic_rule = GenericDomainRule()
        self._rules: dict[str, BaseDomainRule] = {
            "tamanhhospital.vn": TamAnhHospitalRule(),
            "tiemchunglongchau.com.vn": LongChauRule(),
            "hanoimoi.vn": HanoiMoiRule(),
            "vietnamnet.vn": VietNamNetRule(),
            "vov2.vov.vn": VOV2Rule(),
            "baoquangninh.vn": BaoQuangNinhRule(),
            "nhandan.vn": NhanDanRule(),
            "ask.39.net": Ask39QARule(),
            "woman.39.net": Net39ArticleRule(),
            "pf.39.net": Net39ArticleRule(),
            "www.baidu.com": BaiduHealthRule(),
            "baidu.com": BaiduHealthRule(),
        }

    def register(self, domain: str, rule: BaseDomainRule) -> None:
        """Register a custom rule for a domain."""
        self._rules[domain.lower().strip()] = rule

    def get(self, domain: str | None) -> BaseDomainRule:
        """Get cleaning rule for domain, with fallback to generic rule."""
        if not domain:
            return self._generic_rule

        domain_clean = domain.lower().strip()
        if domain_clean in self._rules:
            return self._rules[domain_clean]

        # Check subdomains (e.g. *.39.net)
        for registered_domain, rule in self._rules.items():
            if domain_clean.endswith(f".{registered_domain}"):
                return rule

        return self._generic_rule


# Global default registry instance
default_registry = RuleRegistry()
