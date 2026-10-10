from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


@dataclass
class AuditSamplePair:
    doc_id: int
    domain: str
    url: str
    title: str
    raw_text: str
    clean_text: str
    raw_chars: int
    clean_chars: int
    removed_chars: int
    removed_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StratifiedSampler:
    """Samples a balanced, representative subset of documents across domains."""

    def __init__(
        self,
        raw_corpus_dir: Path | str,
        clean_corpus_dir: Path | str,
        target_total_samples: int = 300,
    ) -> None:
        self.raw_dir = Path(raw_corpus_dir)
        self.clean_dir = Path(clean_corpus_dir)
        self.target_total = target_total_samples

    def sample(self, max_per_domain: int = 30) -> list[AuditSamplePair]:
        clean_shards = sorted(self.clean_dir.glob("*.parquet"))
        if not clean_shards:
            return []

        # Group candidates by domain
        domain_candidates: dict[str, list[AuditSamplePair]] = defaultdict(list)

        for clean_shard in clean_shards:
            raw_shard = self.raw_dir / clean_shard.name
            if not raw_shard.exists():
                continue

            tbl_clean = pq.read_table(
                clean_shard,
                columns=[
                    "doc_id",
                    "domain",
                    "url",
                    "title",
                    "text",
                    "text_chars",
                    "raw_text_chars",
                    "removed_chars",
                    "curation_status",
                ],
            )
            tbl_raw = pq.read_table(
                raw_shard,
                columns=["doc_id", "text"],
            )

            raw_dict = {
                tbl_raw.column("doc_id")[i].as_py(): tbl_raw.column("text")[i].as_py() or ""
                for i in range(tbl_raw.num_rows)
            }

            for i in range(tbl_clean.num_rows):
                status = tbl_clean.column("curation_status")[i].as_py()
                if status != "CLEAN":
                    continue

                did = tbl_clean.column("doc_id")[i].as_py()
                dom = tbl_clean.column("domain")[i].as_py() or "unknown"
                url = tbl_clean.column("url")[i].as_py() or ""
                title = tbl_clean.column("title")[i].as_py() or ""
                clean_t = tbl_clean.column("text")[i].as_py() or ""
                clean_c = tbl_clean.column("text_chars")[i].as_py() or len(clean_t)
                raw_c = tbl_clean.column("raw_text_chars")[i].as_py() or clean_c
                rem_c = tbl_clean.column("removed_chars")[i].as_py() or 0
                raw_t = raw_dict.get(did, clean_t)

                rem_pct = round((rem_c / raw_c * 100), 2) if raw_c > 0 else 0.0

                domain_candidates[dom].append(
                    AuditSamplePair(
                        doc_id=did,
                        domain=dom,
                        url=url,
                        title=title,
                        raw_text=raw_t,
                        clean_text=clean_t,
                        raw_chars=raw_c,
                        clean_chars=clean_c,
                        removed_chars=rem_c,
                        removed_pct=rem_pct,
                    )
                )

        selected_samples: list[AuditSamplePair] = []

        # Stratified selection per domain
        for _dom, candidates in sorted(domain_candidates.items()):
            if len(candidates) <= max_per_domain:
                selected_samples.extend(candidates)
                continue

            # Sort by removed_chars descending to get edge cases
            candidates.sort(key=lambda x: x.removed_chars, reverse=True)

            # Pick top removed, middle removed, and some spread across lengths
            quota = min(len(candidates), max_per_domain)
            top_count = quota // 3
            mid_count = quota // 3
            rest_count = quota - top_count - mid_count

            # Top noise items
            chosen = list(candidates[:top_count])

            # Median noise items
            mid_start = len(candidates) // 2
            chosen.extend(candidates[mid_start : mid_start + mid_count])

            # Length-diverse items from remaining
            remaining = [c for c in candidates if c not in chosen]
            remaining.sort(key=lambda x: x.clean_chars)
            step = max(1, len(remaining) // (rest_count or 1))
            for k in range(rest_count):
                idx = min(k * step, len(remaining) - 1)
                chosen.append(remaining[idx])

            selected_samples.extend(chosen)

        return selected_samples
