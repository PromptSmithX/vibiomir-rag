"""Search engine module executing BM25 queries against the disk index."""

import logging
import time
from pathlib import Path

import pyarrow.parquet as pq
import tantivy

from retrieval.models import QueryItem, RetrievalHit
from retrieval.tokenizer import MultilingualMedicalTokenizer

logger = logging.getLogger(__name__)


class BM25Searcher:
    """Executes multilingual BM25 queries with title boosting against the disk index."""

    TOKENIZER_NAME = "multilingual"

    def __init__(
        self,
        index_dir: Path | str,
        title_boost: float = 2.0,
        tokenizer: MultilingualMedicalTokenizer | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.title_boost = title_boost
        self.tokenizer = tokenizer or MultilingualMedicalTokenizer()

        self._index = tantivy.Index.open(str(self.index_dir))
        self._setup_analyzer()
        self._searcher = self._index.searcher()

    def _setup_analyzer(self) -> None:
        """Register the multilingual analyzer for query token parsing."""
        analyzer = (
            tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.whitespace())
            .filter(tantivy.Filter.lowercase())
            .build()
        )
        self._index.register_tokenizer(self.TOKENIZER_NAME, analyzer)

    def reload(self) -> None:
        """Reload index in case new documents were added."""
        self._index.reload()
        self._searcher = self._index.searcher()

    def search_single_query(
        self,
        query_id: int,
        query_text: str,
        k: int = 1000,
    ) -> list[RetrievalHit]:
        """Search a single query and return top-K RetrievalHit results."""
        query_str = self.tokenizer.build_query_string(query_text)
        if not query_str.strip():
            logger.warning("Empty query string after tokenization for query_id %d", query_id)
            return []

        field_boosts = {"title": self.title_boost, "body": 1.0}
        default_fields = ["title", "body"]

        try:
            parsed = self._index.parse_query(
                query_str,
                default_field_names=default_fields,
                field_boosts=field_boosts,
                conjunction_by_default=False,
                allow_regexes=False,
            )
        except Exception:
            # Fallback to lenient parsing if special symbols encountered
            try:
                parsed = self._index.parse_query_lenient(
                    query_str,
                    default_field_names=default_fields,
                    field_boosts=field_boosts,
                )
            except Exception as e:
                logger.error("Failed to parse query %d ('%s'): %s", query_id, query_str, e)
                return []

        search_result = self._searcher.search(parsed, limit=k)
        hits: list[RetrievalHit] = []

        for rank_0, (score, doc_address) in enumerate(search_result.hits):
            doc = self._searcher.doc(doc_address)
            doc_id_val = int(doc["doc_id"][0])
            hits.append(
                RetrievalHit(
                    query_id=query_id,
                    doc_id=doc_id_val,
                    rank=rank_0 + 1,  # 1-indexed
                    bm25_score=float(score),
                )
            )

        return hits

    def search_queries(
        self,
        queries: list[QueryItem],
        k: int = 1000,
        log_interval: int = 100,
    ) -> list[RetrievalHit]:
        """Execute retrieval across a batch of queries."""
        all_hits: list[RetrievalHit] = []
        total_queries = len(queries)
        logger.info("Executing retrieval for %d queries with K=%d...", total_queries, k)

        start_time = time.time()
        for idx, item in enumerate(queries, 1):
            hits = self.search_single_query(item.query_id, item.query_text, k=k)
            all_hits.extend(hits)

            if idx % log_interval == 0 or idx == total_queries:
                elapsed = time.time() - start_time
                qps = idx / elapsed if elapsed > 0 else 0
                logger.info(
                    "Queries [%d/%d] processed | Total hits: %d | Speed: %.1f queries/s",
                    idx,
                    total_queries,
                    len(all_hits),
                    qps,
                )

        total_time = time.time() - start_time
        avg_ms = (total_time / total_queries * 1000) if total_queries > 0 else 0
        logger.info(
            "Completed %d queries in %.2fs (avg %.2f ms/query). Total hits: %d",
            total_queries,
            total_time,
            avg_ms,
            len(all_hits),
        )

        return all_hits

    @staticmethod
    def load_queries_from_parquet(parquet_path: Path | str) -> list[QueryItem]:
        """Load queries from official BTC query.parquet file."""
        tbl = pq.read_table(parquet_path)
        id_col = "id" if "id" in tbl.schema.names else "query_id"
        text_col = "query" if "query" in tbl.schema.names else "query_text"

        ids = tbl.column(id_col).to_pylist()
        texts = tbl.column(text_col).to_pylist()

        return [
            QueryItem(query_id=int(i), query_text=str(t or ""))
            for i, t in zip(ids, texts, strict=False)
        ]
