from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from retrieval.exporter import CandidateExporter
from retrieval.indexer import BM25Indexer
from retrieval.models import QueryItem
from retrieval.searcher import BM25Searcher
from retrieval.tokenizer import MultilingualMedicalTokenizer


def test_retrieval_end_to_end(tmp_path: Path) -> None:
    # 1. Create mock parquet shard
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    index_dir = tmp_path / "index"
    output_dir = tmp_path / "output"

    shard_path = corpus_dir / "part-000000.parquet"
    data = {
        "doc_id": [10001, 10002, 10003, 10004, 10001],  # Contains duplicate 10001
        "title": [
            "Viêm gan B mãn tính",
            None,  # No title edge case
            "Bệnh tiểu đường type 2",
            "急性扁桃体炎",  # Chinese CJK title
            "Viêm gan B mãn tính trùng",
        ],
        "text": [
            "Bệnh viêm gan B lây qua đường máu và tình dục.",
            "Trả lời: Bạn nên đi khám chuyên khoa tai mũi họng sớm.",  # Short QA doc
            "Điều trị bằng Metformin 500mg và insulin.",
            "患者咽喉肿痛，伴有发热咳嗽。",
            "Nội dung trùng.",
        ],
        "crawl_status": ["SUCCESS", "SUCCESS", "SUCCESS", "SUCCESS", "SUCCESS"],
    }
    tbl = pa.Table.from_pydict(data)
    pq.write_table(tbl, shard_path)

    # 2. Index documents
    tokenizer = MultilingualMedicalTokenizer()
    indexer = BM25Indexer(index_dir=index_dir, tokenizer=tokenizer, heap_size_mb=64)
    stats = indexer.build_index([corpus_dir], recreate=True)

    assert stats.total_docs == 4  # 5 rows minus 1 duplicate
    assert stats.index_size_bytes > 0

    # 3. Searcher
    searcher = BM25Searcher(index_dir=index_dir, title_boost=2.0, tokenizer=tokenizer)

    # Query 1: Vietnamese
    q1 = QueryItem(query_id=1, query_text="viêm gan B")
    hits_q1 = searcher.search_single_query(q1.query_id, q1.query_text, k=10)
    assert len(hits_q1) >= 1
    assert hits_q1[0].doc_id == 10001
    assert hits_q1[0].rank == 1
    assert hits_q1[0].bm25_score > 0

    # Query 2: Chinese CJK
    q2 = QueryItem(query_id=2, query_text="急性扁桃体炎 咽喉肿痛")
    hits_q2 = searcher.search_single_query(q2.query_id, q2.query_text, k=10)
    assert len(hits_q2) >= 1
    assert hits_q2[0].doc_id == 10004

    # Query 3: Short doc without title
    q3 = QueryItem(query_id=3, query_text="tai mũi họng chuyên khoa")
    hits_q3 = searcher.search_single_query(q3.query_id, q3.query_text, k=10)
    assert len(hits_q3) >= 1
    assert hits_q3[0].doc_id == 10002

    # Query 4: Drug Metformin 500mg
    q4 = QueryItem(query_id=4, query_text="Metformin 500mg")
    hits_q4 = searcher.search_single_query(q4.query_id, q4.query_text, k=10)
    assert len(hits_q4) >= 1
    assert hits_q4[0].doc_id == 10003

    # Batch search
    all_hits = searcher.search_queries([q1, q2, q3, q4], k=10)
    assert len(all_hits) >= 4

    # 4. Exporter & Union
    exporter = CandidateExporter(output_dir)
    hits_file = exporter.export_hits(all_hits, "test_hits.parquet")
    assert hits_file.exists()

    level_stats = exporter.compute_level_stats(all_hits, k_levels=[1, 2, 5])
    assert len(level_stats) == 3
    assert level_stats[0].k == 1

    union_file, union_count = exporter.generate_union_candidates(
        all_hits, k=2, filename="candidate_doc_ids.parquet"
    )
    assert union_file.exists()
    assert union_count == 4  # All 4 unique docs retrieved

    # Inspect union parquet content
    union_tbl = pq.read_table(union_file)
    assert union_tbl.num_rows == 4
    assert set(union_tbl.schema.names) == {
        "doc_id",
        "match_count",
        "best_rank",
        "best_score",
        "query_ids",
        "query_matches",
    }
    retrieved_doc_ids = union_tbl.column("doc_id").to_pylist()
    assert set(retrieved_doc_ids) == {10001, 10002, 10003, 10004}


def test_title_boosting_effect(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    index_dir = tmp_path / "index"

    shard_path = corpus_dir / "part-000000.parquet"
    data = {
        "doc_id": [20001, 20002],
        "title": [
            "Thuốc kháng sinh Amoxicillin",  # Amoxicillin in title
            "Thông tin y tế tổng hợp",
        ],
        "text": [
            "Bài viết hướng dẫn sử dụng các loại thuốc thông thường.",
            "Thuốc kháng sinh Amoxicillin được sử dụng phổ biến trong điều trị nhiễm khuẩn.",
        ],
        "crawl_status": ["SUCCESS", "SUCCESS"],
    }
    tbl = pa.Table.from_pydict(data)
    pq.write_table(tbl, shard_path)

    tokenizer = MultilingualMedicalTokenizer()
    indexer = BM25Indexer(index_dir=index_dir, tokenizer=tokenizer, heap_size_mb=64)
    indexer.build_index([corpus_dir], recreate=True)

    # Search with title_boost = 3.0
    searcher = BM25Searcher(index_dir=index_dir, title_boost=3.0, tokenizer=tokenizer)
    hits = searcher.search_single_query(1, "Amoxicillin", k=10)

    assert len(hits) == 2
    # Document with Amoxicillin in title should rank #1 due to boost
    assert hits[0].doc_id == 20001
    assert hits[0].bm25_score > hits[1].bm25_score
