"""Inspect real query text and compare top-10 retrieval results across BM25, Dense, and Fused."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from retrieval.searcher import BM25Searcher
from retrieval.submission.chunk_store import ChunkStore


def inspect_queries(
    query_ids: list[int],
    query_path: str = "data/source/query.parquet",
    fusion_path: str = "data/experiments/exp_a1/fusion/rrf_fusion_top_1000.parquet",
    dense_path: str = "data/experiments/exp_a1/retrieval/dense_retrieval_top_1000.parquet",
    bm25_path: str = "data/retrieval/candidates/retrieval_top_1000.parquet",
    chunks_dir: str = "data/experiments/exp_a1/chunks",
    submission_path: str = "data/experiments/exp_a1/submission/submission.json",
    top_n: int = 10,
) -> None:
    q_file = Path(query_path)
    if not q_file.exists():
        print(f"[ERROR] Query file not found: {q_file}")
        return

    # 1. Load official queries
    all_queries = BM25Searcher.load_queries_from_parquet(q_file)
    query_map = {q.query_id: q.query_text for q in all_queries}

    # 2. Check available result files
    f_file = Path(fusion_path)
    d_file = Path(dense_path)
    b_file = Path(bm25_path)
    s_file = Path(submission_path)
    c_dir = Path(chunks_dir)

    has_fusion = f_file.exists()
    has_dense = d_file.exists()
    has_bm25 = b_file.exists()
    has_submission = s_file.exists()

    # Preload chunk store if chunks exist
    chunk_store = None
    if c_dir.exists():
        chunk_store = ChunkStore(chunks_dir=c_dir)

    # 3. Load retrieval data for target queries
    fusion_by_q: dict[int, list[dict]] = {}
    if has_fusion:
        tbl = pq.read_table(f_file)
        qids = tbl.column("query_id").to_pylist()
        dids = tbl.column("doc_id").to_pylist()
        ranks = tbl.column("final_rank").to_pylist()
        scores = tbl.column("fused_score").to_pylist()
        b_ranks = tbl.column("bm25_rank").to_pylist()
        d_ranks = tbl.column("dense_rank").to_pylist()
        c_ids = tbl.column("dense_best_chunk_id").to_pylist()

        for q, d, r, s, br, dr, cid in zip(qids, dids, ranks, scores, b_ranks, d_ranks, c_ids, strict=False):
            if q in query_ids:
                if q not in fusion_by_q:
                    fusion_by_q[q] = []
                fusion_by_q[q].append(
                    {
                        "doc_id": d,
                        "rank": r,
                        "fused_score": s,
                        "bm25_rank": br,
                        "dense_rank": dr,
                        "chunk_id": cid,
                    }
                )

    bm25_by_q: dict[int, list[dict]] = {}
    if has_bm25:
        tbl = pq.read_table(b_file)
        qids = tbl.column("query_id").to_pylist()
        dids = tbl.column("doc_id").to_pylist()
        ranks = tbl.column("rank").to_pylist()
        scores = tbl.column("bm25_score").to_pylist()

        for q, d, r, s in zip(qids, dids, ranks, scores, strict=False):
            if q in query_ids:
                if q not in bm25_by_q:
                    bm25_by_q[q] = []
                bm25_by_q[q].append({"doc_id": d, "rank": r, "bm25_score": s})

    dense_by_q: dict[int, list[dict]] = {}
    if has_dense:
        tbl = pq.read_table(d_file)
        qids = tbl.column("query_id").to_pylist()
        dids = tbl.column("doc_id").to_pylist()
        ranks = tbl.column("rank").to_pylist()
        scores = tbl.column("score").to_pylist()
        c_ids = tbl.column("best_chunk_id").to_pylist()

        for q, d, r, s, cid in zip(qids, dids, ranks, scores, c_ids, strict=False):
            if q in query_ids:
                if q not in dense_by_q:
                    dense_by_q[q] = []
                dense_by_q[q].append({"doc_id": d, "rank": r, "dense_score": s, "chunk_id": cid})

    sub_by_q: dict[int, dict] = {}
    if has_submission:
        try:
            with open(s_file, encoding="utf-8") as f:
                s_data = json.load(f)
            for item in s_data:
                if item.get("id") in query_ids:
                    sub_by_q[item["id"]] = item
        except Exception:
            pass

    # Collect needed chunk IDs to preload
    if chunk_store:
        needed_cids: set[str] = set()
        needed_dids: set[int] = set()
        for q in query_ids:
            for item in fusion_by_q.get(q, [])[:top_n]:
                needed_dids.add(item["doc_id"])
                if item["chunk_id"]:
                    needed_cids.add(item["chunk_id"])
            for item in dense_by_q.get(q, [])[:top_n]:
                needed_dids.add(item["doc_id"])
                if item["chunk_id"]:
                    needed_cids.add(item["chunk_id"])
        chunk_store.preload_for_retrieval(needed_cids, needed_dids)

    # 4. Render reports
    print("\n" + "=" * 80)
    print("           BẢNG SO SÁNH TRUY VẤN VÀ KẾT QUẢ TÌM KIẾM VIBIOMIR")
    print("=" * 80)

    for q_id in query_ids:
        q_text = query_map.get(q_id, "<Không tìm thấy trong query.parquet>")
        print(f"\n{'─' * 80}")
        print(f"📌 QUERY ID: {q_id}")
        print(f"📝 CÂU HỎI THẬT: \"{q_text}\"")
        print(f"{'─' * 80}")

        # If Fused results available (Experiment A final output)
        if has_fusion and q_id in fusion_by_q:
            f_hits = sorted(fusion_by_q[q_id], key=lambda x: x["rank"])[:top_n]
            print(f"\n[BẢNG KẾT QUẢ FUSION (BM25 + DENSE RRF) - TOP {len(f_hits)}]")
            print(f"{'Rank':<5} | {'Doc ID':<10} | {'RRF Score':<11} | {'BM25 Rank':<10} | {'Dense Rank':<10} | {'Nội dung Chunk trích xuất'}")
            print("-" * 80)
            for h in f_hits:
                bm25_r_str = str(h["bm25_rank"]) if h["bm25_rank"] != -1 else "N/A"
                dense_r_str = str(h["dense_rank"]) if h["dense_rank"] != -1 else "N/A"
                chunk_snippet = ""
                if chunk_store:
                    cid = h["chunk_id"]
                    t = chunk_store.get_chunk_text(cid) if cid else None
                    if not t:
                        fb = chunk_store.get_fallback_chunk_for_doc(h["doc_id"])
                        if fb:
                            t = fb[1]
                    if t:
                        chunk_snippet = t.replace("\n", " ").strip()
                        if len(chunk_snippet) > 80:
                            chunk_snippet = chunk_snippet[:77] + "..."

                print(f"{h['rank']:<5} | {h['doc_id']:<10} | {h['fused_score']:<11.6f} | {bm25_r_str:<10} | {dense_r_str:<10} | {chunk_snippet}")

        # If Dense results available
        elif has_dense and q_id in dense_by_q:
            d_hits = sorted(dense_by_q[q_id], key=lambda x: x["rank"])[:top_n]
            print(f"\n[KẾT QUẢ DENSE RETRIEVAL - TOP {len(d_hits)}]")
            print(f"{'Rank':<5} | {'Doc ID':<10} | {'Dense Cosine':<13} | {'Best Chunk ID':<18} | {'Nội dung Chunk'}")
            print("-" * 80)
            for h in d_hits:
                chunk_snippet = ""
                if chunk_store and h["chunk_id"]:
                    t = chunk_store.get_chunk_text(h["chunk_id"])
                    if t:
                        chunk_snippet = t.replace("\n", " ").strip()
                        if len(chunk_snippet) > 60:
                            chunk_snippet = chunk_snippet[:57] + "..."
                print(f"{h['rank']:<5} | {h['doc_id']:<10} | {h['dense_score']:<13.4f} | {h['chunk_id']:<18} | {chunk_snippet}")

        # Fallback to BM25 if neither dense nor fusion run yet
        elif has_bm25 and q_id in bm25_by_q:
            b_hits = sorted(bm25_by_q[q_id], key=lambda x: x["rank"])[:top_n]
            print(f"\n[KẾT QUẢ BM25 RETRIEVAL - TOP {len(b_hits)}]")
            print(f"{'Rank':<5} | {'Doc ID':<10} | {'BM25 Score':<12}")
            print("-" * 35)
            for h in b_hits:
                print(f"{h['rank']:<5} | {h['doc_id']:<10} | {h['bm25_score']:<12.2f}")
            print("\n* Chú ý: Chưa tìm thấy kết quả Dense/Fusion. Hãy chạy 'dense-search' và 'fuse' để xem so sánh đầy đủ.")

        else:
            print("\n[CHƯA CÓ KẾT QUẢ TÌM KIẾM]")
            print(f"Query {q_id} chưa có kết quả trong các file retrieval.")

        # If submission JSON exists, display its relevant_docs
        if q_id in sub_by_q:
            s_item = sub_by_q[q_id]
            print("\n[OFFICIAL SUBMISSION EXPORT]")
            print(f"  - relevant_docs ({len(s_item.get('relevant_docs', []))} docs): {s_item.get('relevant_docs', [])[:10]}")
            chunks = s_item.get("relevant_chunks", [])
            print(f"  - relevant_chunks ({len(chunks)} chunks):")
            for c_i, c in enumerate(chunks[:3], 1):
                txt = c.get("chunk_text", "").replace("\n", " ")
                if len(txt) > 90:
                    txt = txt[:87] + "..."
                print(f"    {c_i}. [Doc {c.get('doc_id')}]: {txt}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect queries and retrieval results")
    parser.add_argument("--query-ids", nargs="+", type=int, default=[1, 4, 7, 250], help="Query IDs to inspect")
    parser.add_argument("--top-n", type=int, default=10, help="Number of results per query")
    parser.add_argument("--query-path", default="data/source/query.parquet")
    parser.add_argument("--fusion-path", default="data/experiments/exp_a1/fusion/rrf_fusion_top_1000.parquet")
    parser.add_argument("--dense-path", default="data/experiments/exp_a1/retrieval/dense_retrieval_top_1000.parquet")
    parser.add_argument("--bm25-path", default="data/retrieval/candidates/retrieval_top_1000.parquet")
    parser.add_argument("--chunks-dir", default="data/experiments/exp_a1/chunks")
    parser.add_argument("--submission-path", default="data/experiments/exp_a1/submission/submission.json")
    args = parser.parse_args()

    inspect_queries(
        query_ids=args.query_ids,
        query_path=args.query_path,
        fusion_path=args.fusion_path,
        dense_path=args.dense_path,
        bm25_path=args.bm25_path,
        chunks_dir=args.chunks_dir,
        submission_path=args.submission_path,
        top_n=args.top_n,
    )
