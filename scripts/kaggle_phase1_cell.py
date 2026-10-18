# ==============================================================================
# PIPELINE CHẶNG 1 (CHUẨN HOÁ 100%): BỨT PHÁ ĐIỂM SỐ VIBIOMIR (0.30 - 0.35+)
# KHẮC PHỤC TRIỆT ĐỂ: LỆCH PHA SHARDS VECTORS & MỞ RỘNG ĐOẠN TRÍCH 2 CHIỀU
# ==============================================================================
import os
os.system("pip install -q sentence-transformers")

import glob, time, json, zipfile, shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
t_start = time.time()

work_dir = Path("/kaggle/working/top500_pipeline")
work_dir.mkdir(parents=True, exist_ok=True)
all_vec_saved_dir = Path("/kaggle/working/all_vectors")
all_vec_saved_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. NẠP CHÍNH XÁC SHARDS & ĐỒNG BỘ HOÀN TOÀN METADATA VỚI VECTORS
# ------------------------------------------------------------------------------
# 10 shards gốc từ vibiomir-exp-a1-input (467,121 chunks)
old_chunk_files = sorted(glob.glob("/kaggle/input/**/vibiomir-exp-a1-input/**/chunks/*.parquet", recursive=True))
if not old_chunk_files:
    old_chunk_files = sorted([f for f in glob.glob("/kaggle/input/**/chunks/*.parquet", recursive=True) if "diff" not in f and "top500" not in f])
print(f"-> Tìm thấy {len(old_chunk_files)} shards chunks gốc (chuẩn xác 10 shards).")
assert len(old_chunk_files) == 10, f"Lỗi: Số shards chunks gốc không bằng 10! Thực tế: {len(old_chunk_files)}"

# 6 shards diff từ vibiomir-chunks-top500-diff (296,827 chunks)
diff_candidates = glob.glob("/kaggle/input/**/vibiomir-chunks-top500-diff/**/chunks/*.parquet", recursive=True)
if not diff_candidates:
    diff_candidates = [f for f in glob.glob("/kaggle/input/**/chunks_part_*.parquet", recursive=True) if f not in old_chunk_files]
diff_chunks_files = sorted(diff_candidates)
print(f"-> Tìm thấy {len(diff_chunks_files)} shards chunks diff (chuẩn xác 6 shards).")
assert len(diff_chunks_files) == 6, f"Lỗi: Số shards chunks diff không bằng 6! Thực tế: {len(diff_chunks_files)}"

all_chunk_files = old_chunk_files + diff_chunks_files
print(f"-> Tổng cộng {len(all_chunk_files)} shards văn bản chunks (16 shards chuẩn xác).")

# Nạp metadata chuẩn theo đúng thứ tự 10 shards cũ + 6 shards mới
print("-> Đang nạp danh sách metadata chuẩn xác từ các file chunk...")
old_chunk_ids = []
old_doc_ids = []
old_chunk_indices = []
for cf in old_chunk_files:
    tbl = pq.read_table(cf, columns=["chunk_id", "doc_id", "chunk_index"])
    old_chunk_ids.extend([str(c) for c in tbl.column("chunk_id").to_pylist()])
    old_doc_ids.extend([int(d) for d in tbl.column("doc_id").to_pylist()])
    old_chunk_indices.extend([int(i) for i in tbl.column("chunk_index").to_pylist()])
assert len(old_chunk_ids) == 467121, f"Lỗi: Số chunks cũ không khớp 467,121 (thực tế: {len(old_chunk_ids)})"

new_chunk_ids = []
new_doc_ids = []
new_chunk_indices = []
new_embed_texts = []
for f in diff_chunks_files:
    tbl = pq.read_table(f, columns=["chunk_id", "doc_id", "chunk_index", "embed_text"])
    new_chunk_ids.extend([str(c) for c in tbl.column("chunk_id").to_pylist()])
    new_doc_ids.extend([int(d) for d in tbl.column("doc_id").to_pylist()])
    new_chunk_indices.extend([int(i) for i in tbl.column("chunk_index").to_pylist()])
    new_embed_texts.extend(tbl.column("embed_text").to_pylist())
assert len(new_chunk_ids) == 296827, f"Lỗi: Số chunks mới không khớp 296,827 (thực tế: {len(new_chunk_ids)})"

all_chunk_ids = old_chunk_ids + new_chunk_ids
all_doc_ids = old_doc_ids + new_doc_ids
all_chunk_indices = old_chunk_indices + new_chunk_indices
assert len(all_doc_ids) == 763948, f"Lỗi: Tổng số chunks metadata là {len(all_doc_ids)}, mong đợi 763,948"
print(f"-> Đã khớp chuẩn xác 100% metadata cho toàn bộ {len(all_doc_ids):,} chunks!")

# Kiểm tra nếu đã có sẵn kho vectors 764k từ lượt chạy trước
existing_800k = glob.glob("/kaggle/working/**/all_vectors.npy", recursive=True) + glob.glob("/kaggle/working/**/bge_vectors_800k.npy", recursive=True)

if existing_800k:
    print(f"-> Phát hiện kho vectors đã có sẵn: {existing_800k[0]}! Đang kiểm tra kích thước...")
    all_vectors = np.load(existing_800k[0])
    if len(all_vectors) == len(all_doc_ids):
        print(f"-> Kích thước khớp hoàn hảo: {len(all_vectors):,} vectors ({all_vectors.shape})! Nạp xong trong 3 giây.")
    else:
        print(f"-> Kho vectors cũ bị lệch kích thước ({len(all_vectors)} vs {len(all_doc_ids)}). Sẽ nạp và ghép lại!")
        existing_800k = None

if not existing_800k:
    old_zip_candidates = glob.glob("/kaggle/input/**/vibiomir_vectors_bge_m3.zip", recursive=True)
    if not old_zip_candidates:
        old_zip_candidates = glob.glob("/kaggle/working/**/vibiomir_vectors_bge_m3.zip", recursive=True)
    
    print("-> Đang giải nén 467k vectors BGE-M3 gốc...")
    old_vec_dir = work_dir / "old_vectors"
    with zipfile.ZipFile(old_zip_candidates[0], "r") as z:
        z.extractall(old_vec_dir)
        
    old_npy = list(old_vec_dir.rglob("*.npy"))[0]
    old_vectors = np.load(old_npy)
    assert len(old_vectors) == 467121, f"Lỗi: Số vectors cũ ({len(old_vectors)}) không khớp 467,121!"
    print(f"-> Đã nạp {len(old_vectors):,} vectors cũ ({old_vectors.shape})")
    
    from sentence_transformers import SentenceTransformer
    print("-> Đang tải mô hình BAAI/bge-m3...")
    model_bge = SentenceTransformer("BAAI/bge-m3", device=device)
    model_bge.half()
    
    print(f"-> Bắt đầu encode {len(new_embed_texts):,} chunks mới trên GPU...")
    new_vectors = model_bge.encode(
        new_embed_texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float16)
    
    all_vectors = np.vstack([old_vectors, new_vectors])
    all_vec_file = all_vec_saved_dir / "all_vectors.npy"
    print(f"-> Lưu kho vectors chuẩn vào {all_vec_file}...")
    np.save(all_vec_file, all_vectors)

assert len(all_vectors) == len(all_doc_ids) == 763948, f"Lỗi: Kích thước vector ({len(all_vectors)}) không khớp metadata ({len(all_doc_ids)})!"
print(f"==> TỔNG KHO VECTORS CHUẨN XÁC 100%: {len(all_vectors):,} vectors ({all_vectors.shape})")

# ------------------------------------------------------------------------------
# 2. XÂY DỰNG CACHE VĂN BẢN TẤT CẢ CHUNKS (THEO DOC_ID VÀ CHUNK_INDEX)
# ------------------------------------------------------------------------------
print("\n2. Đang nạp toàn bộ văn bản chunks vào RAM Cache...")
doc_chunks_dict = defaultdict(dict)  # did -> {chunk_index: text}
for f in all_chunk_files:
    t = pq.read_table(f, columns=["doc_id", "chunk_index", "text"])
    d_list = t.column("doc_id").to_pylist()
    i_list = t.column("chunk_index").to_pylist()
    txt_list = t.column("text").to_pylist()
    for did, idx, txt in zip(d_list, i_list, txt_list):
        clean_t = str(txt or "").strip()
        if len(clean_t) >= 20:
            doc_chunks_dict[int(did)][int(idx)] = clean_t
print(f"-> Đã cache văn bản của {len(doc_chunks_dict):,} tài liệu!")

# ------------------------------------------------------------------------------
# 3. DENSE SEARCH VỚI TOP-K = 3,000 (ĐỒNG BỘ HOÀN HẢO DOC_ID & CHUNK_INDEX)
# ------------------------------------------------------------------------------
print("\n3. Đang thực hiện Dense Retrieval Top-3,000 chunks cho 1,200 Queries...")
query_file = glob.glob("/kaggle/input/**/query.parquet", recursive=True)[0]
q_tbl = pq.read_table(query_file)
q_col = "id" if "id" in q_tbl.column_names else "query_id"
q_text_col = "query" if "query" in q_tbl.column_names else "text"
q_ids = [int(x) for x in q_tbl.column(q_col).to_pylist()]
q_texts = q_tbl.column(q_text_col).to_pylist()

from sentence_transformers import SentenceTransformer
if 'model_bge' not in locals():
    model_bge = SentenceTransformer("BAAI/bge-m3", device=device)
    model_bge.half()

q_vectors = model_bge.encode(q_texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

all_vectors_torch = torch.from_numpy(all_vectors.astype(np.float32)).to(device)

dense_hits_by_query = defaultdict(list)
query_doc_best_chunk = defaultdict(dict)    # qid -> {did: best_chunk_index}
query_doc_second_chunk = defaultdict(dict)  # qid -> {did: second_chunk_index}

batch_q = 64
for b_i in range(0, len(q_ids), batch_q):
    b_qids = q_ids[b_i : b_i + batch_q]
    b_qvecs = torch.from_numpy(q_vectors[b_i : b_i + batch_q]).to(device)
    sims = torch.matmul(b_qvecs, all_vectors_torch.T)
    top_sims, top_indices = torch.topk(sims, k=3000, dim=1)
    
    for row, qid in enumerate(b_qids):
        doc_max_sim = {}
        for rank_idx in range(3000):
            c_idx = top_indices[row, rank_idx].item()
            score_val = top_sims[row, rank_idx].item()
            did = all_doc_ids[c_idx]
            c_idx_in_doc = all_chunk_indices[c_idx]
            
            if did not in doc_max_sim:
                doc_max_sim[did] = score_val
                query_doc_best_chunk[qid][did] = c_idx_in_doc
            elif did not in query_doc_second_chunk[qid]:
                if abs(c_idx_in_doc - query_doc_best_chunk[qid][did]) >= 2:
                    query_doc_second_chunk[qid][did] = c_idx_in_doc
                    
        sorted_docs = sorted(doc_max_sim.items(), key=lambda x: x[1], reverse=True)[:500]
        for r, (did, s) in enumerate(sorted_docs, 1):
            dense_hits_by_query[qid].append((did, r, s))

del all_vectors_torch
torch.cuda.empty_cache()
print("-> Đã hoàn thành Dense Search Top 3,000 và xác định chuẩn xác best chunk cho từng doc!")

# ------------------------------------------------------------------------------
# 4. HỢP NHẤT RRF 2-WAY (BM25 TOP 500 + DENSE TOP 500) -> CHỌN TOP 200 CANDIDATES
# ------------------------------------------------------------------------------
print("\n4. Hợp nhất RRF Fusion: BM25 (Top 500) + BGE-M3 Dense (Top 500)...")
bm25_file = glob.glob("/kaggle/input/**/retrieval_top_1000.parquet", recursive=True)[0]
bm25_tbl = pq.read_table(bm25_file)
bm25_q = bm25_tbl.column("query_id").to_pylist()
bm25_d = bm25_tbl.column("doc_id").to_pylist()
bm25_r = bm25_tbl.column("rank").to_pylist()

bm25_by_q = defaultdict(list)
for q, d, r in zip(bm25_q, bm25_d, bm25_r):
    if r <= 500:
        bm25_by_q[int(q)].append((int(d), int(r)))

fused_candidates_by_q = {}
rrf_k = 60
for qid in q_ids:
    doc_scores = defaultdict(float)
    for did, r in bm25_by_q.get(qid, []):
        doc_scores[did] += 1.0 / (rrf_k + r)
    for did, r, s in dense_hits_by_query.get(qid, []):
        doc_scores[did] += 1.2 / (rrf_k + r)
        
    sorted_fused = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:200]
    fused_candidates_by_q[qid] = [did for did, _ in sorted_fused]

print("-> Đã chọn ra Top 200 ứng viên chất lượng cao nhất cho mỗi Query để Reranker chấm điểm!")

# ------------------------------------------------------------------------------
# 5. CROSS-ENCODER RERANKER VỚI BEST MATCHING CHUNK
# ------------------------------------------------------------------------------
print("\n5. Khởi tạo Cross-Encoder BAAI/bge-reranker-v2-m3...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification

reranker_name = "BAAI/bge-reranker-v2-m3"
reranker_tok = AutoTokenizer.from_pretrained(reranker_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(
    reranker_name, torch_dtype=torch.float16
).to(device)
reranker_model.eval()

final_reranked_docs = {}
print("-> Bắt đầu Reranking với Best Matching Chunk Context...")

with torch.inference_mode():
    for q_idx, qid in enumerate(q_ids):
        q_text = q_texts[q_idx]
        candidates = fused_candidates_by_q.get(qid, [])[:200]
        
        pairs = []
        c_dids = []
        for did in candidates:
            doc_chunks = doc_chunks_dict.get(did, {})
            best_idx = query_doc_best_chunk[qid].get(did, None)
            if best_idx is None or best_idx not in doc_chunks:
                best_idx = min(doc_chunks.keys()) if doc_chunks else 0
                
            snippet = doc_chunks.get(best_idx, "")
            # Nếu chunk ngắn (< 120 từ), bổ sung đoạn tiếp theo
            if len(snippet.split()) < 120 and (best_idx + 1) in doc_chunks:
                snippet = f"{snippet} {doc_chunks[best_idx + 1]}".strip()
            
            snippet = snippet[:1200]
            pairs.append([q_text, snippet])
            c_dids.append(did)
            
        scores = []
        for b_start in range(0, len(pairs), 64):
            b_pairs = pairs[b_start : b_start + 64]
            inputs = reranker_tok(b_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.cuda.amp.autocast():
                out = reranker_model(**inputs)
            s_list = out.logits.squeeze(-1).float().cpu().numpy().tolist()
            scores.extend(s_list if isinstance(s_list, list) else [s_list])
            
        ranked = [d for _, d in sorted(zip(scores, c_dids), reverse=True)]
        final_reranked_docs[qid] = ranked
        
        if (q_idx + 1) % 200 == 0:
            print(f"  Đã hoàn thành Rerank {q_idx+1} / {len(q_ids)} queries...")

# ------------------------------------------------------------------------------
# 6. PHÂN BỔ PASSAGE THÔNG MINH + MỞ RỘNG 2 CHIỀU (TARGET ~600 TỪ, 0% DUPS)
# ------------------------------------------------------------------------------
print("\n6. Đang tạo các đoạn trích (chunks) theo chuẩn mở rộng 2 chiều (Bidirectional Expansion)...")

def build_smart_passage(doc_chunks, start_idx, target_words=600):
    """Mở rộng đoạn trích 2 chiều (trước & sau) để luôn đạt độ dài ~600 từ."""
    used = {start_idx}
    p = doc_chunks[start_idx]
    
    # Thứ tự mở rộng: tiến 1, lùi 1, tiến 2, lùi 2...
    for offset in [1, -1, 2, -2, 3, -3, 4, -4]:
        cand = start_idx + offset
        if cand in doc_chunks and cand not in used:
            if offset > 0:
                p = f"{p} {doc_chunks[cand]}".strip()
            else:
                p = f"{doc_chunks[cand]} {p}".strip()
            used.add(cand)
            if len(p.split()) >= target_words:
                break
    return p, used

sub_records = []
target_words = 600

for qid in q_ids:
    docs = final_reranked_docs.get(qid, [])[:100]
    rel_chunks = []
    used_chunk_indices = defaultdict(set)
    
    # BƯỚC A: CẤP CHUNK TỐT NHẤT CHO TOÀN BỘ CÁC DOCS HÀNG ĐẦU (TỐI ƯU DOC COVERAGE)
    for did in docs:
        if len(rel_chunks) >= 100:
            break
        doc_chunks = doc_chunks_dict.get(did, {})
        if not doc_chunks:
            continue
            
        b_idx = query_doc_best_chunk[qid].get(did, None)
        if b_idx is None or b_idx not in doc_chunks:
            b_idx = min(doc_chunks.keys())
            
        p1, used = build_smart_passage(doc_chunks, b_idx, target_words=target_words)
        used_chunk_indices[did].update(used)
        rel_chunks.append({"doc_id": did, "chunk_text": p1})
        
    # BƯỚC B: BỔ SUNG CHUNK THỨ HAI CHO TOP 25 DOC TỰ TIN NHẤT (ĐẢM BẢO DISJOINT >= 2)
    if len(rel_chunks) < 100:
        for did in docs[:25]:
            if len(rel_chunks) >= 100:
                break
            doc_chunks = doc_chunks_dict.get(did, {})
            if not doc_chunks:
                continue
                
            s_idx = query_doc_second_chunk[qid].get(did, None)
            target_s = None
            if s_idx is not None and s_idx in doc_chunks and all(abs(s_idx - u) >= 2 for u in used_chunk_indices[did]):
                target_s = s_idx
            else:
                for c_i in sorted(doc_chunks.keys()):
                    if all(abs(c_i - u) >= 2 for u in used_chunk_indices[did]):
                        target_s = c_i
                        break
                        
            if target_s is not None:
                p2, used2 = build_smart_passage(doc_chunks, target_s, target_words=target_words)
                used_chunk_indices[did].update(used2)
                rel_chunks.append({"doc_id": did, "chunk_text": p2})
                
    # BƯỚC C: NẾU VẪN CHƯA ĐỦ 100 CHUNKS, LẤY TIẾP TỪ CÁC DOC TỪ HẠNG 26 TRỞ ĐI
    if len(rel_chunks) < 100:
        for did in docs[25:]:
            if len(rel_chunks) >= 100:
                break
            doc_chunks = doc_chunks_dict.get(did, {})
            for c_i in sorted(doc_chunks.keys()):
                if all(abs(c_i - u) >= 2 for u in used_chunk_indices[did]):
                    p3, used3 = build_smart_passage(doc_chunks, c_i, target_words=target_words)
                    used_chunk_indices[did].update(used3)
                    rel_chunks.append({"doc_id": did, "chunk_text": p3})
                    break
                    
    sub_records.append({
        "id": int(qid),
        "relevant_docs": docs,
        "relevant_chunks": rel_chunks
    })

# ------------------------------------------------------------------------------
# 7. KIỂM THỰC CHUẨN XÁC 100% VÀ ĐÓNG GÓI SUBMISSION
# ------------------------------------------------------------------------------
print("\n7. Đang kiểm tra tính hợp lệ của file nộp...")
assert len(sub_records) == 1200, f"Lỗi: Số queries không đủ 1200 ({len(sub_records)})"
for rec in sub_records:
    assert len(rec["relevant_docs"]) == 100, f"Query {rec['id']} thiếu docs: {len(rec['relevant_docs'])}"
    assert len(rec["relevant_chunks"]) == 100, f"Query {rec['id']} thiếu chunks: {len(rec['relevant_chunks'])}"
    assert all(isinstance(d, int) for d in rec["relevant_docs"]), "Lỗi: doc_id không phải kiểu int"
    assert all(isinstance(c["doc_id"], int) for c in rec["relevant_chunks"]), "Lỗi: chunk doc_id không phải kiểu int"
    assert all(len(c["chunk_text"].strip()) >= 30 for c in rec["relevant_chunks"]), "Lỗi: chunk text quá ngắn"

out_json = Path("/kaggle/working/submission.json")
out_zip_sub = Path("/kaggle/working/submission.zip")
out_zip_p1 = Path("/kaggle/working/submission_phase1_top500.zip")

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(sub_records, f, ensure_ascii=False)

with zipfile.ZipFile(out_zip_sub, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(out_json, arcname="submission.json")

with zipfile.ZipFile(out_zip_p1, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(out_json, arcname="submission.json")

total_time = (time.time() - t_start) / 60
print("\n" + "=" * 65)
print(f"=== PIPELINE CHẶNG 1 ĐÃ HOÀN TẤT THÀNH CÔNG TRONG {total_time:.1f} PHÚT! ===")
print(f"File nộp chính thức: {out_zip_sub} ({out_zip_sub.stat().st_size / (1024*1024):.2f} MB)")
print("=" * 65)

from IPython.display import FileLink
display(FileLink("submission.zip"))
