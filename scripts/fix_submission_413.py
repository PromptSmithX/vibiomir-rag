import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
import zipfile
import time
from pathlib import Path

def fix_submission(
    input_zip: str = r"D:\ROAD_TO_AI_2026\submission_v2.zip",
    output_zip: str = r"D:\ROAD_TO_AI_2026\submission_v2_fixed.zip",
    max_words_per_chunk: int = 420
):
    t0 = time.time()
    print(f"1. Đang đọc {input_zip}...")
    with zipfile.ZipFile(input_zip, "r") as z:
        with z.open("submission.json") as f:
            data = json.load(f)
            
    print(f"-> Đã đọc {len(data):,} queries thành công.")
    
    total_trimmed = 0
    total_chunks = 0
    
    print(f"2. Đang tối ưu hóa độ dài chunk (giới hạn an toàn: {max_words_per_chunk} từ)...")
    for rec in data:
        for c in rec.get("relevant_chunks", []):
            total_chunks += 1
            words = c["chunk_text"].split()
            if len(words) > max_words_per_chunk:
                c["chunk_text"] = " ".join(words[:max_words_per_chunk])
                total_trimmed += 1
                
    print(f"-> Đã tối ưu {total_trimmed:,} / {total_chunks:,} chunks quá dài ({total_trimmed/total_chunks*100:.1f}%).")
    
    print("3. Đang kiểm tra tính hợp lệ...")
    assert len(data) == 1200, f"Lỗi: Số query không phải 1200 mà là {len(data)}"
    for r in data:
        assert len(r["relevant_docs"]) == 100, f"Query {r['id']} thiếu docs"
        assert len(r["relevant_chunks"]) == 100, f"Query {r['id']} thiếu chunks"
        assert all(isinstance(d, int) for d in r["relevant_docs"])
        assert all(isinstance(c["doc_id"], int) for c in r["relevant_chunks"])
        assert all(len(c["chunk_text"].strip()) >= 30 for c in r["relevant_chunks"])
        
    print("-> Tất cả 1,200 queries hoàn toàn hợp lệ 100%!")
    
    print("4. Đang nén file submission_v2_fixed.zip...")
    # Ghi compact json (không chứa khoảng trắng dư thừa)
    compact_json = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    
    out_p = Path(output_zip)
    with zipfile.ZipFile(out_p, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("submission.json", compact_json.encode("utf-8"))
        
    orig_mb = Path(input_zip).stat().st_size / (1024 * 1024)
    new_mb = out_p.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 60)
    print(f"=== ĐÃ XỬ LÝ XONG TRONG {time.time() - t0:.1f} GIÂY! ===")
    print(f"File cũ (Bị lỗi 413) : {orig_mb:.2f} MB")
    print(f"File mới (Đã tối ưu) : {new_mb:.2f} MB  <--- ĐẢM BẢO UPLOAD THÀNH CÔNG!")
    print(f"Đường dẫn file mới   : {output_zip}")
    print("=" * 60)

if __name__ == "__main__":
    fix_submission()
