from __future__ import annotations

import sqlite3
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    pa.field("doc_id", pa.int64(), nullable=False),
    pa.field("url", pa.string(), nullable=False),
    pa.field("domain", pa.string(), nullable=False),
])

def extract_targets_from_db(db_path: str, is_worker_3: bool) -> list[dict]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    
    if is_worker_3:
        query = """
            SELECT doc_id, url, domain
            FROM crawl_tasks
            WHERE domain IN ('vov.vn', 'baolangson.vn', 'article.iiyi.com', 'baothanhhoa.vn', 'wei.39.net', 'gk.39.net')
               OR status IN ('FAILED', 'EMPTY_CONTENT', 'NEEDS_JS')
               OR (domain = 'thanhnien.vn' AND text_chars < 250)
               OR (domain = 'baoquangtri.vn' AND text_chars < 250)
            ORDER BY doc_id
        """
    else:
        query = """
            SELECT doc_id, url, domain
            FROM crawl_tasks
            WHERE domain IN ('baohaiphong.vn', 'www.qdnd.vn', 'fk.39.net', 'cancer.39.net', 'shen.39.net', 'gc.39.net')
               OR status IN ('FAILED', 'EMPTY_CONTENT', 'NEEDS_JS')
               OR (domain = 'thanhnien.vn' AND text_chars < 250)
               OR (domain = 'baoquangtri.vn' AND text_chars < 250)
            ORDER BY doc_id
        """
    cur.execute(query)
    rows = cur.fetchall()
    conn.close()
    return [{"doc_id": r[0], "url": r[1], "domain": r[2]} for r in rows]

def main():
    w3_db = Path("kaggle_outputs/work_3/crawl_worker_3/manifest.sqlite").resolve()
    w4_db = Path("kaggle_outputs/work_4/crawl_worker_4/manifest.sqlite").resolve()
    
    out_dir = Path("data/crawl/recovery").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading targets from Worker 3: {w3_db}")
    w3_rows = extract_targets_from_db(str(w3_db), is_worker_3=True)
    print(f"  Worker 3 targets found: {len(w3_rows)}")
    
    print(f"Reading targets from Worker 4: {w4_db}")
    w4_rows = extract_targets_from_db(str(w4_db), is_worker_3=False)
    print(f"  Worker 4 targets found: {len(w4_rows)}")
    
    # Save Worker 3 targets
    w3_path = out_dir / "recovery_worker_3.parquet"
    w3_table = pa.Table.from_pylist(w3_rows, schema=SCHEMA)
    pq.write_table(w3_table, w3_path)
    print(f"Saved: {w3_path} ({len(w3_rows)} rows)")
    
    # Save Worker 4 targets
    w4_path = out_dir / "recovery_worker_4.parquet"
    w4_table = pa.Table.from_pylist(w4_rows, schema=SCHEMA)
    pq.write_table(w4_table, w4_path)
    print(f"Saved: {w4_path} ({len(w4_rows)} rows)")
    
    # Save combined targets
    all_path = out_dir / "recovery_all.parquet"
    all_rows = w3_rows + w4_rows
    all_table = pa.Table.from_pylist(all_rows, schema=SCHEMA)
    pq.write_table(all_table, all_path)
    print(f"Saved: {all_path} ({len(all_rows)} rows)")
    print("\nTarget extraction completed successfully!")

if __name__ == "__main__":
    main()
