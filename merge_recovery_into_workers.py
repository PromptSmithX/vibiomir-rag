from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
from pathlib import Path
import pyarrow.parquet as pq

SHARD_PATTERN = re.compile(r"part-(\d+)\.parquet")

def merge_recovery_for_worker(worker_id: int, worker_dir: Path, recovery_dir: Path) -> None:
    print(f"\n==========================================")
    print(f"Merging recovery data into Worker {worker_id}")
    print(f"Worker dir:   {worker_dir}")
    print(f"Recovery dir: {recovery_dir}")
    print(f"==========================================")

    if not recovery_dir.exists():
        print(f"[SKIP] Recovery dir {recovery_dir} does not exist.")
        return

    recovery_manifest_path = recovery_dir / "crawl_manifest.sqlite"
    if not recovery_manifest_path.exists():
        recovery_manifest_path = recovery_dir / "manifest.sqlite"
    if not recovery_manifest_path.exists():
        print(f"[SKIP] Manifest not found in {recovery_dir}")
        return

    worker_manifest_path = worker_dir / "manifest.sqlite"
    if not worker_manifest_path.exists():
        raise FileNotFoundError(f"Worker manifest not found: {worker_manifest_path}")

    worker_corpus_dir = worker_dir / "corpus"
    recovery_corpus_dir = recovery_dir / "corpus"

    # Find highest existing shard sequence in worker
    existing_shards = sorted(worker_corpus_dir.glob("part-*.parquet"))
    max_seq = -1
    for s in existing_shards:
        match = SHARD_PATTERN.fullmatch(s.name)
        if match:
            max_seq = max(max_seq, int(match.group(1)))

    print(f"Current highest shard sequence in worker {worker_id}: {max_seq}")

    w_conn = sqlite3.connect(worker_manifest_path)
    w_cur = w_conn.cursor()

    # Find how many recovery shards have already been merged
    w_cur.execute("SELECT COUNT(*) FROM shards WHERE temp_path = ''")
    already_merged_count = w_cur.fetchone()[0]

    recovery_shards = sorted(recovery_corpus_dir.glob("part-*.parquet"))
    new_shards_to_merge = recovery_shards[already_merged_count:]
    if not new_shards_to_merge:
        print(f"[INFO] No new recovery shards to merge for Worker {worker_id} (already merged {already_merged_count} shards).")
        w_conn.close()
        return

    print(f"Worker {worker_id}: Found {len(new_shards_to_merge)} new recovery shard(s) to merge (previously merged {already_merged_count}).")

    copied_shards = 0
    updated_tasks = 0

    for rec_shard in new_shards_to_merge:
        max_seq += 1
        new_shard_name = f"part-{max_seq:06d}.parquet"
        dest_shard_path = worker_corpus_dir / new_shard_name
        
        shutil.copy2(rec_shard, dest_shard_path)
        copied_shards += 1

        # Read parquet to inspect items
        table = pq.read_table(dest_shard_path)
        row_count = table.num_rows

        # Register shard in worker manifest
        w_cur.execute(
            """
            INSERT OR REPLACE INTO shards(shard_name, status, row_count, temp_path, final_path, created_at, committed_at)
            VALUES (?, 'COMMITTED', ?, '', ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (new_shard_name, row_count, str(dest_shard_path))
        )

        # Update tasks in manifest
        records = table.to_pylist()
        for rec in records:
            doc_id = rec["doc_id"]
            crawl_status = rec["crawl_status"]
            text_chars = rec.get("text_chars") or 0
            bytes_down = rec.get("bytes_downloaded") or 0
            http_status = rec.get("http_status")
            final_url = rec.get("final_url")
            content_type = rec.get("content_type")

            # Only overwrite if new status is SUCCESS, or if old status was not SUCCESS
            w_cur.execute(
                """
                UPDATE crawl_tasks
                SET status = ?,
                    output_shard = ?,
                    text_chars = ?,
                    bytes_downloaded = ?,
                    http_status = ?,
                    final_url = ?,
                    content_type = ?,
                    error_type = NULL,
                    error = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE doc_id = ? AND (? = 'SUCCESS' OR status != 'SUCCESS')
                """,
                (crawl_status, new_shard_name, text_chars, bytes_down, http_status, final_url, content_type, doc_id, crawl_status)
            )
            if w_cur.rowcount > 0:
                updated_tasks += 1

    w_conn.commit()
    w_conn.close()

    print(f"Done! Copied {copied_shards} new shards into {worker_corpus_dir}")
    print(f"Updated {updated_tasks} tasks in manifest.")

def main():
    parser = argparse.ArgumentParser(description="Merge recovery run shards into worker directory")
    parser.add_argument("--worker", type=int, choices=[3, 4], help="Worker ID (3 or 4). If not specified, merges both.")
    args = parser.parse_args()

    workers_to_process = [args.worker] if args.worker else [3, 4]

    for wid in workers_to_process:
        w_dir = Path(f"kaggle_outputs/work_{wid}/crawl_worker_{wid}").resolve()
        rec_dir = Path(f"data/crawl/runs/recovery-worker-{wid}").resolve()
        merge_recovery_for_worker(wid, w_dir, rec_dir)

if __name__ == "__main__":
    main()
