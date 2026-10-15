#!/usr/bin/env python3
"""
scripts/recover_zh.py
Prepares run-zh to recover ~12,800 tasks from *.39.net by:
1. Deferring hard-blocked domains (familydoctor, 120ask, zysjonline, iiyi)
2. Resetting recoverable 39.net tasks (NEEDS_JS, EMPTY_CONTENT, etc.) back to PENDING.
3. Keeping existing SUCCESS tasks (7,799) intact.
"""

import sqlite3
from pathlib import Path

MANIFEST_PATH = Path("data/crawl/runs/run-zh/crawl_manifest.sqlite")

UNRECOVERABLE_DOMAINS = [
    "www.familydoctor.com.cn",
    "www.120ask.com",
    "zysjonline.com",
    "article.iiyi.com",
    "bingli.iiyi.com",
    "www.msdmanuals.cn",
    "www.youlai.cn",
    "www.cnkang.com",
]

def main():
    if not MANIFEST_PATH.exists():
        print(f"Error: {MANIFEST_PATH} not found.")
        return

    con = sqlite3.connect(MANIFEST_PATH)
    cur = con.cursor()

    print("=== Initial Status Breakdown in run-zh ===")
    for row in cur.execute("SELECT status, count(*) FROM crawl_tasks GROUP BY status").fetchall():
        print(f"  {row[0]}: {row[1]:,}")

    # 1. Defer unrecoverable domains
    for domain in UNRECOVERABLE_DOMAINS:
        cur.execute(
            "UPDATE fetch_targets SET status='DEFERRED' WHERE domain=? AND status != 'SUCCESS'",
            (domain,),
        )
        cur.execute(
            "UPDATE crawl_tasks SET status='DEFERRED' WHERE domain=? AND status != 'SUCCESS'",
            (domain,),
        )

    # 2. Reset recoverable 39.net tasks to PENDING
    cur.execute(
        """
        UPDATE fetch_targets
        SET status='PENDING', attempts=0, next_retry_at=NULL, error_type=NULL, error=NULL
        WHERE domain LIKE '%.39.net' AND status != 'SUCCESS'
        """
    )
    cur.execute(
        """
        UPDATE crawl_tasks
        SET status='PENDING', attempts=0, error_type=NULL, error=NULL
        WHERE domain LIKE '%.39.net' AND status != 'SUCCESS'
        """
    )
    recovered_count = cur.rowcount

    con.commit()

    print(f"\nReset {recovered_count:,} tasks for 39.net to PENDING!")
    print("\n=== Updated Status Breakdown in run-zh ===")
    for row in cur.execute("SELECT status, count(*) FROM crawl_tasks GROUP BY status").fetchall():
        print(f"  {row[0]}: {row[1]:,}")

    con.close()

if __name__ == "__main__":
    main()
