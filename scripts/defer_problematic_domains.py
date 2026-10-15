#!/usr/bin/env python3
"""
scripts/defer_problematic_domains.py
Temporarily defer problematic/slow domains (dantri, baonghean, pharmacity)
in the SQLite manifest so the crawler can run at full speed on the 287k clean URLs.
Can be reversed anytime by running with --restore.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

DEFAULT_DEFER_DOMAINS = [
    "dantri.com.vn",
    "baonghean.vn",
    "pmc-ecm-healthblog.beta.pharmacity.io",
    "nhathuoclongchau.com.vn",
]

def main():
    parser = argparse.ArgumentParser(description="Defer or restore problematic crawl domains")
    parser.add_argument("--run-name", default="run-vn", help="Run name (default: run-vn)")
    parser.add_argument("--restore", action="store_true", help="Restore deferred domains back to PENDING")
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DEFER_DOMAINS, help="Domains to defer/restore")
    args = parser.parse_args()

    manifest_path = Path(f"data/crawl/runs/{args.run_name}/crawl_manifest.sqlite")
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)

    con = sqlite3.connect(manifest_path)
    cur = con.cursor()

    if args.restore:
        print(f"Restoring domains: {args.domains}")
        for domain in args.domains:
            cur.execute(
                "UPDATE fetch_targets SET status='PENDING' WHERE domain=? AND status='DEFERRED'",
                (domain,),
            )
            cur.execute(
                "UPDATE crawl_tasks SET status='PENDING' WHERE domain=? AND status='DEFERRED'",
                (domain,),
            )
        con.commit()
        print("Restoration complete.")
    else:
        print(f"Deferring slow/problematic domains: {args.domains}")
        # 1. Defer specified domains
        for domain in args.domains:
            cur.execute(
                "UPDATE fetch_targets SET status='DEFERRED' WHERE domain=? AND status IN ('PENDING', 'FETCHING', 'RETRY_WAIT')",
                (domain,),
            )
            count_f = cur.rowcount
            cur.execute(
                "UPDATE crawl_tasks SET status='DEFERRED' WHERE domain=? AND status IN ('PENDING', 'FETCHING', 'RETRY_WAIT')",
                (domain,),
            )
            count_t = cur.rowcount
            print(f"  {domain}: deferred {count_t} tasks")

        # 2. Reset any interrupted FETCHING/RETRY_WAIT of other domains back to PENDING
        cur.execute(
            "UPDATE fetch_targets SET status='PENDING' WHERE status IN ('FETCHING', 'RETRY_WAIT')"
        )
        cur.execute(
            "UPDATE crawl_tasks SET status='PENDING' WHERE status IN ('FETCHING', 'RETRY_WAIT')"
        )
        reset_count = cur.rowcount
        con.commit()
        print(f"Reset {reset_count} active/interrupted tasks for remaining domains back to PENDING.")

    # Status summary
    print("\nCurrent Status Breakdown in Manifest:")
    cur.execute("SELECT status, count(*) FROM crawl_tasks GROUP BY status")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]:,} tasks")

    con.close()

if __name__ == "__main__":
    main()
