from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Reset failed/poisoned crawl tasks in SQLite manifest for recovery."
    )
    parser.add_argument(
        "--manifest",
        default="data/crawl/runs/worker-0-local/crawl_manifest.sqlite",
        help="Path to crawl_manifest.sqlite",
    )
    parser.add_argument(
        "--include-network-errors",
        action="store_true",
        help="Also reset transient NETWORK_ERROR tasks (vietnamnet, longchau, vov2, etc.)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f"Error: Manifest file not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    lock_path = manifest_path.with_suffix(".lock")
    if lock_path.exists():
        try:
            with lock_path.open("a+b") as handle:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            print(
                "ERROR: Crawler process is currently holding the manifest lock!\n"
                "Please stop or wait for the active crawler to finish before running recovery.",
                file=sys.stderr,
            )
            sys.exit(2)

    conn = sqlite3.connect(manifest_path)
    cursor = conn.cursor()

    now = utc_now()
    reset_summary: dict[str, int] = {}

    with conn:
        # 1. woman.39.net and pf.39.net: 2287-byte captcha pages
        for domain in ("woman.39.net", "pf.39.net"):
            cursor.execute(
                """
                SELECT fetch_key FROM crawl_tasks
                WHERE domain = ? AND (
                    (status = 'EMPTY_CONTENT' AND bytes_downloaded = 2287)
                    OR (status = 'NEEDS_JS' AND error_type = 'BOT_CHALLENGE')
                )
                """,
                (domain,),
            )
            keys = [r[0] for r in cursor.fetchall()]
            if keys:
                cursor.execute(
                    """
                    UPDATE crawl_tasks
                    SET status = 'PENDING', attempts = 0, error_type = NULL,
                        error = NULL, updated_at = ?
                    WHERE domain = ? AND (
                        (status = 'EMPTY_CONTENT' AND bytes_downloaded = 2287)
                        OR (status = 'NEEDS_JS' AND error_type = 'BOT_CHALLENGE')
                    )
                    """,
                    (now, domain),
                )
                cursor.executemany(
                    """
                    UPDATE fetch_targets
                    SET status = 'PENDING', attempts = 0, error_type = NULL,
                        error = NULL, updated_at = ?
                    WHERE fetch_key = ?
                    """,
                    [(now, k) for k in keys],
                )
            reset_summary[domain] = len(keys)

        # 2. ask.39.net: 2287-byte backlog
        cursor.execute(
            """
            SELECT fetch_key FROM crawl_tasks
            WHERE domain = 'ask.39.net' AND status = 'EMPTY_CONTENT' AND bytes_downloaded = 2287
            """
        )
        ask_keys = [r[0] for r in cursor.fetchall()]
        if ask_keys:
            cursor.execute(
                """
                UPDATE crawl_tasks
                SET status = 'PENDING', attempts = 0, error_type = NULL,
                    error = NULL, updated_at = ?
                WHERE domain = 'ask.39.net' AND status = 'EMPTY_CONTENT'
                  AND bytes_downloaded = 2287
                """,
                (now,),
            )
            cursor.executemany(
                """
                UPDATE fetch_targets
                SET status = 'PENDING', attempts = 0, error_type = NULL,
                    error = NULL, updated_at = ?
                WHERE fetch_key = ?
                """,
                [(now, k) for k in ask_keys],
            )
        reset_summary["ask.39.net (2287b backlog)"] = len(ask_keys)

        # 3. hanoimoi.vn: HTTP 403
        cursor.execute(
            "SELECT fetch_key FROM crawl_tasks WHERE domain = 'hanoimoi.vn' AND status = 'FAILED'"
        )
        hn_keys = [r[0] for r in cursor.fetchall()]
        if hn_keys:
            cursor.execute(
                """
                UPDATE crawl_tasks
                SET status = 'PENDING', attempts = 0, error_type = NULL,
                    error = NULL, updated_at = ?
                WHERE domain = 'hanoimoi.vn' AND status = 'FAILED'
                """,
                (now,),
            )
            cursor.executemany(
                """
                UPDATE fetch_targets
                SET status = 'PENDING', attempts = 0, error_type = NULL,
                    error = NULL, updated_at = ?
                WHERE fetch_key = ?
                """,
                [(now, k) for k in hn_keys],
            )
        reset_summary["hanoimoi.vn (HTTP 403)"] = len(hn_keys)

        # 4. www.baidu.com: ROBOTS_DENIED
        cursor.execute(
            """
            SELECT fetch_key FROM crawl_tasks
            WHERE domain = 'www.baidu.com' AND status = 'ROBOTS_DENIED'
            """
        )
        baidu_keys = [r[0] for r in cursor.fetchall()]
        if baidu_keys:
            cursor.execute(
                """
                UPDATE crawl_tasks
                SET status = 'PENDING', attempts = 0, error_type = NULL,
                    error = NULL, updated_at = ?
                WHERE domain = 'www.baidu.com' AND status = 'ROBOTS_DENIED'
                """,
                (now,),
            )
            cursor.executemany(
                """
                UPDATE fetch_targets
                SET status = 'PENDING', attempts = 0, error_type = NULL,
                    error = NULL, updated_at = ?
                WHERE fetch_key = ?
                """,
                [(now, k) for k in baidu_keys],
            )
        reset_summary["www.baidu.com (ROBOTS_DENIED)"] = len(baidu_keys)

        # 5. Optional transient network errors
        if args.include_network_errors:
            cursor.execute(
                """
                SELECT fetch_key FROM crawl_tasks
                WHERE status = 'FAILED' AND error_type = 'NETWORK_ERROR'
                """
            )
            net_keys = [r[0] for r in cursor.fetchall()]
            if net_keys:
                cursor.execute(
                    """
                    UPDATE crawl_tasks
                    SET status = 'PENDING', attempts = 0, error_type = NULL,
                        error = NULL, updated_at = ?
                    WHERE status = 'FAILED' AND error_type = 'NETWORK_ERROR'
                    """,
                    (now,),
                )
                cursor.executemany(
                    """
                    UPDATE fetch_targets
                    SET status = 'PENDING', attempts = 0, error_type = NULL,
                        error = NULL, updated_at = ?
                    WHERE fetch_key = ?
                    """,
                    [(now, k) for k in net_keys],
                )
            reset_summary["Transient NETWORK_ERROR (All domains)"] = len(net_keys)

    conn.close()

    total_reset = sum(reset_summary.values())
    print("=" * 65)
    print("RECOVERY RESET SUMMARY:")
    print("=" * 65)
    for category, count in reset_summary.items():
        print(f"  {category:<40}: {count:>8,} tasks reset")
    print("-" * 65)
    print(f"  {'TOTAL TASKS READY FOR RECOVERY':<40}: {total_reset:>8,}")
    print("=" * 65)
    print("\nNext step: Run crawler command to execute recovery crawl.")


if __name__ == "__main__":
    main()
