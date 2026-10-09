from __future__ import annotations

import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path


def get_stats(wid: int) -> dict[str, int]:
    p = Path(f"data/crawl/runs/recovery-worker-{wid}/crawl_manifest.sqlite")
    if not p.exists():
        return {}
    try:
        conn = sqlite3.connect(f"file:{p.resolve()}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT status, COUNT(*) FROM crawl_tasks GROUP BY status")
        res = dict(cur.fetchall())
        conn.close()
        return res
    except Exception:
        return {}


def monitor_progress(stop_event: threading.Event, start_time: float) -> None:
    time.sleep(2)  # Wait for pipelines to initialize and recover tasks
    last_print = 0.0
    while not stop_event.is_set():
        now = time.time()
        if now - last_print >= 5.0:
            last_print = now
            elapsed_sec = int(now - start_time)
            m, s = divmod(elapsed_sec, 60)

            s3 = get_stats(3)
            s4 = get_stats(4)

            w3_success = s3.get("SUCCESS", 0)
            w3_pending = s3.get("PENDING", 0) + s3.get("FETCHING", 0) + s3.get("RETRY_WAIT", 0)
            w3_failed = s3.get("FAILED", 0)

            w4_success = s4.get("SUCCESS", 0)
            w4_pending = s4.get("PENDING", 0) + s4.get("FETCHING", 0) + s4.get("RETRY_WAIT", 0)
            w4_failed = s4.get("FAILED", 0) + s4.get("ROBOTS_DENIED", 0)

            print(
                f"[{m:02d}:{s:02d}] "
                f"W3 (VOV): SUCCESS={w3_success:>5,d} | PENDING={w3_pending:>5,d} | ERR={w3_failed:>5,d}  ||  "
                f"W4 (BHP & QDND): SUCCESS={w4_success:>5,d} | PENDING={w4_pending:>5,d} | ERR={w4_failed:>5,d}"
            )
            sys.stdout.flush()

        time.sleep(1.0)


def stream_logs(process: subprocess.Popen, prefix: str, log_file_path: Path) -> None:
    try:
        with log_file_path.open("a", encoding="utf-8", errors="replace") as f:
            if process.stdout is not None:
                for line in iter(process.stdout.readline, ""):
                    if line:
                        f.write(line)
                        f.flush()
                        # If error line or shard commit, also print to console
                        if "COMMITTED" in line or "Traceback" in line or "Error" in line:
                            sys.stdout.write(f"[{prefix}] {line.strip()}\n")
                            sys.stdout.flush()
    except Exception as exc:
        sys.stdout.write(f"[{prefix}] Log stream error: {exc}\n")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    w3_log = logs_dir / "recovery_w3.log"
    w4_log = logs_dir / "recovery_w4.log"

    print("=" * 75)
    print("KHỞI ĐỘNG CRAWL RECOVERY SONG SONG: WORKER 3 & WORKER 4")
    print("=" * 75)
    print("Worker 3: Phục hồi VOV (bỏ qua WAF PerimeterX bằng headers tối ưu)")
    print("Worker 4: Phục hồi Báo Hải Phòng & Báo QĐND (xử lý cookie 302 + bypass robots)")
    print(f"Chi tiết log được ghi tự động vào: {w3_log} & {w4_log}")
    print("-" * 75)

    cmd_w3 = [
        "uv",
        "run",
        "python",
        "-m",
        "crawler",
        "--run-name",
        "recovery-worker-3",
        "crawl",
        "run",
        "--defer-domain",
        "article.iiyi.com",
    ]

    cmd_w4 = [
        "uv",
        "run",
        "python",
        "-m",
        "crawler",
        "--run-name",
        "recovery-worker-4",
        "crawl",
        "run",
    ]

    start_time = time.time()

    p3 = subprocess.Popen(
        cmd_w3,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    p4 = subprocess.Popen(
        cmd_w4,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    stop_event = threading.Event()
    t_mon = threading.Thread(target=monitor_progress, args=(stop_event, start_time), daemon=True)
    t3 = threading.Thread(target=stream_logs, args=(p3, "W3", w3_log), daemon=True)
    t4 = threading.Thread(target=stream_logs, args=(p4, "W4", w4_log), daemon=True)

    t_mon.start()
    t3.start()
    t4.start()

    p3.wait()
    p4.wait()
    stop_event.set()
    t_mon.join(timeout=2)
    t3.join(timeout=2)
    t4.join(timeout=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"HOÀN THÀNH CẢ 2 WORKER TRONG {elapsed/60:.1f} PHÚT ({int(elapsed)}s)!")
    print(f"Worker 3 exit code: {p3.returncode}")
    print(f"Worker 4 exit code: {p4.returncode}")
    print("Bây giờ bạn có thể chạy: uv run python merge_recovery_into_workers.py")
    print("=" * 75)


if __name__ == "__main__":
    main()
