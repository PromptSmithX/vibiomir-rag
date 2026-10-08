from __future__ import annotations

import subprocess
import sys
import threading
import time

def stream_output(process: subprocess.Popen, prefix: str) -> None:
    try:
        if process.stdout is not None:
            for line in iter(process.stdout.readline, ""):
                if line:
                    sys.stdout.write(f"[{prefix}] {line}")
                    sys.stdout.flush()
    except Exception as exc:
        sys.stdout.write(f"[{prefix}] Stream error: {exc}\n")

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 70)
    print("KHỞI ĐỘNG CRAWL RECOVERY SONG SONG: WORKER 3 & WORKER 4")
    print("=" * 70)
    print("Worker 3: Phục hồi VOV (bỏ delay 20s), Báo Lạng Sơn, v.v. (Hoãn article.iiyi.com bị lỗi 521)")
    print("Worker 4: Phục hồi Hải Phòng, QĐND, Thanh Niên, Quảng Trị, 39.net")
    print("-" * 70)

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

    t3 = threading.Thread(target=stream_output, args=(p3, "W3"), daemon=True)
    t4 = threading.Thread(target=stream_output, args=(p4, "W4"), daemon=True)

    t3.start()
    t4.start()

    p3.wait()
    p4.wait()
    t3.join(timeout=5)
    t4.join(timeout=5)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"HOÀN THÀNH CẢ 2 WORKER TRONG {elapsed/60:.1f} PHÚT!")
    print(f"Worker 3 exit code: {p3.returncode}")
    print(f"Worker 4 exit code: {p4.returncode}")
    print("Bây giờ bạn có thể chạy: uv run python merge_recovery_into_workers.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
