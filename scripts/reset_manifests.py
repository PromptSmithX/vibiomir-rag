import sqlite3

for run_name in ['run-vn', 'run-zh']:
    path = f'data/crawl/runs/{run_name}/crawl_manifest.sqlite'
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("UPDATE crawl_tasks SET status='PENDING' WHERE status IN ('FETCHING', 'RETRY_WAIT')")
    con.commit()
    print(f"{run_name}: reset {cur.rowcount} tasks to PENDING")
    cur.execute("SELECT status, count(*) FROM crawl_tasks GROUP BY status")
    print(f"  current status breakdown: {cur.fetchall()}")
    con.close()
