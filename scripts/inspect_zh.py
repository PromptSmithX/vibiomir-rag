import sqlite3

con = sqlite3.connect('data/crawl/runs/run-zh/crawl_manifest.sqlite')
print("=== Status breakdown in run-zh ===")
for row in con.execute("SELECT status, count(*) FROM crawl_tasks GROUP BY status"):
    print(" ", row)

print("\n=== Error type breakdown in run-zh ===")
for row in con.execute("SELECT status, error_type, http_status, count(*) FROM crawl_tasks WHERE status != 'SUCCESS' GROUP BY status, error_type, http_status ORDER BY count(*) DESC"):
    print(" ", row)

print("\n=== Domain breakdown in run-zh by status & error ===")
for row in con.execute("SELECT domain, status, error_type, count(*) FROM crawl_tasks GROUP BY domain, status, error_type ORDER BY count(*) DESC LIMIT 30"):
    print(" ", row)

con.close()
